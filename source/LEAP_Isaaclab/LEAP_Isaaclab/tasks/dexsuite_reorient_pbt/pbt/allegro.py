"""Measurement-only adapter and count-valued selection for the official Allegro task.

No reward, action, observation or reset result is replaced. Selection uses goals
per completed episode, NOT the native exponentially smoothed logging statistic.
"""

import math
import time
import types


class GoalWindow:
    def __init__(self):
        self.consume()

    def consume(self):
        self.n = 0
        self.total = self.squares = 0.0
        self.started = None

    def add(self, infos, frame, now=None):
        stats = infos.get("allegro_episodes")
        if not stats:
            return
        n, total, squares = stats
        if n <= 0 or not all(math.isfinite(x) and x >= 0 for x in (total, squares)):
            return
        if self.started is None:
            self.started = time.time() if now is None else now
        self.n += n
        self.total += total
        self.squares += squares
        self.end_frame = frame

    def ready(self, cfg, frame, now=None):
        now = time.time() if now is None else now
        if self.started is not None and now - self.started > cfg.max_score_age_seconds:
            self.consume()
        if self.n < cfg.min_completed_episodes:
            return {}
        mean = self.total / self.n
        variance = max(0.0, (self.squares - self.total**2 / self.n) / max(1, self.n - 1))
        return {
            "official_allegro": {
                "completed_episodes": self.n,
                "success_rate": mean,
                "goals_per_episode": mean,
                "variance": variance,
                "progress": 0.0,
                "created_at": self.started,
                "end_frame": self.end_frame,
            }
        }


def select_goal_donor(records, own, cfg, now=None):
    """Count standard error, wall-clock freshness, and sample-own age for async peers.

    Deliberately do not reject a faster/slower peer based on total lifetime frames.
    Each score must instead be fresh relative to its own checkpoint publication.
    """
    now = time.time() if now is None else now
    ours = own.get("evaluations", {}).get("official_allegro")
    if not ours:
        return None
    candidates = []
    for policy, record in records.items():
        if policy == cfg.policy_idx or not record or record.get("schema_version") != 2:
            continue
        theirs = record.get("evaluations", {}).get("official_allegro")
        if not theirs:
            continue
        if any(not 0 <= now - r["created_at"] <= cfg.max_score_age_seconds for r in (own, record)):
            continue
        if any(not 0 <= now - g["created_at"] <= cfg.max_score_age_seconds for g in (ours, theirs)):
            continue
        if min(ours["completed_episodes"], theirs["completed_episodes"]) < cfg.min_completed_episodes:
            continue
        if any(abs(r["frame"] - g["end_frame"]) > cfg.max_frame_lag for r, g in ((own, ours), (record, theirs))):
            continue
        se = math.sqrt(sum(g["variance"] / g["completed_episodes"] for g in (ours, theirs)))
        if theirs["goals_per_episode"] - ours["goals_per_episode"] > max(cfg.threshold_abs, cfg.threshold_std * se):
            candidates.append((theirs["goals_per_episode"], policy))
    return max(candidates)[1] if candidates else None


def attach_measurement(env):
    """Install instance-local hooks after construction, before the initial reset."""
    native_rewards, native_reset = env._get_rewards, env._reset_idx

    def rewards(self):
        self.extras.pop("allegro_episodes", None)
        return native_rewards()

    def reset(self, env_ids):
        # Initial/manual resets are not completed episodes. DirectRLEnv sets
        # reset_buf before reward, then calls reset_idx after reward updates goals.
        valid = env_ids[self.reset_buf[env_ids]]
        if len(valid):
            counts = self.successes[valid].detach()
            n = len(valid)
            self.extras["allegro_episodes"] = (n, counts.sum().item(), counts.square().sum().item())
            self.extras.setdefault("log", {})["Allegro/goals_per_completed_episode"] = counts.mean()
        return native_reset(env_ids)

    env._get_rewards = types.MethodType(rewards, env)
    env._reset_idx = types.MethodType(reset, env)
    # The official task has no evolving curriculum. Do not import Dexsuite state.
    env.get_env_state = lambda: None
    env.set_env_state = lambda state: None
