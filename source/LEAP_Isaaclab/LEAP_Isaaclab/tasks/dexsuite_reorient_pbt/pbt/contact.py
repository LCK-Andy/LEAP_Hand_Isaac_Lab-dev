"""Measurement-only PBT adapter for native, force-gated LEAP contact goals.

Held goals per completed episode dominate selection. Only when BOTH windows
have zero held goals can measured matched-touch occupancy break the tie.
Occupancy is normalized by the maximum episode length, not survival time:
early termination contributes zeros for the unused horizon, never a bonus.
"""

import math
import time
import types


class ContactWindow:
    def __init__(self):
        self.consume()

    def consume(self):
        self.n = 0
        self.totals = [0.0] * 4
        self.started = None

    def add(self, infos, frame, now=None):
        stats = infos.get("contact_episodes")
        if not stats:
            return
        n, *values = stats
        if n <= 0 or len(values) != 4 or not all(math.isfinite(v) and v >= 0 for v in values):
            return
        if self.started is None:
            self.started = time.time() if now is None else now
        self.n += n
        self.totals = [a + b for a, b in zip(self.totals, values)]
        self.end_frame = frame

    def ready(self, cfg, frame, now=None):
        now = time.time() if now is None else now
        if self.started is not None and now - self.started > cfg.max_score_age_seconds:
            self.consume()
        if self.n < cfg.min_completed_episodes:
            return {}
        goals, goals_sq, progress, progress_sq = self.totals

        def variance(total, squares):
            return max(0.0, (squares - total**2 / self.n) / max(1, self.n - 1))

        return {
            "native_contact": {
                "completed_episodes": self.n,
                "success_rate": goals / self.n,
                "goals_per_episode": goals / self.n,
                "variance": variance(goals, goals_sq),
                "progress": progress / self.n,
                "progress_variance": variance(progress, progress_sq),
                "created_at": self.started,
                "end_frame": self.end_frame,
            }
        }


def select_contact_donor(records, own, cfg, now=None):
    """Uncertainty-aware lexicographic selection; no lifetime-frame comparisons."""
    now = time.time() if now is None else now
    ours = own.get("evaluations", {}).get("native_contact")
    if not ours:
        return None
    candidates = []
    for policy, record in records.items():
        if policy == cfg.policy_idx or not record or record.get("schema_version") != 2:
            continue
        theirs = record.get("evaluations", {}).get("native_contact")
        if not theirs:
            continue
        if any(not 0 <= now - r["created_at"] <= cfg.max_score_age_seconds for r in (own, record, ours, theirs)):
            continue
        if min(ours["completed_episodes"], theirs["completed_episodes"]) < cfg.min_completed_episodes:
            continue
        if any(abs(r["frame"] - g["end_frame"]) > cfg.max_frame_lag for r, g in ((own, ours), (record, theirs))):
            continue
        key, var = "goals_per_episode", "variance"
        margin = cfg.threshold_abs
        if ours[key] == theirs[key] == 0:
            key, var = "progress", "progress_variance"
            margin = cfg.contact_progress_threshold_abs
        se = math.sqrt(sum(g[var] / g["completed_episodes"] for g in (ours, theirs)))
        if theirs[key] - ours[key] > max(margin, cfg.threshold_std * se):
            candidates.append((theirs["goals_per_episode"], theirs["progress"], policy))
    return max(candidates)[2] if candidates else None


def attach_measurement(env):
    """Attach to ONE instance before reset, without replacing native MDP outputs."""
    import torch

    if env.cfg.touch_force <= 0 or env._physics_backend != "physx":
        raise ValueError("contact_goals requires the native PhysX measured-force task")
    native_rewards, native_reset = env._get_rewards, env._reset_idx
    touches = torch.zeros(env.num_envs, device=env.device)
    steps = torch.zeros_like(touches)

    def rewards(self):
        self.extras.pop("contact_episodes", None)
        for key in (
            "Contact/held_goals_per_completed_episode",
            "Contact/completed_touch_occupancy",
            "Contact/completed_episode_steps",
        ):
            self.extras.pop(key, None)
        # Native intermediates were refreshed by _get_dones before reward. Capture
        # matched touches now: native reward resamples goals and reset refreshes geometry.
        touches.add_(self.touched.mean(-1))
        steps.add_(1)
        self.extras["Contact/matched_touch_fraction"] = self.touched.mean().item()
        self.extras["Contact/pulp_force_mean_N"] = self.fingertip_contact_forces.norm(dim=-1).mean().item()
        return native_rewards()

    def reset(self, env_ids):
        valid = env_ids[self.reset_buf[env_ids] & (steps[env_ids] > 0)]
        if len(valid):
            goals = self.successes[valid].detach()
            progress = (touches[valid] / self.max_episode_length).clamp(0, 1)
            self.extras["contact_episodes"] = (
                len(valid),
                goals.sum().item(),
                goals.square().sum().item(),
                progress.sum().item(),
                progress.square().sum().item(),
            )
            self.extras["Contact/held_goals_per_completed_episode"] = goals.mean().item()
            self.extras["Contact/completed_touch_occupancy"] = progress.mean().item()
            self.extras["Contact/completed_episode_steps"] = steps[valid].mean().item()
        touches[env_ids] = 0
        steps[env_ids] = 0
        return native_reset(env_ids)

    env._get_rewards = types.MethodType(rewards, env)
    env._reset_idx = types.MethodType(reset, env)
    # No evolving curriculum. A restart begins new simulator episodes; partial
    # episode samples are intentionally not spliced across policy branches.
    env.get_env_state = lambda: None
    env.set_env_state = lambda state: None
