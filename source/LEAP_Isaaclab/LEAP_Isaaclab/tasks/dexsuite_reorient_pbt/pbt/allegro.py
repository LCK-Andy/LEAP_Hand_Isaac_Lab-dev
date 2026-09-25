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
            self.start_frame = frame
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
                "start_frame": self.start_frame,
                "end_frame": self.end_frame,
            }
        }


def select_goal_donor(records, own, cfg, now=None, *, evidence=None, diagnostics=None):
    """Select on fresh count windows; SE margins are heuristics, NOT significance tests.

    Lifetime frame gaps never disqualify asynchronous peers. Two-window evidence
    requires new nonoverlapping windows on BOTH sides, on the same branches.
    ``evidence`` is recipient-owned mutable state persisted with its publication.
    """
    now = time.time() if now is None else now
    evidence = {} if evidence is None else evidence
    diagnostics = [] if diagnostics is None else diagnostics
    ours = own.get("evaluations", {}).get("official_allegro")

    def invalid(record):
        if not record or record.get("schema_version") != 2:
            return "missing_record"
        stats = record.get("evaluations", {}).get("official_allegro")
        if not stats:
            return "missing_window"
        if any(not 0 <= now - r["created_at"] <= cfg.max_score_age_seconds for r in (record, stats)):
            return "stale"
        if stats["completed_episodes"] < cfg.min_completed_episodes:
            return "sample_count"
        if abs(record["frame"] - stats["end_frame"]) > cfg.max_frame_lag:
            return "sample_age"
        if not all(math.isfinite(stats[k]) and stats[k] >= 0 for k in ("goals_per_episode", "variance")):
            return "invalid_score"
        return None

    # Rank only comparable fresh windows. Maturity affects donor eligibility, not
    # population ranking: a strong immature peer must not make a mid-rank recipient weakest.
    peers = {p: r for p, r in records.items() if p != cfg.policy_idx and invalid(r) is None}
    own_reason = invalid(own)
    ranked = {p: r["evaluations"]["official_allegro"]["goals_per_episode"] for p, r in peers.items()}
    if not own_reason:
        ranked[cfg.policy_idx] = ours["goals_per_episode"]
    role_ok = True
    if getattr(cfg, "recipient_role", "any") == "weakest" and not own_reason:
        role_ok = len(ranked) > 1 and ours["goals_per_episode"] == min(ranked.values()) < max(ranked.values())
    candidates = []
    for policy, record in records.items():
        if policy == cfg.policy_idx:
            continue
        theirs = (record or {}).get("evaluations", {}).get("official_allegro")
        detail = {"donor": policy, "reason": own_reason or invalid(record), "accepted": False}
        if ours and theirs:
            detail.update(
                recipient_count=ours["completed_episodes"],
                donor_count=theirs["completed_episodes"],
                score_gap=theirs["goals_per_episode"] - ours["goals_per_episode"],
            )
            if min(ours["completed_episodes"], theirs["completed_episodes"]) > 0:
                se = math.sqrt(max(0, sum(g["variance"] / g["completed_episodes"] for g in (ours, theirs))))
                detail.update(combined_se=se, margin=max(cfg.threshold_abs, cfg.threshold_std * se))
        key = str(policy)
        if not detail["reason"]:
            maturity = getattr(cfg, "donor_maturity_rollouts", 0)
            rollout = record.get("rollout_frames", 0)
            age = record["frame"] - record.get("last_exploit_frame", 0)
            detail["donor_branch_frames"] = age
            if maturity and (rollout <= 0 or age < maturity * rollout or record["frame"] < cfg.warmup_steps):
                detail["reason"] = "donor_immature"
            elif not role_ok:
                detail["reason"] = "recipient_not_weakest"
            elif detail["score_gap"] <= detail["margin"]:
                detail["reason"] = "margin"
            elif getattr(cfg, "evidence_windows", 1) == 2:
                previous = evidence.get(key)
                token = {
                    "recipient_end": ours["end_frame"],
                    "donor_end": theirs["end_frame"],
                    "recipient_branch": own.get("last_exploit_frame", 0),
                    "donor_branch": record.get("last_exploit_frame", 0),
                    "created_at": min(r["created_at"] for r in (own, record, ours, theirs)),
                }
                fresh = previous and 0 <= now - previous["created_at"] <= cfg.max_score_age_seconds
                same = fresh and all(previous[k] == token[k] for k in ("recipient_branch", "donor_branch"))
                nonoverlap = same and (
                    ours.get("start_frame", -1) > previous["recipient_end"]
                    and theirs.get("start_frame", -1) > previous["donor_end"]
                )
                if not nonoverlap:
                    detail["reason"] = "awaiting_fresh_evidence"
                    # Never advance evidence on a reused donor window.
                    if not same:
                        evidence[key] = token
        if detail["reason"] not in (None, "awaiting_fresh_evidence"):
            evidence.pop(key, None)
        if detail["reason"] is None:
            candidates.append((theirs["goals_per_episode"], policy, detail))
        diagnostics.append(detail)
    selected = max(candidates, key=lambda item: item[:2])[1] if candidates else None
    for _, policy, detail in candidates:
        detail.update(
            accepted=policy == selected, reason="accepted" if policy == selected else "lower_qualifying_donor"
        )
    return selected


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
