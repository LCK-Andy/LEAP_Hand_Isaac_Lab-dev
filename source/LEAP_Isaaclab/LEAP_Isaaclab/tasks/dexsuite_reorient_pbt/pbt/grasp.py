"""Completed-episode Stage A grasp selection, independent of shaped reward.

The environment emits ``extras['grasp_hold']`` only for ended episodes, with a
versioned ``condition_key`` and equal-length 1-D tensors:
``completed_valid_fraction``, ``completed_retained``, and
``completed_overload_fraction``. ``completed_retained`` means fully retained
through the nominal horizon AND support-free throughout the episode, as tracked
by the environment; raw survival is diagnostic only. Fractions use the FULL
nominal episode horizon, not survival time. An episode with any overload receives
no retention/validity credit. The environment clears this dict before the next
control step.
"""

import math
import time


def validate_grasp_environment(task, env):
    """Require the Stage A measurement/state contract, without changing its MDP."""
    if task != "Isaac-Grasp-Cube-LEAP-Direct-v0":
        raise ValueError("grasp_hold requires the native Stage A grasp training task")
    if getattr(env, "_physics_backend", None) not in {"physx", "newton"}:
        raise ValueError("grasp_hold backend must support the full grasp measurement contract")
    if getattr(env, "grasp_hold_contract_version", None) != 1:
        raise ValueError("grasp_hold requires environment measurement contract version 1")
    if not all(callable(getattr(env, name, None)) for name in ("get_env_state", "set_env_state")):
        raise ValueError("grasp_hold requires explicit environment checkpoint state hooks")


class GraspHoldWindow:
    """Episode-weighted sufficient statistics for one physical condition."""

    def __init__(self):
        self.consume()

    def consume(self):
        self.n = 0
        self.totals = [0.0] * 5
        self.started = None
        self.condition_key = None

    def add(self, infos, frame, now=None):
        import torch

        stats = infos.get("grasp_hold")
        if not stats:
            return
        if not isinstance(stats, dict) or not isinstance(stats.get("condition_key"), str) or not stats["condition_key"]:
            raise ValueError("grasp_hold requires a nonempty condition_key")
        keys = ("completed_valid_fraction", "completed_retained", "completed_overload_fraction")
        tensors = [stats.get(key) for key in keys]
        if any(not isinstance(value, torch.Tensor) or value.ndim != 1 for value in tensors):
            raise ValueError("grasp_hold completed metrics must be 1-D tensors")
        if len({(value.shape, value.device) for value in tensors}) != 1:
            raise ValueError("grasp_hold completed metrics must have matching shapes and devices")
        if any(not bool((torch.isfinite(value) & (value >= 0) & (value <= 1)).all()) for value in tensors):
            raise ValueError("grasp_hold completed metrics must be finite fractions in [0, 1]")
        valid, retained, overload = [value.detach().double() for value in tensors]
        if not bool(((retained == 0) | (retained == 1)).all()):
            raise ValueError("grasp_hold completed_retained must be binary")
        safe = overload == 0
        retained = retained * safe
        valid = valid * safe
        if not retained.numel():
            return
        if self.condition_key != stats["condition_key"]:
            self.consume()
        if self.started is None:
            self.started = time.time() if now is None else now
        self.condition_key = stats["condition_key"]
        self.n += retained.numel()
        values = (retained.sum(), retained.square().sum(), valid.sum(), valid.square().sum(), (~safe).sum())
        self.totals = [a + b.item() for a, b in zip(self.totals, values)]
        self.end_frame = frame

    def ready(self, cfg, frame, now=None):
        now = time.time() if now is None else now
        if self.started is not None and (
            not 0 <= now - self.started <= cfg.max_score_age_seconds
            or not 0 <= frame - self.end_frame <= cfg.max_frame_lag
        ):
            self.consume()
        if self.n < cfg.min_completed_episodes:
            return {}
        retained, retained_sq, valid, valid_sq, overloaded = self.totals

        def variance(total, squares):
            return max(0.0, (squares - total**2 / self.n) / max(1, self.n - 1))

        return {
            "grasp_hold": {
                "condition_key": self.condition_key,
                "completed_episodes": self.n,
                "success_rate": retained / self.n,
                "variance": variance(retained, retained_sq),
                "progress": valid / self.n,
                "progress_variance": variance(valid, valid_sq),
                "overload_rate": overloaded / self.n,
                "created_at": self.started,
                "end_frame": self.end_frame,
            }
        }


def select_grasp_donor(records, own, cfg, now=None):
    """Compare overload-free, support-free retention; exact ties use safe valid4 occupancy.

    Never trade a higher observed overload incidence for better retention, and
    never use occupancy to overturn an inconclusive retention comparison.
    Asynchronous members need fresh local windows, not matching lifetime frames.
    """
    now = time.time() if now is None else now
    ours = own.get("evaluations", {}).get("grasp_hold")
    if not ours:
        return None
    candidates = []
    for policy, record in records.items():
        if policy == cfg.policy_idx or not record or record.get("schema_version") != 2:
            continue
        theirs = record.get("evaluations", {}).get("grasp_hold")
        if not theirs or theirs["condition_key"] != ours["condition_key"]:
            continue
        if any(not 0 <= now - item["created_at"] <= cfg.max_score_age_seconds for item in (own, record, ours, theirs)):
            continue
        if min(ours["completed_episodes"], theirs["completed_episodes"]) < cfg.min_completed_episodes:
            continue
        if any(not 0 <= r["frame"] - g["end_frame"] <= cfg.max_frame_lag for r, g in ((own, ours), (record, theirs))):
            continue
        if theirs["overload_rate"] > ours["overload_rate"]:
            continue
        key, variance = "success_rate", "variance"
        if theirs[key] == ours[key]:
            key, variance = "progress", "progress_variance"
        se = math.sqrt(sum(item[variance] / item["completed_episodes"] for item in (ours, theirs)))
        if theirs[key] - ours[key] > max(cfg.threshold_abs, cfg.threshold_std * se):
            candidates.append((theirs["success_rate"], theirs["progress"], -theirs["overload_rate"], policy))
    return max(candidates)[-1] if candidates else None
