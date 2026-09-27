"""Import-safe Stage B boundary for the official TD-MPC2 learner.

The task owns physics, reward, and success semantics. Only finite-horizon
transition plumbing and complete-episode accounting are shared with Stage A.
"""

import torch

from .grasp import _FiniteHorizonGraspVectorEnv, _GraspEpisodeMetrics, _matches, _validate_grasp_cfg

_MOTION_CONTRACT = {
    "rotation_amplitude": (0.001745329252,) * 3,
    "translation_amplitude": (0.0,) * 3,
    "frequency_hz": (0.2, 0.3, 0.4),
    "settle_seconds": 1.0,
    "ramp_seconds": 2.0,
    "position_tolerance": 0.0005,
    "rotation_tolerance": 0.0007,
    "linear_velocity_tolerance": 0.002,
    "angular_velocity_tolerance": 0.002,
    "velocity_filter_alpha": 0.2,
    "tracking_reward_scale": 3.0,
    "success_hold_seconds": 1.0,
    "randomize_phase": True,
}


def _validate_cfg(cfg):
    _validate_grasp_cfg(cfg, obs_dim=148, stage="Stage B")
    for name, value in _MOTION_CONTRACT.items():
        if not _matches(getattr(cfg.motion, name, None), value):
            raise ValueError(f"Stage B requires native motion.{name}={value}")


def configure_grasp_motion(cfg, seed):
    """Validate fixed native task semantics; change only seed and final-observation capture."""
    _validate_cfg(cfg)
    cfg.seed = seed
    cfg.compute_final_obs = True
    return cfg


class GraspMotionVectorEnv(_FiniteHorizonGraspVectorEnv):
    """Native 148-state/16-action Stage B with asynchronous same-step resets."""

    obs_dim = 148
    _stage = "Stage B"
    _condition_prefix = "grasp_motion_v1_"
    _report_group = "grasp_motion"
    _report_groups = ("grasp_motion", "grasp_hold")
    _success_key = "completed_tracking_success"
    _info_key = "tdmpc2_grasp_motion"
    _validate_cfg = staticmethod(_validate_cfg)

    def __init__(self, env):
        super().__init__(env)
        if getattr(self.env, "grasp_motion_contract_version", None) != 1:
            raise ValueError("Stage B native runtime contract changed")

    def _native_success(self):
        return self.env._tracking_success


class GraspMotionEpisodeMetrics(_GraspEpisodeMetrics):
    """Episode-weighted native Stage B measurements, with a bounded recent window.

    Success is the task's sticky one-second tracking hold, not grasp success or an
    occupancy threshold. Partial episodes are discarded on simulator restart.
    """

    _stage = "Stage B"
    _condition_prefix = "grasp_motion_v1_"
    _metric_prefix = "grasp_motion"
    _fields = (
        "return",
        "length",
        "tracking_fraction",
        "retained",
        "valid_fraction",
        "overload_fraction",
        "retention_weighted_tracking",
    )

    def _derive(self, row):
        row["retention_weighted_tracking"] = row["tracking_fraction"] * row["retained"]

    def _report(self, infos, count):
        motion, grasp = infos.get("grasp_motion", {}), infos.get("grasp_hold", {})
        condition = motion.get("condition_key")
        if (
            not isinstance(condition, str)
            or not condition.startswith("grasp_motion_v1_")
            or grasp.get("condition_key") != condition
            or self.condition_key not in (None, condition)
        ):
            raise ValueError("Native Stage B report physical conditions differ or are missing")
        tensors = {}
        for name, source, key in (
            ("tracking_fraction", motion, "completed_tracking_fraction"),
            ("retained", motion, "completed_retained"),
            ("valid_fraction", grasp, "completed_valid_fraction"),
            ("overload_fraction", motion, "completed_overload_fraction"),
            ("success", motion, "completed_tracking_success"),
            ("grasp_retained", grasp, "completed_retained"),
            ("grasp_overload", grasp, "completed_overload_fraction"),
        ):
            value = source.get(key)
            if not isinstance(value, torch.Tensor) or value.shape != (count,):
                raise ValueError(f"Native Stage B report {key} must have exactly {count} completed entries")
            value = value.detach().to(device="cpu", dtype=torch.float64)
            if not torch.isfinite(value).all() or not ((value >= 0) & (value <= 1)).all():
                raise ValueError(f"Native Stage B report {key} must be finite and within [0, 1]")
            if name in {"retained", "grasp_retained", "success"} and not ((value == 0) | (value == 1)).all():
                raise ValueError(f"Native Stage B report {key} must be boolean")
            tensors[name] = value
        if not torch.equal(tensors["retained"], tensors.pop("grasp_retained")) or not torch.equal(
            tensors["overload_fraction"], tensors.pop("grasp_overload")
        ):
            raise ValueError("Native Stage B motion/grasp report retention or overload disagrees")
        return {name: value.tolist() for name, value in tensors.items()}, condition
