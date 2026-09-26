"""Simulator-independent control and measurement contract for persistent grasping."""

import hashlib
import json
import torch
from dataclasses import dataclass


def grasp_condition_key(condition: dict) -> str:
    """Stable comparison identity for resolved physical and task conditions."""
    encoded = json.dumps(condition, sort_keys=True, allow_nan=False).encode()
    return "grasp_hold_v1_" + hashlib.sha256(encoded).hexdigest()


@dataclass
class GraspCriteria:
    """Physical thresholds and bounded reward weights, in sensor/finger-name order."""

    force_low: tuple[float, ...] = (0.5, 1.0, 0.5, 0.5)
    force_high: tuple[float, ...] = (8.0, 16.0, 8.0, 8.0)
    force_limit: tuple[float, ...] = (16.0, 24.0, 16.0, 16.0)
    friction_coefficient: float = 1.0
    friction_utilization_limit: float = 0.9
    slip_speed_limit: float = 0.02
    support_force_limit: float = 0.1
    retention_radius: float = 0.08
    approach_length: float = 0.02
    mean_weight: float = 0.5
    weakest_weight: float = 1.0
    valid_weight: float = 2.0
    approach_weight: float = 0.1
    retention_weight: float = 0.1
    overload_weight: float = 2.0
    support_weight: float = 1.0
    slip_weight: float = 0.2
    target_delta_weight: float = 0.01


@dataclass
class GraspScore:
    reward: torch.Tensor
    valid: torch.Tensor
    unsafe: torch.Tensor
    overload: torch.Tensor
    finger_valid: torch.Tensor
    quality: torch.Tensor
    terms: dict[str, torch.Tensor]


def score_grasp(
    normal_forces: torch.Tensor,
    friction_forces: torch.Tensor,
    pulp_contact: torch.Tensor,
    surface_distance: torch.Tensor,
    slip_speed: torch.Tensor,
    support_force: torch.Tensor,
    object_distance: torch.Tensor,
    target_delta: torch.Tensor,
    cfg: GraspCriteria,
) -> GraspScore:
    """Score a control interval; forces are (env, physics sample, finger, xyz)."""
    finite = torch.ones(normal_forces.shape[0], dtype=torch.bool, device=normal_forces.device)
    for data in (
        normal_forces,
        friction_forces,
        surface_distance,
        slip_speed,
        support_force,
        object_distance,
        target_delta,
    ):
        finite &= torch.isfinite(data).reshape(data.shape[0], -1).all(dim=-1)
    normal_forces = torch.nan_to_num(normal_forces, nan=0.0, posinf=0.0, neginf=0.0)
    friction_forces = torch.nan_to_num(friction_forces, nan=0.0, posinf=0.0, neginf=0.0)
    surface_distance = torch.nan_to_num(surface_distance, nan=1.0, posinf=1.0, neginf=1.0)
    slip_speed = torch.nan_to_num(slip_speed, nan=1.0, posinf=1.0, neginf=1.0)
    support_force = torch.nan_to_num(support_force, nan=1.0, posinf=1.0, neginf=1.0)
    object_distance = torch.nan_to_num(object_distance, nan=1.0, posinf=1.0, neginf=1.0)
    target_delta = torch.nan_to_num(target_delta, nan=0.0, posinf=0.0, neginf=0.0)
    low = normal_forces.new_tensor(cfg.force_low)
    high = normal_forces.new_tensor(cfg.force_high)
    limit = normal_forces.new_tensor(cfg.force_limit)
    normal = torch.linalg.vector_norm(normal_forces, dim=-1)
    friction = torch.linalg.vector_norm(friction_forces, dim=-1)
    in_band = ((normal >= low) & (normal <= high)).all(dim=1)
    friction_ok = (friction <= cfg.friction_utilization_limit * cfg.friction_coefficient * normal).all(dim=1)
    finger_valid = in_band & friction_ok & pulp_contact & (slip_speed <= cfg.slip_speed_limit)
    retained = object_distance < cfg.retention_radius
    supported = support_force.amax(dim=1) > cfg.support_force_limit
    valid = finger_valid.all(dim=-1) & retained & ~supported & finite
    overload = (normal > high).any(dim=(1, 2)) | ~finite
    total_force = torch.linalg.vector_norm(normal_forces + friction_forces, dim=-1)
    unsafe = (total_force > limit).any(dim=(1, 2)) | ~finite
    overload |= unsafe
    valid &= ~unsafe
    quality = torch.minimum(normal / low, high / normal.clamp_min(1e-6)).clamp(0.0, 1.0).amin(dim=1)
    quality = quality * pulp_contact.float()
    approach = torch.exp(-surface_distance.clamp_min(0.0) / cfg.approach_length).mean(dim=-1)
    excess = ((normal / high - 1.0).clamp(0.0, 1.0)).amax(dim=1).mean(dim=-1)
    terms = {
        "mean_contact": cfg.mean_weight * quality.mean(dim=-1),
        "weakest_contact": cfg.weakest_weight * quality.amin(dim=-1),
        "valid_hold": cfg.valid_weight * valid.float(),
        "approach": cfg.approach_weight * approach * retained.float(),
        "retention": cfg.retention_weight * retained.float(),
        "overload": -cfg.overload_weight * excess,
        "support": -cfg.support_weight * supported.float(),
        "slip": -cfg.slip_weight * (slip_speed / cfg.slip_speed_limit).clamp(0.0, 1.0).mean(dim=-1),
        "target_delta": -cfg.target_delta_weight * target_delta.square().mean(dim=-1),
    }
    return GraspScore(sum(terms.values()), valid, unsafe, overload, finger_valid, quality, terms)


class GraspEpisodeState:
    """Persistent hold and horizon-normalized completed-episode measurements."""

    def __init__(self, num_envs: int, device: str | torch.device):
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.valid_steps = torch.zeros_like(self.steps)
        self.overload_steps = torch.zeros_like(self.steps)
        self.hold_seconds = torch.zeros(num_envs, device=device)
        self.longest_hold = torch.zeros_like(self.hold_seconds)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.support_seen = torch.zeros_like(self.success)

    def advance(
        self,
        valid: torch.Tensor,
        overload: torch.Tensor,
        dt: float,
        hold_seconds: float,
        supported: torch.Tensor | None = None,
    ) -> None:
        if supported is not None:
            self.support_seen |= supported
        self.steps += 1
        self.valid_steps += valid.long()
        self.overload_steps += overload.long()
        self.hold_seconds = torch.where(valid, self.hold_seconds + dt, 0.0)
        self.longest_hold = torch.maximum(self.longest_hold, self.hold_seconds)
        self.success |= self.hold_seconds >= hold_seconds - 1e-6

    def completed(self, env_ids: torch.Tensor, retained: torch.Tensor, horizon: int) -> dict[str, torch.Tensor]:
        return {
            "completed_valid_fraction": (self.valid_steps[env_ids].float() / horizon).clamp_max(1.0),
            "completed_retained": (retained & ~self.support_seen[env_ids]).float(),
            "completed_overload_fraction": (self.overload_steps[env_ids].float() / horizon).clamp_max(1.0),
        }

    def reset(self, env_ids: torch.Tensor) -> None:
        for tensor in (
            self.steps,
            self.valid_steps,
            self.overload_steps,
            self.hold_seconds,
            self.longest_hold,
            self.success,
            self.support_seen,
        ):
            tensor[env_ids] = 0


def advance_joint_targets(
    previous: torch.Tensor,
    actions: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    max_delta: float,
) -> torch.Tensor:
    """Integrate one bounded target increment per control step, without accumulating windup."""
    return torch.clamp(previous + actions.clamp(-1.0, 1.0) * max_delta, min=lower, max=upper)
