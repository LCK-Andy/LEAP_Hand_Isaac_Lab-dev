"""Simulator-free bounds for the Cartesian arm action interface."""

import math
import torch

ARM_JOINT_NAMES = tuple(f"iiwa7_joint_{index}" for index in range(1, 8))
FINGER_JOINT_NAMES = tuple(
    f"{finger}_joint_{index}" for finger in ("index", "middle", "ring", "thumb") for index in range(4)
)
ACTION_DIM = 22
CONTROLLER_OBS_DIM = 22
OBS_DIM = 1762
CONTROL_VERSION = "palm_delta_dls_v1"


def bounded_pose_delta(actions: torch.Tensor, translation_scale: float, rotation_scale: float) -> torch.Tensor:
    """Bound translation and axis-angle increments by separate Euclidean radii."""
    if actions.ndim < 1 or actions.shape[-1] != 6:
        raise ValueError("Palm actions must have six channels")
    if not all(math.isfinite(value) and value > 0 for value in (translation_scale, rotation_scale)):
        raise ValueError("Palm translation and rotation scales must be finite and positive")
    actions = actions.clamp(-1.0, 1.0)
    translation, rotation = actions[..., :3], actions[..., 3:]
    translation = translation / translation.norm(dim=-1, keepdim=True).clamp_min(1.0)
    rotation = rotation / rotation.norm(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.cat((translation * translation_scale, rotation * rotation_scale), dim=-1)


def bounded_joint_target(
    desired: torch.Tensor,
    current: torch.Tensor,
    previous: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    max_step: torch.Tensor,
    max_tracking_error: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prioritize position and command-slew bounds over the soft tracking-error bound.

    A contact can move the measured joint faster than the command slew limit. In
    that case both slew and tracking-error bounds cannot necessarily be satisfied.
    These are target bounds, not bounds on actual physical velocity or torque.
    """
    tracking_bounded = torch.maximum(
        torch.minimum(desired, current + max_tracking_error),
        current - max_tracking_error,
    )
    position_bounded = torch.maximum(torch.minimum(tracking_bounded, upper), lower)
    previous = torch.maximum(torch.minimum(previous, upper), lower)
    target = previous + torch.maximum(torch.minimum(position_bounded - previous, max_step), -max_step)
    target = torch.maximum(torch.minimum(target, upper), lower)
    limited = (target - desired).abs().amax(dim=-1) > 1.0e-6
    return target, limited
