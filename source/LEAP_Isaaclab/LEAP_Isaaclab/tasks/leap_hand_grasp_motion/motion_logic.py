"""Simulator-independent smooth object references and grasp-gated tracking."""

import math
import torch
from dataclasses import dataclass


@dataclass
class MotionCriteria:
    """Reference offsets are expressed in the fixed initial object frame."""

    rotation_amplitude: tuple[float, float, float] = (0.001745329252, 0.001745329252, 0.001745329252)
    translation_amplitude: tuple[float, float, float] = (0.0, 0.0, 0.0)
    frequency_hz: tuple[float, float, float] = (0.20, 0.30, 0.40)
    settle_seconds: float = 1.0
    ramp_seconds: float = 2.0
    position_tolerance: float = 0.0005
    rotation_tolerance: float = 0.0007
    linear_velocity_tolerance: float = 0.002
    angular_velocity_tolerance: float = 0.002
    velocity_filter_alpha: float = 0.2
    tracking_reward_scale: float = 3.0
    success_hold_seconds: float = 1.0
    randomize_phase: bool = True


def validate_motion_criteria(cfg: MotionCriteria) -> None:
    for name in ("rotation_amplitude", "translation_amplitude", "frequency_hz"):
        values = getattr(cfg, name)
        if len(values) != 3 or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"{name} must contain three finite nonnegative values")
    if any(f <= 0 for f in cfg.frequency_hz):
        raise ValueError("Each excitation frequency must be positive")
    for name in (
        "ramp_seconds",
        "position_tolerance",
        "rotation_tolerance",
        "linear_velocity_tolerance",
        "angular_velocity_tolerance",
        "success_hold_seconds",
    ):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(cfg.settle_seconds) or cfg.settle_seconds < 0:
        raise ValueError("settle_seconds must be finite and nonnegative")
    if not 0 < cfg.velocity_filter_alpha <= 1:
        raise ValueError("velocity_filter_alpha must be inside (0,1]")
    if not any(cfg.rotation_amplitude) and not any(cfg.translation_amplitude):
        raise ValueError("Stage B requires a nonzero motion command")


@dataclass
class MotionReference:
    position: torch.Tensor
    quaternion: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    linear_acceleration: torch.Tensor
    angular_acceleration: torch.Tensor
    phase: torch.Tensor
    envelope: torch.Tensor
    active: torch.Tensor


@dataclass
class TrackingScore:
    reward: torch.Tensor
    tracking_valid: torch.Tensor
    position_error: torch.Tensor
    rotation_error: torch.Tensor
    linear_velocity_error: torch.Tensor
    angular_velocity_error: torch.Tensor


def rotation_error_vector(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Shortest actual-to-target axis-angle error in fixed/world axes (xyzw)."""
    inverse = actual * actual.new_tensor((-1.0, -1.0, -1.0, 1.0))
    difference = multiply_quaternions(target, inverse)
    difference = difference / difference.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    difference = torch.where(difference[..., 3:] < 0, -difference, difference)
    sine = difference[..., :3].norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sine, difference[..., 3:].clamp_min(0.0))
    scale = torch.where(sine > 1e-8, angle / sine.clamp_min(1e-12), 2.0)
    return difference[..., :3] * scale


class PoseMotionState:
    """Once-per-control-step pose differences, with explicit reset and EMA state."""

    def __init__(self, num_envs: int, device):
        self.position = torch.zeros(num_envs, 3, device=device)
        self.quaternion = torch.zeros(num_envs, 4, device=device)
        self.quaternion[:, 3] = 1.0
        self.linear_velocity = torch.zeros_like(self.position)
        self.angular_velocity = torch.zeros_like(self.position)
        self.angular_acceleration = torch.zeros_like(self.position)
        self.valid = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._previous_finite = torch.ones_like(self.valid)

    def reset(self, ids: torch.Tensor, position: torch.Tensor, quaternion: torch.Tensor):
        self.position[ids] = position
        self.quaternion[ids] = quaternion
        self.linear_velocity[ids] = 0.0
        self.angular_velocity[ids] = 0.0
        self.angular_acceleration[ids] = 0.0
        self.valid[ids] = True
        self._previous_finite[ids] = True

    def update(self, position: torch.Tensor, quaternion: torch.Tensor, dt: float, alpha: float):
        finite = torch.isfinite(position).all(-1) & torch.isfinite(quaternion).all(-1)
        finite &= quaternion.norm(dim=-1) > 0.5
        self.valid = finite & self._previous_finite
        p = torch.where(finite[:, None], position, self.position)
        q = torch.where(finite[:, None], quaternion, self.quaternion)
        self._previous_finite[:] = finite
        linear = (p - self.position) / dt
        angular = rotation_error_vector(self.quaternion, q) / dt
        old_angular = self.angular_velocity.clone()
        self.linear_velocity = alpha * linear + (1.0 - alpha) * self.linear_velocity
        self.angular_velocity = alpha * angular + (1.0 - alpha) * old_angular
        self.angular_acceleration = (self.angular_velocity - old_angular) / dt
        for value in (self.linear_velocity, self.angular_velocity, self.angular_acceleration):
            value[~self.valid] = 0.0
        self.position[:] = p
        self.quaternion[:] = q


def score_tracking(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    reference: MotionReference,
    grasp_valid: torch.Tensor,
    cfg: MotionCriteria,
) -> TrackingScore:
    """Bounded pose/velocity tracking bonus only during active, valid grasp windows."""
    p_error = (reference.position - position).norm(dim=-1)
    r_error = rotation_error_vector(quaternion, reference.quaternion).norm(dim=-1)
    v_error = (reference.linear_velocity - linear_velocity).norm(dim=-1)
    w_error = (reference.angular_velocity - angular_velocity).norm(dim=-1)
    errors = torch.stack(
        (
            p_error / cfg.position_tolerance,
            r_error / cfg.rotation_tolerance,
            v_error / cfg.linear_velocity_tolerance,
            w_error / cfg.angular_velocity_tolerance,
        ),
        -1,
    )
    finite = torch.isfinite(errors).all(-1)
    allowed = grasp_valid & reference.active & finite
    tracking_valid = allowed & (errors <= 1.0).all(-1)
    quality = torch.exp(-0.5 * torch.nan_to_num(errors, nan=1e6).square().clamp_max(1e6).mean(-1))
    reward = cfg.tracking_reward_scale * quality * allowed.float()
    return TrackingScore(reward, tracking_valid, p_error, r_error, v_error, w_error)


def multiply_quaternions(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product with xyzw ordering, preserving broadcast batch dimensions."""
    xyz = left[..., 3:] * right[..., :3] + right[..., 3:] * left[..., :3]
    xyz = xyz + torch.cross(left[..., :3], right[..., :3], dim=-1)
    scalar = left[..., 3:] * right[..., 3:] - (left[..., :3] * right[..., :3]).sum(-1, keepdim=True)
    return torch.cat((xyz, scalar), dim=-1)


def rotate_vectors(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit xyzw quaternions."""
    cross = 2.0 * torch.cross(quaternion[..., :3], vector, dim=-1)
    return vector + quaternion[..., 3:] * cross + torch.cross(quaternion[..., :3], cross, dim=-1)


def motion_reference(time: torch.Tensor, phase: torch.Tensor, cfg: MotionCriteria) -> MotionReference:
    """C2-starting periodic pose with analytic spatial velocities/accelerations.

    Rotation is Rz(yaw) Ry(pitch) Rx(roll). Angular quantities are in the fixed
    reference frame, not Euler-angle rates and not the moving object frame.
    """
    validate_motion_criteria(cfg)
    if time.ndim != 1 or phase.shape != (time.shape[0], 3):
        raise ValueError("Reference requires time (N,) and phase (N,3)")
    elapsed = (time - cfg.settle_seconds).clamp_min(0.0)
    u = (elapsed / cfg.ramp_seconds).clamp(0.0, 1.0)
    envelope = 10 * u**3 - 15 * u**4 + 6 * u**5
    ramping = (elapsed > 0) & (elapsed < cfg.ramp_seconds)
    d_envelope = torch.where(ramping, (30 * u**2 - 60 * u**3 + 30 * u**4) / cfg.ramp_seconds, 0.0)
    dd_envelope = torch.where(ramping, (60 * u - 180 * u**2 + 120 * u**3) / cfg.ramp_seconds**2, 0.0)
    frequency = time.new_tensor(cfg.frequency_hz) * (2 * math.pi)
    angle = elapsed[:, None] * frequency + phase
    sine, cosine = torch.sin(angle), torch.cos(angle)
    wave = envelope[:, None] * sine
    d_wave = d_envelope[:, None] * sine + envelope[:, None] * frequency * cosine
    dd_wave = dd_envelope[:, None] * sine + 2 * d_envelope[:, None] * frequency * cosine
    dd_wave -= envelope[:, None] * frequency.square() * sine
    amplitude = time.new_tensor(cfg.rotation_amplitude)
    euler, euler_dot, euler_ddot = wave * amplitude, d_wave * amplitude, dd_wave * amplitude
    roll, pitch, yaw = euler.unbind(-1)
    half = euler * 0.5
    sr, sp, sy = torch.sin(half).unbind(-1)
    cr, cp, cy = torch.cos(half).unbind(-1)
    quaternion = torch.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dim=-1,
    )
    zero = torch.zeros_like(roll)
    x_axis = torch.stack((torch.cos(yaw) * torch.cos(pitch), torch.sin(yaw) * torch.cos(pitch), -torch.sin(pitch)), -1)
    y_axis = torch.stack((-torch.sin(yaw), torch.cos(yaw), zero), -1)
    z_axis = torch.stack((zero, zero, torch.ones_like(zero)), -1)
    rdot, pdot, ydot = euler_dot.unbind(-1)
    rddot, pddot, yddot = euler_ddot.unbind(-1)
    omega = rdot[:, None] * x_axis + pdot[:, None] * y_axis + ydot[:, None] * z_axis
    x_dot = torch.cross(ydot[:, None] * z_axis + pdot[:, None] * y_axis, x_axis, dim=-1)
    y_dot = torch.cross(ydot[:, None] * z_axis, y_axis, dim=-1)
    alpha = rddot[:, None] * x_axis + rdot[:, None] * x_dot + pddot[:, None] * y_axis
    alpha += pdot[:, None] * y_dot + yddot[:, None] * z_axis
    translation = time.new_tensor(cfg.translation_amplitude)
    active = time >= cfg.settle_seconds + cfg.ramp_seconds
    return MotionReference(
        wave * translation,
        quaternion,
        d_wave * translation,
        omega,
        dd_wave * translation,
        alpha,
        angle,
        envelope,
        active,
    )
