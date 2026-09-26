"""Scripted mechanical motion baseline on the unmodified Stage A PhysX grasp.

This is NOT a Stage B task-accuracy evaluation: it uses a settled measured pose,
own world-axis rotation-vector reference, and a longer no-reset horizon. Default
commands are 0.1 deg XYZ at 0.2/0.3/0.4 Hz, without translation. The original
30-second sweep retained all4 validity but measured only 0.098/0.034/0.0066 deg
XYZ motion: mostly X, not faithful three-axis tracking. Larger motions are unvalidated.

Two environments compare stationary preloaded targets (env 0) with coherent
contact-point DLS (env 1). Preparation physically settles 2 s, ramps bounded
inward preload 2 s, then settles 2 s. Only initial reset writes object state;
subsequent object pose/velocity/force writers raise, including on autoreset.
No physics, reward, contact thresholds, or actuator gains are patched.

Example:
    /home/andy/env_isaaclab/bin/python scripts/probes/leap_grasp_motion_probe.py \
        --steps 900 --output /tmp/leap_grasp_motion.json

Coordinates: world axes (env origins translate only), xyzw quaternion, meters,
radians, seconds. Contact offsets use LINK origins, not COMs. Raw contact points
are positive-normal-force weighted centroids, not exact centers of pressure.
The JSON summary is accompanied by an NPZ with raw measured vectors/poses.
"""

# flake8: noqa: E402

import argparse
import json
import math
import numpy as np
import torch
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--steps",
    type=int,
    default=900,
    help="30 Hz scored rollout steps, after preparation",
)
parser.add_argument("--output", type=Path, default=Path("/tmp/leap_grasp_motion.json"))
parser.add_argument(
    "--rotation_deg",
    type=float,
    default=0.1,
    help="Per-axis command, not guaranteed actual motion",
)
parser.add_argument(
    "--preload_mm",
    type=float,
    default=2.0,
    help="Geometric inward target offset, not indentation",
)
AppLauncher.add_app_launcher_args(parser)
if "--headless" not in parser._option_string_actions:
    parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401


def rotation_matrix(v):
    """Rotation vector to matrix, including zero angle."""
    angle = v.norm(dim=-1, keepdim=True)
    axis = v / angle.clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)
    return (
        torch.eye(3, device=v.device, dtype=v.dtype)
        + torch.sin(angle)[..., None] * skew
        + (1 - torch.cos(angle))[..., None] * (skew @ skew)
    )


class ContactPositionController:
    """Fixed-linearization DLS around measured contacts, preserving settled PD preload."""

    def __init__(self, env):
        self.env = env
        self.targets0 = env.cur_targets.clone()
        self.q0 = env.hand.data.joint_pos.torch.clone()
        self.pos0 = env.object.data.root_link_pos_w.torch.clone()
        self.quat0 = env.object.data.root_link_quat_w.torch.clone()
        self.contacts0 = env._contact_positions.clone()
        self.levers = self.contacts0 - self.pos0[:, None]
        tippos = env.hand.data.body_link_pos_w.torch[:, env._tip_indices]
        jac_all = env.hand.data.body_link_jacobian_w.torch
        offset = len(env.hand.body_names) - jac_all.shape[1]
        jac = jac_all[:, [index - offset for index in env._tip_indices]].clone()
        lever = self.contacts0 - tippos
        # v_contact = v_link_origin + omega cross (contact - link_origin).
        jac_position = jac[:, :, :3] + torch.cross(
            jac[:, :, 3:].transpose(-1, -2),
            lever[:, :, None].expand(-1, -1, jac.shape[-1], -1),
            dim=-1,
        ).transpose(-1, -2)
        self.jac_position = jac_position.reshape(env.num_envs, 12, 16)
        damping = 0.001
        self.preload_pinv = self.jac_position.transpose(1, 2) @ torch.linalg.inv(
            self.jac_position @ self.jac_position.transpose(1, 2) + damping**2 * torch.eye(12, device=env.device)
        )
        # Same zero-orientation-weight system used in the measured final sweep.
        jac_full = torch.cat((jac_position, torch.zeros_like(jac[:, :, 3:])), dim=2).reshape(env.num_envs, 24, 16)
        self.pinv = torch.linalg.solve(
            jac_full.transpose(1, 2) @ jac_full + damping**2 * torch.eye(16, device=env.device),
            jac_full.transpose(1, 2),
        )
        self.frequencies = torch.tensor([0.2, 0.3, 0.4], device=env.device)

    def preload_delta(self, millimeters):
        """Bounded target increment; callers must ramp through physical env.step calls."""
        normal = self.env._normal
        inward = -normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        delta = inward * millimeters / 1000
        delta[:, 1] = 0.0  # Do not increase the already-loaded thumb.
        return (self.preload_pinv @ delta.flatten(1)[..., None]).squeeze(-1).clamp(-0.1, 0.1)

    def action(self, time_s, rotation_deg):
        u = min(max(time_s / 2.0, 0.0), 1.0)
        ramp = u**3 * (10 - 15 * u + 6 * u * u)
        wave = torch.sin(2 * math.pi * self.frequencies * time_s) * ramp
        rv = (rotation_deg * math.pi / 180 * wave).expand(self.env.num_envs, 3).clone()
        rv[0] = 0  # Stationary comparator, same preload procedure.
        desired = self.pos0[:, None] + torch.einsum("nij,nkj->nki", rotation_matrix(rv), self.levers)
        delta = desired - self.contacts0
        twist = torch.cat((delta, torch.zeros_like(delta)), dim=-1)
        dq = (self.pinv @ twist.flatten(1)[..., None]).squeeze(-1).clamp(-0.12, 0.12)
        target = (self.targets0 + dq).clamp(self.env.hand_dof_lower_limits, self.env.hand_dof_upper_limits)
        target[0] = self.targets0[0]
        return ((target - self.env.cur_targets) / self.env.cfg.max_joint_delta).clamp(-1, 1), rv


def relative_rotvec(quat, quat0):
    """Principal log(q * inverse(q0)); xyzw, world-axis rotation from initial pose."""
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    quat0 = quat0 / np.linalg.norm(quat0, axis=-1, keepdims=True)
    a, aw = quat[..., :3], quat[..., 3:]
    b, bw = -quat0[..., :3], quat0[..., 3:]
    vector = aw * b + bw * a + np.cross(a, b)
    scalar = aw * bw - (a * b).sum(-1, keepdims=True)
    sign = np.where(scalar < 0, -1, 1)
    vector, scalar = vector * sign, scalar * sign
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector * (2 * np.arctan2(norm, scalar) / np.maximum(norm, 1e-12))


def summarize(raw, index, dt):
    """Frequency-locked pose acceleration excludes broadband/contact-solver jitter."""
    mask = raw["time"] >= 3.0
    t = raw["time"][mask]

    def data(key):
        return raw[key][mask, index]

    rot = data("rotvec")
    reference = data("reference_rotvec")
    frequencies = (0.2, 0.3, 0.4)
    design = np.stack(
        [
            np.ones(len(t)),
            *[fn(2 * np.pi * hz * t) for hz in frequencies for fn in (np.sin, np.cos)],
        ],
        axis=1,
    )
    fit = np.linalg.lstsq(design, rot, rcond=None)[0]
    alpha = np.zeros_like(rot)
    amplitudes = []
    for k, hz in enumerate(frequencies):
        amplitudes.append(float(np.hypot(fit[1 + 2 * k, k], fit[2 + 2 * k, k]) * 180 / np.pi))
        alpha -= (2 * np.pi * hz) ** 2 * (
            np.sin(2 * np.pi * hz * t)[:, None] * fit[1 + 2 * k] + np.cos(2 * np.pi * hz * t)[:, None] * fit[2 + 2 * k]
        )
    unit = alpha / np.maximum(np.linalg.norm(alpha, axis=-1, keepdims=True), 1e-12)
    error = np.linalg.norm(rot - reference, axis=-1)
    return {
        "role": "stationary_preloaded" if index == 0 else "scripted_motion",
        "samples": len(t),
        "valid4_fraction": float(data("valid").mean()),
        "whole_rollout_valid4_fraction": float(raw["valid"][:, index].mean()),
        "finger_valid_fraction": data("finger_valid").mean(0).tolist(),
        "unsafe_count": int(data("unsafe").sum()),
        "overload_count": int(data("overload").sum()),
        "normal_force_max_N": data("normal_peak").max(0).tolist(),
        "full_force_max_N": data("full_peak").max(0).tolist(),
        "support_force_max_N": float(data("support").max()),
        "slip_max_m_s": float(data("slip").max()),
        "rotation_rms_error_deg": float(np.sqrt(np.mean(error**2)) * 180 / np.pi),
        "rotation_axis_rms_error_deg": (np.sqrt(np.mean((rot - reference) ** 2, axis=0)) * 180 / np.pi).tolist(),
        "rotation_max_error_deg": float(error.max() * 180 / np.pi),
        "position_rms_error_mm": float(np.sqrt(np.mean(np.sum(data("position_delta") ** 2, axis=-1))) * 1000),
        "frequency_matched_rotation_amplitudes_deg": amplitudes,
        "pose_frequency_alpha_rms_rad_s2": np.sqrt(np.mean(alpha**2, axis=0)).tolist(),
        "pose_frequency_alpha_positive_fraction": (alpha > 0).mean(0).tolist(),
        "pose_frequency_alpha_direction_eigenvalues": np.linalg.eigvalsh(unit.T @ unit / len(unit)).tolist(),
        "raw_angular_velocity_mean_rad_s": data("angular_velocity").mean(0).tolist(),
        "raw_velocity_difference_alpha_rms_rad_s2": (
            np.sqrt(np.mean(np.gradient(data("angular_velocity"), dt, axis=0) ** 2, axis=0)).tolist()
        ),
        "max_target_offset_rad": float(data("target_offset").max()),
        "max_target_increment_rad": float(data("target_increment").max()),
        "hand_root_max_drift_m": float(data("hand_drift").max()),
    }


def main():
    if args.steps < 180 or args.rotation_deg < 0 or args.preload_mm < 0:
        raise ValueError("Require steps >= 180 and nonnegative amplitudes")
    cfg = parse_env_cfg("Isaac-Grasp-Cube-LEAP-Direct-v0", device=args.device, num_envs=2)
    cfg.seed = 42
    cfg.episode_length_s = 1000.0
    env = gym.make("Isaac-Grasp-Cube-LEAP-Direct-v0", cfg=cfg).unwrapped
    if env._physics_backend != "physx":
        raise RuntimeError("This mechanical baseline was measured only on PhysX")
    env.reset(seed=42)

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("Object state/force writer or autoreset called after initial reset; probe aborted")

    for name in dir(env.object):
        if name.startswith("write_") and any(word in name for word in ("pose", "velocity", "state")):
            setattr(env.object, name, forbidden)
    for name in (
        "set_external_force_and_torque",
        "set_external_force_and_torque_index",
    ):
        if hasattr(env.object, name):
            setattr(env.object, name, forbidden)

    def step(action):
        _, _, terminated, truncated, _ = env.step(action)
        if (terminated | truncated).any():
            raise RuntimeError("Episode ended; refusing to hide failure with an autoreset")

    zero = torch.zeros(env.num_envs, 16, device=env.device)
    for _ in range(60):
        step(zero)
    initial = ContactPositionController(env)
    preload = initial.preload_delta(args.preload_mm)
    for k in range(120):
        u = min((k + 1) / 60, 1.0)
        blend = u**3 * (10 - 15 * u + 6 * u * u)
        target = initial.targets0 + preload * blend
        step(((target - env.cur_targets) / cfg.max_joint_delta).clamp(-1, 1))
    controller = ContactPositionController(env)
    root0 = env.hand.data.root_link_pos_w.torch.clone()
    records = {}
    for k in range(args.steps):
        action, reference = controller.action((k + 1) * env.step_dt, args.rotation_deg)
        step(action)
        normal_history = torch.stack(
            [
                sensor.data.normal_force_matrix_w_history.torch[:, :, 0, 0]
                for sensor in env.fingertip_contact_sensors.values()
            ],
            dim=2,
        )
        friction_history = torch.stack(
            [
                sensor.data.friction_force_matrix_w_history.torch[:, :, 0, 0]
                for sensor in env.fingertip_contact_sensors.values()
            ],
            dim=2,
        )
        values = {
            "valid": env._score.valid,
            "finger_valid": env._score.finger_valid,
            "unsafe": env._score.unsafe,
            "overload": env._score.overload,
            "normal": env._normal,
            "friction": env._friction,
            "normal_peak": normal_history.norm(dim=-1).amax(dim=1),
            "full_peak": (normal_history + friction_history).norm(dim=-1).amax(dim=1),
            "support": env._support_force.amax(1),
            "slip": env._slip_speed,
            "pulp": env._pulp_contact,
            "position_delta": env.object.data.root_link_pos_w.torch - controller.pos0,
            "quaternion": env.object.data.root_link_quat_w.torch,
            "angular_velocity": env.object.data.root_link_vel_w.torch[:, 3:],
            "reference_rotvec": reference,
            "target_offset": (env.cur_targets - controller.targets0).abs().amax(-1),
            "target_increment": (env.target_delta * cfg.max_joint_delta).abs().amax(-1),
            "hand_drift": (env.hand.data.root_link_pos_w.torch - root0).norm(dim=-1),
        }
        for key, value in values.items():
            records.setdefault(key, []).append(value.detach().cpu().numpy().copy())
    raw = {key: np.asarray(value) for key, value in records.items()}
    raw["time"] = (np.arange(args.steps) + 1) * env.step_dt
    raw["rotvec"] = relative_rotvec(raw["quaternion"], controller.quat0.cpu().numpy())
    output = {
        "status": "completed",
        "scope": "Stage A mechanical baseline; own measured-pose reference, NOT Stage B task-accuracy evaluation",
        "task": "Isaac-Grasp-Cube-LEAP-Direct-v0",
        "backend": env._physics_backend,
        "condition_key": env.condition_key,
        "fixture": cfg.pregrasp_fixture,
        "steps": args.steps,
        "control_dt_s": env.step_dt,
        "physics_dt_s": env.physics_dt,
        "rotation_command_deg_each_axis": args.rotation_deg,
        "frequencies_hz": [0.2, 0.3, 0.4],
        "translation_command_m": 0,
        "reference_ramp_s": 2,
        "preparation_s": {"settle": 2, "preload_ramp": 2, "preload_settle": 2},
        "preload_geometric_command_mm": args.preload_mm,
        "preload_joint_increment_rad": preload.cpu().tolist(),
        "initial_target_minus_measured_joint_rad": (controller.targets0 - controller.q0).cpu().tolist(),
        "rollout_object_state_writes": 0,
        "external_object_forces": False,
        "force_maxima": "maximum over all four physics-tick history samples in each control interval",
        "acceleration_caveat": (
            "Pose-frequency second derivative is a small-angle estimate. Raw angular velocity may"
            " contain solver residual bias/jitter; do not infer physical excitation from its derivative alone."
        ),
        "steady_results": [summarize(raw, i, env.step_dt) for i in range(env.num_envs)],
        "analysis_start_s": 3,
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), **raw)
    print(json.dumps(output, indent=2))
    env.close()


exit_code = 0
try:
    main()
except Exception:
    traceback.print_exc()
    args.output.write_text(json.dumps({"status": "error", "traceback": traceback.format_exc()}, indent=2))
    exit_code = 1
finally:
    app.close(exit_code=exit_code)
