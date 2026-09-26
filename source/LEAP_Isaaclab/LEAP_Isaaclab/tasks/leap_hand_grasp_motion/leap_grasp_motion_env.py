"""Stage B: track a smooth object reference while maintaining a four-pulp grasp."""

import math
import torch
from dataclasses import asdict, replace

from ..leap_hand_grasp.grasp_logic import grasp_condition_key
from ..leap_hand_grasp.leap_grasp_env import LeapGraspEnv
from .leap_grasp_motion_env_cfg import LeapGraspMotionEnvCfg
from .motion_logic import (
    MotionReference,
    PoseMotionState,
    motion_reference,
    multiply_quaternions,
    rotate_vectors,
    rotation_error_vector,
    score_tracking,
)


class LeapGraspMotionEnv(LeapGraspEnv):
    """Physical object motion is produced by joint actions, never by reference-pose writes."""

    cfg: LeapGraspMotionEnvCfg
    grasp_motion_contract_version = 1

    def __init__(self, cfg: LeapGraspMotionEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._motion_phase = torch.zeros(self.num_envs, 3, device=self.device)
        self._tracking_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._tracking_time = torch.zeros(self.num_envs, device=self.device)
        self._tracking_longest = torch.zeros_like(self._tracking_time)
        self._tracking_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._motion_errors = torch.zeros(self.num_envs, 4, device=self.device)
        self._active_samples = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._angular_gram = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.motion_state = PoseMotionState(self.num_envs, self.device)
        self._filtered_ref_linear_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self._filtered_ref_angular_velocity = torch.zeros_like(self._filtered_ref_linear_velocity)
        self._rejected_motion_samples = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.active_start_step = max(
            1, math.ceil((cfg.motion.settle_seconds + cfg.motion.ramp_seconds) / self.step_dt - 1e-9)
        )
        self.active_horizon = self.max_episode_length - self.active_start_step + 1
        self.reference = None
        self.tracking = None
        self.grasp_condition_key = self.condition_key
        self.condition_key = grasp_condition_key(
            {
                "stage": "B",
                "version": 1,
                "grasp_condition": self.grasp_condition_key,
                "motion": asdict(cfg.motion),
                "observation_dim": cfg.observation_space,
            }
        ).replace("grasp_hold_v1_", "grasp_motion_v1_", 1)
        self._reference_local = None
        self._refresh_reference()

    def _refresh_reference(self):
        time = self.episode_length_buf.float() * self.step_dt
        local = motion_reference(time, self._motion_phase, self.cfg.motion)
        local.active = self.episode_length_buf >= self.active_start_step
        base_rot = self._fixture_pose[3:].expand(self.num_envs, -1)
        self._reference_local = local
        self.reference = MotionReference(
            self.in_hand_pos + rotate_vectors(base_rot, local.position),
            multiply_quaternions(base_rot, local.quaternion),
            rotate_vectors(base_rot, local.linear_velocity),
            rotate_vectors(base_rot, local.angular_velocity),
            rotate_vectors(base_rot, local.linear_acceleration),
            rotate_vectors(base_rot, local.angular_acceleration),
            local.phase,
            local.envelope,
            local.active,
        )

    def _pre_physics_step(self, actions):
        self.extras["grasp_motion"] = {}
        super()._pre_physics_step(actions)

    def _get_rewards(self):
        grasp_reward = super()._get_rewards()
        self._refresh_reference()
        smoothing = self.cfg.motion.velocity_filter_alpha
        self.motion_state.update(self.object_pos, self.object_rot, self.step_dt, smoothing)
        self._filtered_ref_linear_velocity.lerp_(self.reference.linear_velocity, smoothing)
        self._filtered_ref_angular_velocity.lerp_(self.reference.angular_velocity, smoothing)
        tracking_reference = replace(
            self.reference,
            linear_velocity=self._filtered_ref_linear_velocity,
            angular_velocity=self._filtered_ref_angular_velocity,
        )
        self.tracking = score_tracking(
            self.object_pos,
            self.object_rot,
            self.motion_state.linear_velocity,
            self.motion_state.angular_velocity,
            tracking_reference,
            self._score.valid & self.motion_state.valid,
            self.cfg.motion,
        )
        valid = self.tracking.tracking_valid
        self._tracking_steps += valid.long()
        self._tracking_time = torch.where(valid, self._tracking_time + self.step_dt, 0.0)
        self._tracking_longest = torch.maximum(self._tracking_longest, self._tracking_time)
        self._tracking_success |= self._tracking_time >= self.cfg.motion.success_hold_seconds - 1e-6
        active = self.reference.active
        errors = torch.stack(
            (
                self.tracking.position_error,
                self.tracking.rotation_error,
                self.tracking.linear_velocity_error,
                self.tracking.angular_velocity_error,
            ),
            -1,
        )
        finite = torch.isfinite(errors).all(-1) & self.motion_state.valid
        accepted = active & finite
        self._motion_errors += torch.where(accepted[:, None], errors, 0.0)
        self._active_samples += accepted.long()
        self._rejected_motion_samples += (active & ~finite).long()
        # This is a filtered pose-motion diagnostic, not Fisher information or an inertia estimate.
        alpha = torch.where((accepted & self._score.valid)[:, None], self.motion_state.angular_acceleration, 0.0)
        self._angular_gram += alpha[:, :, None] * alpha[:, None, :] * self.step_dt
        log = self.extras.setdefault("log", {})
        log["Motion/tracking_reward"] = self.tracking.reward.mean()
        log["Motion/tracking_valid"] = valid.float().mean()
        log["Motion/position_error_m"] = self.tracking.position_error.mean()
        log["Motion/rotation_error_rad"] = self.tracking.rotation_error.mean()
        log["Motion/linear_velocity_error"] = self.tracking.linear_velocity_error.mean()
        log["Motion/angular_velocity_error"] = self.tracking.angular_velocity_error.mean()
        log["Motion/raw_solver_angular_speed"] = self.object_angvel.norm(dim=-1).mean()
        log["Motion/angular_speed"] = self.motion_state.angular_velocity.norm(dim=-1).mean()
        log["Motion/angular_acceleration"] = self.motion_state.angular_acceleration.norm(dim=-1).mean()
        log["Motion/reference_angular_speed"] = self.reference.angular_velocity.norm(dim=-1).mean()
        log["Motion/tracking_hold_seconds"] = self._tracking_time.mean()
        return grasp_reward + self.tracking.reward

    def _reset_idx(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        ended = env_ids[self.grasp_state.steps[env_ids] > 0]
        if ended.numel():
            horizon = self.active_horizon
            retained = self.reset_time_outs[ended] & ~self.reset_terminated[ended]
            metrics = self.grasp_state.completed(ended, retained, self.max_episode_length)
            self.extras["grasp_motion"] = {
                "condition_key": self.condition_key,
                "completed_tracking_fraction": (self._tracking_steps[ended].float() / horizon).clamp_max(1.0),
                "completed_retained": metrics["completed_retained"],
                "completed_overload_fraction": metrics["completed_overload_fraction"],
            }
            log = self.extras.setdefault("log", {})
            log["MotionEpisode/tracking_fraction"] = self.extras["grasp_motion"]["completed_tracking_fraction"].mean()
            log["MotionEpisode/tracking_success"] = self._tracking_success[ended].float().mean()
            log["MotionEpisode/longest_tracking_hold"] = self._tracking_longest[ended].mean()
            measured = ended[self._active_samples[ended] > 0]
            log["MotionEpisode/measured_episodes"] = float(measured.numel())
            log["MotionEpisode/rejected_samples"] = self._rejected_motion_samples[ended].float().mean()
            for key in ("MotionEpisode/position_error_m", "MotionEpisode/rotation_error_rad"):
                log.pop(key, None)
            if measured.numel():
                averages = self._motion_errors[measured] / self._active_samples[measured, None]
                log["MotionEpisode/position_error_m"] = averages[:, 0].mean()
                log["MotionEpisode/rotation_error_rad"] = averages[:, 1].mean()
            eigenvalues = torch.linalg.eigvalsh(self._angular_gram[ended]).clamp_min(0.0)
            log["MotionEpisode/angular_coverage_min"] = eigenvalues[:, 0].mean()
            log["MotionEpisode/angular_coverage_max"] = eigenvalues[:, -1].mean()
        super()._reset_idx(env_ids)
        for tensor in (
            self._tracking_steps,
            self._tracking_time,
            self._tracking_longest,
            self._tracking_success,
            self._motion_errors,
            self._active_samples,
            self._angular_gram,
            self._filtered_ref_linear_velocity,
            self._filtered_ref_angular_velocity,
            self._rejected_motion_samples,
        ):
            tensor[env_ids] = 0
        self.motion_state.reset(env_ids, self.object_pos[env_ids], self.object_rot[env_ids])
        if self.cfg.motion.randomize_phase:
            self._motion_phase[env_ids] = 2 * math.pi * torch.rand(len(env_ids), 3, device=self.device)
        else:
            self._motion_phase[env_ids] = 0.0
        self._refresh_reference()

    def _get_observations(self):
        observations = super()._get_observations()
        self._refresh_reference()
        position_scale = max(max(self.cfg.motion.translation_amplitude), self.cfg.motion.position_tolerance)
        rotation_scale = max(max(self.cfg.motion.rotation_amplitude), self.cfg.motion.rotation_tolerance)
        reference_rot_error = rotation_error_vector(self.object_rot, self.reference.quaternion)
        base_rot_inverse = self._fixture_pose[3:].expand(self.num_envs, -1) * self._fixture_pose.new_tensor(
            (-1, -1, -1, 1)
        )
        local_error = rotate_vectors(base_rot_inverse, reference_rot_error)
        commands = torch.cat(
            (
                ((self.reference.position - self.object_pos) / position_scale).clamp(-10, 10),
                (local_error / rotation_scale).clamp(-10, 10),
                rotate_vectors(base_rot_inverse, self._filtered_ref_linear_velocity - self.motion_state.linear_velocity)
                / max(position_scale, 1e-6),
                rotate_vectors(
                    base_rot_inverse, self._filtered_ref_angular_velocity - self.motion_state.angular_velocity
                )
                / max(rotation_scale, 1e-6),
                self._reference_local.linear_acceleration / max(position_scale, 1e-6),
                self._reference_local.angular_acceleration / max(rotation_scale, 1e-6),
                torch.sin(self._reference_local.phase),
                torch.cos(self._reference_local.phase),
            ),
            -1,
        )
        observations["policy"] = torch.cat((observations["policy"], commands), -1)
        return observations
