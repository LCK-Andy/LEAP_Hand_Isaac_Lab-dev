"""Direct-RL port of the official manager-based ``Isaac-Dexsuite-Kuka-Allegro-Lift-v0``.

The official task is manager-based and splits its observation across the
``policy`` / ``proprio`` / ``perception`` groups of the dexsuite cfg, which
off-policy stacks that only read ``"policy"`` (FlashSAC's ``flash_rl`` wrapper)
cannot consume. This env re-implements the same MDP as a single ``DirectRLEnv``
with one flat 785-d policy observation (157 x 5-step history).

Port map (official -> here):
  * actions: RelativeJointPositionAction over all 23 joints, scale 0.1
    (target = current joint pos + 0.1 * action, recomputed each physics substep)
  * observations: object_quat_b (4) + goal pose in robot-base frame (7) +
    last action (23) + joint pos (23) + joint vel (23) + palm/tip body states
    in the base frame (5 x 13 = 65) + per-tip contact forces in the base frame
    (4 x 3 = 12, from the four fingertip ContactSensors filtered against the
    object, clipped to +-20 N), with the official ADR-scaled uniform noises
    and clips
  * contact gating: the official per-fingertip ContactSensors with the official
    force thresholds -- fingers_to_object's contact bonus gates at 1.0 N,
    good_finger_contact / position_tracking at 0.1 N, success (and its sticky
    flag) and contact_count at 0.01 N; "good contact" = thumb AND at least one
    finger above the threshold
  * goal command: uniform pose in the robot-base frame (official ranges),
    resampled per env on a 2-3 s timer
  * rewards: the official Lift terms with the official weights, each scaled by
    the step dt like the manager's RewardManager does -- action L2 / action
    rate L2 (clamped), fingers-to-object tanh (max-distance kernel with the
    contact bonus), good finger contact, contact count fraction, position
    tracking tanh, contact-gated success (sticky flag), abnormal-termination
    penalty
  * ADR curriculum: per-env difficulty 0..10, +1 on episode success /
    -1 otherwise; scene gravity 0 -> -9.81 m/s^2 with the MEAN difficulty
    (the robot asset has ``disable_gravity=True``, so only the object feels
    it -- identical to the official behavior); obs-noise scales grow linearly
    with the PER-ENV difficulty
  * terminations: time-out, object out of bounds (env-relative official
    ranges), abnormal robot state (|joint vel| > 2x limits)
  * resets: joints default +- U(-0.5, 0.5) (wrist joint 7 +- U(-3, 3)), object
    at (-0.55, 0.1, 0.35) + U(-0.2, 0.2) xy + U(0, 0.4) z with a uniform
    random rotation

Known v1 parity gaps (official startup randomization NOT ported; all stay at
their authored defaults): rigid-body friction buckets, object mass scaling,
object inertia offset, actuator gain scaling, and joint-friction scaling. The
object is the official default cuboid -- the multi-shape spawner's
``random_choice`` is a no-op in this Isaac Lab build anyway.
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
)

from .dexsuite_lift_direct_env_cfg import DexsuiteKukaAllegroLiftDirectEnvCfg

# per-step observation layout (see module docstring)
OBS_PER_STEP = 4 + 7 + 23 + 23 + 23 + 5 * 13 + 4 * 3  # 157


class DexsuiteKukaAllegroLiftDirectEnv(DirectRLEnv):
    """KUKA+Allegro pick-and-lift with the official dexsuite Lift MDP as a DirectRLEnv."""

    cfg: DexsuiteKukaAllegroLiftDirectEnvCfg

    def __init__(self, cfg: DexsuiteKukaAllegroLiftDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._robot = self.scene["robot"]
        self._object = self.scene["object"]
        # per-fingertip contact sensors filtered against the object, in
        # cfg.fingertip_links order (thumb last, like the official FINGERTIP_LIST)
        self._sensors = [self.scene.sensors[f"{link}_object_s"] for link in self.cfg.fingertip_links]
        self._thumb_sensor_col = self.cfg.fingertip_links.index("thumb_link_3")

        # palm + fingertips for the body-state obs and the fingers_to_object reward
        self._hand_body_ids, hand_body_names = self._robot.find_bodies(["palm_link", ".*_tip"])
        assert len(self._hand_body_ids) == 5, f"expected palm + 4 fingertips, got {hand_body_names}"
        # wrist joint gets a wider reset range in the official task
        self._wrist_joint_ids, _ = self._robot.find_joints("iiwa7_joint_7")

        n = self.num_envs
        # goal command in the robot-base frame: (x, y, z, qx, qy, qz, qw), like the
        # official ObjectUniformPoseCommand buffer (position-only in the Lift task)
        self._goal_pose_b = torch.zeros(n, 7, device=self.device)
        self._goal_pose_b[:, 6] = 1.0
        self._goal_time_left = torch.zeros(n, device=self.device)
        # ADR difficulty + sticky episode-success flag
        self._difficulty = torch.full((n,), float(self.cfg.adr_init_difficulty), device=self.device)
        self._succeeded = torch.zeros(n, dtype=torch.bool, device=self.device)
        # raw previous-step action (rate penalty) + EMA state for action smoothing
        self._prev_action = torch.zeros(n, self.cfg.action_space, device=self.device)
        self._ema_action = torch.zeros(n, self.cfg.action_space, device=self.device)
        # rolling observation history, oldest first
        self._obs_history = torch.zeros(n, self.cfg.history_length, OBS_PER_STEP, device=self.device)

        # gravity starts at the difficulty-scheduled value (zero at difficulty 0)
        self._update_scene_gravity()

    ##
    # Helpers
    ##

    def _contact_force_mags(self) -> torch.Tensor:
        """Net contact-force magnitude per fingertip sensor. Shape (num_envs, 4).

        Each sensor's force matrix has a single filtered partner (the object),
        so its first column is the net contact force on that tip.
        """
        mags = [sensor.data.force_matrix_w.torch.view(self.num_envs, 3).norm(dim=-1) for sensor in self._sensors]
        return torch.stack(mags, dim=1)

    def _good_contact(self, threshold: float) -> torch.Tensor:
        """Official ``contacts`` gate: thumb AND at least one finger above threshold."""
        mags = self._contact_force_mags()
        thumb = mags[:, self._thumb_sensor_col]
        fingers = mags[:, : self._thumb_sensor_col].any(dim=1)
        return (thumb > threshold) & fingers

    def _update_scene_gravity(self) -> None:
        """Set scene-wide gravity from the mean ADR difficulty (official gravity ADR).

        The KUKA+Allegro asset has ``disable_gravity=True`` on its rigid bodies, so
        this only ever changes the object's weight -- exactly like the official task.
        """
        import carb  # noqa: PLC0415 -- only importable once the sim app exists
        from isaaclab.sim import SimulationContext  # noqa: PLC0415

        span = self.cfg.adr_max_difficulty - self.cfg.adr_min_difficulty
        frac = ((self._difficulty.float().mean() - self.cfg.adr_min_difficulty) / span).clamp(0.0, 1.0)
        view = SimulationContext.instance().physics_sim_view
        view.set_gravity(carb.Float3(0.0, 0.0, -9.81 * float(frac)))

    def _resample_goal(self, env_ids: torch.Tensor) -> None:
        """Sample goal poses (robot-base frame) and resample timers for the given envs."""
        cfg = self.cfg
        n = len(env_ids)
        self._goal_pose_b[env_ids, 0] = sample_uniform(*cfg.goal_pos_x, (n,), self.device)
        self._goal_pose_b[env_ids, 1] = sample_uniform(*cfg.goal_pos_y, (n,), self.device)
        self._goal_pose_b[env_ids, 2] = sample_uniform(*cfg.goal_pos_z, (n,), self.device)
        # official ranges: roll/pitch uniform, yaw fixed at 0; stored xyzw like the command buffer
        roll = sample_uniform(-3.14159265, 3.14159265, (n,), self.device)
        pitch = sample_uniform(-3.14159265, 3.14159265, (n,), self.device)
        quat = quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll))
        self._goal_pose_b[env_ids, 3:7] = quat
        self._goal_time_left[env_ids] = sample_uniform(*cfg.goal_resample_time_range, (n,), self.device)

    def _compute_current_obs(self) -> torch.Tensor:
        """One step of the (noisy) observation. Shape (num_envs, 157)."""
        cfg = self.cfg
        robot, obj = self._robot, self._object
        n = self.num_envs
        # per-env ADR noise fraction in [0, 1]
        frac = (self._difficulty - cfg.adr_min_difficulty) / (cfg.adr_max_difficulty - cfg.adr_min_difficulty)
        frac = frac.clamp(0.0, 1.0).unsqueeze(1)

        def noise(shape, scale):
            if scale <= 0.0:
                return 0.0
            return (torch.rand(shape, device=self.device) * 2.0 - 1.0) * (scale * frac)

        root_pos = robot.data.root_pos_w.torch
        root_quat = robot.data.root_quat_w.torch

        object_quat_b = quat_mul(quat_inv(root_quat), obj.data.root_quat_w.torch)
        object_quat_b = object_quat_b + noise(object_quat_b.shape, cfg.object_quat_noise)

        joint_pos = robot.data.joint_pos.torch + noise((n, self.cfg.action_space), cfg.joint_pos_noise)
        joint_vel = robot.data.joint_vel.torch + noise((n, self.cfg.action_space), cfg.joint_vel_noise)

        # palm/tip body states in the robot base frame (official body_state_b port)
        body_ids = self._hand_body_ids
        num_bodies = len(body_ids)
        body_pos_w = robot.data.body_pos_w.torch[:, body_ids].reshape(-1, 3)
        body_quat_w = robot.data.body_quat_w.torch[:, body_ids].reshape(-1, 4)
        body_lin_vel_w = robot.data.body_lin_vel_w.torch[:, body_ids].reshape(-1, 3)
        body_ang_vel_w = robot.data.body_ang_vel_w.torch[:, body_ids].reshape(-1, 3)
        root_pos_rep = root_pos.unsqueeze(1).repeat_interleave(num_bodies, dim=1).reshape(-1, 3)
        root_quat_rep = root_quat.unsqueeze(1).repeat_interleave(num_bodies, dim=1).reshape(-1, 4)
        body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_rep, root_quat_rep, body_pos_w, body_quat_w)
        body_lin_vel_b = quat_apply_inverse(root_quat_rep, body_lin_vel_w)
        body_ang_vel_b = quat_apply_inverse(root_quat_rep, body_ang_vel_w)
        tips = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1).view(n, -1)
        tips = (tips + noise(tips.shape, cfg.tip_state_noise)).clamp(-2.0, 2.0)

        # per-tip contact forces rotated into the robot base frame (official
        # fingers_contact_force_b; flat shapes for the compiled quat helpers)
        num_tips = len(self._sensors)
        forces_w = torch.stack(
            [sensor.data.force_matrix_w.torch.view(n, 3) for sensor in self._sensors], dim=1
        ).reshape(-1, 3)
        root_quat_f = root_quat.unsqueeze(1).repeat(1, num_tips, 1).reshape(-1, 4)
        forces_b = quat_apply_inverse(root_quat_f, forces_w).view(n, -1).clamp(-20.0, 20.0)

        return torch.cat(
            (object_quat_b, self._goal_pose_b, self.actions, joint_pos, joint_vel, tips, forces_b), dim=1
        )

    ##
    # DirectRLEnv implementation
    ##

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_action = self.actions.clone()
        self.actions = actions.clone()
        alpha = self.cfg.act_moving_average
        if alpha >= 1.0:
            self._ema_action = self.actions.clone()
        else:
            self._ema_action.mul_(alpha).add_(self.actions, alpha=1.0 - alpha)

    def _apply_action(self) -> None:
        # official RelativeJointPositionAction: the target follows the current joint
        # positions, so it is recomputed at every physics substep
        robot = self._robot
        targets = robot.data.joint_pos.torch + self.cfg.action_scale * self._ema_action
        limits = robot.data.joint_pos_limits.torch  # (num_envs, num_joints, 2)
        targets = targets.clamp(limits[..., 0], limits[..., 1])
        robot.set_joint_position_target_index(target=targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # goal resampling timer (rewards/observations below see the fresh goal)
        self._goal_time_left -= self.step_dt
        expired = self._goal_time_left <= 0.0
        if expired.any():
            self._resample_goal(expired.nonzero(as_tuple=False).squeeze(-1))

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        obj_pos = self._object.data.root_pos_w.torch
        origins = self.scene.env_origins
        lower = origins + obj_pos.new_tensor(
            [
                self.cfg.object_out_of_bound_x[0],
                self.cfg.object_out_of_bound_y[0],
                self.cfg.object_out_of_bound_z[0],
            ]
        )
        upper = origins + obj_pos.new_tensor(
            [
                self.cfg.object_out_of_bound_x[1],
                self.cfg.object_out_of_bound_y[1],
                self.cfg.object_out_of_bound_z[1],
            ]
        )
        out_of_bound = ((obj_pos < lower) | (obj_pos > upper)).any(dim=1)

        joint_vel = self._robot.data.joint_vel.torch
        joint_vel_limits = self._robot.data.joint_vel_limits.torch
        abnormal = (joint_vel.abs() > joint_vel_limits * 2.0).any(dim=1)

        return out_of_bound | abnormal, time_out

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        robot, obj = self._robot, self._object
        dt = self.step_dt

        mags = self._contact_force_mags()

        # fingers_to_object: max distance from palm/tips to the object, tanh kernel,
        # full reward in good contact (1.0 N gate) and 0.1x otherwise (official object_ee_distance)
        hand_pos_w = robot.data.body_pos_w.torch[:, self._hand_body_ids]
        obj_pos_w = obj.data.root_pos_w.torch
        dist_max = (hand_pos_w - obj_pos_w.unsqueeze(1)).norm(dim=-1).max(dim=-1).values
        good_at_1 = self._good_contact(cfg.fingers_to_object_contact_threshold)
        contact_bonus = torch.where(good_at_1, 1.0, 0.1)
        r_fingers = (1.0 - torch.tanh(dist_max / cfg.fingers_to_object_std)) * contact_bonus

        good_at_01 = self._good_contact(cfg.good_finger_contact_threshold)
        r_good_contact = good_at_01.float()
        r_contact_count = (mags > cfg.contact_count_threshold).float().sum(dim=1) / mags.shape[1]

        # goal position in world (robot-base-frame command), official position tracking
        root_pos = robot.data.root_pos_w.torch
        root_quat = robot.data.root_quat_w.torch
        des_pos_w = quat_apply(root_quat, self._goal_pose_b[:, :3]) + root_pos
        pos_dist = (obj_pos_w - des_pos_w).norm(dim=-1)
        r_position = (1.0 - torch.tanh(pos_dist / cfg.position_tracking_std)) * good_at_01.float()

        good_at_001 = self._good_contact(cfg.success_contact_threshold)
        r_success = (1.0 - torch.tanh(pos_dist / cfg.success_pos_std)) ** 2 * good_at_001.float()
        self._succeeded |= good_at_001 & (pos_dist < cfg.success_pos_std)

        joint_vel = robot.data.joint_vel.torch
        joint_vel_limits = robot.data.joint_vel_limits.torch
        abnormal = (joint_vel.abs() > joint_vel_limits * 2.0).any(dim=1)

        r_action = torch.sum(self.actions.square(), dim=1).clamp(-1000.0, 1000.0)
        r_action_rate = torch.sum((self.actions - self._prev_action).square(), dim=1).clamp(-1000.0, 1000.0)

        return dt * (
            cfg.action_l2_scale * r_action
            + cfg.action_rate_l2_scale * r_action_rate
            + cfg.fingers_to_object_scale * r_fingers
            + cfg.good_finger_contact_scale * r_good_contact
            + cfg.contact_count_scale * r_contact_count
            + cfg.position_tracking_scale * r_position
            + cfg.success_scale * r_success
            + cfg.abnormal_termination_scale * abnormal.float()
        )

    def _get_observations(self) -> dict:
        current = self._compute_current_obs()
        self._obs_history = torch.roll(self._obs_history, shifts=-1, dims=1)
        self._obs_history[:, -1] = current
        return {"policy": self._obs_history.reshape(self.num_envs, -1)}

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        cfg = self.cfg
        n = len(env_ids)

        # ADR: promote on episode success, demote otherwise (official DifficultyScheduler)
        promoted = self._succeeded[env_ids]
        self._difficulty[env_ids] = torch.where(
            promoted, self._difficulty[env_ids] + 1, self._difficulty[env_ids] - 1
        ).clamp(cfg.adr_min_difficulty, cfg.adr_max_difficulty)
        log = self.extras.setdefault("log", {})
        log["Metrics/success_rate"] = promoted.float().mean().item()
        log["Metrics/adr_difficulty"] = self._difficulty.float().mean().item()
        self._succeeded[env_ids] = False
        self._update_scene_gravity()
        self._resample_goal(env_ids)

        # robot joints: default + U(-0.5, 0.5), wrist joint with the wider official range
        robot = self._robot
        joint_pos = robot.data.default_joint_pos[env_ids].clone()
        joint_pos += sample_uniform(-cfg.joint_reset_range, cfg.joint_reset_range, joint_pos.shape, self.device)
        joint_pos[:, self._wrist_joint_ids] += sample_uniform(
            -cfg.wrist_joint_reset_range, cfg.wrist_joint_reset_range, (n, len(self._wrist_joint_ids)), self.device
        )
        robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)

        # object: init pos + uniform xy/z offsets, uniform random rotation, zero velocity
        base = self.scene.env_origins[env_ids] + torch.tensor(cfg.object_init_pos, device=self.device)
        base[:, 0] += sample_uniform(-cfg.object_reset_pos_xy, cfg.object_reset_pos_xy, (n,), self.device)
        base[:, 1] += sample_uniform(-cfg.object_reset_pos_xy, cfg.object_reset_pos_xy, (n,), self.device)
        base[:, 2] += sample_uniform(*cfg.object_reset_pos_z, (n,), self.device)
        roll = sample_uniform(-3.14159265, 3.14159265, (n,), self.device)
        pitch = sample_uniform(-3.14159265, 3.14159265, (n,), self.device)
        yaw = sample_uniform(-3.14159265, 3.14159265, (n,), self.device)
        obj_quat = quat_from_euler_xyz(roll, pitch, yaw)
        root_pose = torch.cat((base, obj_quat), dim=1)
        self._object.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self._object.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(n, 6, device=self.device), env_ids=env_ids
        )

        # restart action smoothing from neutral and fill the obs history (filled, not
        # zeroed, to avoid a fake spike at the episode boundary)
        self._ema_action[env_ids] = 0.0
        current = self._compute_current_obs()
        self._obs_history[env_ids] = current[env_ids].unsqueeze(1).expand(-1, cfg.history_length, -1)
