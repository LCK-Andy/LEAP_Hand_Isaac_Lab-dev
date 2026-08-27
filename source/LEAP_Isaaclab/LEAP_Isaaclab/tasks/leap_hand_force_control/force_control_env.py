# --------------------------------------------------------
# LEAP Hand Force Control Task
# Given desired fingertip positions (on a table surface) and
# desired normal contact forces, control the LEAP hand joints.
# --------------------------------------------------------

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import saturate, sample_uniform

if TYPE_CHECKING:
    from LEAP_Isaaclab.tasks.leap_hand_force_control.leap_hand_force_env_cfg import LeapHandForceEnvCfg


class ForceControlEnv(DirectRLEnv):
    """LEAP hand presses a fixed table with specified normal forces at specified fingertip locations."""

    cfg: LeapHandForceEnvCfg

    def __init__(self, cfg: LeapHandForceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # actuated joints and fingertip bodies
        self.actuated_dof_indices = [self.hand.joint_names.index(j) for j in self.cfg.actuated_joint_names]
        self.actuated_dof_indices.sort()
        self.finger_bodies = [self.hand.body_names.index(b) for b in self.cfg.fingertip_body_names]
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # joint limits (Isaac Lab 3.0: accessed via data.joint_limits)
        joint_pos_limits = self.hand.data.joint_limits.torch.to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # default joint pose
        self.override_default_joint_pos = torch.tensor(
            self.cfg.default_joint_pos, device=self.device
        ).repeat(self.num_envs, 1)

        # task targets: desired fingertip positions [N, 4, 3] (env-local) and forces [N, 4]
        self.target_pos = torch.zeros((self.num_envs, self.num_fingertips, 3), dtype=torch.float, device=self.device)
        self.target_force = torch.zeros((self.num_envs, self.num_fingertips), dtype=torch.float, device=self.device)
        # anchor: default-pose fingertip positions (env-local), measured once at first step
        self.tip_anchor: torch.Tensor | None = None

        # fingertip positions / forces read each step
        self.fingertip_pos = torch.zeros((self.num_envs, self.num_fingertips, 3), dtype=torch.float, device=self.device)
        self.fingertip_force = torch.zeros((self.num_envs, self.num_fingertips), dtype=torch.float, device=self.device)

        # observation history (64 per timestep, hist_len 3 -> 192)
        self.obs_hist_buf = torch.zeros(
            (self.num_envs, self.cfg.observation_space // self.cfg.hist_len, self.cfg.hist_len),
            device=self.device, dtype=torch.double,
        )

        # success tracking: rolling window of force errors
        self.force_err_window = torch.zeros(
            (self.num_envs, self.cfg.success_window), dtype=torch.float, device=self.device
        )
        self.success_rate = torch.zeros(1, dtype=torch.float, device=self.device)

        # extras logging
        if not hasattr(self, "extras") or self.extras is None:
            self.extras = {}
        if "log" not in self.extras:
            self.extras["log"] = {}

        # bind write methods for performance (Isaac Lab 3.0 API)
        self._set_joint_pos_target = self.hand.set_joint_position_target_index
        self._write_hand_joint_pos = self.hand.write_joint_position_to_sim_index
        self._write_hand_joint_vel = self.hand.write_joint_velocity_to_sim_index

    def _setup_scene(self):
        # add hand, table, contact sensors (one per fingertip)
        self.hand = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.contact_sensors = [ContactSensor(cfg) for cfg in self.cfg.contact_sensor_cfgs]
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["table"] = self.table
        for i, sensor in enumerate(self.contact_sensors):
            self.scene.sensors[f"contact_{i}"] = sensor
        # lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # enlarge the PhysX contact-offset patch-pairs buffer (not exposed via PhysxCfg;
        # default 32768 overflows when 4 fingertips press the table across 2048+ envs)
        scene_prim = self.sim.stage.GetPrimAtPath(self.sim.cfg.physics_prim_path)
        sim_utils.safe_set_attribute_on_usd_prim(
            scene_prim, "physxScene:gpuMaxNumContactOffsetPatchPairs", 2**20, camel_case=False
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions.clone(), -1.0, 1.0)

    def _apply_action(self) -> None:
        targets = self.prev_targets[:, self.actuated_dof_indices] + self.cfg.act_moving_average * self.actions
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            targets,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self._set_joint_pos_target(
            target=self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        # initialize anchor and targets on the first step (hand is at default pose)
        if self.tip_anchor is None:
            self.tip_anchor = self.fingertip_pos.detach().clone()[0]
            self._sample_targets(torch.arange(self.num_envs, device=self.device))

        # 64-d per timestep: joints (scaled) + fingertip pos + force + pos error + force target
        frame = torch.cat(
            (
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),  # 16
                self.hand_dof_vel,  # 16
                self.fingertip_pos.reshape(self.num_envs, -1),  # 12 (env-local)
                self.fingertip_force,  # 4 (normal force magnitude)
                (self.target_pos - self.fingertip_pos).reshape(self.num_envs, -1),  # 12
                self.target_force,  # 4
            ),
            dim=-1,
        )

        self.obs_hist_buf[:, :, :-1] = self.obs_hist_buf[:, :, 1:]
        self.obs_hist_buf[:, :, -1] = frame
        obs = self.obs_hist_buf.transpose(1, 2).reshape(self.num_envs, -1)
        return {"policy": obs.float()}

    def _get_rewards(self) -> torch.Tensor:
        force_err = (self.fingertip_force - self.target_force).abs().mean(dim=-1)  # [N]
        pos_err = torch.linalg.norm(self.fingertip_pos - self.target_pos, ord=2, dim=-1).mean(dim=-1)  # [N]
        action_penalty = torch.sum(self.actions**2, dim=-1)

        reward = (
            -self.cfg.force_reward_scale * force_err
            - self.cfg.pos_reward_scale * pos_err
            + self.cfg.action_penalty_scale * action_penalty
        )

        # rolling success window: mean force error < 15% of mean target force
        self.force_err_window[:, :-1] = self.force_err_window[:, 1:]
        self.force_err_window[:, -1] = force_err

        # episode success = last `success_window` steps within tolerance
        mean_target = self.target_force.mean(dim=-1)
        window_ok = (self.force_err_window.mean(dim=-1) < self.cfg.success_tolerance * mean_target).float()
        # count successes only for envs that just finished an episode
        finished = self.episode_length_buf >= self.max_episode_length - 1
        num_finished = finished.sum()
        if num_finished > 0:
            success_count = (window_ok * finished).sum()
            self.success_rate = (self.success_rate * 0.9) + (success_count / num_finished) * 0.1

        self.extras["log"]["mean_force_err"] = force_err.mean()
        self.extras["log"]["mean_pos_err"] = pos_err.mean()
        self.extras["log"]["mean_force"] = self.fingertip_force.mean()
        self.extras["log"]["success_rate"] = self.success_rate.mean()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids)

        # reset hand to default pose
        dof_pos = self.override_default_joint_pos[env_ids]
        dof_vel = self.hand.data.default_joint_vel.torch[env_ids]

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self._set_joint_pos_target(target=dof_pos, env_ids=env_ids)
        self._write_hand_joint_pos(position=dof_pos, env_ids=env_ids)
        self._write_hand_joint_vel(velocity=dof_vel, env_ids=env_ids)

        # resample desired fingertip positions and forces
        self._sample_targets(env_ids)

    def _sample_targets(self, env_ids) -> None:
        if self.tip_anchor is None:
            return  # anchor not measured yet; targets sampled on first observation step

        n = len(env_ids)
        # desired fingertip positions: default-pose positions + xy noise (reachability by construction)
        pos_noise = sample_uniform(-1.0, 1.0, (n, self.num_fingertips, 3), device=self.device)
        pos_noise[:, :, 2] = 0.0  # keep z at default pose height (slightly above table top)
        self.target_pos[env_ids] = self.tip_anchor + self.cfg.target_pos_noise * pos_noise

        # desired normal forces: uniform in [force_target_range]
        self.target_force[env_ids] = sample_uniform(
            self.cfg.force_target_range[0], self.cfg.force_target_range[1], (n, self.num_fingertips), device=self.device
        )

    def _compute_intermediate_values(self):
        # fingertip positions (env-local) and contact forces (normal component)
        self.fingertip_pos = self.hand.data.body_pos_w.torch[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )

        # normal contact force per fingertip: stack the 4 per-fingertip sensors
        forces_w = torch.stack(
            [s.data.net_forces_w.torch[:, 0] for s in self.contact_sensors], dim=1
        )  # [N, 4, 3] world frame
        self.fingertip_force = forces_w[:, :, 2].abs()  # normal (table normal = +z)

        self.hand_dof_pos = self.hand.data.joint_pos.torch
        self.hand_dof_vel = self.hand.data.joint_vel.torch


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
