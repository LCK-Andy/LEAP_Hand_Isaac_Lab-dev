"""
Franka Panda arm with a welded LEAP hand.

Single articulated robot (23 DOF). The policy controls joint position deltas
for the whole arm+hand. This is a minimal working environment intended as a
base for task development (grasping, in-hand manipulation, tabletop tasks).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import saturate

if TYPE_CHECKING:
    from LEAP_Isaaclab.tasks.franka_leap.franka_leap_env_cfg import FrankaLeapEnvCfg


class FrankaLeapEnv(DirectRLEnv):
    cfg: FrankaLeapEnvCfg

    def __init__(self, cfg: FrankaLeapEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # joint index sets (robust to USD joint ordering; Warp wants int lists)
        self.arm_dof_idx = [self.robot.joint_names.index(n) for n in self.cfg.arm_joint_names]
        self.hand_dof_idx = [self.robot.joint_names.index(n) for n in self.cfg.hand_joint_names]
        self.actuated_dof_idx = sorted(self.arm_dof_idx + self.hand_dof_idx)

        # fingertip bodies for world-space reporting
        self.fingertip_bodies = [self.robot.body_names.index(n) for n in self.cfg.fingertip_body_names]

        # joint limits (data.joint_limits: [num_envs, num_joints, 2])
        limits = self.robot.data.joint_limits.torch.to(self.device)
        self.joint_lower_limits = limits[..., 0]
        self.joint_upper_limits = limits[..., 1]

        # default joint position target (arm ready pose + hand open)
        self.default_joint_pos = torch.zeros(
            (self.num_envs, self.robot.num_joints), dtype=torch.float, device=self.device
        )
        for name, pos in dict(
            zip(self.cfg.arm_joint_names, [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
        ).items():
            self.default_joint_pos[:, self.robot.joint_names.index(name)] = pos

        # position targets for PD control
        self.dof_targets = self.default_joint_pos.clone()

        self._set_joint_pos_target = self.robot.set_joint_position_target_index
        self._write_joint_pos = self.robot.write_joint_position_to_sim_index
        self._write_joint_vel = self.robot.write_joint_velocity_to_sim_index

        if not hasattr(self, "extras") or self.extras is None:
            self.extras = {}
        if "log" not in self.extras:
            self.extras["log"] = {}

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # relative joint-position control with saturation at joint limits
        targets = self.dof_targets + self.cfg.actions_scale * self.actions
        self.dof_targets = saturate(targets, self.joint_lower_limits, self.joint_upper_limits)

    def _apply_action(self) -> None:
        self._set_joint_pos_target(target=self.dof_targets, joint_ids=self.actuated_dof_idx)

    def _get_observations(self) -> dict:
        qpos = self.robot.data.joint_pos.torch
        qvel = self.robot.data.joint_vel.torch
        obs = torch.cat([qpos, qvel], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        total_reward = self.cfg.action_penalty_scale * torch.sum(self.actions**2, dim=-1)
        total_reward += self.cfg.velocity_penalty_scale * torch.sum(
            self.robot.data.joint_vel.torch**2, dim=-1
        )
        self.extras["log"]["action_penalty"] = torch.sum(self.actions**2, dim=-1).mean()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        dof_pos = self.default_joint_pos[env_ids].clone() + 0.02 * torch.randn(
            len(env_ids), self.robot.num_joints, device=self.device
        )
        dof_pos = saturate(dof_pos, self.joint_lower_limits[env_ids], self.joint_upper_limits[env_ids])
        dof_vel = torch.zeros_like(dof_pos)

        self.dof_targets[env_ids] = dof_pos

        self._set_joint_pos_target(target=dof_pos, env_ids=env_ids)
        self._write_joint_pos(position=dof_pos, env_ids=env_ids)
        self._write_joint_vel(velocity=dof_vel, env_ids=env_ids)

    def fingertip_positions_world(self) -> torch.Tensor:
        """World-space fingertip positions [num_envs, 4, 3] (for demos/checks)."""
        return self.robot.data.body_pos_w.torch[:, self.fingertip_bodies]
