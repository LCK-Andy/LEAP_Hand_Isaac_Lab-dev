# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Original DexPBT task runtime on EA PhysX, separate from the Dexsuite task family.

The released Gym task returns terminal observations and resets at the NEXT pre-step.
DirectRLEnv.step auto-resets immediately, so the explicit loop below is intentional.
"""

from __future__ import annotations

import hashlib
import json
import math
import torch
from dataclasses import fields
from pathlib import Path

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import CuboidCfg, DomeLightCfg, GroundPlaneCfg, MassPropertiesCfg, MultiAssetSpawnerCfg

from . import mdp
from .catalog import original_cube_catalog
from .env_cfg import (
    DEFAULT_JOINT_POSITION,
    FINGERTIP_NAMES,
    JOINT_NAMES,
    DexPBTOriginalEnvCfg,
    original_robot_cfg,
)

MUTABLE_KEYS = (
    "liftingRewScale",
    "liftingBonus",
    "liftingBonusThreshold",
    "keypointRewScale",
    "distanceDeltaRewScale",
    "reachGoalBonus",
    "kukaActionsPenaltyScale",
    "allegroActionsPenaltyScale",
    "fallDistance",
)


class DexPBTOriginalEnv(DirectRLEnv):
    cfg: DexPBTOriginalEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        if not isinstance(cfg.sim.physics, PhysxCfg):
            raise NotImplementedError("Original DexPBT parity targets Isaac Sim PhysX only")
        if cfg.original_profile != mdp.PROFILE:
            raise ValueError(f"Only the audited released-source profile is implemented: {mdp.PROFILE}")
        if cfg.events or cfg.action_noise_model or cfg.observation_noise_model:
            raise ValueError("Released task.randomize=False; additional events/noise would change the task")
        unknown = set(cfg.original_mutable_params) - set(MUTABLE_KEYS)
        if unknown:
            raise ValueError(f"Not original PBT mutation parameters: {sorted(unknown)}")
        self.reward_cfg = mdp.RewardConfig.from_mapping(dict(cfg.original_mutable_params))
        self.reward_cfg.success_tolerance = cfg.initial_success_tolerance
        self.original_mutable_params = self.get_original_mutable_params()
        if abs(cfg.sim.dt * cfg.decimation - 0.01667) > 1e-9:
            raise ValueError("Original released action timestep must remain 0.01667 seconds")
        if abs(cfg.episode_length_s - 600 * 0.01667) > 1e-9:
            raise ValueError("Released episodeLength is 600 control steps")
        asset = Path(cfg.robot_usd_path)
        if not asset.is_file():
            raise FileNotFoundError(
                f"Build original assets with scripts/assets/build_dexpbt_original_assets.py: {asset}"
            )
        manifest_path = asset.parent / "manifest.json"
        self.asset_manifest = json.loads(manifest_path.read_text())
        if self.asset_manifest["source_commit"] != mdp.SOURCE_COMMIT:
            raise ValueError("Robot asset source does not match the released task")
        self.asset_sha256 = hashlib.sha256(asset.read_bytes()).hexdigest()
        if self.asset_sha256 != self.asset_manifest["converted"]["sha256"]:
            raise ValueError("Robot USD bytes differ from the recorded conversion hash")
        self.cube_catalog = original_cube_catalog(
            cfg.randomize_object_dimensions,
            cfg.with_small_cuboids,
            cfg.with_big_cuboids,
            cfg.with_sticks,
        )
        super().__init__(cfg, render_mode, **kwargs)
        if self._physics_handles_decimation:
            raise RuntimeError("Original runtime requires explicit two-substep PhysX stepping")
        if tuple(self.robot.joint_names) != JOINT_NAMES:
            raise RuntimeError(f"Original joint order was not preserved: {self.robot.joint_names}")
        if not self.robot.is_fixed_base:
            raise RuntimeError("Original KUKA base must be fixed")
        self.fingertip_ids = [self.robot.body_names.index(name) for name in FINGERTIP_NAMES]
        self.palm_id = self.robot.body_names.index("iiwa7_link_7")
        self.joint_lower = self.robot.data.joint_pos_limits.torch[0, :, 0].clone()
        self.joint_upper = self.robot.data.joint_pos_limits.torch[0, :, 1].clone()
        self.default_joint_position = torch.tensor(DEFAULT_JOINT_POSITION, device=self.device)
        self.joint_targets = self.default_joint_position.repeat(self.num_envs, 1)
        self.state = mdp.make_state(self.num_envs, self.device)
        self.episode_length_buf = self.state.progress_buf
        self.successes = self.state.successes
        self.actions = torch.zeros(self.num_envs, 23, device=self.device)
        catalog_scales = torch.tensor([entry[1] for entry in self.cube_catalog], device=self.device)
        self.object_variant_ids = torch.arange(self.num_envs, device=self.device) % len(catalog_scales)
        self.object_scales = catalog_scales[self.object_variant_ids]
        self.object_initial_z = torch.full((self.num_envs,), 0.63, device=self.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self.goal_pose[:, 6] = 1.0
        self.reset_buf.fill_(True)
        self.reset_goal_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_episode_successes = torch.zeros(self.num_envs, device=self.device)
        self.true_objective = torch.zeros(self.num_envs, device=self.device)
        self.prev_episode_true_objective = torch.zeros(self.num_envs, device=self.device)
        self.local_object_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        probability_range = torch.tensor(cfg.force_probability_range, device=self.device)
        self.random_force_prob = torch.exp(
            (torch.log(probability_range[0]) - torch.log(probability_range[1]))
            * torch.rand(self.num_envs, device=self.device)
            + torch.log(probability_range[1])
        )
        # Preserve the released source quirk: env-0 mass is the force scale for ALL variants.
        self.reference_object_mass = self.object.data.body_mass.torch[0, 0].clone()
        self.frame_since_restart = 0
        self.last_curriculum_update = 0
        self.reward_buf = torch.zeros(self.num_envs, device=self.device)
        self._episode_reward = torch.zeros_like(self.reward_buf)

    def reset(self, seed=None, options=None):
        """Released startup returns zero observations; actual resets wait until pre-physics."""
        if seed is not None:
            self.seed(seed)
        self.reset_buf.fill_(True)
        self.reset_goal_buf.fill_(True)
        self.obs_buf = {"policy": torch.zeros(self.num_envs, mdp.OBSERVATION_DIM, device=self.device)}
        self.extras = {}
        return self.obs_buf, self.extras

    def get_original_mutable_params(self):
        values = self.reward_cfg.to_source_mapping()
        return {name: float(values[name]) for name in MUTABLE_KEYS}

    def get_pbt_env_state(self):
        """Released donor contract: tolerance only, not donor physics or episode memory."""
        return {"success_tolerance": self.success_tolerance}

    def set_pbt_env_state(self, state):
        """Restore only source tolerance; leave recipient rewards and fresh episode state intact."""
        if not isinstance(state, dict) or set(state) != {"success_tolerance"}:
            raise ValueError("Original PBT state must contain exactly success_tolerance")
        value = state["success_tolerance"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Original PBT success_tolerance must be a finite number")
        if not self.cfg.target_success_tolerance <= value <= self.cfg.initial_success_tolerance:
            raise ValueError("Original PBT success_tolerance is outside the configured curriculum bounds")
        self.success_tolerance = value

    @property
    def success_tolerance(self):
        return self.reward_cfg.success_tolerance

    @success_tolerance.setter
    def success_tolerance(self, value):
        self.reward_cfg.success_tolerance = float(value)

    def _setup_scene(self):
        robot_cfg = original_robot_cfg(self.cfg.robot_usd_path)
        material = PhysxRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        rigid_props = PhysxRigidBodyPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            max_depenetration_velocity=1000.0,
        )
        collision_props = PhysxCollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)
        variants = [
            CuboidCfg(
                size=tuple(self.cfg.object_base_size * value for value in scale),
                rigid_props=rigid_props,
                collision_props=collision_props,
                mass_props=MassPropertiesCfg(density=self.cfg.object_density),
                physics_material=material,
            )
            for _, scale in self.cube_catalog
        ]
        object_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_[^/]+/Object",
            spawn=MultiAssetSpawnerCfg(assets_cfg=variants),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.63)),
        )
        table_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_[^/]+/Table",
            spawn=CuboidCfg(
                size=(0.475, 0.4, 0.3),
                rigid_props=PhysxRigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=collision_props,
                physics_material=material,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38)),
        )
        # Explicit EA plan supplies segment-safe source paths and deterministic per-env variants.
        with cloner.ReplicateSession(
            [robot_cfg, object_cfg, table_cfg],
            num_clones=self.num_envs,
            env_spacing=self.cfg.scene.env_spacing,
            device=self.device,
            global_paths=("/World/ground", "/World/Light"),
            env_template="/World/envs/env_{}",
            stage=self.scene.stage,
            clone_strategy=cloner.sequential,
            replicate_physics=True,
        ):
            self.robot = Articulation(robot_cfg)
            self.object = RigidObject(object_cfg)
            self.table = RigidObject(table_cfg)
            ground = GroundPlaneCfg(physics_material=material)
            ground.func("/World/ground", ground)
            light = DomeLightCfg(intensity=1500.0)
            light.func("/World/Light", light)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def _random_quaternion(self, count):
        uvw = torch.rand(count, 3, device=self.device)
        a = torch.sqrt(1.0 - uvw[:, 0])
        b = torch.sqrt(uvw[:, 0])
        return torch.stack(
            (
                a * torch.cos(2 * math.pi * uvw[:, 1]),
                b * torch.sin(2 * math.pi * uvw[:, 2]),
                b * torch.cos(2 * math.pi * uvw[:, 2]),
                a * torch.sin(2 * math.pi * uvw[:, 1]),
            ),
            dim=-1,
        )

    def _reset_target_pose(self, env_ids):
        count = len(env_ids)
        minimum = self.goal_pose.new_tensor((-0.4, 0.0, 0.68))
        span = self.goal_pose.new_tensor((0.8, 0.35, 0.37))
        self.goal_pose[env_ids, :3] = minimum + torch.rand(count, 3, device=self.device) * span
        self.goal_pose[env_ids, 3:] = self._random_quaternion(count)
        mdp.reset_state(self.state, env_ids, goal_only=True)
        self.reset_goal_buf[env_ids] = False

    def _reset_idx(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.prev_episode_successes[env_ids] = self.state.successes[env_ids]
        self.prev_episode_true_objective[env_ids] = self.true_objective[env_ids]
        super()._reset_idx(env_ids)
        self._reset_target_pose(env_ids)
        count = len(env_ids)
        noise = self.joint_targets.new_tensor(self.cfg.reset_position_noise)
        position = (
            self.joint_targets.new_tensor((0.0, 0.0, 0.63)) + (2 * torch.rand(count, 3, device=self.device) - 1) * noise
        )
        pose = torch.cat((position + self.scene.env_origins[env_ids], self._random_quaternion(count)), dim=-1)
        self.object.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(count, 6, device=self.device), env_ids=env_ids
        )
        low, high = self.cfg.force_probability_range
        self.random_force_prob[env_ids] = torch.exp(
            (math.log(low) - math.log(high)) * torch.rand(count, device=self.device) + math.log(high)
        )
        random_joint = self.joint_lower + (self.joint_upper - self.joint_lower) * torch.rand(
            count, 23, device=self.device
        )
        coefficient = self.joint_targets.new_tensor(
            [self.cfg.reset_dof_position_noise_arm] * 7 + [self.cfg.reset_dof_position_noise_fingers] * 16
        )
        position = self.default_joint_position + coefficient * (random_joint - self.default_joint_position)
        velocity = self.cfg.reset_dof_velocity_noise * (2 * torch.rand(count, 23, device=self.device) - 1)
        self.robot.write_joint_state_to_sim_index(position=position, velocity=velocity, env_ids=env_ids)
        self.robot.set_joint_position_target_index(target=position, env_ids=env_ids)
        self.joint_targets[env_ids] = position
        self.local_object_forces[env_ids] = 0.0
        self.object.permanent_wrench_composer.reset(env_ids=env_ids)
        mdp.reset_state(self.state, env_ids)
        self.true_objective[env_ids] = 0.0
        self.reset_buf[env_ids] = False
        self._episode_reward[env_ids] = 0.0

    def _pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        # Goal resets precede full resets, even if both flags are set (source random-stream order).
        self._reset_target_pose(self.reset_goal_buf.nonzero(as_tuple=False).flatten())
        ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(ids):
            self._reset_idx(ids)
        self.joint_targets.copy_(
            mdp.joint_targets(
                self.actions,
                self.joint_targets,
                self.joint_lower,
                self.joint_upper,
                dt=self.step_dt,
                speed_scale=self.cfg.dof_speed_scale,
                moving_average=self.cfg.actions_moving_average,
            )
        )
        if self.cfg.force_scale > 0:
            self.local_object_forces *= self.cfg.force_decay ** (self.step_dt / self.cfg.force_decay_interval)
            selected = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero().flatten()
            self.local_object_forces[selected] = torch.randn(len(selected), 1, 3, device=self.device) * (
                self.reference_object_mass * self.cfg.force_scale
            )
        else:
            self.local_object_forces.zero_()
        self.object.permanent_wrench_composer.reset()
        self.object.permanent_wrench_composer.add_forces_and_torques(
            self.local_object_forces,
            torch.zeros_like(self.local_object_forces),
            is_global=False,
        )

    def _apply_action(self):
        self.robot.set_joint_position_target_index(target=self.joint_targets)

    def _measure(self):
        object_state = self.object.data.root_state_w.torch.clone()
        body_states = self.robot.data.body_state_w.torch
        palm_state = body_states[:, self.palm_id].clone()
        fingertip_states = body_states[:, self.fingertip_ids].clone()
        object_state[:, :3] -= self.scene.env_origins
        palm_state[:, :3] -= self.scene.env_origins
        fingertip_states[:, :, :3] -= self.scene.env_origins[:, None, :]
        geometry = mdp.compute_geometry(
            object_state,
            palm_state,
            fingertip_states,
            self.goal_pose,
            self.object_scales,
            self.cfg.object_base_size,
            self.reward_cfg.keypoint_scale,
        )
        mdp.initialize_distances(self.state, geometry)
        observation = mdp.pack_observations(
            self.robot.data.joint_pos.torch,
            self.robot.data.joint_vel.torch,
            self.joint_lower,
            self.joint_upper,
            palm_state,
            object_state,
            self.object_scales,
            geometry,
            self.state,
        )
        return observation, object_state, geometry

    def _get_observations(self):
        observation, _, _ = self._measure()
        mdp.finish_observations(observation, self.reward_buf)
        return {"policy": observation}

    def _get_rewards(self):
        return self.reward_buf

    def _get_dones(self):
        return self.reset_terminated, self.reset_time_outs

    def step(self, action):
        """Return terminal state intact; perform pending resets on the NEXT invocation."""
        self._pre_physics_step(action.clamp(-1.0, 1.0))
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and self.sim.is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.physics_dt)
        self.state.progress_buf += 1
        self.common_step_counter += 1
        self.frame_since_restart += 1
        self.success_tolerance, self.last_curriculum_update = mdp.tolerance_curriculum(
            self.last_curriculum_update,
            self.frame_since_restart,
            self.cfg.tolerance_curriculum_interval,
            self.prev_episode_successes,
            self.success_tolerance,
            self.cfg.initial_success_tolerance,
            self.cfg.target_success_tolerance,
            self.cfg.tolerance_curriculum_increment,
        )
        observation, object_state, geometry = self._measure()
        result = mdp.step_rewards(
            self.state,
            geometry,
            object_state[:, :3],
            self.object_initial_z,
            self.robot.data.joint_vel.torch,
            self.reward_cfg,
        )
        self.reward_buf = result.reward
        self._episode_reward += result.reward
        self.reset_goal_buf.copy_(result.is_success)
        self.reset_buf.copy_(result.reset)
        self.reset_time_outs.copy_(result.timeout)
        self.reset_terminated.copy_(result.reset & ~result.timeout)
        mdp.finish_observations(observation, result.reward)
        self.obs_buf = {"policy": observation}
        self.true_objective = mdp.tolerance_successes_objective(
            self.success_tolerance,
            self.cfg.initial_success_tolerance,
            self.cfg.target_success_tolerance,
            self.state.successes,
        )
        self.extras = {
            "true_objective": self.true_objective.clone(),
            "successes": self.state.successes.clone(),
            "time_outs": self.reset_time_outs.clone(),
            "scalars": {"success_tolerance": self.success_tolerance},
            "log": {
                "Original/success_tolerance": self.success_tolerance,
                "Original/successes": self.state.successes.mean(),
                "Original/true_objective": self.true_objective.mean(),
            },
        }
        if result.reset.any():
            self.extras["episode"] = {
                "reward": self._episode_reward[result.reset].mean(),
                "successes": self.state.successes[result.reset].mean(),
            }
        for recorder in self.video_recorders:
            recorder.step()
        if self.sim.consume_reset_request():
            self.reset_buf.fill_(True)
        return self.obs_buf, self.reward_buf, self.reset_terminated.clone(), self.reset_time_outs.clone(), self.extras

    def get_env_state(self):
        """Full task/runtime checkpoint; reward mutations are intentionally NOT donor-restored."""
        names = (
            "goal_pose",
            "joint_targets",
            "local_object_forces",
            "random_force_prob",
            "reset_buf",
            "reset_goal_buf",
            "prev_episode_successes",
            "prev_episode_true_objective",
            "true_objective",
            "reward_buf",
            "_episode_reward",
        )
        return {
            "profile": mdp.PROFILE,
            "num_envs": self.num_envs,
            "success_tolerance": self.success_tolerance,
            "last_curriculum_update": self.last_curriculum_update,
            "frame_since_restart": self.frame_since_restart,
            "common_step_counter": self.common_step_counter,
            "mdp_state": {
                field.name: getattr(self.state, field.name).detach().cpu().clone() for field in fields(self.state)
            },
            "buffers": {name: getattr(self, name).detach().cpu().clone() for name in names},
            "joint_pos": self.robot.data.joint_pos.torch.detach().cpu().clone(),
            "joint_vel": self.robot.data.joint_vel.torch.detach().cpu().clone(),
            "object_pose": self.object.data.root_link_pose_w.torch.detach().cpu().clone(),
            "object_velocity": self.object.data.root_com_vel_w.torch.detach().cpu().clone(),
        }

    def set_env_state(self, checkpoint):
        if checkpoint is None:
            return
        if checkpoint.get("profile") != mdp.PROFILE or checkpoint.get("num_envs") != self.num_envs:
            raise ValueError("Original-task restore requires matching profile and environment count")
        self.success_tolerance = checkpoint["success_tolerance"]
        self.last_curriculum_update = int(checkpoint["last_curriculum_update"])
        self.frame_since_restart = int(checkpoint["frame_since_restart"])
        self.common_step_counter = int(checkpoint["common_step_counter"])
        for field in fields(self.state):
            getattr(self.state, field.name).copy_(checkpoint["mdp_state"][field.name].to(self.device))
        for name, value in checkpoint["buffers"].items():
            getattr(self, name).copy_(value.to(self.device))
        self.robot.write_joint_state_to_sim_index(
            position=checkpoint["joint_pos"].to(self.device), velocity=checkpoint["joint_vel"].to(self.device)
        )
        self.robot.set_joint_position_target_index(target=self.joint_targets)
        self.object.write_root_pose_to_sim_index(root_pose=checkpoint["object_pose"].to(self.device))
        self.object.write_root_velocity_to_sim_index(root_velocity=checkpoint["object_velocity"].to(self.device))
        self.object.permanent_wrench_composer.reset()
        self.object.permanent_wrench_composer.add_forces_and_torques(
            self.local_object_forces,
            torch.zeros_like(self.local_object_forces),
            is_global=False,
        )
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)
