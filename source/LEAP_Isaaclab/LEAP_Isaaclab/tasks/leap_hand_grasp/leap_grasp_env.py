"""Persistent four-pulp grasp with full contact forces and bounded target increments."""

from __future__ import annotations

import json
import torch
from dataclasses import asdict
from pathlib import Path

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from ..leap_hand_repose.leap_direct_env import LeapHandDirectEnv
from .contact_forces import read_contact_sample, read_support_load
from .grasp_logic import GraspEpisodeState, advance_joint_targets, grasp_condition_key, score_grasp
from .leap_grasp_env_cfg import LeapGraspEnvCfg


class LeapGraspEnv(LeapHandDirectEnv):
    """Hold a cube using all four pulps, without point goals or repeated success bonuses."""

    cfg: LeapGraspEnvCfg
    grasp_hold_contract_version = 1

    def __init__(self, cfg: LeapGraspEnvCfg, render_mode: str | None = None, **kwargs):
        self.fixture = json.loads(Path(cfg.pregrasp_fixture).read_text())
        self._validate_cfg(cfg)
        self.support_sensor = None
        super().__init__(cfg, render_mode, **kwargs)
        if self._physics_handles_decimation:
            raise RuntimeError("Stage A requires physics-step contact history; set sim.use_newton_actuators=False")
        if self._physics_backend == "newton":
            from isaaclab_newton.physics import NewtonManager

            from .newton_geometry import validate_newton_collision_coverage

            self._newton_collision_coverage = validate_newton_collision_coverage(
                NewtonManager.get_model(), [*self.fingertip_contact_sensors.values(), self.support_sensor]
            )
        expected_support = len(self.support_sensor.cfg.filter_prim_paths_expr)
        actual_support = self.support_sensor.data.normal_force_matrix_w.torch.shape[2]
        if actual_support != expected_support:
            raise RuntimeError(
                f"Non-pulp support filters resolved {actual_support} partners, expected {expected_support}"
            )
        self._tip_indices = [self.hand.body_names.index(name) for name in cfg.fingertip_body_names]
        self._fixture_q = torch.tensor(
            [self.fixture["joint_position"][name] for name in self.hand.joint_names],
            device=self.device,
        )
        self._fixture_targets = torch.tensor(
            [self.fixture["joint_target"][name] for name in self.hand.joint_names],
            device=self.device,
        )
        self._fixture_pose = torch.tensor(self.fixture["object_pose"], device=self.device)
        if self.fixture["object_pose"][2] - cfg.grasp.retention_radius <= 2.0 * cfg.cube_half_extent:
            raise ValueError("Retention region must terminate the episode before the cube can reach the ground")
        self._pad_centers = torch.tensor(
            [self.fixture["pulps"][name]["center"] for name in cfg.fingertip_body_names],
            device=self.device,
        )
        self._pad_half_extents = torch.tensor(
            [self.fixture["pulps"][name]["half_extents"] for name in cfg.fingertip_body_names],
            device=self.device,
        )
        self._pad_normals = torch.tensor(
            [self.fixture["pulps"][name]["normal"] for name in cfg.fingertip_body_names],
            device=self.device,
        )
        self.grasp_state = GraspEpisodeState(self.num_envs, self.device)
        self.target_delta = torch.zeros_like(self.actions)
        self._fresh_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._normal = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self._friction = torch.zeros_like(self._normal)
        self._contact_positions = torch.zeros_like(self._normal)
        self._pulp_contact = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._slip_speed = torch.zeros(self.num_envs, 4, device=self.device)
        self._support_force = torch.zeros(self.num_envs, cfg.decimation, device=self.device)
        self._score = None
        self.in_hand_pos[:] = self._fixture_pose[:3]
        self.goal_rot.zero_()
        self.goal_rot[:, 3] = 1.0
        condition = {
            "version": 1,
            "grasp": asdict(cfg.grasp),
            "fixture": self.fixture,
            "dt": self.step_dt,
            "physics_dt": self.physics_dt,
            "horizon": self.max_episode_length,
            "max_joint_delta": cfg.max_joint_delta,
            "mass": cfg.object_mass,
            "backend": self._physics_backend,
            "hold_duration": cfg.hold_duration,
            "position_noise": cfg.reset_position_noise,
            "joint_noise": cfg.reset_dof_pos_noise,
            "pulp_tolerance": cfg.pulp_contact_tolerance,
            "pulp_alignment": cfg.pulp_normal_alignment,
            "cube_half_extent": cfg.cube_half_extent,
            "simulation": cfg.sim.to_dict(),
            "robot": cfg.robot_cfg.to_dict(),
            "object": cfg.object_cfg.to_dict(),
        }
        condition["simulation"].pop("device", None)
        for key in ("fixture", "robot", "object", "simulation"):
            if not condition[key]:
                raise ValueError(f"Empty physical comparison condition: {key}")
        self.condition_key = grasp_condition_key(condition)

    @staticmethod
    def _validate_cfg(cfg):
        backend = type(cfg.sim.physics).__name__.removesuffix("Cfg").lower()
        if backend == "ovphysx":
            raise NotImplementedError(
                "The installed isaaclab_ov ContactSensor does not implement friction-force or contact-point tracking; "
                "Stage A requires both. Use PhysX or a validated Newton configuration, not a normal-only fallback."
            )
        if backend == "newton" and not cfg.allow_unvalidated_backend:
            raise NotImplementedError(
                "Newton's full-force API is wired, but this LEAP collision/hold setup is not validated. "
                "Run leap_grasp_newton_smoke.py for explicit experimental validation; production training defaults to PhysX."
            )
        if not cfg.fingertip_contact_sensors:
            raise ValueError("Stage A requires enabled full-force contact sensors")
        if cfg.max_joint_delta <= 0 or cfg.hold_duration <= 0 or cfg.force_observation_scale <= 0:
            raise ValueError("Control increment, hold duration and force observation scale must be positive")
        if cfg.hold_duration > cfg.episode_length_s:
            raise ValueError("Hold duration must fit within the episode")
        if any(
            not 0 < lo < hi < limit
            for lo, hi, limit in zip(
                cfg.grasp.force_low,
                cfg.grasp.force_high,
                cfg.grasp.force_limit,
                strict=True,
            )
        ):
            raise ValueError("Each finger requires 0 < force_low < force_high < force_limit")

    def _spawn_hand_articulation(self):
        super()._spawn_hand_articulation()
        if self._physics_backend == "newton":
            from isaaclab.sim.utils.stage import get_current_stage

            from .newton_geometry import prepare_newton_hand_collisions

            self._newton_geometry_report = prepare_newton_hand_collisions(get_current_stage())

    def _activate_fingertip_contact_reporters(self):
        if self._physics_backend != "physx":
            return
        from pxr import UsdPhysics

        from isaaclab.sim.schemas.schemas import safe_set_attribute_on_usd_prim
        from isaaclab.sim.utils.stage import get_current_stage

        for prim in get_current_stage().Traverse():
            if not prim.GetPath().pathString.startswith("/World/envs/env_0/"):
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if "PhysxContactReportAPI" not in prim.GetAppliedSchemas():
                    prim.AddAppliedSchema("PhysxContactReportAPI")
                safe_set_attribute_on_usd_prim(prim, "physxContactReport:threshold", 0.0, camel_case=False)

    def _create_fingertip_contact_sensors(self, object_filter_exprs):
        if self._physics_backend == "physx":
            from isaaclab_physx.sensors.contact_sensor import (
                ContactSensor,
                ContactSensorCfg,
            )
        elif self._physics_backend == "newton":
            from isaaclab_newton.sensors.contact_sensor import (
                ContactSensor,
                ContactSensorCfg,
            )
        else:
            raise NotImplementedError(f"No complete contact measurement contract for {self._physics_backend}")
        for name in self.cfg.fingertip_body_names:
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_[^/]+/Robot/{name}",
                    filter_prim_paths_expr=list(object_filter_exprs),
                    track_contact_points=True,
                    track_friction_forces=True,
                    history_length=self.cfg.decimation,
                    update_period=0.0,
                    max_contact_data_count_per_prim=32,
                )
            )
            self.fingertip_contact_sensors[name] = sensor
            self.scene.sensors[f"grasp:{name}"] = sensor
        from pxr import UsdPhysics

        from isaaclab.sim.utils.stage import get_current_stage

        filters = []
        for prim in get_current_stage().Traverse():
            path = prim.GetPath().pathString
            if path.startswith("/World/envs/env_0/Robot/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if prim.GetName() not in self.cfg.fingertip_body_names:
                    filters.append(path.replace("/env_0/", "/env_[^/]+/"))
        if not filters:
            raise RuntimeError("No non-pulp hand bodies found for support-contact monitoring")
        self.support_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path=self.cfg.object_cfg.prim_path,
                filter_prim_paths_expr=filters,
                track_friction_forces=True,
                history_length=self.cfg.decimation,
                update_period=0.0,
                max_contact_data_count_per_prim=128,
            )
        )
        self.scene.sensors["grasp:non_pulp_support"] = self.support_sensor

    @property
    def fingertip_contact_forces(self):
        """Full normal + friction contact forces ON the pulps, in world axes [N]."""
        return self._normal + self._friction

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("Non-finite grasp action")
        self.extras["grasp_hold"] = {}
        self.actions = actions.clamp(-1.0, 1.0).clone()
        previous = self.cur_targets.clone()
        self.cur_targets = advance_joint_targets(
            previous,
            self.actions,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
            self.cfg.max_joint_delta,
        )
        self.prev_targets[:] = previous
        self.target_delta = (self.cur_targets - previous) / self.cfg.max_joint_delta
        self._fresh_reset.zero_()

    def _apply_action(self):
        self._set_joint_pos_target(
            target=self.cur_targets[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
        )

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()

    def _measure_grasp(self):
        self._compute_intermediate_values()
        samples = [
            read_contact_sample(self.fingertip_contact_sensors[name], self.num_envs, self.physics_dt)
            for name in self.cfg.fingertip_body_names
        ]
        self._normal = torch.stack([sample.normal for sample in samples], dim=1)
        self._friction = torch.stack([sample.friction for sample in samples], dim=1)
        self._contact_positions = torch.stack([sample.position for sample in samples], dim=1)
        point_valid = torch.stack([sample.position_valid for sample in samples], dim=1)
        normal_history = torch.stack([sample.history_normal for sample in samples], dim=2)
        friction_history = torch.stack([sample.history_friction for sample in samples], dim=2)
        self._support_force = read_support_load(self.support_sensor, self.num_envs)
        tip_pos = self.hand.data.body_link_pos_w.torch[:, self._tip_indices]
        tip_rot = self.hand.data.body_link_quat_w.torch[:, self._tip_indices]
        self._contact_positions = torch.where(point_valid[..., None], self._contact_positions, tip_pos)
        contact_local = quat_apply_inverse(tip_rot.flatten(0, 1), (self._contact_positions - tip_pos).flatten(0, 1))
        contact_local = contact_local.reshape(self.num_envs, 4, 3)
        normal_local = quat_apply_inverse(tip_rot.flatten(0, 1), self._normal.flatten(0, 1)).reshape(
            self.num_envs, 4, 3
        )
        normal_mag = torch.linalg.vector_norm(self._normal, dim=-1)
        alignment = -(normal_local * self._pad_normals).sum(dim=-1) / normal_mag.clamp_min(1e-6)
        in_pad = (
            (contact_local - self._pad_centers).abs() <= self._pad_half_extents + self.cfg.pulp_contact_tolerance
        ).all(-1)
        self._pulp_contact = point_valid & in_pad & (alignment >= self.cfg.pulp_normal_alignment)
        tip_vel = self.hand.data.body_link_vel_w.torch[:, self._tip_indices]
        tip_contact_vel = tip_vel[..., :3] + torch.cross(tip_vel[..., 3:], self._contact_positions - tip_pos, dim=-1)
        obj_pos = self.object.data.root_link_pos_w.torch
        obj_vel = self.object.data.root_link_vel_w.torch
        object_contact_vel = obj_vel[:, None, :3] + torch.cross(
            obj_vel[:, None, 3:].expand(-1, 4, -1),
            self._contact_positions - obj_pos[:, None],
            dim=-1,
        )
        relative = tip_contact_vel - object_contact_vel
        normal_unit = self._normal / normal_mag.unsqueeze(-1).clamp_min(1e-6)
        tangential = relative - (relative * normal_unit).sum(-1, keepdim=True) * normal_unit
        self._slip_speed = torch.where(point_valid, torch.linalg.vector_norm(tangential, dim=-1), 0.0)
        pad_world = tip_pos + quat_apply(
            tip_rot.flatten(0, 1),
            self._pad_centers.expand(self.num_envs, -1, -1).flatten(0, 1),
        ).reshape(self.num_envs, 4, 3)
        obj_rot = self.object_rot[:, None].expand(-1, 4, -1).reshape(-1, 4)
        pad_object = quat_apply_inverse(obj_rot, (pad_world - obj_pos[:, None]).flatten(0, 1)).reshape(
            self.num_envs, 4, 3
        )
        surface_distance = torch.linalg.vector_norm(
            (pad_object.abs() - self.cfg.cube_half_extent).clamp_min(0.0), dim=-1
        )
        distance = torch.linalg.vector_norm(self.object_pos - self.in_hand_pos, dim=-1)
        self._score = score_grasp(
            normal_history,
            friction_history,
            self._pulp_contact,
            surface_distance,
            self._slip_speed,
            self._support_force,
            distance,
            self.target_delta,
            self.cfg.grasp,
        )
        self._score.valid &= ~self._fresh_reset

    def _get_dones(self):
        self._measure_grasp()
        distance = torch.linalg.vector_norm(self.object_pos - self.in_hand_pos, dim=-1)
        dropped = distance >= self.cfg.grasp.retention_radius
        self._dropped = dropped | ~torch.isfinite(distance)
        return (
            self._dropped | self._score.unsafe,
            self.episode_length_buf >= self.max_episode_length,
        )

    def _get_rewards(self):
        self.grasp_state.advance(
            self._score.valid,
            self._score.overload,
            self.step_dt,
            self.cfg.hold_duration,
            supported=self._support_force.amax(dim=1) > self.cfg.grasp.support_force_limit,
        )
        self.successes[:] = self.grasp_state.success.float()
        reward = self._score.reward + self.cfg.fall_penalty * self._dropped.float()
        reward += self.cfg.unsafe_penalty * self._score.unsafe.float()
        log = self.extras.setdefault("log", {})
        log.update({f"GraspReward/{name}": value.mean() for name, value in self._score.terms.items()})
        log["Grasp/valid_fraction"] = self._score.valid.float().mean()
        log["Grasp/contact_count"] = self._score.finger_valid.float().sum(-1).mean()
        log["Grasp/hold_seconds"] = self.grasp_state.hold_seconds.mean()
        log["Grasp/normal_force_max"] = torch.linalg.vector_norm(self._normal, dim=-1).amax()
        log["Grasp/full_force_max"] = torch.linalg.vector_norm(self.fingertip_contact_forces, dim=-1).amax()
        log["Grasp/support_force"] = self._support_force.amax(dim=1).mean()
        log["Grasp/slip_speed"] = self._slip_speed.mean()
        return reward

    def _reset_target_pose(self, env_ids):
        self.reset_goal_buf[env_ids] = False

    def _reset_idx(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        ended = env_ids[self.grasp_state.steps[env_ids] > 0]
        if ended.numel():
            retained = self.reset_time_outs[ended] & ~self.reset_terminated[ended]
            metrics = self.grasp_state.completed(ended, retained, self.max_episode_length)
            metrics["condition_key"] = self.condition_key
            self.extras["grasp_hold"] = metrics
            log = self.extras.setdefault("log", {})
            log["GraspEpisode/raw_retained"] = retained.float().mean()
            log["GraspEpisode/retained"] = metrics["completed_retained"].mean()
            log["GraspEpisode/valid_fraction"] = metrics["completed_valid_fraction"].mean()
            log["GraspEpisode/overload_fraction"] = metrics["completed_overload_fraction"].mean()
            log["GraspEpisode/success_rate"] = self.grasp_state.success[ended].float().mean()
            log["GraspEpisode/longest_hold"] = self.grasp_state.longest_hold[ended].mean()
        DirectRLEnv._reset_idx(self, env_ids)
        count = len(env_ids)
        q = self._fixture_q.expand(count, -1).clone()
        q += (2.0 * torch.rand_like(q) - 1.0) * self.cfg.reset_dof_pos_noise
        q = torch.clamp(q, self.hand_dof_lower_limits[env_ids], self.hand_dof_upper_limits[env_ids])
        targets = self._fixture_targets.expand(count, -1).clone()
        pose = self._fixture_pose.expand(count, -1).clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        pose[:, :3] += (2.0 * torch.rand(count, 3, device=self.device) - 1.0) * self.cfg.reset_position_noise
        self._write_obj_root_pose(root_pose=pose, env_ids=env_ids)
        self._write_obj_root_vel(root_velocity=torch.zeros(count, 6, device=self.device), env_ids=env_ids)
        self._write_hand_joint_pos(position=q, env_ids=env_ids)
        self._write_hand_joint_vel(velocity=torch.zeros_like(q), env_ids=env_ids)
        self._set_joint_pos_target(target=targets, env_ids=env_ids)
        self.cur_targets[env_ids] = targets
        self.prev_targets[env_ids] = targets
        self.actions[env_ids] = 0.0
        self.target_delta[env_ids] = 0.0
        self.grasp_state.reset(env_ids)
        self.successes[env_ids] = 0
        self.reset_goal_buf[env_ids] = False
        self._fresh_reset[env_ids] = True
        for tensor in (
            self._normal,
            self._friction,
            self._contact_positions,
            self._pulp_contact,
            self._slip_speed,
            self._support_force,
        ):
            tensor[env_ids] = 0
        self._compute_intermediate_values()

    def _get_observations(self):
        self._compute_intermediate_values()
        q = (
            2.0
            * (self.hand_dof_pos - self.hand_dof_lower_limits)
            / (self.hand_dof_upper_limits - self.hand_dof_lower_limits)
            - 1.0
        )
        targets = (
            2.0
            * (self.cur_targets - self.hand_dof_lower_limits)
            / (self.hand_dof_upper_limits - self.hand_dof_lower_limits)
            - 1.0
        )
        point_relative = self._contact_positions - (self.object_pos + self.scene.env_origins)[:, None]
        point_relative = torch.where(self._pulp_contact[..., None], point_relative, 0.0)
        obs = torch.cat(
            (
                q,
                self.hand_dof_vel * 0.1,
                targets,
                self.actions,
                self.object_pos - self.in_hand_pos,
                self.object_rot,
                self.object_linvel,
                self.object_angvel * 0.1,
                self.fingertip_contact_forces.flatten(1) / self.cfg.force_observation_scale,
                self._normal.flatten(1) / self.cfg.force_observation_scale,
                point_relative.flatten(1),
                self._pulp_contact.float(),
                self._slip_speed,
                self._support_force.amax(dim=1, keepdim=True) / self.cfg.force_observation_scale,
                (self.grasp_state.hold_seconds / self.cfg.hold_duration).clamp_max(1.0)[:, None],
                (1.0 - self.episode_length_buf.float() / self.max_episode_length).clamp_min(0.0)[:, None],
            ),
            dim=-1,
        )
        return {"policy": obs}

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        if state is not None:
            raise ValueError("Stage A has no checkpointed curriculum; refusing incompatible environment state")
