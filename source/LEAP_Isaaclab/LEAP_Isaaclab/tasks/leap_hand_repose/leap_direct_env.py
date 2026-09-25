# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Direct in-hand cube reorientation with the LEAP hand, using the 7-term
# reference reward function (see the manager-based sibling mdp/rewards.py):
#
#   r_t = c1 * 1(task successful)                (reach_goal_bonus)
#       + c2 / (|theta_t| + eps_theta)           (rot_reward_scale / rot_eps)
#       + c3 * sum_i ||p^f_i - p^o||^2           (fingertip_dist_penalty_scale)
#       + c4 * |qdot_t|^T |tau_t|                (joint_energy_penalty_scale)
#       + c5 * 1(||p^o||^2 > p_bar)              (object_away_penalty_scale)
#       + c6 * 1(object contacts the table)      (object_drop_penalty_scale, REMOVED: 0.0)
#       + c7 * sum_i 1(fingertip over-curled)    (fingertip_overcurl_penalty_scale, REMOVED: 0.0)
#       + c8 * sum_i 1(pulp force > f_contact)  (fingertip_contact_scale)
#       + c9 * 1(object pinned on the palm)     (palm_contact_penalty_scale)
#       + c10 * 1(at goal, not palm-pinned)     (goal_press_scale)
#         * sum_i 1(pulp force > f_press)
#
# Terms 6/7 are computed but zero-scaled by default (removed at user request,
# 2026-09-08) -- set their cfg scales nonzero to restore any of them. The
# tactile terms 8/10 are gated on the MEASURED per-pulp contact forces from
# the PhysX ContactSensors (filtered vs the in-hand object) -- exactly the
# signal the real tactile pulps will report -- and the observations carry the
# same forces as 12 extra channels (4 pulps x 3 components, scaled by
# force_torque_obs_scale; observation_space 124 -> 136). c9 + the c10 palm
# gate together make palm-crushing (object sandwiched against the palm) earn
# zero press bonus and a penalty, so a symmetric fingertip grasp dominates.
#
# Observations, actions, resets, and goal bookkeeping are inherited unchanged
# from the official shared InHandManipulationEnv; only _get_rewards differs.
# --------------------------------------------------------

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim import spawners as sim_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from LEAP_Isaaclab.compat.inhand_direct.inhand_manipulation_env import (
    InHandManipulationEnv,
    rotation_distance,
)

from .leap_direct_env_cfg import LeapHandDirectEnvCfg


class LeapHandDirectEnv(InHandManipulationEnv):
    """InHandManipulationEnv with the LEAP hand and the reference reward function.

    Identical to the official ``Isaac-Repose-Cube-LEAP-Direct-v0`` environment
    except for the reward: the official dense object-position term is disabled
    by default (``dist_reward_scale = 0``), the five shaping terms of the
    reference reward are added, and two tactile-oriented terms reward
    fingertip-contact manipulation (see ``leap_direct_env.py`` header). The
    official behavior is restored by setting ``dist_reward_scale = -10.0`` and
    zeroing the new scales in the cfg.
    """

    cfg: LeapHandDirectEnvCfg

    def __init__(self, cfg: LeapHandDirectEnvCfg, render_mode: str | None = None, **kwargs):
        # populated inside _setup_scene (which runs within super().__init__)
        self.fingertip_contact_sensors: dict = {}
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        # official InHand scene, plus PhysX contact reporters on the pulp links
        # (pre-clone) and per-pulp ContactSensors (post-clone) -- see the
        # tactile notes in the module header
        self._spawn_hand_articulation()
        self.object: Articulation | RigidObject = self.cfg.object_cfg.class_type(self.cfg.object_cfg)
        self._joint_wrench_sensor = None
        if self.cfg.asymmetric_obs:
            self._joint_wrench_sensor = self._create_joint_wrench_sensor()
        self._activate_fingertip_contact_reporters()
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (reporter schemas replicate to every env)
        from LEAP_Isaaclab.compat.scene import clone_environments

        clone_environments(self.scene)
        # register entities (same keys as the official env)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        if self._joint_wrench_sensor is not None:
            self.scene.sensors["joint_wrench"] = self._joint_wrench_sensor
        self._create_fingertip_contact_sensors([self.cfg.object_cfg.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_hand_articulation(self) -> None:
        """Spawn the hand with backend-appropriate root fixing.

        Shared by this env's ``_setup_scene`` and the objects env's override
        (which does not call ``super()._setup_scene``).
        """
        if self._physics_backend != "physx":
            # The stock fix_root_link path needs omni.physx (Kit-only) to author the
            # world joint. Under kit-less backends we drop it and instead pin the
            # hand root pxr-style after spawn (see _fix_root_link_kitless).
            articulation_props = getattr(self.cfg.robot_cfg.spawn, "articulation_props", None)
            if articulation_props is not None:
                articulation_props.fix_root_link = False
        self.hand = Articulation(self.cfg.robot_cfg)
        if self._physics_backend != "physx":
            self._fix_root_link_kitless()

    def _fix_root_link_kitless(self) -> None:
        """pxr-only replacement for ``ArticulationRootPropertiesCfg.fix_root_link``.

        The stock fix_root_link path authors the world joint via omni.physx
        (Kit-only). Under kit-less backends the same USD authoring is done
        directly: a ``UsdPhysics.FixedJoint`` welding the hand root link to the
        world frame, authored on the env_0 template BEFORE cloning so every env
        gets its own joint. Newton's USD importer compiles the jointed root as a
        static base -- the same convention as the official Shadow Hand newton
        USD (``shadow_hand_instanceable_newton.usd``).
        """
        from pxr import UsdPhysics

        from isaaclab.sim.utils.stage import get_current_stage

        stage = get_current_stage()
        template = "/World/envs/env_0/Robot"
        root_prim = None
        for prim in stage.Traverse():
            if not prim.GetPath().pathString.startswith(template):
                continue
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                root_prim = prim
                break
        if root_prim is None:
            raise RuntimeError(f"[leap-repose] no ArticulationRootAPI prim found under {template}")
        if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise NotImplementedError(
                f"[leap-repose] articulation prim '{root_prim.GetPath()}' has no RigidBodyAPI; "
                "cannot weld it to the world (same limitation as the stock fix_root_link)"
            )
        joint = UsdPhysics.FixedJoint.Define(stage, root_prim.GetPath().AppendChild("world_fixed_joint"))
        # body0 unset = the world frame; identity local poses weld the root exactly
        # at its authored spawn transform (the reset root-pose writes still land).
        joint.CreateBody1Rel().SetTargets([root_prim.GetPath()])
        print(f"[leap-repose] kit-less root weld: {joint.GetPath()} body1 -> {root_prim.GetPath()}")

    def _activate_fingertip_contact_reporters(self) -> None:
        """Author PhysxContactReportAPI on the four pulp links of env_0's template.

        Must run BEFORE cloning -- the cloner replicates applied schemas to
        every env, so one authoring pass on the template suffices. Imports pxr
        lazily: module-level pxr imports before SimulationApp corrupt the PhysX
        Kit bindings (same trap as assets/kuka_leap.py).
        """
        if self._physics_backend != "physx":
            return  # PhysxContactReportAPI is a PhysX schema; kit-less backends have no reporter to activate
        from isaaclab.sim.schemas.schemas import safe_set_attribute_on_usd_prim
        from isaaclab.sim.utils.stage import get_current_stage

        stage = get_current_stage()
        template = "/World/envs/env_0/Robot"
        num_activated = 0
        for prim in stage.Traverse():
            if not prim.GetPath().pathString.startswith(template):
                continue
            if prim.GetName() in self.cfg.fingertip_body_names:
                if "PhysxContactReportAPI" not in prim.GetAppliedSchemas():
                    prim.AddAppliedSchema("PhysxContactReportAPI")
                safe_set_attribute_on_usd_prim(prim, "physxContactReport:threshold", 0.0, camel_case=False)
                num_activated += 1
        if num_activated != len(self.cfg.fingertip_body_names):
            raise RuntimeError(
                f"Expected {len(self.cfg.fingertip_body_names)} pulp links under {template}, "
                f"found {num_activated} ({self.cfg.fingertip_body_names})"
            )

    def _create_fingertip_contact_sensors(self, object_filter_exprs: list[str]) -> None:
        """One filtered ContactSensor per pulp link; run AFTER cloning.

        Each sensor watches a single body per env (required for filtered
        per-partner force reporting) and reports the net contact force on that
        pulp against ``object_filter_exprs`` -- the in-hand object(s). Registered
        in ``scene.sensors`` so ``scene.update(dt)`` refreshes the buffers.
        """
        if not getattr(self.cfg, "fingertip_contact_sensors", True):
            return
        if self._physics_backend == "physx":
            from isaaclab_physx.sensors.contact_sensor import ContactSensor, ContactSensorCfg
        else:
            # kit-less backends: the isaaclab_newton sensor takes the same
            # prim_path + filter_prim_paths_expr interface (it warns and ignores
            # PhysX-only fields like max_contact_data_count_per_prim)
            from isaaclab_newton.sensors.contact_sensor import ContactSensor, ContactSensorCfg

        for body_name in self.cfg.fingertip_body_names:
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{body_name}",
                    filter_prim_paths_expr=list(object_filter_exprs),
                    history_length=0,
                    update_period=0.0,
                    max_contact_data_count_per_prim=16,
                )
            )
            self.fingertip_contact_sensors[body_name] = sensor
            self.scene.sensors[f"fingertip_contact:{body_name}"] = sensor
        print(
            f"[leap-repose] pulp contact sensors ON: {len(self.fingertip_contact_sensors)} tips x "
            f"{self.num_envs} envs, filtered vs {len(object_filter_exprs)} object expr(s)"
        )

    @property
    def _physics_backend(self) -> str:
        """"physx" (Isaac Sim PhysX), "newton" or "ovphysx" (kit-less) from the sim cfg's physics class."""
        return type(self.cfg.sim.physics).__name__.removesuffix("Cfg").lower()

    @property
    def fingertip_contact_forces(self) -> torch.Tensor:
        """(num_envs, num_fingertips, 3) net contact forces on the pulps [N].

        Zero when the sensors are disabled (``fingertip_contact_sensors`` off).
        Each sensor reports (num_envs, 1, 3) -- one body per env -- so the
        singleton body dim is squeezed here (leaving it in would broadcast the
        reward counts (N, 1) against (N,) terms into an (N, N) outer product).
        """
        if not self.fingertip_contact_sensors:
            return torch.zeros((self.num_envs, len(self.cfg.fingertip_body_names), 3), device=self.device)
        forces = []
        for name in self.cfg.fingertip_body_names:
            data = self.fingertip_contact_sensors[name].data
            # EA separates normal/friction forces. PhysX reports normal forces;
            # the partner matrix, unlike net_forces_w, excludes hand/ground contact.
            matrix = data.normal_force_matrix_w if self._physics_backend == "physx" else data.force_matrix_w
            if matrix is None:
                raise RuntimeError(f"Filtered fingertip contact data unavailable for {name} on {self._physics_backend}")
            forces.append(matrix.torch.reshape(self.num_envs, -1, 3).sum(dim=1))
        return torch.stack(forces, dim=1)

    def _get_observations(self) -> dict:
        # official 124-d "full" obs + 12 tactile channels (4 pulps x 3 filtered
        # contact-force components), scaled like the official force/torque obs
        observations = super()._get_observations()
        tactile = self.fingertip_contact_forces.flatten(1) * self.cfg.force_torque_obs_scale
        observations["policy"] = torch.cat([observations["policy"], tactile], dim=-1)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        hand_root_pos = self.hand.data.root_pos_w.torch - self.scene.env_origins
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_leap_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
            # reference shaping terms
            self.fingertip_pos,
            self.hand_dof_vel,
            self.hand.data.applied_torque.torch,
            hand_root_pos,
            self.cfg.fingertip_dist_penalty_scale,
            self.cfg.joint_energy_penalty_scale,
            self.cfg.object_away_penalty_scale,
            self.cfg.object_away_dist,
            self.cfg.object_drop_penalty_scale,
            self.cfg.object_drop_z_offset,
            self.cfg.fingertip_overcurl_penalty_scale,
            self.cfg.fingertip_overcurl_z_offset,
            # tactile terms (trailing args; see the fn docstring)
            self.fingertip_contact_forces,
            self.cfg.fingertip_contact_scale,
            self.cfg.fingertip_contact_force,
            self.cfg.palm_contact_dist,
            self.cfg.palm_contact_penalty_scale,
            self.cfg.goal_press_scale,
            self.cfg.goal_press_tolerance,
            self.cfg.goal_press_force,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward


@torch.jit.script
def compute_leap_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
    fingertip_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    applied_torque: torch.Tensor,
    hand_root_pos: torch.Tensor,
    fingertip_dist_penalty_scale: float,
    joint_energy_penalty_scale: float,
    object_away_penalty_scale: float,
    object_away_dist: float,
    object_drop_penalty_scale: float,
    object_drop_z_offset: float,
    fingertip_overcurl_penalty_scale: float,
    fingertip_overcurl_z_offset: float,
    fingertip_forces: torch.Tensor,
    fingertip_contact_scale: float,
    fingertip_contact_force: float,
    palm_contact_dist: float,
    palm_contact_penalty_scale: float,
    goal_press_scale: float,
    goal_press_tolerance: float,
    goal_press_force: float,
):
    """Official in-hand reward extended with the reference shaping terms.

    All positions are relative to the env origins. ``fingertip_pos`` is
    (num_envs, num_fingertips, 3) and ``object_pos``/``hand_root_pos`` are
    (num_envs, 3). ``fingertip_forces`` is (num_envs, num_fingertips, 3) --
    the MEASURED filtered net contact force on each pulp [N]; the tactile
    terms 8/10 are gated on its magnitude, i.e. exactly the signal the real
    tactile pulps will report.

    NOTE: ``success_tolerance`` must stay at positional index 14 (the FlashSAC
    hold-to-confirm hook relies on it and passes trailing args through).
    """
    goal_dist = torch.linalg.norm(object_pos - target_pos, ord=2, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale
    action_penalty = torch.sum(actions**2, dim=-1)

    # term 3: sum of squared fingertip-to-object distances
    fingertip_dist = torch.sum(torch.sum((fingertip_pos - object_pos.unsqueeze(1)) ** 2, dim=2), dim=1)
    # term 4: mechanical joint energy |qdot|^T |tau|
    joint_energy = torch.sum(torch.abs(joint_vel) * torch.abs(applied_torque), dim=-1)
    # term 5: object pushed away from the in-hand position
    object_away = (goal_dist > object_away_dist).to(action_penalty.dtype)
    # term 6: object dropped below the hand (no table in this scene)
    object_drop = ((object_pos[:, 2] - hand_root_pos[:, 2]) < object_drop_z_offset).to(action_penalty.dtype)
    # term 7: fingertips over-curled below the palm-base plane (the LEAP palm is
    # vertical: fingers extend along world +z and curl back toward the palm base
    # through the hand root; tips tucked below that plane are closed past the
    # object, which then rests on the penultimate joints -- see mdp/rewards.py)
    fingertip_overcurl = torch.sum(
        (fingertip_pos[:, :, 2] - hand_root_pos[:, 2:3] < fingertip_overcurl_z_offset).to(action_penalty.dtype), dim=1
    )
    # term 8: pulp contact -- reward each tip whose MEASURED filtered contact
    # force (vs the in-hand object) exceeds the threshold; tactile sensors
    # only sense on contact, so rich multi-tip contact coverage is what makes
    # the task tactile-driven
    tip_force_mag = torch.sqrt(torch.sum(fingertip_forces**2, dim=-1))  # (N, F)
    fingertip_contact = torch.sum(
        (tip_force_mag > fingertip_contact_force).to(action_penalty.dtype), dim=1
    )
    # term 9: object pinned against the palm -- the uninstrumented palm can
    # carry the object without any fingertip involvement, defeating the
    # tactile-purpose of the task (object held by the fingers rides ~9 cm from
    # the hand root; below palm_contact_dist it is tucked into the palm)
    palm_dist = torch.sqrt(torch.sum((object_pos - hand_root_pos) ** 2, dim=-1))
    palm_contact = (palm_dist < palm_contact_dist).to(action_penalty.dtype)
    # term 10: goal-state press -- with the repose DONE (within
    # goal_press_tolerance of the goal rotation) AND the object NOT palm-
    # pinned, reward each pulp whose contact force exceeds the press
    # threshold, i.e. finishing with every pulp preloaded on the object
    # (tactile preload). The palm gate makes the palm-crush degenerate mode
    # earn ZERO press bonus (it still pays the term-9 penalty), so a symmetric
    # fingertip grasp strictly dominates it. Gated on its OWN tolerance knob:
    # the FlashSAC hold-hook overwrites success_tolerance (positional arg 14)
    # with -1 inside this function.
    fingertip_press = torch.sum((tip_force_mag > goal_press_force).to(action_penalty.dtype), dim=1)
    at_goal = (torch.abs(rot_dist) <= goal_press_tolerance).to(action_penalty.dtype)
    press_gate = at_goal * (1.0 - palm_contact)

    # Total reward is: position distance + orientation alignment + action regularization
    # + the reference shaping terms + tactile-oriented terms + success bonus + fall penalty
    reward = (
        dist_rew
        + rot_rew
        + action_penalty * action_penalty_scale
        + fingertip_dist * fingertip_dist_penalty_scale
        + joint_energy * joint_energy_penalty_scale
        + object_away * object_away_penalty_scale
        + object_drop * object_drop_penalty_scale
        + fingertip_overcurl * fingertip_overcurl_penalty_scale
        + fingertip_contact * fingertip_contact_scale
        + palm_contact * palm_contact_penalty_scale
        + fingertip_press * goal_press_scale * press_gate
    )

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes
