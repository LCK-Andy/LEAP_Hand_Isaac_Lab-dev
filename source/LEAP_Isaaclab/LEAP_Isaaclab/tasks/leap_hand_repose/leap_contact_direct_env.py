# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Contact-point dexterity benchmark: instead of a goal cube orientation, 4
# contact points are sampled uniformly on the cube's surface (cube-local
# frame, minimum pairwise separation) and the hand must press a fingertip
# pulp onto each of them:
#
#   r_t = -c1 * sum_j d(matched pulp_j, point_j)   (point_dist_reward_scale)
#       + c2 * sum_j 1(point_j touched)            (touch_bonus_scale)
#       + 250  on all-4 held for the hold time     (reach_goal_bonus)
#       - 0.0002 * ||a||^2                          (action_penalty_scale)
#       - 0.001 * |qdot|^T |tau|                    (joint_energy_penalty_scale)
#       - 1.0 * 1(object pushed away)               (object_away_penalty_scale)
#
# "Touched" = the one-to-one MATCHED pulp is within `touch_distance` of the
# point AND reports more than `touch_force` of cube-filtered contact force --
# the same per-pulp ContactSensors the repose tasks use, so a pulp hovering a
# centimeter off the surface never counts. The matching is exact: the tip->point
# assignment is the argmin over all 4! = 24 one-to-one pairings, so no pulp
# covers two points and no pairing is forced. Success uses a NATIVE hold timer
# (per-env duration sampled from `hold_time_range`): all 4 points must be
# touched CONTINUOUSLY for that long before the bonus fires and the points
# resample (mid-episode, like the base task's goal resample).
#
# All 6 faces are sampled, including regions facing the palm: those are only
# reachable by REGRASPING the cube with the free fingers, which is the point
# of the benchmark. The orientation goal is gone entirely -- the inherited
# goal_rot channels in the obs stay at identity and degrade to a fixed-
# reference orientation encoding.
#
# Observations: the 136-d repose obs (124 official "full" + 12 tactile) plus
# the 4x4 tip->point relative-vector matrix in env frame (48 d) = 184 d. The
# reward fn is deliberately named `compute_contact_rewards` (NOT
# `compute_leap_rewards`): the patched FlashSAC hold-hook resolves the latter
# name and would double-apply hold semantics -- hold is native here, so the
# hook stays inactive by construction. Backends: physx (reference); ovphysx
# (kit-less PhysX, experimental -- sensors re-plumbed to isaaclab_ovphysx,
# force reporting verified by the kit-less probe); newton only in geometry-only
# mode (touch_force <= 0 drops the force gate -- sensor forces read zero there).
# --------------------------------------------------------

from __future__ import annotations

import itertools

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply, sample_uniform
from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import randomize_rotation

from .leap_contact_direct_env_cfg import LeapContactDirectEnvCfg
from .leap_direct_env import LeapHandDirectEnv


class LeapContactDirectEnv(LeapHandDirectEnv):
    """Touch-4-contact-points dexterity benchmark on the in-hand cube.

    The MDP of ``Isaac-Repose-Cube-LEAP-Direct-v0`` with the orientation goal
    replaced by 4 cube-surface contact points: the policy must press a pulp on
    each point (one-to-one matched, force-gated) and hold all 4 contacts for a
    per-env sampled duration to score and resample the points.
    """

    cfg: LeapContactDirectEnvCfg

    def __init__(self, cfg: LeapContactDirectEnvCfg, render_mode: str | None = None, **kwargs):
        backend = type(cfg.sim.physics).__name__.removesuffix("Cfg").lower()
        if backend != "physx" and backend != "ovphysx" and cfg.touch_force > 0:
            raise NotImplementedError(
                "[leap-contact] the touch test gates on ContactSensor forces, which read zero "
                f"under the '{backend}' kit-less backend (upstream reporting gap). Use the physx "
                "backend, or set touch_force <= 0 for the geometry-only touch test."
            )
        # contact-point buffers are created lazily by _ensure_contact_state():
        # _reset_target_pose runs inside super().__init__ (via _reset_idx), and
        # num_envs/device only exist once DirectRLEnv.__init__ is underway.
        self._contact_state_ready = False
        super().__init__(cfg, render_mode, **kwargs)

        # fingertip_pos columns follow self.finger_bodies (sorted body indices)
        # while fingertip_contact_forces stacks in cfg.fingertip_body_names
        # order -- permute the forces into finger_bodies order so the per-tip
        # pairing (distance <-> force) in the touch test is correct.
        name_to_sensor = {name: i for i, name in enumerate(self.cfg.fingertip_body_names)}
        self._contact_force_perm = torch.tensor(
            [name_to_sensor[self.hand.body_names[b]] for b in self.finger_bodies],
            device=self.device,
            dtype=torch.long,
        )

        # debug markers: 4 colored spheres glued to the sampled points, world
        # poses recomputed every step (the cube moves -- regrasping is the
        # task). No backend exists headless, so visualize() no-ops there.
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
        self._contact_point_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/contact_points",
                markers={
                    f"point_{i}": sim_utils.SphereCfg(
                        radius=self.cfg.contact_marker_radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    )
                    for i, color in enumerate(colors)
                },
            )
        )

    ##
    # Contact-point state
    ##

    def _setup_scene(self):
        super()._setup_scene()
        if self._physics_backend != "physx":
            return
        # Author the PhysX GPU scene buffer sizes as USD attributes directly:
        # this isaaclab generation never consumes the PhysxCfg
        # gpu_max_rigid_patch_count fields (zero references in the physx
        # manager), and the default 163840-patch buffer DROPS contact patches
        # from ~12k envs on (~250k needed at 16384), silently corrupting the
        # force-gated touch test. Authored pre-sim-reset like the objects env
        # (leap_objects_direct_env: partially effective there; verified here
        # by the absence of "Patch buffer overflow" in the training log).
        from isaaclab.sim.schemas.schemas import safe_set_attribute_on_usd_prim
        from isaaclab.sim.utils.stage import get_current_stage

        scene_prim = get_current_stage().GetPrimAtPath(self.cfg.sim.physics_prim_path)
        if not scene_prim.IsValid():
            print(f"[leap-contact] WARNING: physics scene prim '{self.cfg.sim.physics_prim_path}' invalid; "
                  "PhysX buffer attrs not authored")
            return
        safe_set_attribute_on_usd_prim(scene_prim, "physxScene:maxRigidPatchCount", 2**19, camel_case=False)
        safe_set_attribute_on_usd_prim(scene_prim, "physxScene:maxRigidContactCount", 2**24, camel_case=False)

    def _activate_fingertip_contact_reporters(self) -> None:
        # ovphysx honors the same PhysX-style contact-report schema as physx
        # (authoring an unrecognized applied schema is inert metadata anyway);
        # every other backend defers to the parent (a no-op for newton).
        if self._physics_backend != "ovphysx":
            super()._activate_fingertip_contact_reporters()
            return
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
        # the parent falls back to the isaaclab_newton sensor for EVERY non-physx
        # backend; ovphysx ships its own ContactSensor with the same interface
        if self._physics_backend != "ovphysx":
            super()._create_fingertip_contact_sensors(object_filter_exprs)
            return
        if not getattr(self.cfg, "fingertip_contact_sensors", True):
            return
        from isaaclab_ovphysx.sensors.contact_sensor import ContactSensor, ContactSensorCfg

        for body_name in self.cfg.fingertip_body_names:
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{body_name}",
                    filter_prim_paths_expr=list(object_filter_exprs),
                    history_length=0,
                    update_period=0.0,
                )
            )
            self.fingertip_contact_sensors[body_name] = sensor
            self.scene.sensors[f"fingertip_contact:{body_name}"] = sensor
        print(
            f"[leap-contact] ovphysx pulp contact sensors ON: {len(self.fingertip_contact_sensors)} tips x "
            f"{self.num_envs} envs, filtered vs {len(object_filter_exprs)} object expr(s)"
        )

    def _ensure_contact_state(self) -> None:
        """Allocate the contact-point buffers once (lazily, inside super().__init__)."""
        if self._contact_state_ready:
            return
        self.contact_points_local = torch.zeros((self.num_envs, 4, 3), device=self.device)
        self.hold_timer = torch.zeros(self.num_envs, device=self.device)
        self.hold_target = torch.full((self.num_envs,), float(self.cfg.hold_time_range[0]), device=self.device)
        # per-step intermediates (filled by _compute_intermediate_values)
        self.contact_points_env = torch.zeros((self.num_envs, 4, 3), device=self.device)
        self.tip_point_deltas = torch.zeros((self.num_envs, 4, 4, 3), device=self.device)
        self.matched_dist = torch.zeros((self.num_envs, 4), device=self.device)
        self.touched = torch.zeros((self.num_envs, 4), device=self.device)
        # all 4! one-to-one tip->point pairings for the exact assignment
        self._contact_perms = torch.tensor(
            list(itertools.permutations(range(4))), dtype=torch.long, device=self.device
        )
        self._contact_state_ready = True

    def _draw_face_points(self, count: int) -> torch.Tensor:
        """Uniform sampling on the cube surface: face uniform over the 6 (equal-area) faces, (u, v) uniform."""
        half = self.cfg.cube_half_extent
        faces = torch.randint(0, 6, (count, 4), device=self.device)
        uv = (torch.rand(count, 4, 2, device=self.device) * 2.0 - 1.0) * half
        points = torch.zeros(count, 4, 3, device=self.device)
        axis = torch.div(faces, 2, rounding_mode="floor")
        signed = torch.where(faces % 2 == 0, half, -half)
        for a in range(3):
            on_face = axis == a
            others = [d for d in range(3) if d != a]
            points[..., others[0]] = torch.where(on_face, uv[..., 0], points[..., others[0]])
            points[..., others[1]] = torch.where(on_face, uv[..., 1], points[..., others[1]])
            points[..., a] = torch.where(on_face, signed, points[..., a])
        return points

    def _sample_contact_points(self, env_ids) -> None:
        """Sample a min-separated 4-point set per env (rejection resampling, capped)."""
        points = self._draw_face_points(len(env_ids))
        for _ in range(self.cfg.max_point_resamples):
            pairwise = torch.cdist(points, points)  # (n, 4, 4)
            diag = torch.arange(4, device=self.device)
            pairwise[:, diag, diag] = float("inf")
            too_close = pairwise.amin(dim=(1, 2)) < self.cfg.min_point_separation
            if not too_close.any():
                break
            points[too_close] = self._draw_face_points(int(too_close.sum()))
        self.contact_points_local[env_ids] = points

    ##
    # DirectRLEnv overrides
    ##

    def _reset_target_pose(self, env_ids) -> None:
        # Intentionally does NOT call super(): there is no orientation goal to
        # resample and no ghost-cube marker to draw. goal_rot keeps its
        # identity value from the parent __init__ (the inherited obs channels
        # become a fixed-reference orientation encoding).
        self._ensure_contact_state()
        self._sample_contact_points(env_ids)
        low, high = self.cfg.hold_time_range
        self.hold_target[env_ids] = low + (high - low) * torch.rand(len(env_ids), device=self.device)
        self.hold_timer[env_ids] = 0.0
        self.reset_goal_buf[env_ids] = 0

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        rot_scale = float(self.cfg.reset_rot_scale)
        if rot_scale >= 1.0:
            return  # parent already applied the full official X/Y randomization
        # re-write the object pose with the spawn rotation SCALED DOWN
        # (reset_rot_scale 0 = axis-aligned cube; raise toward 1.0 as curriculum).
        # The random rotation dropped into open fingers against the vertical palm
        # is a big part of the instant-drop death spiral, so v2 starts aligned.
        # Position/noise replicate the parent exactly; velocity stays zeroed.
        pose = self.object.data.default_root_pose.torch.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        pose[:, 0:3] = pose[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        pose[:, 3:7] = randomize_rotation(
            rot_noise[:, 0] * rot_scale,
            rot_noise[:, 1] * rot_scale,
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )
        self._write_obj_root_pose(root_pose=pose, env_ids=env_ids)
        self._compute_intermediate_values()

    def _compute_intermediate_values(self) -> None:
        super()._compute_intermediate_values()
        self._ensure_contact_state()

        # cube-local -> env-frame contact points (object_pos is env-relative).
        # The compiled quat helpers need FLAT 2-D inputs (mid-dim broadcasting
        # fails inside TorchScript), so expand the per-env quat and flatten.
        quat = self.object_rot.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
        self.contact_points_env = self.object_pos.unsqueeze(1) + quat_apply(
            quat, self.contact_points_local.reshape(-1, 3)
        ).reshape(self.num_envs, 4, 3)
        # pulp forces in the fingertip_pos column order (see _contact_force_perm)
        force_mag = torch.linalg.norm(self.fingertip_contact_forces[:, self._contact_force_perm], dim=-1)

        # tip->point relative vectors and pairwise distance matrix (tips x points)
        self.tip_point_deltas = self.fingertip_pos.unsqueeze(2) - self.contact_points_env.unsqueeze(1)
        dist_mat = torch.linalg.norm(self.tip_point_deltas, dim=-1)  # (N, tips, points)

        # exact one-to-one assignment: argmin over all 24 tip-permutations
        perm_index = self._contact_perms.view(1, 24, 4)
        cost = torch.gather(dist_mat, 1, perm_index.expand(self.num_envs, 24, 4)).sum(-1)  # (N, 24)
        best = cost.argmin(-1)
        matched_tip = self._contact_perms[best]  # (N, 4) tip index per point
        self.matched_dist = dist_mat.gather(1, matched_tip.unsqueeze(-1)).squeeze(-1)  # (N, 4)
        matched_force = force_mag.gather(1, matched_tip)

        # touch test: distance gate AND cube-filtered force on the matched pulp.
        # touch_force <= 0 = geometry-only mode (kit-less newton fallback: the
        # force gate reads zero there, so the gate must drop out entirely --
        # `force > 0.0` against all-zero readings would never fire).
        dist_ok = self.matched_dist < self.cfg.touch_distance
        if self.cfg.touch_force > 0:
            self.touched = (dist_ok & (matched_force > self.cfg.touch_force)).float()
        else:
            self.touched = dist_ok.float()
        all_touched = self.touched.sum(-1) == self.contact_points_local.shape[1]
        # native hold-to-confirm: accumulate only while all points are touched
        self.hold_timer = torch.where(
            all_touched, self.hold_timer + self.step_dt, torch.zeros_like(self.hold_timer)
        )

        self._update_contact_point_markers()

    def _update_contact_point_markers(self) -> None:
        markers = getattr(self, "_contact_point_markers", None)
        if markers is None:
            return
        translations = (self.contact_points_env + self.scene.env_origins.unsqueeze(1)).reshape(-1, 3)
        orientations = torch.zeros(self.num_envs * 4, 4, device=self.device)  # xyzw identity
        orientations[:, 3] = 1.0
        marker_indices = torch.arange(4, device=self.device).repeat(self.num_envs)
        markers.visualize(translations=translations, orientations=orientations, marker_indices=marker_indices)

    def _get_observations(self) -> dict:
        # 136-d repose obs (124 official "full" + 12 tactile) + 4x4 tip->point
        # relative vectors in env frame; point order is fixed per goal -- the
        # one-to-one matching is reward-side only, the network sees all pairs
        observations = super()._get_observations()
        observations["policy"] = torch.cat(
            [observations["policy"], self.tip_point_deltas.reshape(self.num_envs, 48)], dim=-1
        )
        return observations

    def _get_rewards(self) -> torch.Tensor:
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_contact_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.hold_timer >= self.hold_target - 1e-6,
            self.matched_dist,
            self.touched,
            self.actions,
            self.hand_dof_vel,
            self.hand.data.applied_torque.torch,
            self.object_pos,
            self.in_hand_pos,
            self.cfg.point_dist_reward_scale,
            self.cfg.touch_bonus_scale,
            self.cfg.action_penalty_scale,
            self.cfg.joint_energy_penalty_scale,
            self.cfg.object_away_penalty_scale,
            self.cfg.object_away_dist,
            self.cfg.reach_goal_bonus,
            self.cfg.av_factor,
            # v2/v3 retention terms
            self.cfg.cube_pull_scale,
            self.cfg.fall_penalty,
            self.reset_terminated,
            self.cfg.alive_bonus_scale,
            self.cfg.alive_distance,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # points resample the step the hold confirms (mid-episode goal resample)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward


@torch.jit.script
def compute_contact_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    confirmed: torch.Tensor,
    matched_dist: torch.Tensor,
    touched: torch.Tensor,
    actions: torch.Tensor,
    joint_vel: torch.Tensor,
    applied_torque: torch.Tensor,
    object_pos: torch.Tensor,
    in_hand_pos: torch.Tensor,
    point_dist_reward_scale: float,
    touch_bonus_scale: float,
    action_penalty_scale: float,
    joint_energy_penalty_scale: float,
    object_away_penalty_scale: float,
    object_away_dist: float,
    reach_goal_bonus: float,
    av_factor: float,
    object_pull_scale: float,
    fall_penalty: float,
    terminated: torch.Tensor,
    alive_bonus_scale: float,
    alive_distance: float,
):
    """Contact-point reward with the v2 retention terms; hold logic lives in the env.

    ``matched_dist``/``touched`` are (num_envs, num_points) -- the exact
    one-to-one pulp->point distances and touch indicators computed in
    ``_compute_intermediate_values``. ``confirmed`` is (num_envs,) bool: all
    points touched continuously for the env's hold duration. ``terminated``
    is (num_envs,) bool (the out-of-reach terminations this step) and gates
    the fall penalty. Positions are env-relative, as in the parent task.
    Deliberately NOT named ``compute_leap_rewards`` -- the FlashSAC hold-hook
    patches that name and must stay inactive (hold is native here).

    v2 (2026-09-21): the v1 lean stack never taught KEEPING the cube -- there
    was no retention gradient (the dense point distances are tiny, the away
    penalty fires only post-drop, and the fall cost ~2 penalty steps), so
    episodes died at ~10 steps and the +250 was never observed. Added: the
    continuous cube-pull to the anchor (cheap to hold, expensive to drift) and
    a terminal fall penalty.
    """
    point_dist = torch.sum(matched_dist, dim=-1)
    touch_bonus = torch.sum(touched, dim=-1)
    action_penalty = torch.sum(actions**2, dim=-1)
    joint_energy = torch.sum(torch.abs(joint_vel) * torch.abs(applied_torque), dim=-1)
    goal_dist = torch.linalg.norm(object_pos - in_hand_pos, ord=2, dim=-1)
    object_away = (goal_dist > object_away_dist).to(action_penalty.dtype)
    # v2 retention: continuous drift direction signal (v3: scaled down to a minor
    # stream -- the exploit-proof positive incentive is the alive bonus below)
    object_pull = goal_dist * object_pull_scale
    # v3 alive bonus: holding the cube near the anchor EARNS, so dropping loses the
    # stream and pays the fall penalty -- strictly dominated at any episode length
    alive = (goal_dist < alive_distance).to(action_penalty.dtype) * alive_bonus_scale

    reward = (
        point_dist * point_dist_reward_scale
        + touch_bonus * touch_bonus_scale
        + action_penalty * action_penalty_scale
        + joint_energy * joint_energy_penalty_scale
        + object_pull
        + alive
        + object_away * object_away_penalty_scale
    )

    # held all-4 success: bonus, count, and (env-side) point resample
    goal_resets = torch.where(confirmed, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # v2 terminal fall cost: charged on the out-of-reach terminations this step
    reward = reward + terminated.float() * fall_penalty

    # consecutive-successes EMA over episodes finished this step (reset_buf is
    # current: _get_dones ran before _get_rewards)
    num_resets = torch.sum(reset_buf)
    finished_cons_successes = torch.sum(successes * reset_buf.float())
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes
