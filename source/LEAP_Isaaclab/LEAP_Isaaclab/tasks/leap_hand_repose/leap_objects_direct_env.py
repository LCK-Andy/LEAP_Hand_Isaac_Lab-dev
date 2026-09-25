# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Objects variant env: identical to LeapHandDirectEnv (v0: official shared
# InHandManipulationEnv mechanics + the 7-term reference reward) except the
# in-hand object is resampled every episode from a catalog of YCB-style USDs
# (spawn-all-and-toggle design, cf. Chen et al. 2022):
#
#   * every env spawns the whole (palm-size-filtered) catalog as a
#     RigidObjectCollection under its env prim;
#   * inactive members are parked in a column far below the ground plane;
#   * each episode reset randomly activates ONE member per env and places it
#     at the LEAP palm;
#   * the goal marker mirrors the ACTIVE object: one VisualizationMarkers
#     prototype per catalog entry (official DexCube cube replaced), per-env
#     prototype selection via `marker_indices == _active_idx`, plus a
#     frame_prim orientation triad at the goal pose;
#   * `self.object` is a duck-typed view of the ACTIVE member, so the official
#     obs/reward/reset mechanics work unchanged.
# --------------------------------------------------------

from __future__ import annotations

import math
import os
import random

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.assets.rigid_object_collection import RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_from_usd, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from LEAP_Isaaclab.compat.inhand_direct.inhand_manipulation_env import randomize_rotation

from .leap_direct_env import LeapHandDirectEnv
from .leap_direct_env_cfg import LeapHandDirectEnvCfg

import LEAP_Isaaclab.tasks.leap_hand_reorient.leap_hand_reorient_ycb_env_cfg as ycb_catalog

# parked (inactive) objects rest in a ground row beyond the WHOLE env grid --
# parking below the ground plane does not work (PhysX depenetrates the two-sided
# ground collision and pushes objects back to the surface within one step), and a
# fixed local offset would land inside other envs' cells at large env counts.
# The row offset is computed from num_envs in _setup_scene (park_y0).
_PARK_MARGIN = 2.0  # extra clearance beyond the grid edge [m]
_PARK_GAP = 0.4  # spacing between parked objects [m]
_PARK_Z = 0.05  # height of the parked row above the ground [m]


def usd_extents(usd_path: str) -> tuple[float, float, float]:
    """The USD's world-axis-aligned bounding-box side lengths, in meters."""
    from pxr import Usd, UsdGeom

    # open as a side stage; must not touch the live Kit stage context
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open USD for size measurement: {usd_path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], False)
    bbox = cache.ComputeWorldBound(stage.GetPseudoRoot())
    extent = bbox.ComputeAlignedRange()
    extents = tuple(float(extent.max[i] - extent.min[i]) for i in range(3))
    del stage  # release the side stage; no explicit Close in this pxr build
    return extents  # type: ignore[return-value]


class _TensorBox:
    """Minimal `.torch` accessor wrapper over a plain tensor."""

    __slots__ = ("torch",)

    def __init__(self, tensor: torch.Tensor):
        self.torch = tensor


class _ActiveObjectView:
    """Duck-typed single-RigidObject view over each env's ACTIVE collection member.

    Only the pieces used by the official env's __init__ are provided; the
    state gathering itself is done in the overridden `_compute_intermediate_values`.
    """

    def __init__(self, collection: RigidObjectCollection, active_idx: torch.Tensor, num_envs: int):
        self._collection = collection
        self._active_idx = active_idx
        self._num_envs = num_envs

    @property
    def data(self) -> "_ActiveObjectView":
        return self

    @property
    def default_root_pose(self) -> _TensorBox:
        poses = self._collection.data.default_body_pose.torch
        rows = torch.arange(self._num_envs, device=poses.device)
        return _TensorBox(poses[rows, self._active_idx].clone())

    # never called: `_reset_idx` is fully overridden and writes through the collection
    def write_root_pose_to_sim_index(self, *args, **kwargs) -> None:
        raise RuntimeError("_ActiveObjectView.write_root_pose_to_sim_index must not be called")

    def write_root_velocity_to_sim_index(self, *args, **kwargs) -> None:
        raise RuntimeError("_ActiveObjectView.write_root_velocity_to_sim_index must not be called")


class LeapObjectsDirectEnv(LeapHandDirectEnv):
    """v0 direct task with the in-hand object resampled from a catalog every episode."""

    cfg: LeapHandDirectEnvCfg | object  # LeapObjectsDirectEnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # the base class derives the fall-termination anchor (in_hand_pos) from the
        # object's DEFAULT pose -- which for this env is a far-away park slot, not the
        # palm. Pin it to the palm spawn (same anchor the cube task uses).
        palm_anchor = torch.tensor((0.0, -0.10, 0.62 - 0.04), device=self.device)
        self.in_hand_pos = palm_anchor.repeat(self.num_envs, 1)

        # RViz-style orientation triad at the goal pose (official frame_prim
        # widget: red/green/blue x/y/z arrows). Complements the object-shaped
        # goal marker, whose orientation alone can be hard to read for
        # near-symmetric objects (cans, fruit).
        self.goal_axes_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/goal_axes",
                markers={
                    "frame": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                        scale=(self.cfg.goal_axes_scale,) * 3,
                    )
                },
            )
        )

    def _setup_scene(self):
        # hand: spawned on the template env and cloned, exactly like the official env
        # (kit-less backends get the pxr root weld -- see _spawn_hand_articulation)
        self._spawn_hand_articulation()
        self._joint_wrench_sensor = None
        if self.cfg.asymmetric_obs:
            self._joint_wrench_sensor = self._create_joint_wrench_sensor()
        # PhysX contact reporters on the four pulp links (pre-clone)
        self._activate_fingertip_contact_reporters()

        # palm-size filter over the candidate catalog (measured from the USDs)
        max_extent = self.cfg.max_object_extent_m
        catalog = []
        for name in self.cfg.object_names:
            extents = usd_extents(ycb_catalog._OBJECTS[name])
            if max(extents) <= max_extent:
                catalog.append((name, max(extents)))
        if not catalog:
            raise RuntimeError(
                f"No catalog object fits max_object_extent_m={max_extent}; "
                f"candidates: {self.cfg.object_names}"
            )
        self.object_catalog = catalog  # [(name, max_extent_m)], body order == member order
        print(
            f"[objects] catalog filtered to {len(catalog)} palm-size objects: "
            + ", ".join(f"{n}({e * 100:.0f}cm)" for n, e in catalog)
        )

        # goal marker: one prototype per catalog object (dict order == member
        # order), so every env's goal marker shows its ACTIVE object instead of
        # the inherited DexCube. Rebuilt here because the filtered catalog is
        # only known after measuring the USDs; the base __init__ constructs the
        # VisualizationMarkers from this cfg after _setup_scene returns. The
        # wrapper (visual) USDs are used -- the marker is display-only.
        self.cfg.goal_object_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/goal_marker",
            markers={name: sim_utils.UsdFileCfg(usd_path=ycb_catalog._OBJECTS[name]) for name, _ in catalog},
        )

        # spawn every catalog object on the template env, then clone
        from .leap_objects_direct_env_cfg import object_spawn_cfg

        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import UsdPhysics

        stage = get_current_stage()
        # this Isaac Lab beta defines the PhysX GPU buffer cfg fields but never
        # applies them -- author the physxScene attributes directly on the scene
        # prim (before the first physics step) or large env counts overflow the
        # patch buffer and corrupt the simulation (no PhysxSchema module in this
        # build, hence the raw attributes)
        from pxr import Sdf

        scene_prim = stage.GetPrimAtPath(self.cfg.sim.physics_prim_path)
        for attr_name, value in (
            ("physxScene:maxRigidPatchCount", 2**22),
            ("physxScene:maxRigidContactCount", 2**24),
        ):
            attr = scene_prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Int, True)
            attr.Set(value)

        # parked rows sit beyond the WHOLE env grid so they never enter another
        # env's cell: the offset must exceed the full grid span (envs at the
        # negative edge would otherwise park into the positive half)
        grid_side = math.ceil(math.sqrt(self.num_envs))
        park_y0 = grid_side * self.cfg.scene.env_spacing + _PARK_MARGIN

        # sphere collision proxies (MorphIt) when available; wrapper USD otherwise
        sphere_dir = (
            getattr(self.cfg, "sphere_proxy_dir", None) or ycb_catalog._YCB_DIR + "/spheres"
        )
        use_spheres = getattr(self.cfg, "collision_proxy", "hull") == "spheres"

        def _object_usd(name: str) -> str:
            proxy = f"{sphere_dir}/{name}_spheres.usd"
            if use_spheres and os.path.isfile(proxy):
                return proxy
            return ycb_catalog._OBJECTS[name]

        members: dict[str, RigidObjectCfg] = {}
        for member_idx, (name, _) in enumerate(catalog):
            # Objects are nested one level deep (env_*/obj/<name>): the newton
            # RigidObjectCollection builds ONE fnmatch pattern over all members,
            # wildcarding the segment that differs -- siblings directly under
            # env_* would give /World/envs/env_*/* whose '*' CROSSES '/' and
            # would also match the hand articulation (breaking the collection's
            # floating-base detection and buffer strides). The extra 'obj'
            # segment pins the pattern to /World/envs/env_*/obj/*.
            spawn_from_usd(f"/World/envs/env_0/obj/{name}", object_spawn_cfg(_object_usd(name)))
            prim = stage.GetPrimAtPath(f"/World/envs/env_0/obj/{name}")
            self._author_object_mass(prim, name)
            # hull mode: the wrapper USDs author CollisionAPI on BOTH the object
            # root and the mesh child -- PhysX parses both into TWO identical
            # hulls per object (~2x mass via density x summed shape volumes, and
            # every contact generated twice). Keep collision on the mesh only,
            # with an explicit hull approximation (also silences the
            # "falling back to convexHull" parse-error spam).
            if not use_spheres:
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
                for child in prim.GetChildren():
                    if child.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr("convexHull")
            members[name] = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/obj/{name}",
                spawn=None,  # template spawned above; cloning replicates it
                # the member's default pose IS its parked slot in the far row
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, park_y0 + _PARK_GAP * member_idx, _PARK_Z), rot=(0.0, 0.0, 0.0, 1.0)
                ),
            )

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (hand + all catalog objects)
        self.scene.clone_environments(copy_from_source=False)

        # one collection handle over all per-env objects; buffers are (N, K, ...)
        self.objects: Articulation | RigidObject | RigidObjectCollection = RigidObjectCollection(
            RigidObjectCollectionCfg(rigid_objects=members)
        )
        self._active_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.object = _ActiveObjectView(self.objects, self._active_idx, self.num_envs)
        # palm spawn pose (local) used for the active member at each reset
        self.object_palm_spawn_pose = torch.tensor((0.0, -0.10, 0.62), device=self.device)

        # register entities (same keys as the official env)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.objects
        if self._joint_wrench_sensor is not None:
            self.scene.sensors["joint_wrench"] = self._joint_wrench_sensor
        # per-pulp contact sensors filtered against EVERY catalog member (only
        # the active one is ever in reach, so the filtered net force is exactly
        # the pulp-vs-object contact force)
        self._create_fingertip_contact_sensors([f"/World/envs/env_.*/obj/{name}" for name, _ in catalog])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _author_object_mass(self, prim, name: str) -> None:
        """Author the USD mass properties of one catalog template prim (pre-clone).

        v0: a uniform density. The YCB wrapper USDs have no mass attributes, so
        MassPropertiesCfg(density=...) in the spawn cfg is silently ignored
        ("Could not perform 'modify_mass_properties'") and PhysX would fall back
        to its default density (1000 kg/m^3). The cloner replicates applied
        schemas, so the authored attributes land on every env's copy. v1
        (leap_objects_direct_v1_env.py) overrides this with full ground-truth
        (mass, COM, inertia) from the sibling repo's inertial manifest.
        """
        from pxr import UsdPhysics

        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateDensityAttr(self.cfg.object_density, True)

    def _reset_idx(self, env_ids):
        # episode bookkeeping only (the official in-hand reset is reimplemented below
        # for the collection, so skip InHandManipulationEnv._reset_idx)
        self._last_episode_success[env_ids] = self.successes[env_ids] >= self.cfg.success_count_threshold
        self.extras.setdefault("log", {})["Metrics/success_rate"] = (
            self._last_episode_success[env_ids].float().mean().item()
        )
        from isaaclab.envs.direct_rl_env import DirectRLEnv

        DirectRLEnv._reset_idx(self, env_ids)

        # resample the active object per env FIRST -- the goal-marker prototype
        # picked by _reset_target_pose below must show the NEW object
        num_resets = len(env_ids)
        num_objects = len(self.object_catalog)
        self._active_idx[env_ids] = torch.randint(num_objects, (num_resets,), device=self.device)
        self._reset_target_pose(env_ids)
        active_ids = self._active_idx[env_ids]
        rows = torch.arange(num_resets, device=self.device)

        default_poses = self.objects.data.default_body_pose.torch[env_ids]  # (n, K, 7)
        poses = default_poses.clone()
        # inactive members: back to their own parked slot in the far row
        # (default poses are local; writes take WORLD poses, so add the env origins)
        poses[..., 0:3] = default_poses[..., 0:3] + self.scene.env_origins[env_ids].unsqueeze(1)
        poses[..., 3:6] = 0.0
        poses[..., 6] = 1.0
        # active member: palm pose + position noise + env origin, random rotation
        pos_noise = sample_uniform(-1.0, 1.0, (num_resets, 3), device=self.device)
        rot_noise = sample_uniform(-1.0, 1.0, (num_resets, 2), device=self.device)
        active_pose = torch.zeros_like(default_poses[rows, active_ids])
        active_pose[:, 0:3] = (
            self.object_palm_spawn_pose[0:3]
            + self.cfg.reset_position_noise * pos_noise
            + self.scene.env_origins[env_ids]
        )
        active_pose[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )
        poses[rows, active_ids] = active_pose

        self.objects.write_body_pose_to_sim_index(body_poses=poses, env_ids=env_ids)
        self.objects.write_body_velocity_to_sim_index(
            body_velocities=torch.zeros(num_resets, num_objects, 6, device=self.device), env_ids=env_ids
        )

        # reset hand (identical to the official env)
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos.torch[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos.torch[env_ids]
        dof_pos_noise = sample_uniform(-1.0, 1.0, (num_resets, self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos.torch[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta
        dof_vel_noise = sample_uniform(-1.0, 1.0, (num_resets, self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel.torch[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos
        self._set_joint_pos_target(target=dof_pos, env_ids=env_ids)
        self._write_hand_joint_pos(position=dof_pos, env_ids=env_ids)
        self._write_hand_joint_vel(velocity=dof_vel, env_ids=env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values()
        # the pose buffers are stale until the next physics step, which would make the
        # freshly-activated object report its far-away park slot for one observation --
        # patch the active object's state from the poses just written
        self.object_pos[env_ids] = poses[rows, active_ids][:, 0:3] - self.scene.env_origins[env_ids]
        self.object_rot[env_ids] = poses[rows, active_ids][:, 3:7]
        self.object_velocities[env_ids] = 0.0
        self.object_linvel[env_ids] = 0.0
        self.object_angvel[env_ids] = 0.0

    def _reset_target_pose(self, env_ids):
        # same goal sampling as the official env, but every env's goal marker
        # prototype is switched to its ACTIVE catalog object (prototype order
        # == collection member order, so `_active_idx` indexes both), and an
        # orientation triad is drawn at the goal pose. Called both on episode
        # reset (after the resample above) and mid-episode on goal reach
        # (active object unchanged there, so the indices stay valid).
        super()._reset_target_pose(env_ids)
        self.goal_markers.visualize(marker_indices=self._active_idx)
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_axes_markers.visualize(goal_pos, self.goal_rot)

    def _compute_intermediate_values(self):
        # gather the state of each env's ACTIVE object member
        poses = self.objects.data.body_link_pose_w.torch  # (N, K, 7)
        vels = self.objects.data.body_link_vel_w.torch  # (N, K, 6)
        rows = torch.arange(self.num_envs, device=self.device)
        active_pose = poses[rows, self._active_idx]
        active_vel = vels[rows, self._active_idx]

        # data for hand (identical to the official env)
        self.fingertip_pos = self.hand.data.body_pos_w.torch[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w.torch[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w.torch[:, self.finger_bodies]
        self.hand_dof_pos = self.hand.data.joint_pos.torch
        self.hand_dof_vel = self.hand.data.joint_vel.torch

        # data for object (active member)
        self.object_pos = active_pose[:, 0:3] - self.scene.env_origins
        self.object_rot = active_pose[:, 3:7]
        self.object_velocities = active_vel
        self.object_linvel = active_vel[:, 0:3]
        self.object_angvel = active_vel[:, 3:6]

    # -- introspection helpers (smoke tests / logging) --

    @property
    def active_object_names(self) -> list[str]:
        return [self.object_catalog[i][0] for i in self._active_idx.tolist()]

    @property
    def active_object_ids(self) -> torch.Tensor:
        return self._active_idx
