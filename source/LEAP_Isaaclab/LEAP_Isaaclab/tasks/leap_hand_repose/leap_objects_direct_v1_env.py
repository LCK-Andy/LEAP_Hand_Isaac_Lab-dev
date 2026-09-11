# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Objects v1 env: the v0 catalog task (leap_objects_direct_env.py) with the
# in-hand object's inertial properties drawn from the sibling
# Estimate_weight_distribution ground-truth manifest instead of density x
# collision-volume:
#
#   * the catalog is restricted to manifest-covered objects (the manifest is
#     baked to assets/ycb/inertial_manifest.json by
#     scripts/generate_ycb_inertial_manifest.py; each object carries one
#     ground-truth variant per random density strategy);
#   * template prims are authored the FULL (mass, COM, diagonal inertia +
#     principal axes) MassAPI state at spawn (route A: explicit mass overrides
#     density, so the dynamics no longer depend on the collision-proxy volume);
#   * every episode reset then RESAMPLES (m, COM, I) per env from the object's
#     manifest variants via the RigidObjectCollection runtime writers
#     set_masses/coms/inertias_index (route B: per-episode randomization with
#     known ground truth -- the excitation-for-estimation setup, cf. ASID).
#
# The ground-truth inertials are deliberately NOT part of the observation
# space: the 136-d policy obs is inherited unchanged from v0.
# --------------------------------------------------------

from __future__ import annotations

import json
import os

import torch

from .leap_objects_direct_env import LeapObjectsDirectEnv
from .leap_objects_direct_env_cfg import LeapObjectsDirectEnvCfg

import LEAP_Isaaclab.tasks.leap_hand_reorient.leap_hand_reorient_ycb_env_cfg as ycb_catalog


class LeapObjectsDirectV1Env(LeapObjectsDirectEnv):
    """v0 catalog task with per-episode (mass, COM, inertia) from the ground-truth manifest."""

    cfg: LeapObjectsDirectEnvCfg | object  # LeapObjectsDirectV1EnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # the manifest records must be loaded BEFORE super().__init__: both the
        # template mass authoring (_setup_scene) and the construction-time
        # _reset_idx (runtime inertial resampling) run inside it
        self._inertial_records = self._load_inertial_records(cfg)
        self._inertial_tensors_built = False
        super().__init__(cfg, render_mode, **kwargs)

    @staticmethod
    def _load_inertial_records(cfg) -> dict[str, list[dict]]:
        path = getattr(cfg, "inertial_manifest", None) or os.path.join(
            ycb_catalog._YCB_DIR, "inertial_manifest.json"
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"inertial manifest not found at {path} -- generate it with "
                f"scripts/generate_ycb_inertial_manifest.py (needs the sibling "
                f"Estimate_weight_distribution repo) or point cfg.inertial_manifest at one"
            )
        with open(path) as f:
            records = json.load(f)["objects"]
        n = sum(len(v) for v in records.values())
        print(f"[objects-v1] inertial manifest: {n} variants over {len(records)} objects ({path})")
        return records

    def _author_object_mass(self, prim, name: str) -> None:
        # route A (spawn-time state): author the full inertial state from the
        # manifest's deterministic "uniform" strategy variant. The
        # construction-time reset immediately re-randomizes via the runtime
        # writers, so this only defines the parse-time defaults. Explicit mass
        # overrides density, and explicit diagonalInertia + principalAxes make
        # the body's inertia independent of the collision-proxy volume (hull
        # and sphere proxies yield identical dynamics).
        from pxr import Gf, UsdPhysics

        if name not in self._inertial_records:
            raise RuntimeError(
                f"object {name!r} has no inertial-manifest entry -- this task requires every "
                f"catalog object to be covered; extend it via "
                f"scripts/generate_ycb_inertial_manifest.py (or drop the name from object_names)"
            )
        variants = self._inertial_records[name]
        entry = next((v for v in variants if v["strategy"] == "uniform"), variants[0])
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(entry["mass"], True)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*entry["com"]), True)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*entry["principal_moments"]), True)
        # principalAxes quat == the eigenvector matrix whose columns are the
        # descending-moment axes, so PhysX reconstructs inertia = R diag R^T
        w, x, y, z = entry["principal_axes_quat_wxyz"]
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(w, x, y, z), True)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._resample_inertials(env_ids)

    def _resample_inertials(self, env_ids):
        """Resample (m, COM, I) per reset env from the manifest variants and write to sim.

        A fresh variant is drawn for EVERY (env, member) pair -- parked members
        included (one batched write over all bodies; a parked member's variant
        then simply persists until it is next activated).
        """
        if not self._inertial_tensors_built:
            self._build_inertial_tensors()
        num_resets = len(env_ids)
        num_objects = len(self.object_catalog)
        vids = torch.stack(
            [
                torch.randint(int(self._variant_counts[k]), (num_resets,), device=self.device)
                for k in range(num_objects)
            ],
            dim=1,
        )
        kk = torch.arange(num_objects, device=self.device)
        rows = torch.arange(num_resets, device=self.device)
        masses = self._inertial_mass[kk, vids]  # (n, K)
        coms = self._inertial_com[kk, vids]  # (n, K, 3)
        # inertia: full 3x3 about the COM in the body-local frame, row-major flatten
        inertias = self._inertial_inertia[kk, vids].reshape(num_resets, num_objects, 9)  # (n, K, 9)
        self.objects.set_masses_index(masses=masses, env_ids=env_ids)
        if self._physics_backend == "physx":
            # the physx backend's set_coms_index takes the full COM POSE (7 floats,
            # pos + quat xyzw): the manifest COM is expressed in the body frame, so
            # the COM frame keeps the body orientation (identity quat)
            com_poses = torch.zeros(num_resets, num_objects, 7, device=self.device)
            com_poses[..., 0:3] = coms
            com_poses[..., 6] = 1.0
            self._write_com_poses(com_poses, env_ids)
            self._write_inertias(inertias, env_ids)
        else:
            # kit-less newton backend: the collection's NATIVE writers take exactly
            # the manifest quantities (COM position (n, K, 3) -- Newton pins the COM
            # frame to the body frame -- and the full body-local 3x3 inertia (n, K, 9));
            # the physx raw-view workaround writers don't apply
            self.objects.set_coms_index(coms=coms, env_ids=env_ids)
            self.objects.set_inertias_index(inertias=inertias, env_ids=env_ids)
        # ground-truth bookkeeping (smoke tests / logging / estimation work)
        self._inertial_variant_idx[env_ids] = vids
        active = self._active_idx[env_ids]
        self._active_mass[env_ids] = masses[rows, active]
        self._active_com[env_ids] = coms[rows, active]
        self._active_inertia[env_ids] = inertias.reshape(num_resets, num_objects, 3, 3)[rows, active]

    def _write_com_poses(self, com_poses: torch.Tensor, env_ids) -> None:
        """Write COM poses to the sim, working around an isaaclab 3.0.0b2 bug.

        The physx backend's ``RigidObjectCollection.set_coms_index`` hands a
        transformf-TYPED warp array to the omni.physics tensor frontend, which
        has no dtype mapping for it (KeyError: transformf) -- the working pose
        writers re-view theirs as plain ``wp.float32`` first. This replicates
        the setter exactly (kernel write into the cached COM buffer, then
        ``root_view.set_coms`` in view order on CPU) with the dtype fix.
        """
        import warp as wp
        from isaaclab_physx.assets import kernels as physx_kernels

        objects = self.objects
        data = objects.data
        env_ids_resolved = objects._resolve_env_ids(env_ids)
        body_ids_resolved = objects._resolve_body_ids(None)
        wp.launch(
            physx_kernels.write_body_com_pose_to_buffer,
            dim=(env_ids_resolved.shape[0], body_ids_resolved.shape[0]),
            inputs=[com_poses, env_ids_resolved, body_ids_resolved, False],
            outputs=[data._body_com_pose_b.data],
            device=self.device,
        )
        data._body_com_pose_b.timestamp = data._sim_timestamp
        com_view_order = objects.reshape_data_to_view_2d(data._body_com_pose_b.data, device="cpu")
        view_ids = objects._env_body_ids_to_view_ids(env_ids_resolved, body_ids_resolved, device="cpu")
        objects.root_view.set_coms(com_view_order.view(wp.float32), indices=view_ids)

    def _write_inertias(self, inertias: torch.Tensor, env_ids) -> None:
        """Write body inertias to the sim, working around an isaaclab 3.0.0b2 bug.

        The physx backend's ``RigidObjectCollection.set_inertias_index`` feeds
        its (I, B, 9) buffer through ``reshape_data_to_view_2d``, which only
        understands (I, B) data -- the trailing 9 gets scrambled and omni.physx
        rejects the size ("expected total size 648, received 72"). The 3D
        variant of the same helper exists for exactly this; replicate the setter
        with ``reshape_data_to_view_3d(..., 9)``.
        """
        import warp as wp

        from isaaclab_physx.assets import kernels as physx_kernels

        objects = self.objects
        data = objects.data
        env_ids_resolved = objects._resolve_env_ids(env_ids)
        body_ids_resolved = objects._resolve_body_ids(None)
        wp.launch(
            physx_kernels.write_body_inertia_to_buffer,
            dim=(env_ids_resolved.shape[0], body_ids_resolved.shape[0]),
            inputs=[inertias, env_ids_resolved, body_ids_resolved, False],
            outputs=[data._body_inertia],
            device=self.device,
        )
        inertia_view_order = objects.reshape_data_to_view_3d(data._body_inertia, 9, device="cpu")
        view_ids = objects._env_body_ids_to_view_ids(env_ids_resolved, body_ids_resolved, device="cpu")
        objects.root_view.set_inertias(inertia_view_order, indices=view_ids)

    def _build_inertial_tensors(self):
        """Materialize the manifest as per-member sampling tables on the env device."""
        num_objects = len(self.object_catalog)
        counts = [len(self._inertial_records[name]) for name, _ in self.object_catalog]
        vmax = max(counts)
        mass = torch.zeros(num_objects, vmax)
        com = torch.zeros(num_objects, vmax, 3)
        inertia = torch.zeros(num_objects, vmax, 3, 3)
        for k, (name, _) in enumerate(self.object_catalog):
            for v, entry in enumerate(self._inertial_records[name]):
                mass[k, v] = entry["mass"]
                com[k, v] = torch.tensor(entry["com"], dtype=torch.float32)
                inertia[k, v] = torch.tensor(entry["inertia"], dtype=torch.float32)
            for v in range(counts[k], vmax):  # pad with variant 0 (never sampled)
                mass[k, v] = mass[k, 0]
                com[k, v] = com[k, 0]
                inertia[k, v] = inertia[k, 0]
        self._variant_counts = torch.tensor(counts, device=self.device)
        self._inertial_mass = mass.to(self.device)
        self._inertial_com = com.to(self.device)
        self._inertial_inertia = inertia.to(self.device)
        self._inertial_variant_idx = torch.zeros(self.num_envs, num_objects, dtype=torch.long, device=self.device)
        self._active_mass = torch.zeros(self.num_envs, device=self.device)
        self._active_com = torch.zeros(self.num_envs, 3, device=self.device)
        self._active_inertia = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self._inertial_tensors_built = True

    @property
    def active_inertials(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sampled ground truth of each env's ACTIVE member: (mass (N,), COM (N,3), inertia (N,3,3))."""
        return self._active_mass, self._active_com, self._active_inertia
