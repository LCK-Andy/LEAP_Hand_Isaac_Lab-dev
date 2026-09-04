# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact sensor variant for nested body prims (URDF importer 3.0).

The URDF importer (isaacsim.asset.importer.urdf, Isaac Lab 3.0) generates nested
Xform body prims, one level per link::

    /World/envs/env_0/Robot/Geometry/iiwa7_base_link/.../palm_lower/mcp_joint/pip/dip/fingertip

The stock :class:`isaaclab_physx.sensors.contact_sensor.ContactSensor` builds
the contact-view glob ``Robot/(fingertip)`` from resolved leaf names, but
``create_rigid_contact_view`` matches a named leaf only against *direct
children*, so the view comes up empty for deeply nested bodies. A naive fix —
inserting ``**`` (e.g. ``Robot/**/(fingertip)``) — over-matches: the URDF
converter creates same-named mesh prims under each link, so the view picks up
non-body prims and the filter pairing breaks.

This class instead passes the *exact resolved primitive paths* (only the env
segment wildcarded) as the pattern list. Each pattern names exactly one body
prim, no recursion, no over-match.

Use it by setting ``class_type`` in ``ContactSensorCfg``:

    ContactSensorCfg(..., class_type="LEAP_Isaaclab.tasks.kuka_leap_dexsuite.contact_sensor:NestedContactSensor")

Only the PhysX backend is customized; the Newton backend keeps its own factory.
"""

from __future__ import annotations

import re

from isaaclab.sensors.contact_sensor import BaseContactSensor
from isaaclab.sim.utils.queries import get_all_matching_child_prims, resolve_matching_prims_from_source

# NOTE: isaaclab_physx.sensors.contact_sensor transitively imports
# pxr.UsdPhysics. Importing it at package level loads pxr before
# SimulationApp/AppLauncher starts and breaks the PhysX extension python
# bindings. The task-registration import path must therefore never import this
# module eagerly: `scripts/kuka_leap_dexsuite_random_agent.py` and
# `kuka_leap_dexsuite_smoke.py` both launch Kit BEFORE importing the task
# package, which makes this import safe.
#
# We still import PhysXContactSensor lazily inside the module __getattr__ below
# so that merely importing the task cfg (e.g. class_type string resolution)
# stays pxr-free.
from isaaclab_physx.sensors.contact_sensor.contact_sensor import ContactSensor as PhysXContactSensor


class NestedContactSensor(PhysXContactSensor):
    """PhysX contact sensor whose view patterns are the exact resolved prim paths."""

    def _initialize_impl(self):
        # Only run SensorBase-level init; do NOT call PhysXContactSensor's
        # _initialize_impl (it builds view globs that fail on nested bodies).
        BaseContactSensor._initialize_impl(self)

        # Lazy import: isaaclab_physx pulls in pxr.UsdPhysics (schemas.py). At
        # import time (before SimulationApp) that breaks PhysX extension python
        # bindings; at runtime (after Kit) it is safe.
        from isaaclab_physx.physics import PhysxManager as SimulationManager

        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        # -- resolve contact-reporting bodies (same as stock) --
        parent_expr, leaf_pattern = self.cfg.prim_path.rsplit("/", 1)
        name_pattern = re.compile(leaf_pattern)

        def has_contact_report(prim) -> bool:
            return bool(name_pattern.fullmatch(prim.GetName())) and (
                "PhysxContactReportAPI" in prim.GetAppliedSchemas()
            )

        # Resolve the asset subtree (clone-plan aware) and collect contact-reporting descendants.
        matches = resolve_matching_prims_from_source(parent_expr)
        if not matches:
            raise RuntimeError(f"No prim found at '{parent_expr}'.")
        asset_prim, body_parent = matches[0]
        walk_root = asset_prim.GetPath().pathString
        prims = get_all_matching_child_prims(walk_root, predicate=has_contact_report, traverse_instance_prims=False)
        if not prims:
            raise RuntimeError(
                f"Sensor at path '{self.cfg.prim_path}' could not find any bodies with contact reporter API."
                "\nHINT: Make sure to enable 'activate_contact_sensors' in the corresponding asset spawn configuration."
            )

        # -- build exact per-prim patterns: wildcard only the env segment --
        # prim paths look like /World/envs/env_0/Robot/Geometry/.../fingertip;
        # body_parent looks like /World/envs/env_*/Robot. Strip the "/Robot"
        # suffix of body_parent and re-append the direct child path.
        prim0_path = prims[0].GetPath().pathString
        # "/World/envs/env_0" prefix -> leave, remainder starts with "/Robot"
        suffix_from_env0 = prim0_path.split("/World/envs/env_0", 1)[-1]  # /Robot/Geometry/...
        # body_parent ends with the artic root name; rebuild from body_parent root + suffix
        root_prim = "/Robot"
        body_paths_glob = [f"{body_parent}{suffix_from_env0[len(root_prim):]}" for _ in prims]
        body_names = [p.GetPath().pathString.rsplit("/", 1)[-1] for p in prims]
        body_paths_glob = list(dict.fromkeys(body_paths_glob))
        # Filters: the env segment must be explicit per-env paths (env_* wildcard
        # makes the tensors plugin count filter entries wrongly). Each sensor
        # pattern gets its own filter list; generate /World/envs/env_*/Object
        # concretized per env.
        filter_exprs = [expr.replace(".*", "*") for expr in self.cfg.filter_prim_paths_expr]
        concrete_filters: list[str] = []
        if filter_exprs:
            for expr in filter_exprs:
                if expr.count("*") == 1 and "/env_" in expr:
                    # concretize the single env wildcard: /World/envs/env_*/Object -> per env
                    for env_idx in range(self._num_envs):
                        concrete_filters.append(expr.replace("*", str(env_idx)))
                else:
                    concrete_filters.append(expr)
        # one filter list per sensor pattern (all sensors share the same filters)
        filter_prim_paths_glob = [concrete_filters] * len(body_paths_glob)

        print(f"[NestedContactSensor] sensor paths: {body_paths_glob}")
        print(f"[NestedContactSensor] filters: {filter_prim_paths_glob}")
        # create a rigid prim view for the sensor (list of exact patterns)
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(body_paths_glob)
        print(f"[NestedContactSensor] body view count={self._body_physx_view.count}")
        self._contact_view = self._physics_sim_view.create_rigid_contact_view(
            body_paths_glob,
            filter_patterns=filter_prim_paths_glob,
            max_contact_data_count=self.cfg.max_contact_data_count_per_prim * len(body_names) * self._num_envs,
        )
        if self._contact_view is None:
            raise RuntimeError(
                "Failed to create rigid contact view (returned None). "
                f"\n\tSensor paths : {body_paths_glob}"
                f"\n\tFilters      : {filter_prim_paths_glob}"
            )

        # resolve the true count of bodies
        self._num_sensors = self.body_physx_view.count // self._num_envs
        # check that contact reporter succeeded
        if self._num_sensors != len(body_names):
            raise RuntimeError(
                "Failed to initialize contact reporter for specified bodies."
                f"\n\tInput prim path    : {self.cfg.prim_path}"
                f"\n\tResolved prim paths: {body_paths_glob}"
            )

        # -- buffers (same as stock tail of _initialize_impl) --
        # check if filter paths are valid
        if self.cfg.track_contact_points or self.cfg.track_friction_forces:
            if not self.cfg.filter_prim_paths_expr:
                raise ValueError(
                    "The 'filter_prim_paths_expr' is empty. Please specify a valid filter pattern to track"
                    f" {'contact points' if self.cfg.track_contact_points else 'friction forces'}."
                )
            if self.cfg.max_contact_data_count_per_prim < 1:
                raise ValueError(
                    f"The 'max_contact_data_count_per_prim' is {self.cfg.max_contact_data_count_per_prim}. "
                    "Please set it to a value greater than 0 to track"
                    f" {'contact points' if self.cfg.track_contact_points else 'friction forces'}."
                )

        self._create_buffers()
