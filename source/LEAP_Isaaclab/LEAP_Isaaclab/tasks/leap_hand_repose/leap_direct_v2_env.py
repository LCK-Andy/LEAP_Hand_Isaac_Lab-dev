# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# v2 env: the official shared InHandManipulationEnv observed through geometry.
# The downward depth camera is backprojected to a sparse point cloud, split into
# object/hand sets with ground-truth instance segmentation, and exposed as the
# "points" observation group (DP3-style object-centric geometry).
# --------------------------------------------------------

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import Camera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv

from .leap_direct_v2_env_cfg import LeapHandDirectV2EnvCfg


class LeapHandDirectV2Env(InHandManipulationEnv):
    """InHandManipulationEnv + object-centric point-cloud observation (v2).

    The official direct in-hand environment (``Isaac-Repose-Cube-LEAP-Direct-v0``)
    with an added ``"points"`` observation group: (N, P, 4) per-point features
    (x, y, z, flag), sampled from the depth camera, object points first
    (flag=1) then hand points (flag=0), unused slots all-zero. The ``"policy"``
    group stays the official 124-d "full" vector.
    """

    cfg: LeapHandDirectV2EnvCfg

    # semantic labels used to split the point cloud into object / hand sets
    OBJECT_SEMANTIC_LABEL = "inhand_object"
    HAND_SEMANTIC_LABEL = "leap_hand"

    def __init__(self, cfg: LeapHandDirectV2EnvCfg, render_mode: str | None = None, **kwargs):
        # tag the hand spawn so semantic segmentation can label it
        # (instance segmentation annotators hang this Isaac Sim build; semantics do not)
        cfg.robot_cfg.spawn.semantic_tags = [("class", self.HAND_SEMANTIC_LABEL)]
        super().__init__(cfg, render_mode, **kwargs)

        p = self.cfg.num_object_points + self.cfg.num_hand_points
        c = self.cfg.point_feature_dim
        self.single_observation_space["points"] = gym.spaces.Box(
            -float("inf"), float("inf"), (p, c), dtype=np.float32
        )
        self.observation_space = gym.vector.utils.batch_space(
            gym.spaces.Dict(self.single_observation_space.spaces), self.num_envs
        )

        # cache for the object/hand semantic colors, resolved lazily from the
        # camera's segmentation info on the first observation
        self._seg_ids_resolved = False
        self._object_color: torch.Tensor | None = None
        self._hand_color: torch.Tensor | None = None

    def _setup_scene(self):
        # same as InHandManipulationEnv._setup_scene, plus the depth camera
        # (created before cloning so every env gets an instance)
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object: Articulation | RigidObject = self.cfg.object_cfg.class_type(self.cfg.object_cfg)
        self._joint_wrench_sensor = None
        if self.cfg.asymmetric_obs:
            self._joint_wrench_sensor = self._create_joint_wrench_sensor()
        # tag the object template prim so semantic segmentation labels it
        # (tags on the preset spawn cfg do not survive the PresetCfg instantiation;
        # tagging the template prim before cloning propagates to every env)
        from isaaclab.sim.utils.semantics import add_labels
        from isaaclab.sim.utils.stage import get_current_stage

        stage = get_current_stage()
        object_prim = stage.GetPrimAtPath("/World/envs/env_0/object")
        if not object_prim.IsValid():
            raise RuntimeError("Could not find the object template prim /World/envs/env_0/object")
        add_labels(object_prim, labels=[self.OBJECT_SEMANTIC_LABEL], instance_name="class")
        # add the downward-facing depth camera
        self.camera = Camera(self.cfg.camera)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["camera"] = self.camera
        if self._joint_wrench_sensor is not None:
            self.scene.sensors["joint_wrench"] = self._joint_wrench_sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _resolve_segmentation_ids(self) -> None:
        """Resolve the object/hand colors of the color-coded semantic segmentation."""
        seg_info = self.camera.data.info.get("semantic_segmentation")
        if seg_info is None:
            raise RuntimeError(f"No semantic_segmentation info from camera: {self.camera.data.info}")

        # the mapping is {'idToLabels': {'(R, G, B, A)': {'class': label}, ...}} and the
        # segmentation image holds those colors per pixel (N, H, W, 4) uint8
        raw_labels = seg_info.get("idToLabels", {}) if isinstance(seg_info, dict) else {}
        color_to_label: dict[tuple[int, int, int], str] = {}
        for key, label in raw_labels.items():
            try:
                rgb = tuple(int(v.strip()) for v in str(key).strip("()").split(",")[:3])
                label_str = str(label.get("class", "")) if isinstance(label, dict) else str(label)
                color_to_label[rgb] = label_str
            except (ValueError, IndexError):
                continue
        if not color_to_label:
            raise RuntimeError(f"Unrecognized segmentation info structure: {seg_info}")

        object_color = next((c for c, l in color_to_label.items() if self.OBJECT_SEMANTIC_LABEL in l), None)
        hand_color = next((c for c, l in color_to_label.items() if self.HAND_SEMANTIC_LABEL in l), None)
        if object_color is None or hand_color is None:
            raise RuntimeError(f"Could not resolve object/hand colors from mapping: {color_to_label}")

        self._object_color = torch.tensor(object_color, dtype=torch.uint8, device=self.device)
        self._hand_color = torch.tensor(hand_color, dtype=torch.uint8, device=self.device)
        self._seg_ids_resolved = True
        print(
            f"[v2] semantic colors resolved: object rgb={object_color}, hand rgb={hand_color} "
            f"(all labels: {color_to_label})"
        )

    def _compute_point_observation(self) -> torch.Tensor:
        """Sample the (N, P, 4) object/hand point set from the depth camera."""
        if not self._seg_ids_resolved:
            self._resolve_segmentation_ids()

        depth = self.camera.data.output["depth"]  # (N, H, W)
        seg = self.camera.data.output["semantic_segmentation"]  # (N, H, W, 4) uint8 colors
        depth = depth.torch if hasattr(depth, "torch") else depth
        seg = seg.torch if hasattr(seg, "torch") else seg
        intrinsics = self.camera.data.intrinsic_matrices  # (N, 3, 3)
        intrinsics = intrinsics.torch if hasattr(intrinsics, "torch") else intrinsics

        if depth.ndim == 4:
            depth = depth.squeeze(-1)  # camera depth arrives as (N, H, W, 1)
        n, h, w = depth.shape
        device = depth.device
        seg_rgb = seg[..., :3]
        valid = (depth > 0.05) & (depth < self.cfg.rgbd_depth_clip)
        object_mask = valid & (seg_rgb == self._object_color).all(dim=-1)
        hand_mask = valid & (seg_rgb == self._hand_color).all(dim=-1)

        # backproject to camera-frame points (same math as isaaclab's
        # create_pointcloud_from_depth, vectorized over envs)
        uu, vv = torch.meshgrid(
            torch.arange(w, device=device, dtype=depth.dtype),
            torch.arange(h, device=device, dtype=depth.dtype),
            indexing="xy",
        )
        fx = intrinsics[:, 0, 0].view(n, 1, 1)
        fy = intrinsics[:, 1, 1].view(n, 1, 1)
        cx = intrinsics[:, 0, 2].view(n, 1, 1)
        cy = intrinsics[:, 1, 2].view(n, 1, 1)
        x = (uu.view(1, h, w) - cx) * depth / fx
        y = (vv.view(1, h, w) - cy) * depth / fy
        points_cam = torch.stack((x, y, depth), dim=-1)  # (N, H, W, 3)

        # camera frame -> env frame (batched), then anchor+scale to a canonical frame
        cam_pos_w = self.camera.data.pos_w  # (N, 3)
        # the intrinsics-unprojected points live in the pinhole/CV frame (x right,
        # y down, z forward), which is the ROS camera convention
        cam_quat_w = self.camera.data.quat_w_ros  # (N, 4), xyzw
        cam_pos_w = cam_pos_w.torch if hasattr(cam_pos_w, "torch") else cam_pos_w
        cam_quat_w = cam_quat_w.torch if hasattr(cam_quat_w, "torch") else cam_quat_w
        points_flat = points_cam.reshape(n, h * w, 3)
        quats = cam_quat_w.unsqueeze(1).expand(n, h * w, 4).reshape(n * h * w, 4)
        points_flat = math_utils.quat_apply(quats, points_flat.reshape(n * h * w, 3)).reshape(n, h * w, 3)
        anchor = torch.tensor(self.cfg.point_anchor, device=device) + self.scene.env_origins
        points_flat = (points_flat + cam_pos_w.unsqueeze(1) - anchor.unsqueeze(1)) / self.cfg.point_scale

        def _sample(mask: torch.Tensor, count: int, flag: float) -> torch.Tensor:
            scores = torch.rand(n, h * w, device=device)
            scores = torch.where(mask.flatten(1), scores, torch.full_like(scores, -1.0))
            top = scores.topk(count, dim=1)  # (N, count): invalid slots have score < 0
            flat_points = points_flat[torch.arange(n, device=device).unsqueeze(1), top.indices]
            is_valid = (top.values >= 0.0).unsqueeze(-1)
            feats = torch.cat(
                (flat_points, torch.full((n, count, 1), flag, device=device)), dim=-1
            )
            return torch.where(is_valid, feats, torch.zeros_like(feats))

        object_points = _sample(object_mask, self.cfg.num_object_points, 1.0)
        hand_points = _sample(hand_mask, self.cfg.num_hand_points, 0.0)
        return torch.cat((object_points, hand_points), dim=1)  # (N, P, 4)

    def _get_observations(self) -> dict:
        observations = super()._get_observations()
        observations["points"] = self._compute_point_observation()
        return observations
