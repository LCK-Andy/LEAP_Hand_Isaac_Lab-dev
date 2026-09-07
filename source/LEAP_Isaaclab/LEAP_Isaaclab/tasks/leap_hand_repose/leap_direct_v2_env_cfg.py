# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# v2: the direct in-hand reorientation task (Isaac-Repose-Cube-LEAP-Direct-v0)
# observed through geometry: the downward-facing depth camera is backprojected to a
# sparse point cloud split into object/hand sets via instance segmentation
# (DP3-style). The "points" observation (N, P, 4) replaces the RGBD image.
# --------------------------------------------------------

from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

import isaaclab.sim as sim_utils

from .leap_direct_env_cfg import LeapHandDirectEnvCfg


@configclass
class LeapHandDirectV2EnvCfg(LeapHandDirectEnvCfg):
    """Direct in-hand reorientation with an object-centric point-cloud observation (v2).

    Identical MDP to the official ``Isaac-Repose-Cube-Allegro-Direct-v0`` (the
    124-d "policy" obs, actions, rewards, resets are unchanged). Instead of an
    RGBD image, the added ``"points"`` observation group holds a sparse point
    set sampled from a downward-facing depth camera and split into object
    points and hand points via ground-truth instance segmentation: shape
    (N, P, 4) with per-point features (x, y, z, flag) where flag=1 marks
    object points, flag=0 hand points, and all-zero rows mark unused slots.
    Coordinates are in the env frame, centered on the LEAP palm and scaled by
    ``point_scale``.
    """

    # v2 stays on the official reward (its env class uses InHandManipulationEnv's
    # compute_rewards), so re-pin the official dense position scale that the
    # v0 cfg disables for the reference reward. To give v2 the reference reward
    # instead, parent LeapHandDirectV2Env on LeapHandDirectEnv and drop this.
    dist_reward_scale = -10.0

    # world-fixed depth camera above the LEAP palm (same framing as the v1 RGBD camera)
    camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.1, 0.95),
            rot=(0.0, 0.7071, 0.0, 0.7071),  # look straight down (-Z)
            convention="world",
        ),
        data_types=["depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.5),
        ),
        width=128,
        height=128,
        renderer_cfg=MultiBackendRendererCfg(),
    )

    # point-cloud sampling
    num_object_points = 128  # points sampled from the in-hand object per env
    num_hand_points = 128  # points sampled from the hand per env
    point_scale = 0.3  # m; coordinates are divided by this before the encoder
    point_anchor = (0.0, -0.1, 0.56)  # env-frame anchor (LEAP palm) subtracted from points
    point_feature_dim = 4  # (x, y, z, class flag)
    rgbd_depth_clip = 1.5  # m; depths beyond this are invalid
