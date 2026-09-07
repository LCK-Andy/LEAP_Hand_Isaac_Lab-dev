# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# v1: the direct in-hand reorientation task (Isaac-Repose-Cube-LEAP-Direct-v0)
# plus a downward-facing RGBD camera observation, mirroring the RGBD variant of
# Isaac-Reorient-Cube-Leap (tasks/leap_hand_reorient/leap_hand_rgbd_env_cfg.py).
# --------------------------------------------------------

from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

import isaaclab.sim as sim_utils

from .leap_direct_env_cfg import LeapHandDirectEnvCfg


@configclass
class LeapHandDirectV1EnvCfg(LeapHandDirectEnvCfg):
    """Direct in-hand reorientation with a downward-facing RGBD camera (v1).

    Identical to the official ``Isaac-Repose-Cube-Allegro-Direct-v0`` reward
    (124-d "full" policy obs, 16-d actions, all rewards/resets unchanged) plus
    a world-fixed RGBD camera above the LEAP palm exposed as a separate
    ``"image"`` observation group (rgb + clipped/normalized depth, (4, H, W)).
    """

    # v1 stays on the official reward (its env class uses InHandManipulationEnv's
    # compute_rewards), so re-pin the official dense position scale that the
    # v0 cfg disables for the reference reward. To give v1 the reference reward
    # instead, parent LeapHandDirectV1Env on LeapHandDirectEnv and drop this.
    dist_reward_scale = -10.0

    # world-fixed camera framing the hand and the in-hand cube from above.
    # convention "world": forward axis is +X, so rot (0, 0.7071, 0, 0.7071)
    # (90 deg about Y) turns the camera to look straight down (-Z).
    camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.1, 0.95),  # directly above the in-hand cube spawn
            rot=(0.0, 0.7071, 0.0, 0.7071),
            convention="world",
        ),
        data_types=["rgb", "depth"],
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

    # depth is clipped to [0, rgbd_depth_clip] m and normalized to [0, 1] in the observation
    rgbd_depth_clip = 1.5  # m
    rgbd_channels = 4  # rgb (3) + depth (1)
