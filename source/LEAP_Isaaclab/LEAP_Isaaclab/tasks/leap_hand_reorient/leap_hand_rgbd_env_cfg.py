# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

import isaaclab.sim as sim_utils

from .leap_hand_env_cfg import LeapHandEnvCfg


@configclass
class LeapHandRGBDCfg(LeapHandEnvCfg):
    """Reorient cube env config with a world-fixed RGBD camera in the policy observation.

    The policy observation becomes a dict of the original 96-dim proprio history
    (``policy``) plus a normalized RGBD image (``image``, shape (H, W, 4)).
    """

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
