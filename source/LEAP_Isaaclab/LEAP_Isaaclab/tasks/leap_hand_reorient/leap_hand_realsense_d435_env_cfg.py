# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from .leap_hand_rgbd_env_cfg import LeapHandRGBDCfg

D435_INTRINSICS_320X240 = [229.8, 0.0, 160.0, 0.0, 229.8, 120.0, 0.0, 0.0, 1.0]


@configclass
class LeapHandRealsenseD435Cfg(LeapHandRGBDCfg):
    """LEAP cube reorientation with a D435-like aligned RGBD observation.

    This models one pinhole optical center that provides aligned RGB and metric
    Z-depth. It does not emulate RealSense stereo disparity, IR patterning,
    invalid depth pixels, or depth-noise characteristics.
    """

    camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.1, 0.95),
            rot=(0.0, 0.7071, 0.0, 0.7071),
            convention="world",
        ),
        data_types=["rgb", "depth"],
        update_period=1.0 / 30.0,
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=D435_INTRINSICS_320X240,
            width=320,
            height=240,
            clipping_range=(0.105, 3.0),
        ),
        width=320,
        height=240,
        renderer_cfg=MultiBackendRendererCfg(),
    )

    # A 32-step PPO rollout stores about 1.17 GiB of float32 RGBD images for
    # 32 environments; inheriting the proprio task's 8192-env default is unsafe.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=0.75, replicate_physics=False)
    rgbd_depth_clip = 3.0  # m
