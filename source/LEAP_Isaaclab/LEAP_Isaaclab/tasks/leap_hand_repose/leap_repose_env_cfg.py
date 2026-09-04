# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on the official Isaac Lab in-hand reorientation task:
# https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/allegro_env_cfg.py
# --------------------------------------------------------

from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.manipulation.inhand.inhand_env_cfg as inhand_env_cfg

##
# Pre-defined configs
##
from LEAP_Isaaclab.assets import LEAP_HAND_CFG  # isort: skip


@configclass
class LeapCubeEnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
    """In-hand cube reorientation with the LEAP hand.

    Identical to ``Isaac-Repose-Cube-Allegro-v0`` except that the Allegro hand is
    substituted with the 16-DOF LEAP hand (right, v1). All MDP terms (actions,
    observations, events, rewards, terminations, commands) are inherited unchanged
    from the official environment since they address the robot via regex joint names.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to LEAP hand
        self.scene.robot = LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # enable clone in fabric
        self.scene.clone_in_fabric = True
        # place the cube in the LEAP palm: the LEAP palm center sits closer to the
        # hand origin than the Allegro's (-0.19 -> -0.10 along y in the world frame)
        self.scene.object.init_state.pos = (0.0, -0.10, 0.56)


@configclass
class LeapCubeEnvCfg_PLAY(LeapCubeEnvCfg):
    """Play configuration of the LEAP cube reorientation environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        self.terminations.time_out = None
