# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from isaaclab.utils.configclass import configclass

from LEAP_Isaaclab.tasks.leap_hand_reorient.agents.rsl_rl_ppo_rgbd_cfg import LeapHandRGBDPPORunnerCfg


@configclass
class LeapDirectV1PPORunnerCfg(LeapHandRGBDPPORunnerCfg):
    """CNN-on-RGBD PPO runner for Isaac-Repose-Cube-LEAP-Direct-v1.

    Reuses the RGBD runner of ``Isaac-Reorient-Cube-Leap-RGBD``: the actor/critic
    fuse the CNN embedding of the (4, 128, 128) ``image`` group with the 124-d
    ``policy`` group (obs_groups are inherited). The base class already strips
    the deprecated rsl_rl<4 model fields from its serialized dict.
    """

    experiment_name = "leap_cube_direct_v1"


# @configclass resets to_dict on this subclass, shadowing the parent's patched
# serializer (which strips the deprecated rsl_rl<4 model fields); re-attach it.
LeapDirectV1PPORunnerCfg.to_dict = LeapHandRGBDPPORunnerCfg.to_dict
