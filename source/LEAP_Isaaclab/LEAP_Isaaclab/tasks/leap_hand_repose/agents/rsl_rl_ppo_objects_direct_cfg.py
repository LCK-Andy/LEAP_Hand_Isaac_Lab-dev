# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_direct_cfg import LeapHandDirectPPORunnerCfg


@configclass
class LeapObjectsDirectPPORunnerCfg(LeapHandDirectPPORunnerCfg):
    """Same PPO recipe as the cube direct task, separate experiment log."""

    experiment_name = "leap_objects_direct"
