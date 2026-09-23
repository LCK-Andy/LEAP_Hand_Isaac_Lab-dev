# Copyright (c) 2025, The Isaac Lab Project Developers.
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_direct_cfg import LeapHandDirectPPORunnerCfg


@configclass
class LeapContactDirectPPORunnerCfg(LeapHandDirectPPORunnerCfg):
    """RSL-RL PPO runner for Isaac-Contact-Cube-LEAP-Direct-v0 (same hyperparams as the repose direct task)."""

    experiment_name = "leap_hand_contact"

    # rsl_rl 5.0.1's MLPModel rejects the deprecated rsl_rl<4 fields that RslRlMLPModelCfg
    # still declares (they serialize into cfg["actor"]/cfg["critic"] and reach the model
    # constructor as unexpected kwargs). The parent's __post_init__ strips them
    # (@configclass keeps inherited __post_init__ for subclasses).
