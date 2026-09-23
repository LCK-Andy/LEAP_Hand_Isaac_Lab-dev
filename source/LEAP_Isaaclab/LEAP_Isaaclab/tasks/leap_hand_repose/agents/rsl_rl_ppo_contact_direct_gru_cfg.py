# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlRNNModelCfg

from .rsl_rl_ppo_direct_cfg import LeapHandDirectPPORunnerCfg


@configclass
class LeapContactDirectGRUPPORunnerCfg(LeapHandDirectPPORunnerCfg):
    """Recurrent (GRU) RSL-RL PPO runner for Isaac-Contact-Cube-LEAP-Direct-GRU-v0.

    Same env, same PPO hyperparams as the MLP runner (clean A/B); the actor and
    critic become recurrent: obs -> GRU (256, 1 layer) -> MLP [512, 256] heads.
    Recurrence lets the policy integrate contact-force history and cube motion
    over time -- useful because the hold-to-confirm success is a temporal
    predicate and the tactile channels are impulse-like.
    """

    experiment_name = "leap_hand_contact_gru"

    actor = RslRlRNNModelCfg(
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        hidden_dims=[512, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlRNNModelCfg(
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        hidden_dims=[512, 256],
        activation="elu",
        obs_normalization=True,
    )

    # the parent's __post_init__ strips the deprecated rsl_rl<4 model fields
    # (RslRlRNNModelCfg inherits them from RslRlMLPModelCfg)
