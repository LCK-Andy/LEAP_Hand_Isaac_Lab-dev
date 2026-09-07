# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class LeapPointSetModelCfg:
    """Config for LEAP_Isaaclab.tasks.leap_hand_repose.point_model:LeapPointSetModel."""

    class_name: str = "LEAP_Isaaclab.tasks.leap_hand_repose.point_model:LeapPointSetModel"
    hidden_dims: list = [512, 256, 128]
    activation: str = "elu"
    obs_normalization: bool = True
    point_group: str = "points"
    # must match num_object_points in LeapHandDirectV2EnvCfg (leading object rows)
    point_split: int = 128
    point_hidden_dims: list = [64, 64]
    point_latent_dim: int = 64
    distribution_cfg: object = None


@configclass
class LeapDirectV2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """DP3-style point-cloud PPO runner for Isaac-Repose-Cube-LEAP-Direct-v2.

    The actor/critic encode the (256, 4) ``points`` group (object points first,
    then hand points) with a shared per-point MLP + masked max-pool per set,
    concatenated with the normalized 124-d ``policy`` group.
    """

    seed = 42
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 200
    experiment_name = "leap_cube_direct_v2"
    device = "cuda:0"

    obs_groups = {
        "actor": ["policy", "points"],
        "critic": ["policy", "points"],
    }

    actor = LeapPointSetModelCfg(distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0))
    critic = LeapPointSetModelCfg()

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=4.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=1,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=True,
    )
