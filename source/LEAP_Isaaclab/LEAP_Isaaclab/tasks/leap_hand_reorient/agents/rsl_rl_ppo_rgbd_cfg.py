# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LeapHandRGBDPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the RGBD variant: CNN on the camera image + MLP on proprio.

    The actor fuses the CNN embedding of the config-defined ``(C, H, W)`` RGBD
    ``"image"`` group with the 96-dim ``"policy"`` group. The critic sees the
    same groups (symmetric actor-critic).
    """

    seed = 42
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 200
    experiment_name = "leap_hand_reorient_rgbd"
    device = "cuda:0"

    # environment observation groups -> rsl-rl observation sets
    obs_groups = {
        "actor": ["policy", "image"],
        "critic": ["policy", "image"],
    }

    actor = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=(32, 64, 64),
            kernel_size=(8, 4, 3),
            stride=(4, 2, 1),
            activation="relu",
        ),
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    critic = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=(32, 64, 64),
            kernel_size=(8, 4, 3),
            stride=(4, 2, 1),
            activation="relu",
        ),
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
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


# Note: @configclass overwrites any "to_dict" defined in the class body
# (setattr at decoration time), so the override is attached after the class exists.
_configclass_to_dict = LeapHandRGBDPPORunnerCfg.to_dict
_DEPRECATED_MODEL_KEYS = ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std")


def _to_dict_without_deprecated(self) -> dict:
    """rsl-rl <5.0 model fields leak through configclass.to_dict() (unset fields
    serialize as "{}"), and the 5.x model constructors reject them as kwargs."""
    cfg_dict = _configclass_to_dict(self)
    for model_key in ("actor", "critic"):
        for key in _DEPRECATED_MODEL_KEYS:
            cfg_dict[model_key].pop(key, None)
    return cfg_dict


LeapHandRGBDPPORunnerCfg.to_dict = _to_dict_without_deprecated
