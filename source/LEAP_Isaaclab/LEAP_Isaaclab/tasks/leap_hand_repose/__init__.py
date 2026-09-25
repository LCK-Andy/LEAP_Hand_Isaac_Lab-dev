"""
LEAP hand in-hand cube reorientation environment.

Identical to the official ``Isaac-Repose-Cube-Allegro-v0`` environment, with the
Allegro hand substituted by the 16-DOF LEAP hand (right, v1). The environment is
manager-based and reuses the official in-hand manipulation MDP from Isaac Lab:

https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Repose-Cube-LEAP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_repose_env_cfg:LeapCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeapCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-LEAP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_repose_env_cfg:LeapCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeapCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

##
# Direct (DirectRLEnv) variant: official Isaac-Repose-Cube-Allegro-Direct-v0 with the
# LEAP hand, restructured into the 7-term reference reward (leap_direct_env.py).
##

gym.register(
    id="Isaac-Repose-Cube-LEAP-Direct-v0",
    entry_point=f"{__name__}.leap_direct_env:LeapHandDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_direct_env_cfg:LeapHandDirectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_direct_cfg:LeapHandDirectPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_direct_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_direct_cfg.yaml",
    },
)

##
# v1: the direct task plus a downward-facing RGBD camera observation ("image" group,
# (4, 128, 128)); the 124-d "policy" obs is unchanged. rsl_rl only -- rl_games/skrl
# configs cannot consume mixed vector+image observations.
##

gym.register(
    id="Isaac-Repose-Cube-LEAP-Direct-v1",
    entry_point=f"{__name__}.leap_direct_v1_env:LeapHandDirectV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_direct_v1_env_cfg:LeapHandDirectV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_direct_v1_cfg:LeapDirectV1PPORunnerCfg",
    },
)

##
# v2: the direct task observed through geometry -- the downward depth camera is
# backprojected to a sparse point cloud split into object/hand sets via instance
# segmentation (DP3-style "points" group, (256, 4)); the 124-d "policy" obs is
# unchanged. rsl_rl only, via the custom LeapPointSetModel.
##

gym.register(
    id="Isaac-Repose-Cube-LEAP-Direct-v2",
    entry_point=f"{__name__}.leap_direct_v2_env:LeapHandDirectV2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_direct_v2_env_cfg:LeapHandDirectV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_direct_v2_cfg:LeapDirectV2PPORunnerCfg",
    },
)

##
# Objects variant: the direct task (same LeapHandDirectEnv / 7-term reference
# reward as v0) generalized to a catalog of different objects (YCB and others)
# via MultiUsdFileCfg heterogeneous cloning -- each env gets a randomly chosen
# object; the MDP is inherited unchanged.
##

gym.register(
    id="Isaac-Repose-Objects-LEAP-Direct-v0",
    entry_point=f"{__name__}.leap_objects_direct_env:LeapObjectsDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_objects_direct_env_cfg:LeapObjectsDirectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_objects_direct_cfg:LeapObjectsDirectPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_direct_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_direct_cfg.yaml",
    },
)

##
# Objects v1: the objects task with the in-hand object's inertial properties
# (mass, COM, inertia) drawn from the sibling Estimate_weight_distribution
# ground-truth manifest -- authored on the template prims at spawn and
# RESAMPLED per episode via the RigidObjectCollection runtime writers. The
# ground truth stays hidden from the policy (obs unchanged); the catalog is
# restricted to manifest-covered objects.
##

gym.register(
    id="Isaac-Repose-Objects-LEAP-Direct-v1",
    entry_point=f"{__name__}.leap_objects_direct_v1_env:LeapObjectsDirectV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_objects_direct_v1_env_cfg:LeapObjectsDirectV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_objects_direct_v1_cfg:LeapObjectsDirectV1PPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_direct_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_direct_cfg.yaml",
    },
)

##
# Contact-point dexterity benchmark: the repose direct MDP with the orientation
# goal replaced by 4 contact points sampled uniformly on the cube surface. A
# pulp within the distance threshold AND pressing above the force gate counts
# as touching a point (one-to-one matched); all 4 held continuously for a
# per-env sampled duration scores and resamples the points. PhysX-only (the
# force gate reads the per-pulp ContactSensors, which are dead kit-less).
##

gym.register(
    id="Isaac-Contact-Cube-LEAP-Direct-v0",
    entry_point=f"{__name__}.leap_contact_direct_env:LeapContactDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_contact_direct_env_cfg:LeapContactDirectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_contact_direct_cfg:LeapContactDirectPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_contact_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Contact-Cube-LEAP-Direct-Play-v0",
    entry_point=f"{__name__}.leap_contact_direct_env:LeapContactDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_contact_direct_env_cfg:LeapContactDirectEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_contact_direct_cfg:LeapContactDirectPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_contact_cfg.yaml",
    },
)

##
# GRU variant: the SAME env/MDP with a recurrent (GRU-256) actor-critic -- the
# policy can integrate tactile/force history over time (the hold-to-confirm
# success is a temporal predicate). PPO hyperparams identical to the MLP task
# for a clean A/B.
##

gym.register(
    id="Isaac-Contact-Cube-LEAP-Direct-GRU-v0",
    entry_point=f"{__name__}.leap_contact_direct_env:LeapContactDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_contact_direct_env_cfg:LeapContactDirectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_contact_direct_gru_cfg:LeapContactDirectGRUPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Contact-Cube-LEAP-Direct-GRU-Play-v0",
    entry_point=f"{__name__}.leap_contact_direct_env:LeapContactDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_contact_direct_env_cfg:LeapContactDirectEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_contact_direct_gru_cfg:LeapContactDirectGRUPPORunnerCfg",
    },
)
