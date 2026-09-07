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
