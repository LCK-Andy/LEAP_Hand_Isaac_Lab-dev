"""
Kuka iiwa7 arm + LEAP hand environments.
"""

import gymnasium as gym

from . import agents
from .dexsuite_kuka_leap_env_cfg import DexsuiteKukaLeapReorientEnvCfg, DexsuiteKukaLeapReorientEnvCfg_PLAY
from .dexsuite_kuka_leap_env_cfg import DexsuiteKukaLeapLiftEnvCfg, DexsuiteKukaLeapLiftEnvCfg_PLAY

## Kuka LEAP Reorient
gym.register(
    id="Isaac-Dexsuite-Kuka-LEAP-Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_leap_env_cfg:DexsuiteKukaLeapReorientEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaLeapPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-Kuka-LEAP-Reorient-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_leap_env_cfg:DexsuiteKukaLeapReorientEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaLeapPPORunnerCfg",
    },
)

## Kuka LEAP Lift
gym.register(
    id="Isaac-Dexsuite-Kuka-LEAP-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_leap_env_cfg:DexsuiteKukaLeapLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaLeapPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-Kuka-LEAP-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_leap_env_cfg:DexsuiteKukaLeapLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaLeapPPORunnerCfg",
    },
)
