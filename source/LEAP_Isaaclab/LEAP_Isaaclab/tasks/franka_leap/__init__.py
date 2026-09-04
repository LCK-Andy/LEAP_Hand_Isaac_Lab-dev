"""
Franka Panda arm + LEAP hand environment registration.
"""

import gymnasium as gym

from . import agents, franka_leap_env_cfg, franka_leap_env  # noqa: F401  (register via tasks __init__)

gym.register(
    id="Isaac-Franka-Leap",
    entry_point="LEAP_Isaaclab.tasks.franka_leap.franka_leap_env:FrankaLeapEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_leap_env_cfg:FrankaLeapEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
