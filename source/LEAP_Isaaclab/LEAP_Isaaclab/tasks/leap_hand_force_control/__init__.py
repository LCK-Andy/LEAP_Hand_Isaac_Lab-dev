"""
LEAP Hand Force Control environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

force_task_entry = "LEAP_Isaaclab.tasks.leap_hand_force_control"

gym.register(
    id="Isaac-ForceControl-Leap",
    entry_point=f"{force_task_entry}.force_control_env:ForceControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_force_env_cfg:LeapHandForceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
