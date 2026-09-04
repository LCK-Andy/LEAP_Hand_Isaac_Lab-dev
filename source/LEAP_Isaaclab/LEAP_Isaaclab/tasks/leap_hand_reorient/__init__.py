"""
Leap Inhand Manipulation environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

reorient_task_entry = "LEAP_Isaaclab.tasks.leap_hand_reorient"

gym.register(
    id="Isaac-Reorient-Cube-Leap",
    entry_point=f"{reorient_task_entry}.reorientation_env:ReorientationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeapHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reorient-Cube-Leap-RGBD",
    entry_point=f"{reorient_task_entry}.reorientation_env:ReorientationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_rgbd_env_cfg:LeapHandRGBDCfg",
        # rsl-rl only: rl_games' actor_critic network cannot consume mixed vector+image obs
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_rgbd_cfg:LeapHandRGBDPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reorient-Cube-Leap-Realsense-D435",
    entry_point=f"{reorient_task_entry}.reorientation_env:ReorientationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_realsense_d435_env_cfg:LeapHandRealsenseD435Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_rgbd_cfg:LeapHandRGBDPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reorient-YCB-Leap",
    entry_point=f"{reorient_task_entry}.reorientation_env:ReorientationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_reorient_ycb_env_cfg:LeapHandReorientYCBCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ycb_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeapHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)