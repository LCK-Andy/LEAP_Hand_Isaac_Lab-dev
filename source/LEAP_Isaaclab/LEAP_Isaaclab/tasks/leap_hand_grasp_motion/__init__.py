"""Stage B fixed-hand cube motion tracking, separate from Stage A grasp holding."""

import gymnasium as gym

from . import agents

for task_id, config_name in (
    ("Isaac-Grasp-Motion-Cube-LEAP-Direct-v0", "LeapGraspMotionEnvCfg"),
    ("Isaac-Grasp-Motion-Cube-LEAP-Direct-Play-v0", "LeapGraspMotionEnvCfg_PLAY"),
):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.leap_grasp_motion_env:LeapGraspMotionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.leap_grasp_motion_env_cfg:{config_name}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_grasp_motion_cfg.yaml",
        },
    )
