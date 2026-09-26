"""Persistent, force-regulated four-pulp grasp tasks, separate from contact-point acquisition."""

import gymnasium as gym

from . import agents

for task_id, config_name in (
    ("Isaac-Grasp-Cube-LEAP-Direct-v0", "LeapGraspEnvCfg"),
    ("Isaac-Grasp-Cube-LEAP-Direct-Play-v0", "LeapGraspEnvCfg_PLAY"),
):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.leap_grasp_env:LeapGraspEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.leap_grasp_env_cfg:{config_name}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_grasp_cfg.yaml",
        },
    )
