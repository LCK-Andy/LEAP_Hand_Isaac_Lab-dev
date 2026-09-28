"""Separate original released DexPBT task registrations; no Dexsuite MDP inheritance."""

import gymnasium as gym

gym.register(
    id="Isaac-DexPBT-Allegro-Kuka-Reorientation-Direct-v0",
    entry_point=f"{__name__}.environment:DexPBTOriginalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DexPBTOriginalEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo.yaml",
    },
)

gym.register(
    id="Isaac-DexPBT-Allegro-Kuka-Reorientation-Direct-Play-v0",
    entry_point=f"{__name__}.environment:DexPBTOriginalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DexPBTOriginalEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo.yaml",
    },
)
