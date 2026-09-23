"""DexPBT replication task registration: official DexSuite Reorient MDP + rl-games PPO population.

``Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0`` -- the official manager-based
Reorient env cfg (plus the DexPBT success-tolerance annealing curriculum)
paired with an rl-games PPO agent cfg carrying a ``pbt:`` block. Launched one
process per member via ``scripts/launch/dexpbt_reorient.sh``; the box's
IsaacLab rl-games train script auto-attaches ``PbtAlgoObserver`` when it sees
``pbt.enabled``.
"""

import gymnasium as gym

gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0",
    entry_point=f"{__name__}.environment:DexsuiteReorientPbtEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_reorient_pbt_env_cfg:DexsuiteReorientPbtEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_pbt.yaml",
    },
)

gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-Play-v0",
    entry_point=f"{__name__}.environment:DexsuiteReorientPbtEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_reorient_pbt_env_cfg:DexsuiteReorientPbtEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_pbt.yaml",
    },
)
