"""Cartesian-palm control ablation; legacy joint-control registrations are unchanged."""

import gymnasium as gym

for suffix, config in (
    ("v0", "DexsuiteReorientCartesianEnvCfg"),
    ("Play-v0", "DexsuiteReorientCartesianEnvCfg_PLAY"),
):
    gym.register(
        id=f"Isaac-Dexsuite-Kuka-Allegro-Reorient-Cartesian-{suffix}",
        entry_point=f"{__name__}.environment:DexsuiteReorientCartesianEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{config}",
            "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo.yaml",
        },
    )
