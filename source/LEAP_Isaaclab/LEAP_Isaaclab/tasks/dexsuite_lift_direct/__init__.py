"""Direct-RL port of the official Dexsuite KUKA+Allegro Lift task.

Registered ids:
  - ``Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0``      (train, ADR from difficulty 0)
  - ``Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-Play-v0`` (eval, ADR at max difficulty)

See ``dexsuite_lift_direct_env.py`` for the port map and the v1 parity gaps.
Train with FlashSAC via ``scripts/flashsac_train_dexsuite_lift.sh``.
"""

import gymnasium as gym

from .dexsuite_lift_direct_env import DexsuiteKukaAllegroLiftDirectEnv
from .dexsuite_lift_direct_env_cfg import (
    DexsuiteKukaAllegroLiftDirectEnvCfg,
    DexsuiteKukaAllegroLiftDirectEnvCfg_PLAY,
)

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0",
    entry_point=f"{__name__}.dexsuite_lift_direct_env:DexsuiteKukaAllegroLiftDirectEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": DexsuiteKukaAllegroLiftDirectEnvCfg},
)

gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-Play-v0",
    entry_point=f"{__name__}.dexsuite_lift_direct_env:DexsuiteKukaAllegroLiftDirectEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": DexsuiteKukaAllegroLiftDirectEnvCfg_PLAY},
)
