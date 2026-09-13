"""Direct-RL port of the official Dexsuite KUKA+Allegro Lift task.

Registered ids:
  - ``Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0``      (train, ADR from difficulty 0)
  - ``Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-Play-v0`` (eval, ADR at max difficulty)

See ``dexsuite_lift_direct_env.py`` for the port map and the v1 parity gaps.
Train with FlashSAC via ``scripts/launch/flashsac.sh`` (``train --task Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0``).
"""

import gymnasium as gym

# Keep registration fully lazy (string entry points only, no module imports):
# importing the env class eagerly pulls in isaaclab.envs.DirectRLEnv and with it
# isaacsim/carb/omni.kit.app BEFORE SimulationApp exists, which natively crashes
# Kit startup (all sibling task packages are imported together at registration).

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0",
    entry_point=f"{__name__}.dexsuite_lift_direct_env:DexsuiteKukaAllegroLiftDirectEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.dexsuite_lift_direct_env_cfg:DexsuiteKukaAllegroLiftDirectEnvCfg"},
)

gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-Play-v0",
    entry_point=f"{__name__}.dexsuite_lift_direct_env:DexsuiteKukaAllegroLiftDirectEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.dexsuite_lift_direct_env_cfg:DexsuiteKukaAllegroLiftDirectEnvCfg_PLAY"},
)
