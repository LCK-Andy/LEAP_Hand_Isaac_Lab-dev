"""Package containing task implementations for various robotic environments."""

import os
import toml

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

# EA renamed and changed the native Allegro MDP. Keep existing PBT experiments
# on their original semantics, without replacing EA's new task registration.
import gymnasium as gym

if "Isaac-Repose-Cube-Allegro-Direct-v0" not in gym.registry:
    gym.register(
        id="Isaac-Repose-Cube-Allegro-Direct-v0",
        entry_point="LEAP_Isaaclab.compat.inhand_direct.inhand_manipulation_env:InHandManipulationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "LEAP_Isaaclab.compat.allegro.allegro_hand_env_cfg:AllegroHandEnvCfg",
            "rl_games_cfg_entry_point": "LEAP_Isaaclab.compat.allegro.agents:rl_games_ppo_cfg.yaml",
            "rsl_rl_cfg_entry_point": "LEAP_Isaaclab.compat.allegro.agents.rsl_rl_ppo_cfg:AllegroHandPPORunnerCfg",
            "skrl_cfg_entry_point": "LEAP_Isaaclab.compat.allegro.agents:skrl_ppo_cfg.yaml",
        },
    )
