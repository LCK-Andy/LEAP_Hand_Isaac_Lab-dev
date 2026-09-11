# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Train an RL agent with RL-Games, launching Kit ONLY if the task needs it.

The rl_games counterpart of ``scripts/rsl_rl/train_kitless.py``: the official
``scripts/rl_games/train.py`` builds an ``AppLauncher`` first, so it can never run
without Isaac Sim. Here the environment AND agent configs are resolved BEFORE
launching anything and handed to
:func:`isaaclab_tasks.utils.sim_launcher.launch_simulation`:

* Tasks whose physics is a kit-less backend (e.g. a ``NewtonCfg`` preset) run
  WITHOUT Isaac Sim / Omniverse Kit entirely -- seconds-fast startup, no
  Vulkan/AppLauncher machinery.
* Tasks with Kit physics (PhysX) still launch Kit transparently, so the same
  script covers both.

Passing ``presets=<name>`` as a hydra-style argument selects the matching
``PresetCfg`` alternatives in the task config, e.g.::

    python scripts/rl_games/train_kitless.py --task Isaac-Humanoid-Direct-v0 \
        --headless presets=newton_mjwarp

Only tasks registered in ``isaaclab_tasks`` are covered (the LEAP tasks import
the Kit-bound asset chain and are PhysX-only anyway -- run those through
``scripts/rl_games/train.py``).

Differences from the official entrypoint: video recording is not offered
(kit-less has no renderer -- use ``train.py`` for that), ``--distributed`` is
not offered (no AppLauncher ranks), and a ``NanGuardWrapper`` sits between the
env and the rl-games wrapper so a rare Newton solver divergence resets the
affected envs instead of killing the run.
"""

import argparse
import math
import os
import random
import sys
from datetime import datetime

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher  # noqa: F401  (adds the launcher CLI args)

parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games (kit-aware launcher).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations (epochs).")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume training from an rl-games checkpoint.")
parser.add_argument(
    "--sigma", type=str, default=None, help="Initial noise std for rl-games (e.g. '1.0' or a range '1.0,1.0')."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.task is None:
    parser.error("--task is required")

# clear out sys.argv for the hydra-backed config resolution (presets=... etc.)
sys.argv = [sys.argv[0]] + hydra_args

"""Resolve the configs first, then launch whatever runtime the task needs."""

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.sim_launcher import launch_simulation

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class NanGuardWrapper(gym.Wrapper):
    """Reset envs that produced non-finite data instead of crashing training.

    Same contract as the guard in ``scripts/rsl_rl/train_kitless.py``: beta GPU
    physics backends (Newton / OVPhysX) can diverge in rare states; a single NaN
    would otherwise abort the whole run. Diverged envs are reset immediately,
    their poisoned transition is replaced with a sanitized reset observation,
    and a running counter is printed. The replacement values are explicitly
    sanitized: the first observation read after an Isaac Lab reset can STILL be
    non-finite (data buffers are stale until the next physics step), so raw
    post-reset values must never be trusted for the patched rows.
    """

    def __init__(self, env):
        super().__init__(env)
        self._nan_reset_count = 0

    def step(self, action):
        obs, rew, terminated, truncated, extras = self.env.step(action)
        obs_t = obs["policy"] if isinstance(obs, dict) else obs
        bad = ~torch.isfinite(obs_t).all(dim=-1)
        bad |= ~torch.isfinite(rew)
        if bad.any():
            bad_ids = bad.nonzero(as_tuple=False).squeeze(-1)
            self._nan_reset_count += int(bad_ids.numel())
            print(f"[NanGuard] {int(bad_ids.numel())} env(s) diverged; resetting (total {self._nan_reset_count})")
            # unwrap past gymnasium's OrderEnforcing, which drops custom kwargs like env_ids
            fresh_obs, _ = self.env.unwrapped.reset(env_ids=bad_ids)
            fresh = fresh_obs["policy"] if isinstance(fresh_obs, dict) else fresh_obs
            fresh = torch.nan_to_num(fresh, nan=0.0, posinf=0.0, neginf=0.0)
            if isinstance(obs, dict):
                obs["policy"][bad_ids] = fresh[bad_ids]
            else:
                obs[bad_ids] = fresh[bad_ids]
            rew[bad_ids] = 0.0
            terminated[bad_ids] = True
        # final fallback: a divergence must never reach the RL loop
        if isinstance(obs, dict):
            for value in obs.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0, out=value)
        elif torch.is_tensor(obs) and not torch.isfinite(obs).all():
            torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0, out=obs)
        if not torch.isfinite(rew).all():
            torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0, out=rew)
        return obs, rew, terminated, truncated, extras


def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict) -> None:
    """Train with RL-Games agent (body mirrors the official train.py)."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    resume_path = None
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # set the environment seed (certain randomizations occur during env initialization)
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # keep sporadic physics divergences from killing the run (Newton backends)
    env = NanGuardWrapper(env)

    # wrap around environment for rl-games
    # image observations need dict mode so rl-games routes "image" through its CNN
    if "image" in env.unwrapped.single_observation_space:
        env = RlGamesVecEnvWrapper(
            env,
            rl_device,
            clip_obs,
            clip_actions,
            obs_groups={"obs": ["policy", "image"], "states": []},
            concate_obs_group=False,
        )
    else:
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent
    if resume_path is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # close the simulator
    env.close()


if __name__ == "__main__":
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rl_games_cfg_entry_point")
    with launch_simulation(env_cfg, args_cli):
        main(env_cfg, agent_cfg)
