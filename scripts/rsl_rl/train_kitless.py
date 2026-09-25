# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Train an RL agent with RSL-RL, launching Kit ONLY if the task needs it.

Unlike ``train.py`` (which always launches the Isaac Sim app first), this
script resolves the environment config BEFORE launching anything and hands it
to :func:`isaaclab.app.sim_launcher.launch_simulation`:

* Tasks whose physics is a kit-less backend (e.g. a ``NewtonCfg`` preset) run
  WITHOUT Isaac Sim / Omniverse Kit entirely -- seconds-fast startup, no
  Vulkan/AppLauncher machinery.
* Tasks with Kit physics (PhysX) or Kit-rendered cameras still launch Kit
  transparently, so the same script covers both.

Passing ``presets=<name>`` as a hydra-style argument selects the matching
``PresetCfg`` alternatives in the task config. The classic tasks ship a
``newton_mjwarp`` preset, e.g.::

    python scripts/rsl_rl/train_kitless.py --task Isaac-Humanoid-v0 \
        --num_envs 2048 presets=newton_mjwarp
"""

import argparse
import os
import sys
from datetime import datetime

import gymnasium as gym
import torch

# local imports
import cli_args  # isort: skip

from isaaclab.app import AppLauncher  # noqa: F401  (adds the launcher CLI args)

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL (kit-aware launcher).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for the hydra-backed config resolution (presets=... etc.)
sys.argv = [sys.argv[0]] + hydra_args

"""Resolve the configs first, then launch whatever runtime the task needs."""

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab.app.sim_launcher import launch_simulation

import LEAP_Isaaclab.tasks  # noqa: F401  (registers the custom gym ids before config resolution)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class NanGuardWrapper(gym.Wrapper):
    """Reset envs that produced non-finite data instead of crashing training.

    Beta GPU physics backends (Newton / OVPhysX) can diverge in rare states
    (e.g. after a dropped-constraint overflow or a solver blow-up); a single
    NaN would otherwise abort the whole run via rsl_rl's ``check_nan``.
    Diverged envs are reset immediately, their poisoned transition is replaced
    with a sanitized reset observation, and a running counter is printed.

    The replacement values are explicitly sanitized: the first observation
    read after an Isaac Lab reset can STILL be non-finite (data buffers are
    stale until the next physics step, and a hard-diverged backend state may
    not be fully cured by the reset writes), so the raw post-reset values
    must never be trusted for the patched rows. A final sweep sanitizes any
    remaining non-finite values in every obs group and the rewards so a
    divergence costs one env-reset instead of the run.
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
            print(
                f"[NanGuard] {int(bad_ids.numel())} env(s) diverged; resetting "
                f"(total {self._nan_reset_count})"
            )
            # unwrap past gymnasium's OrderEnforcing, which drops custom kwargs
            # like env_ids (safe mid-episode: ordering is reset-before-step, and
            # every env has already stepped)
            fresh_obs, _ = self.env.unwrapped.reset(env_ids=bad_ids)
            fresh = fresh_obs["policy"] if isinstance(fresh_obs, dict) else fresh_obs
            fresh = torch.nan_to_num(fresh, nan=0.0, posinf=0.0, neginf=0.0)
            if isinstance(obs, dict):
                obs["policy"][bad_ids] = fresh[bad_ids]
            else:
                obs[bad_ids] = fresh[bad_ids]
            rew[bad_ids] = 0.0
            terminated[bad_ids] = True
        # final fallback: a divergence must never reach the RL loop. Sanitize
        # any straggler non-finite values (all obs groups + rewards) in-place.
        if isinstance(obs, dict):
            for value in obs.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0, out=value)
        elif torch.is_tensor(obs) and not torch.isfinite(obs).all():
            torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0, out=obs)
        if not torch.isfinite(rew).all():
            torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0, out=rew)
        return obs, rew, terminated, truncated, extras


def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg) -> None:
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # keep sporadic physics divergences from killing the run
    env = NanGuardWrapper(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # isaaclab_rl 3.0.0b2 still serializes the deprecated rsl_rl<4 model fields,
    # which rsl-rl 5.0.1's MLPModel rejects (same fix as the LEAP task cfgs,
    # but applied generically to the serialized dict so ANY stock task works)
    _DEPRECATED_MODEL_FIELDS = ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std")
    agent_dict = agent_cfg.to_dict()
    for section in ("actor", "critic"):
        model_cfg = agent_dict.get(section)
        if isinstance(model_cfg, dict):
            for field in _DEPRECATED_MODEL_FIELDS:
                model_cfg.pop(field, None)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    with launch_simulation(env_cfg, args_cli):
        main(env_cfg, agent_cfg)
