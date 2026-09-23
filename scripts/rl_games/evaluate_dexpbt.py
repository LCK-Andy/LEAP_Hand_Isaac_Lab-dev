# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate a trusted DexPBT checkpoint at fixed difficulty without changing its policy."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, help="Trusted RL-Games checkpoint produced by this project.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument(
    "--episodes", type=int, default=512, help="Minimum completed episodes; finish the last vector step."
)
parser.add_argument("--max_steps", type=int, default=20000)
parser.add_argument("--seed", type=int, default=1042)
parser.add_argument("--adr_fraction", type=float, default=0.0, help="Fixed difficulty fraction: 0 acquisition, 1 full.")
parser.add_argument("--stochastic", action="store_true", help="Sample the saved policy instead of using its mean.")
parser.add_argument("--output", type=Path, required=True, help="New JSON report; existing files are not overwritten.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1 or args.num_envs < 1 or args.max_steps < 1 or not 0 <= args.adr_fraction <= 1:
    parser.error("episodes, num_envs and max_steps must be positive; adr_fraction must be in [0, 1]")
if args.output.exists():
    parser.error(f"Report already exists: {args.output}")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import hashlib
import json
import time
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401

TASK = "Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0"


def main():
    """Collect episode-weighted, reward-independent statistics at fixed evaluation conditions."""
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(TASK, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["config"].update(device=args.device, device_name=args.device, num_actors=args.num_envs)
    env = gym.make(TASK, cfg=cfg)
    try:
        groups = agent_cfg["params"]["env"]["obs_groups"]
        try:
            wrapped = RlGamesVecEnvWrapper(env, args.device, 100.0, 1.0, groups, True)
        except TypeError:
            wrapped = RlGamesVecEnvWrapper(env, args.device, 100.0, 1.0, {"obs": groups, "states": groups})
        vecenv.register(
            "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors)
        )
        env_configurations.register(
            "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped}
        )
        runner = Runner()
        runner.load(agent_cfg)
        player = runner.create_player()
        # Evaluation restores policy/normalizers, never the donor's training curriculum.
        saved = torch.load(checkpoint, map_location=player.device, weights_only=False)
        weights = {key.removeprefix("_orig_mod."): value for key, value in saved["model"].items()}
        player.model.load_state_dict(weights)
        if player.normalize_input and "running_mean_std" in saved:
            player.model.running_mean_std.load_state_dict(saved["running_mean_std"])
        del saved
        player.reset()
        env.unwrapped.configure_fixed_evaluation(args.adr_fraction)
        torch.manual_seed(args.seed)
        obs = wrapped.reset()
        player.get_batch_size(obs["obs"] if isinstance(obs, dict) else obs, 1)
        if player.is_rnn:
            player.init_rnn()
        totals = dict.fromkeys(
            [
                "completed_episodes",
                "success_count",
                "position_error_sum",
                "rotation_error_sum",
                "good_contact_fraction_sum",
                "episode_length_sum",
                "adr_difficulty_sum",
            ],
            0.0,
        )
        start = time.monotonic()
        steps = 0
        while simulation_app.is_running() and steps < args.max_steps and totals["completed_episodes"] < args.episodes:
            with torch.inference_mode():
                actions = player.get_action(player.obs_to_torch(obs), is_deterministic=not args.stochastic)
                obs, _, dones, info = wrapped.step(actions)
                if player.is_rnn and player.states is not None:
                    for state in player.states:
                        state[:, dones.bool(), :] = 0
            metrics = info["dexpbt"]
            if metrics["protocol"] != "heldout_fixed_definition_v1":
                raise RuntimeError("Environment is not in fixed evaluation mode")
            for key in totals:
                totals[key] += float(metrics[key])
            steps += 1
        n = int(totals["completed_episodes"])
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report = {
            "protocol": "heldout_fixed_definition_v1",
            "task": TASK,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "adr_fraction": args.adr_fraction,
            "deterministic": not args.stochastic,
            "condition": env.unwrapped.get_evaluation_condition(),
            "requested_episodes": args.episodes,
            "completed": n >= args.episodes,
            "control_steps": steps,
            "elapsed_seconds": time.monotonic() - start,
            "totals": totals,
            "success_rate": totals["success_count"] / n if n else None,
            "mean_position_error_m": totals["position_error_sum"] / n if n else None,
            "mean_rotation_error_rad": totals["rotation_error_sum"] / n if n else None,
            "mean_good_contact_fraction": totals["good_contact_fraction_sum"] / n if n else None,
            "mean_episode_length": totals["episode_length_sum"] / n if n else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(json.dumps(report, indent=2, allow_nan=False))
        if not report["completed"]:
            raise RuntimeError("Evaluation ended before the requested episode count; report marked incomplete")
    finally:
        env.close()


if __name__ == "__main__":
    import sys
    import traceback

    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close(exit_code=exit_code)
    sys.exit(exit_code)
