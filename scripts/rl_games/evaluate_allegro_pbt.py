"""Evaluate native Allegro goals/episode, or smoke-test measurement with --smoke."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint")
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--max_steps", type=int, default=20000)
parser.add_argument("--seed", type=int, default=1042)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
if "--headless" not in parser._option_string_actions:
    parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
if args.output.exists() or not (args.smoke or args.checkpoint):
    parser.error("Use --smoke or --checkpoint, and a new output path")
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import hashlib
import json
import time
import torch
import traceback

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401
from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt.allegro import attach_measurement

TASK = "Isaac-Repose-Cube-Allegro-Direct-v0"


def main():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(TASK, cfg=cfg)
    base = env.unwrapped
    attach_measurement(base)
    try:
        if args.smoke:
            env.reset()
            base._compute_intermediate_values()
            # Set exact native orientation target, then invoke the unchanged native
            # reward. No simulation motion can spoil this known-success assertion.
            base.goal_rot[:] = base.object_rot
            before = base.successes.clone()
            base.reset_buf[:] = False
            result = base._get_rewards()
            assert torch.isfinite(result).all()
            assert torch.equal(base.successes, before + 1)
            expected = base.successes.sum().item()
            base.reset_buf[:] = True
            base._reset_idx(torch.arange(base.num_envs, device=base.device))
            assert base.extras["allegro_episodes"][1] == expected
            assert not base.successes.any()
            for _ in range(5):
                obs, reward, _, _, _ = env.step(torch.zeros(base.num_envs, cfg.action_space, device=base.device))
                assert torch.isfinite(reward).all() and torch.isfinite(obs["policy"]).all()
            report = {
                "task": TASK,
                "forced_goal_smoke": "passed",
                "counted_goals": expected,
                "observation_shape": list(obs["policy"].shape),
            }
        else:
            agent = load_cfg_from_registry(TASK, "rl_games_cfg_entry_point")
            agent["params"]["seed"] = args.seed
            agent["params"]["config"].update(device=args.device, device_name=args.device, num_actors=args.num_envs)
            groups = {"obs": ["policy"], "states": ["policy"]}
            try:
                wrapped = RlGamesVecEnvWrapper(env, args.device, 5.0, 1.0, groups, True)
            except TypeError:
                wrapped = RlGamesVecEnvWrapper(env, args.device, 5.0, 1.0, groups)
            vecenv.register(
                "IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors)
            )
            env_configurations.register(
                "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped}
            )
            runner = Runner()
            runner.load(agent)
            player = runner.create_player()
            checkpoint = Path(args.checkpoint).resolve(strict=True)
            saved = torch.load(checkpoint, map_location=player.device, weights_only=False)
            player.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in saved["model"].items()})
            if player.normalize_input and "running_mean_std" in saved:
                player.model.running_mean_std.load_state_dict(saved["running_mean_std"])
            player.reset()
            torch.manual_seed(args.seed)
            obs = wrapped.reset()
            player.get_batch_size(obs["obs"] if isinstance(obs, dict) else obs, 1)
            n, goals, squares = 0, 0.0, 0.0
            start = time.monotonic()
            steps = 0
            while steps < args.max_steps and n < args.episodes:
                with torch.inference_mode():
                    action = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
                    obs, _, _, info = wrapped.step(action)
                count, total, total_sq = info.get("allegro_episodes", (0, 0, 0))
                n += count
                goals += total
                squares += total_sq
                steps += 1
            report = dict(
                task=TASK,
                checkpoint=str(checkpoint),
                seed=args.seed,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                completed_episodes=n,
                goals=goals,
                goals_squared=squares,
                goals_per_episode=goals / n if n else None,
                requested_episodes=args.episodes,
                completed=n >= args.episodes,
                deterministic=True,
                elapsed_seconds=time.monotonic() - start,
                control_steps=steps,
                success_tolerance=cfg.success_tolerance,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report), flush=True)
        if not args.smoke and not report["completed"]:
            raise RuntimeError("Incomplete evaluation")
    finally:
        env.close()


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        app.close(exit_code=code)
