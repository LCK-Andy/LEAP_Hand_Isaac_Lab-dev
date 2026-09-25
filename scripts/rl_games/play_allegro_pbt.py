"""Continuously play a native Allegro PBT policy (no training-state restore)."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=1042)
AppLauncher.add_app_launcher_args(parser)
if "--headless" not in parser._option_string_actions:
    parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
args.checkpoint = args.checkpoint.resolve(strict=True)
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
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
    """Load only policy/normalizers and render until the user closes Kit."""
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(TASK, cfg=cfg)
    base = env.unwrapped
    attach_measurement(base)
    try:
        agent = load_cfg_from_registry(TASK, "rl_games_cfg_entry_point")
        agent["params"]["seed"] = args.seed
        agent["params"]["config"].update(device=args.device, device_name=args.device, num_actors=args.num_envs)
        wrapped = RlGamesVecEnvWrapper(env, args.device, 5.0, 1.0, obs_groups={"obs": ["policy"], "states": ["policy"]})
        vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors))
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
        runner = Runner()
        runner.load(agent)
        player = runner.create_player()
        saved = torch.load(args.checkpoint, map_location=player.device, weights_only=False)
        player.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in saved["model"].items()})
        if player.normalize_input and "running_mean_std" in saved:
            player.model.running_mean_std.load_state_dict(saved["running_mean_std"])
        player.model.eval()
        player.reset()
        torch.manual_seed(args.seed)
        obs = wrapped.reset()
        player.get_batch_size(obs["obs"] if isinstance(obs, dict) else obs, 1)
        if player.is_rnn:
            player.init_rnn()
        steps, episodes, goals = 0, 0, 0.0
        print(f"[PLAY] Loaded {args.checkpoint}; task={TASK}; envs={base.num_envs}", flush=True)
        while app.is_running():
            start = time.monotonic()
            with torch.inference_mode():
                action = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
                obs, reward, dones, info = wrapped.step(action)
                if player.is_rnn and player.states is not None:
                    for state in player.states:
                        state[:, dones.bool(), :] = 0.0
            count, total, _ = info.get("allegro_episodes", (0, 0, 0))
            episodes += count
            goals += total
            steps += 1
            if steps % 100 == 0:
                print(
                    f"[PLAY] steps={steps} reward={reward.mean().item():.3f} "
                    f"completed_episodes={episodes} native_goals={goals:.0f}",
                    flush=True,
                )
            time.sleep(max(0.0, base.step_dt - (time.monotonic() - start)))
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
