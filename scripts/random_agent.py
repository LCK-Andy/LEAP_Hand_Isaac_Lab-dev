# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Random-action agent for the LEAP repose environments.

Same CLI API and default behavior as the official Isaac Lab script
(scripts/environments/random_agent.py): opens the Kit visualizer by default and
runs uniform random actions until the window is closed; `--headless` for
terminal runs. On top of that, `--verify` runs the environment-correctness
checks (finite obs/rewards, per-episode object resampling, parked-row
invariant, activation histogram) for `--num_steps` steps and prints a summary.

Examples:
  ~/env_isaaclab/bin/python scripts/random_agent.py                       # GUI, run forever
  ~/env_isaaclab/bin/python scripts/random_agent.py --headless            # terminal, run forever
  ~/env_isaaclab/bin/python scripts/random_agent.py --headless --verify   # verification run
  ~/env_isaaclab/bin/python scripts/random_agent.py --headless --verify --num_steps 1000
"""

import argparse
import contextlib
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for LEAP repose environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Repose-Objects-LEAP-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--verify",
    action="store_true",
    default=False,
    help="Run environment-correctness checks (finite obs/rewards, object resampling, parked-row invariant).",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=500,
    help="Number of steps for --verify (ignored without it; without --verify the agent runs forever).",
)
# append AppLauncher cli args
add_launcher_args(parser)
# simple agents should open Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args
# reconcile the official "visualizer on by default" with --headless: the kit default
# plus headless crashes Kit startup in this build when a full env cfg is resolved
if getattr(args_cli, "headless", False):
    args_cli.visualizer = []


def _verify_step(unwrapped, obs, reward, state):
    """Accumulate verification stats; returns an error string or None."""
    catalog = state["catalog"]
    if not torch.isfinite(obs["policy"]).all():
        return "non-finite observation"
    if not torch.isfinite(reward).all():
        return "non-finite reward"
    state["episode_returns"] += reward
    state["episode_lengths"] += 1
    done = state["done"]
    if done.any():
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        state["completed_returns"].extend(state["episode_returns"][done_ids].tolist())
        state["completed_lengths"].extend(state["episode_lengths"][done_ids].tolist())
        state["episode_returns"][done_ids] = 0.0
        state["episode_lengths"][done_ids] = 0
        if catalog is not None:
            active = unwrapped.active_object_ids
            state["resample_total"] += len(done_ids)
            state["resample_changed"] += int((state["prev_active"][done_ids] != active[done_ids]).sum())
            state["prev_active"] = active.clone()
            for i in active[done_ids].tolist():
                state["activation_counts"][i] += 1
            # inactive members must stay in the far parked row (checked for envs reset
            # in the PREVIOUS step: pose buffers refresh only at the next physics step)
            if state["prev_done_ids"] is not None:
                poses = unwrapped.objects.data.body_link_pose_w.torch  # (N, K, 7)
                rows = torch.arange(unwrapped.num_envs, device=unwrapped.device)
                palm_w = unwrapped.scene.env_origins + torch.tensor(
                    (0.0, -0.10, 0.5), device=unwrapped.device
                )
                dist = torch.linalg.norm(poses[..., 0:3] - palm_w.unsqueeze(1), dim=-1)
                inactive = torch.ones_like(dist, dtype=torch.bool)
                inactive[rows, active] = False
                inactive[state["prev_done_ids"]] = False
                if inactive.any() and dist[inactive].min() < 1.0:
                    bad = (dist < 1.0) & inactive
                    env_id, body_id = bad.nonzero(as_tuple=False)[0].tolist()
                    return (
                        f"inactive object inside the workspace: env {env_id} "
                        f"{catalog[body_id]} at {poses[env_id, body_id, 0:3].tolist()}"
                    )
            state["prev_done_ids"] = done_ids
    return None


def main():
    """Random actions agent with Isaac Lab environment."""

    torch.manual_seed(42)

    # Launch the simulation app FIRST and resolve the task config inside the launch
    # context: the LEAP env cfg import chain (isaaclab_newton/ovphysx presets) pulls
    # pxr, which corrupts Kit startup in this Isaac Sim build if imported before the
    # app exists. (The upstream script resolves before launching; that order only
    # works there because their pxr tolerates pre-app imports.)
    with launch_simulation(None, args_cli):
        import LEAP_Isaaclab.tasks  # noqa: F401

        # parse configuration via Hydra (supports preset selection, e.g. env.sim.physics=newton_mjwarp)
        env_cfg, _ = resolve_task_config(args_cli.task, "")

        # override with CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg)

        # print info (this is vectorized environment)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        # reset environment
        env.reset()

        # verification state (only used with --verify)
        unwrapped = env.unwrapped
        is_objects_env = hasattr(unwrapped, "object_catalog")
        state = {
            "catalog": [name for name, _ in unwrapped.object_catalog] if is_objects_env else None,
            "episode_returns": torch.zeros(unwrapped.num_envs, device=unwrapped.device),
            "episode_lengths": torch.zeros(unwrapped.num_envs, dtype=torch.int32, device=unwrapped.device),
            "completed_returns": [],
            "completed_lengths": [],
            "resample_total": 0,
            "resample_changed": 0,
            "activation_counts": torch.zeros(
                len(unwrapped.object_catalog) if is_objects_env else 1, dtype=torch.long
            ),
            "prev_active": unwrapped.active_object_ids.clone() if is_objects_env else None,
            "prev_done_ids": None,
            "done": None,
        }

        # simulate environment
        sim = env.unwrapped.sim
        step = 0
        while True:
            if sim.visualizers:
                # visualizer mode: run until the visualizer window is closed
                if not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break
            if args_cli.verify and step >= args_cli.num_steps:
                break
            # run everything in inference mode
            with torch.inference_mode():
                # sample actions from -1 to 1
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                # apply actions
                obs, reward, terminated, truncated, _ = env.step(actions)
                if args_cli.verify:
                    state["done"] = terminated | truncated
                    error = _verify_step(unwrapped, obs, reward, state)
                    assert error is None, f"[random-agent] verification failed at step {step}: {error}"
                step += 1

        if args_cli.verify:
            print("\n[random-agent] ===== verification summary =====")
            print(f"steps run: {step}, episodes completed: {len(state['completed_returns'])}")
            if state["completed_returns"]:
                t = torch.tensor(state["completed_returns"])
                l = torch.tensor(state["completed_lengths"])
                print(f"episode return: mean {t.mean():8.2f}  min {t.min():8.2f}  max {t.max():8.2f}")
                print(f"episode length: mean {l.float().mean():6.1f}  min {int(l.min()):4d}  max {int(l.max()):4d}")
            if state["catalog"] is not None:
                k = len(state["catalog"])
                print(
                    f"object resampled on {state['resample_changed']}/{state['resample_total']} episode resets "
                    f"({100 * state['resample_changed'] / max(state['resample_total'], 1):.0f}%, "
                    f"expect ~{100 * (1 - 1 / k):.0f}% for K={k})"
                )
                print("activation histogram:")
                for name, count in zip(state["catalog"], state["activation_counts"].tolist()):
                    bar = "#" * int(40 * count / max(max(state["activation_counts"].tolist()), 1))
                    print(f"  {name:>16}: {count:5d} {bar}")
            print("[random-agent] ALL CHECKS PASSED")

        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
