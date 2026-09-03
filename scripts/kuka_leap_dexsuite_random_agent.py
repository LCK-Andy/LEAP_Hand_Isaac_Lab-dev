# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Random agent for the KUKA-LEAP DexSuite environments.

Same as IsaacLab's ``scripts/environments/random_agent.py`` but launches the
SimulationApp (Kit) BEFORE importing LEAP task modules:

* LEAP task files import ``isaaclab.sensors`` / ``isaaclab_physx`` modules that
  transitively import ``pxr`` (UsdPhysics / PhysxSchema). If those load before
  ``SimulationApp`` starts, the PhysX extension python bindings fail with
  ``PhysxSchema.Tokens`` / ``GfVec3f converter`` errors at Kit startup.
* ``resolve_task_config`` needs the Gymnasium registry entry, so the task
  registration (``import LEAP_Isaaclab.tasks``) must happen after Kit is up but
  before config resolution.

Usage:
    /home/andy/env_isaaclab/bin/python scripts/kuka_leap_dexsuite_random_agent.py \
        --task Isaac-Dexsuite-Kuka-LEAP-Reorient-v0 --num_envs 1 --viz Kit
"""

import argparse
import contextlib
import sys

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, resolve_task_config, setup_preset_cli

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
add_launcher_args(parser)
# simple agents should open Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main():
    """Random actions agent with Isaac Lab environment."""
    torch.manual_seed(42)

    # When --headless is given (or no display), Kit visualizer cannot start;
    # strip it from the launcher args so AppLauncher does not try to configure.
    visualizer = list(args_cli.visualizer or [])
    if args_cli.headless:
        visualizer = []

    # Launch Kit BEFORE importing the task package (see module docstring).
    args_cli.visualizer = visualizer
    app_launcher = AppLauncher(args_cli)
    sim_app = app_launcher.app

    # Task registration: must be AFTER AppLauncher (pxr import safety) and
    # BEFORE resolve/gym.make (registry lookup).
    import LEAP_Isaaclab.tasks  # noqa: F401  <-- registers Isaac-Dexsuite-Kuka-LEAP-*

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
    # simulate environment
    sim = env.unwrapped.sim
    num_steps = 0
    max_steps = args_cli.num_envs * 200 if args_cli.num_envs else 200
    while True:
        if sim.visualizers:
            # visualizer mode: run until the visualizer window is closed
            if not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                break
        elif num_steps >= max_steps:
            # headless mode: run a bounded number of steps then exit
            print(f"[INFO]: headless mode: reached {num_steps} steps, exiting.")
            break
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            env.step(actions)
        num_steps += 1

    # close the simulator
    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
