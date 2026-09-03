"""
Short rendering demo of the Franka+LEAP environment (for visual verification).

The arm sweeps while the fingers slowly close and open. Records an MP4 via
gymnasium RecordVideo (headless, offscreen render).

Run: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
     __NV_PRIME_RENDER_OFFLOAD=1 \
     ~/env_isaaclab/bin/python scripts/franka_leap_demo.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka+LEAP demo video.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401

N_ARM = 7
N_HAND = 16
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "videos", "franka_leap_demo")


def main():
    env_cfg = parse_env_cfg("Isaac-Franka-Leap", device=args_cli.device, num_envs=args_cli.num_envs)
    # close-up 3/4 view of the arm + hand
    env_cfg.viewer.eye = (0.9, -1.4, 1.0)
    env_cfg.viewer.lookat = (-0.05, -0.55, 0.45)
    env_cfg.viewer.resolution = (1280, 720)
    env = gym.make("Isaac-Franka-Leap", cfg=env_cfg, render_mode="rgb_array")
    video_kwargs = {
        "video_folder": OUT_DIR,
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.steps,
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    obs, _ = env.reset()
    for t in range(args_cli.steps):
        # arm: slow sweep; hand: sinusoidal close/open (all fingers in sync)
        import math

        tt = float(t)
        if tt < 90.0:  # hold the ready pose (camera check frame at t~1s)
            actions = torch.zeros((env_cfg.scene.num_envs, env_cfg.action_space))
            env.step(actions)
            continue
        arm = 0.15 * torch.tensor(
            [
                0.0,
                -0.3 + 0.3 * math.cos(0.03 * tt + 1.0),
                0.0,
                -0.3 + 0.25 * math.cos(0.03 * tt),
                0.0,
                0.3 * math.cos(0.02 * tt),
                0.0,
            ]
        ).unsqueeze(0)
        hand = (0.8 * math.sin(0.04 * tt)) * torch.ones(1, N_HAND)
        actions = torch.cat([arm, hand], dim=-1).repeat(env_cfg.scene.num_envs, 1)
        env.step(actions)

    env.close()
    print(f"demo video written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
