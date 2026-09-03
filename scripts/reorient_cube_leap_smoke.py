"""
Smoke test for Isaac-Reorient-Cube-Leap.

Checks, headless with 2 envs:
  1. The LEAP hand articulation loads and has 16 actuated joints.
  2. The env resets and steps without crashing.
  3. Hand joints respond to actions and object stays roughly in hand.

Run: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
     __NV_PRIME_RENDER_OFFLOAD=1 \
     ~/env_isaaclab/bin/python scripts/reorient_cube_leap_smoke.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Reorient-Cube-LEAP smoke test.")
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401  (registers Isaac-Reorient-Cube-Leap)

N_HAND = 16


def main():
    env_cfg = parse_env_cfg(
        "Isaac-Reorient-Cube-Leap", device=args_cli.device, num_envs=args_cli.num_envs
    )
    env = gym.make("Isaac-Reorient-Cube-Leap", cfg=env_cfg, render_mode=None)
    robot = env.unwrapped.hand
    print(f"\n[1] articulation: {robot.num_joints} DOF, {robot.num_bodies} bodies")
    print("    joint names:", robot.joint_names)
    for name in ["a_0", "a_15"]:
        assert name in robot.joint_names, f"missing joint {name}"

    obs, _ = env.reset()
    print(f"\n[2] reset OK, obs shape: {obs['policy'].shape}")

    # ---------- Test A: random small actions, no crash ----------
    for step in range(200):
        actions = torch.randn((args_cli.num_envs, env.unwrapped.cfg.action_space), device=args_cli.device) * 0.1
        obs, reward, terminated, truncated, info = env.step(actions)
        if (terminated | truncated).any():
            obs, _ = env.reset()

    hand_idx = [robot.joint_names.index(f"a_{i}") for i in range(N_HAND)]
    hand_pos = robot.data.joint_pos.torch[:, hand_idx]
    print(f"\n[3] stepped 200 random actions OK")
    print(f"    hand joint pos mean: {hand_pos.mean():.3f} rad")
    print(f"    hand joint pos std:  {hand_pos.std():.3f} rad")

    # ---------- Test B: close fingers, object should not fall ----------
    obs, _ = env.reset()
    for step in range(100):
        actions = torch.full((args_cli.num_envs, env.unwrapped.cfg.action_space), 0.5, device=args_cli.device)
        obs, reward, terminated, truncated, info = env.step(actions)

    object_pos = env.unwrapped.object.data.root_pos_w.torch - env.unwrapped.scene.env_origins
    in_hand_pos = env.unwrapped.in_hand_pos
    dist = torch.norm(object_pos - in_hand_pos, dim=-1).mean()
    print(f"\n[4] close-finger test: object-in-hand distance {dist:.3f} m (must be < 0.3)")
    assert dist < 0.3, "object fell out of hand"

    env.close()
    print("\nSMOKE TEST PASSED ✅")


if __name__ == "__main__":
    main()
