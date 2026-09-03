"""
Smoke test for the Franka+LEAP environment (Isaac-Franka-Leap).

Checks, headless with 2 envs:
  1. The articulation has 23 DOF (Franka 7 + LEAP 16) with expected joint names.
  2. The hand follows the flange when the arm moves (weld integrity).
  3. Hand joints are independently controllable (fingers close) while the arm stays.

Run: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
     __NV_PRIME_RENDER_OFFLOAD=1 \
     ~/env_isaaclab/bin/python scripts/franka_leap_smoke.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka+LEAP environment smoke test.")
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

import LEAP_Isaaclab.tasks  # noqa: F401  (registers Isaac-Franka-Leap)

N_ARM = 7
N_HAND = 16


def main():
    env_cfg = parse_env_cfg("Isaac-Franka-Leap", device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make("Isaac-Franka-Leap", cfg=env_cfg, render_mode=None)
    robot = env.unwrapped.robot
    print(f"\n[1] articulation: {robot.num_joints} DOF, {robot.num_bodies} bodies")
    print("    joint names:", robot.joint_names)
    print("    body names:", robot.body_names)
    assert robot.num_joints == N_ARM + N_HAND, "expected 23 DOF"
    for name in ["panda_joint1", "panda_joint7", "a_0", "a_15"]:
        assert name in robot.joint_names, f"missing joint {name}"
    flange_idx = robot.body_names.index("panda_link7")  # importer fuses link8(base+palm) into the wrist link7
    print(
        f"    flange proxy body: panda_link7 (idx {flange_idx}); "
        f"palm_lower idx {robot.body_names.index('palm_lower')}"
    )

    obs, _ = env.reset()
    print(f"\n[2] reset OK, obs shape: {obs['policy'].shape}")

    def fingertips():
        return env.unwrapped.fingertip_positions_world()

    def flange_pos():
        return robot.data.body_pos_w.torch[:, flange_idx]

    def hand_joint_pos():
        idx = [robot.joint_names.index(f"a_{i}") for i in range(16)]
        return robot.data.joint_pos.torch[:, idx]

    def arm_joint_pos():
        idx = [robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
        return robot.data.joint_pos.torch[:, idx]

    # ---------- Test A: arm moves, hand must follow (weld) ----------
    tip0 = fingertips().clone()
    fl0 = flange_pos().clone()
    arm0 = arm_joint_pos().clone()
    hands0 = hand_joint_pos().clone()

    actions = torch.zeros((args_cli.num_envs, env.unwrapped.cfg.action_space))
    actions[:, :N_ARM] = 0.5  # move the arm
    for _ in range(100):
        env.step(actions)

    tip1 = fingertips()
    fl1 = flange_pos()
    arm1 = arm_joint_pos()
    hands1 = hand_joint_pos()

    flange_moved = (fl1 - fl0).norm(dim=-1).mean()
    tip_moved = (tip1 - tip0).norm(dim=-1).mean()
    hand_drift = (hands1 - hands0).abs().max()
    arm_moved = (arm1 - arm0).abs().max()
    print(f"\n[3] arm move: flange moved {flange_moved:.3f} m, arm joints delta {arm_moved:.3f} rad")
    print(f"    fingertips followed: {tip_moved:.3f} m (must be > 0.05)")
    print(f"    hand joints drift (uncommanded): {hand_drift:.4f} rad (must be < 0.15)")
    assert flange_moved > 0.05 and tip_moved > 0.05, "hand did not follow the arm"
    assert hand_drift < 0.15, "hand joints drifted while uncommanded"

    # ---------- Test B: fingers close, arm stays ----------
    arm1_before = arm_joint_pos().clone()
    actions = torch.zeros((args_cli.num_envs, env.unwrapped.cfg.action_space))
    actions[:, N_ARM:] = 1.0  # close all fingers (relative target +)
    for _ in range(100):
        env.step(actions)

    hands2 = hand_joint_pos()
    arm2 = arm_joint_pos()
    hand_moved = (hands2 - hands1).abs().mean()
    arm_moved2 = (arm2 - arm1_before).abs().max()
    print(f"\n[4] hand control: hand joints delta {hand_moved:.3f} rad (must be > 0.05)")
    print(f"    arm drift during hand close: {arm_moved2:.4f} rad (must be < 0.15)")
    assert hand_moved > 0.05, "hand joints did not respond to actions"
    assert arm_moved2 < 0.15, "arm drifted while uncommanded"

    # ---------- Test C: fingertip-flange rigid distance ----------
    d0 = (tip0 - fl0[:, None, :]).norm(dim=-1)
    d1 = (tip1 - fl1[:, None, :]).norm(dim=-1)
    print(f"\n[5] weld rigid check: max |dist(tip,flange) delta| = {(d1 - d0).abs().max():.5f} m (must be < 5e-3)")
    assert (d1 - d0).abs().max() < 5e-3, "fingertip-flange distance changed: weld broken"

    env.close()
    print("\nSMOKE TEST PASSED ✅")


if __name__ == "__main__":
    main()
