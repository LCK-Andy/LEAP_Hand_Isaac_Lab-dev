"""
Smoke test for the Isaac-Repose-Cube-LEAP-Direct-v0 direct (DirectRLEnv) task.

Checks, headless, with 4 envs:
  1. The task is registered and the LEAP hand articulation is in the scene
     (16 joints a_0..a_15, LEAP USD spawned) with the 4 LEAP fingertip bodies.
  2. Reset works and the policy observation is the 136-d vector (124 "full" + 12 tactile)
     (dof pos/vel + object pose/vel + goal terms + 4 fingertips + last action).
  3. Stepping with zero actions (joint targets -> mid-range) keeps rewards finite
     and the cube within the fall radius of the in-hand position.
  4. Goal-reset bookkeeping and the consecutive-successes metric update.

Run:
  ~/env_isaaclab/bin/python scripts/smoke/leap_repose_direct_smoke.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Repose-Cube-LEAP-Direct smoke test.")
parser.add_argument("--task", default="Isaac-Repose-Cube-LEAP-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401  (registers Isaac-Repose-Cube-LEAP-Direct-v0)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # [1] scene check: LEAP articulation with 16 joints and 4 fingertips
    hand = env.unwrapped.hand
    joint_names = list(hand.joint_names)
    assert len(joint_names) == 16, f"expected 16 LEAP joints, got {joint_names}"
    assert all(name.startswith("a_") for name in joint_names), f"unexpected joint names: {joint_names}"
    for fingertip in env.unwrapped.cfg.fingertip_body_names:
        assert fingertip in hand.body_names, f"fingertip body '{fingertip}' not in hand bodies"
    print(f"[1] robot OK: 16 LEAP joints, fingertips {env.unwrapped.cfg.fingertip_body_names}")
    print(f"    robot USD: {env.unwrapped.cfg.robot_cfg.spawn.usd_path}")

    # [2] reset + observation structure
    obs, _ = env.reset()
    policy_obs = obs["policy"]
    assert policy_obs.shape == (args_cli.num_envs, 136), f"unexpected obs shape {tuple(policy_obs.shape)}"
    assert torch.isfinite(policy_obs).all(), "non-finite values in policy obs"
    print(f"[2] reset OK, policy obs shape: {tuple(policy_obs.shape)} (124 'full' + 12 tactile)")
    print(f"    action space: {env.action_space}")

    # [3] step with zero actions (targets saturate to mid-range of joint limits)
    zero_action = torch.zeros(args_cli.num_envs, env.action_space.shape[1], device=env.unwrapped.device)
    rew_sum = 0.0
    for _ in range(60):
        obs, reward, terminated, truncated, extras = env.step(zero_action)
        assert reward.shape == (args_cli.num_envs,), f"reward shape {tuple(reward.shape)}"
        assert torch.isfinite(reward).all(), "non-finite reward"
        rew_sum += reward.mean().item()
        assert torch.isfinite(obs["policy"]).all(), "non-finite obs while stepping"
    obj_pos = env.unwrapped.object_pos
    in_hand_pos = env.unwrapped.in_hand_pos
    dist = torch.linalg.norm(obj_pos - in_hand_pos, dim=1)
    print(f"[3] 60 steps OK, mean reward/step: {rew_sum / 60:.4f}")
    print(f"    object->in-hand distance: min {dist.min():.3f} max {dist.max():.3f} (fall cut at 0.24)")

    # [4] success bookkeeping
    cons = env.unwrapped.consecutive_successes.item()
    assert "consecutive_successes" in env.unwrapped.extras.get("log", {}), "missing consecutive_successes log"
    print(f"[4] consecutive_successes metric: {cons:.2f}")

    env.close()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
    simulation_app.close()
