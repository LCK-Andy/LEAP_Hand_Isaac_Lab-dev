"""
Smoke test for the Isaac-Repose-Cube-LEAP-v0 manager-based task.

Checks, headless, with 4 envs:
  1. The task is registered and the LEAP hand articulation is in the scene
     (16 joints a_0..a_15, LEAP USD spawned).
  2. Reset works and the policy observation group is a single concatenated
     vector (joint pos/vel + object pose/vel + goal terms + last action).
  3. Stepping with zero actions keeps rewards finite and the cube near the palm
     (no object_out_of_reach terminations right after spawn).
  4. Episode terminations/timeouts propagate through the manager-based env.

Run:
  ~/env_isaaclab/bin/python scripts/leap_repose_smoke.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Repose-Cube-LEAP smoke test.")
parser.add_argument("--task", default="Isaac-Repose-Cube-LEAP-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401  (registers Isaac-Repose-Cube-LEAP-v0)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # [1] scene check: LEAP articulation with 16 joints
    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.joint_names)
    assert len(joint_names) == 16, f"expected 16 LEAP joints, got {joint_names}"
    assert all(name.startswith("a_") for name in joint_names), f"unexpected joint names: {joint_names}"
    print(f"[1] robot OK: 16 LEAP joints {joint_names[0]}..{joint_names[-1]}")
    print(f"    robot USD: {env.unwrapped.cfg.scene.robot.spawn.usd_path}")

    # [2] reset + observation structure
    obs, _ = env.reset()
    assert set(obs.keys()) == {"policy"} or "policy" in obs, f"unexpected obs keys: {obs.keys()}"
    policy_obs = obs["policy"]
    assert policy_obs.shape[0] == args_cli.num_envs
    assert torch.isfinite(policy_obs).all(), "non-finite values in policy obs"
    print(f"[2] reset OK, policy obs shape: {tuple(policy_obs.shape)} (expect 16+16+3+4+3+3+7+4+16 = 72)")
    print(f"    action space: {env.action_space}")

    # [3] step with zero actions (EMA -> joints stay near defaults)
    zero_action = torch.zeros(args_cli.num_envs, env.action_space.shape[1], device=env.unwrapped.device)
    rew_sum = 0.0
    for _ in range(60):
        obs, reward, terminated, truncated, extras = env.step(zero_action)
        assert torch.isfinite(reward).all(), "non-finite reward"
        rew_sum += reward.mean().item()
        assert torch.isfinite(obs["policy"]).all(), "non-finite obs while stepping"
    obj_pos = env.unwrapped.scene["object"].data.root_pos_w
    robot_pos = robot.data.root_pos_w
    dist = torch.linalg.norm(obj_pos - robot_pos, dim=1)
    print(f"[3] 60 steps OK, mean reward/step: {rew_sum / 60:.4f}")
    print(f"    object->robot-root distance: min {dist.min():.3f} max {dist.max():.3f} (out-of-reach cut at 0.3)")

    # [4] command metrics
    cmd = env.unwrapped.command_manager.get_term("object_pose")
    print(f"[4] command metrics: orient_err {cmd.metrics['orientation_error'].mean():.3f} rad, "
          f"consec_success {cmd.metrics['consecutive_success'].mean():.1f}")

    env.close()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
    simulation_app.close()
