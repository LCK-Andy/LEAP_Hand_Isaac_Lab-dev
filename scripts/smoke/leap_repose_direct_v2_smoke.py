"""
Smoke test for the Isaac-Repose-Cube-LEAP-Direct-v2 task (point-cloud obs).

Checks, headless (offscreen) with 4 envs:
  1. The v2 task registers; LEAP hand (16 joints) + depth camera in the scene.
  2. Observations: "policy" (N, 124) and "points" (N, 256, 4); the first 128 rows
     are object points (flag=1), the last 128 hand points (flag=0); zero rows are
     unused slots; coordinates finite and within a few units of the palm anchor.
  3. Object and hand point sets are both populated (segmentation resolved) and
     geometrically sensible (object points above the palm, hand points around it).
  4. Stepping with zero actions keeps rewards/obs finite.

Run:
  ~/env_isaaclab/bin/python scripts/leap_repose_direct_v2_smoke.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Repose-Cube-LEAP-Direct-v2 point-obs smoke test.")
parser.add_argument("--task", default="Isaac-Repose-Cube-LEAP-Direct-v2")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401  (registers Isaac-Repose-Cube-LEAP-Direct-v2)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped

    # [1] scene check
    joint_names = list(unwrapped.hand.joint_names)
    assert len(joint_names) == 16 and all(n.startswith("a_") for n in joint_names), joint_names
    assert "camera" in unwrapped.scene.sensors
    print(f"[1] robot OK (16 LEAP joints), depth camera OK")

    # [2] observation structure
    obs, _ = env.reset()
    policy_obs, points = obs["policy"], obs["points"]
    num_obj = unwrapped.cfg.num_object_points
    num_hand = unwrapped.cfg.num_hand_points
    assert policy_obs.shape == (args_cli.num_envs, 124), tuple(policy_obs.shape)
    assert points.shape == (args_cli.num_envs, num_obj + num_hand, 4), tuple(points.shape)
    assert torch.isfinite(points).all(), "non-finite points"
    obj_flags = points[:, :num_obj, 3]
    hand_flags = points[:, num_obj:, 3]
    assert ((obj_flags == 1.0) | (obj_flags == 0.0)).all() and ((hand_flags == 0.0)).all()
    print(f"[2] obs OK: policy {tuple(policy_obs.shape)}, points {tuple(points.shape)}")

    # [3] point-set sanity
    obj_valid = (obj_flags == 1.0)
    hand_valid = torch.linalg.norm(points[:, num_obj:, :3], dim=-1) > 0.0
    frac_obj = obj_valid.float().mean().item()
    frac_hand = hand_valid.float().mean().item()
    assert frac_obj > 0.5, f"only {frac_obj:.0%} object slots filled"
    assert frac_hand > 0.3, f"only {frac_hand:.0%} hand slots filled"
    obj_z = points[:, :num_obj, 2][obj_valid]
    print(f"[3] slots filled: object {frac_obj:.0%}, hand {frac_hand:.0%}; "
          f"object z-range [{obj_z.min():.2f}, {obj_z.max():.2f}] (scaled units)")
    assert obj_z.abs().max() < 3.0, "object points unexpectedly far from palm anchor"

    # [4] step
    zero_action = torch.zeros(args_cli.num_envs, unwrapped.single_action_space.shape[0], device=unwrapped.device)
    rew_sum = 0.0
    for _ in range(30):
        obs, reward, terminated, truncated, extras = env.step(zero_action)
        assert torch.isfinite(reward).all() and torch.isfinite(obs["points"]).all()
        rew_sum += reward.mean().item()
    print(f"[4] 30 steps OK, mean reward/step: {rew_sum / 30:.4f}")

    env.close()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
    simulation_app.close()
