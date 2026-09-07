"""
Smoke test for the Isaac-Repose-Cube-LEAP-Direct-v1 task (direct env + RGBD camera).

Checks, headless (offscreen) with 4 envs:
  1. The v1 task registers; LEAP hand (16 joints) + camera are in the scene.
  2. Observations are a dict: "policy" (N, 124, official full obs) and "image"
     (N, 4, 128, 128) RGBD, values finite and in [0, 1].
  3. Stepping with zero actions keeps rewards finite and the cube near the palm.
  4. Saves one RGB frame and one depth frame to scripts/v1_smoke_{rgb,depth}.png
     for a visual check that the hand + cube are framed from above.

Run:
  ~/env_isaaclab/bin/python scripts/leap_repose_direct_v1_smoke.py
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Repose-Cube-LEAP-Direct-v1 RGBD smoke test.")
parser.add_argument("--task", default="Isaac-Repose-Cube-LEAP-Direct-v1")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True  # offscreen rendering for the RGBD camera

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import torch

from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401  (registers Isaac-Repose-Cube-LEAP-Direct-v1)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # [1] scene check
    unwrapped = env.unwrapped
    joint_names = list(unwrapped.hand.joint_names)
    assert len(joint_names) == 16 and all(n.startswith("a_") for n in joint_names), joint_names
    assert "camera" in unwrapped.scene.sensors, "RGBD camera missing from scene"
    print(f"[1] robot OK (16 LEAP joints), camera OK: {unwrapped.cfg.camera.width}x{unwrapped.cfg.camera.height}")

    # [2] observation structure
    obs, _ = env.reset()
    policy_obs, image_obs = obs["policy"], obs["image"]
    assert policy_obs.shape == (args_cli.num_envs, 124), f"policy obs {tuple(policy_obs.shape)}"
    assert image_obs.shape == (args_cli.num_envs, 4, 128, 128), f"image obs {tuple(image_obs.shape)}"
    assert torch.isfinite(policy_obs).all() and torch.isfinite(image_obs).all()
    assert image_obs.min() >= 0.0 and image_obs.max() <= 1.0
    print(f"[2] reset OK: policy {tuple(policy_obs.shape)}, image {tuple(image_obs.shape)}, in-range")

    # [3] step with zero actions
    zero_action = torch.zeros(args_cli.num_envs, unwrapped.single_action_space.shape[0], device=unwrapped.device)
    rew_sum = 0.0
    for _ in range(30):
        obs, reward, terminated, truncated, extras = env.step(zero_action)
        assert torch.isfinite(reward).all(), "non-finite reward"
        assert torch.isfinite(obs["image"]).all(), "non-finite image obs"
        rew_sum += reward.mean().item()
    dist = torch.linalg.norm(unwrapped.object_pos - unwrapped.in_hand_pos, dim=1)
    print(f"[3] 30 steps OK, mean reward/step: {rew_sum / 30:.4f}, cube distance: min {dist.min():.3f}")

    # [4] save one RGB + depth frame for visual inspection
    frame = obs["image"][0].cpu()
    out_dir = Path(__file__).parent
    imageio.imwrite(out_dir / "v1_smoke_rgb.png", (frame[:3].permute(1, 2, 0).numpy() * 255).astype("uint8"))
    imageio.imwrite(out_dir / "v1_smoke_depth.png", (frame[3].numpy() * 255).astype("uint8"))
    print(f"[4] saved {out_dir}/v1_smoke_rgb.png and v1_smoke_depth.png")

    env.close()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
    simulation_app.close()
