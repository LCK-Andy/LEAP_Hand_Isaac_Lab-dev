"""
Smoke test for LEAP reorientation RGBD tasks.

Checks, headless (offscreen) with 2 envs:
  1. Env reset/step works with the RGBD camera in the scene.
  2. Observations are a dict: "policy" (N, 96) + "image" (N, 4, H, W), values in range.
  3. Object spawn pose and all fingertips project inside the camera frame.
  4. The rsl-rl wrapper turns the obs dict into a TensorDict with the same keys.
  5. One full PPO learning iteration (CNN model build + rollout + update) runs.
  6. Saves one RGB frame and one depth frame to scripts/rgbd_smoke_{rgb,depth}.png
     for a visual check that the hand + cube are in view.

Run the D435-like variant:
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  __NV_PRIME_RENDER_OFFLOAD=1 \
  ~/env_isaaclab/bin/python scripts/reorient_cube_leap_rgbd_smoke.py \
    --task Isaac-Reorient-Cube-Leap-Realsense-D435
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Reorient-Cube-LEAP RGBD smoke test.")
parser.add_argument(
    "--task",
    default="Isaac-Reorient-Cube-Leap-Realsense-D435",
    choices=["Isaac-Reorient-Cube-Leap-RGBD", "Isaac-Reorient-Cube-Leap-Realsense-D435"],
)
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True  # offscreen rendering for the RGBD camera

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.sensors import save_images_to_file
from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401  (registers Isaac-Reorient-Cube-Leap-RGBD)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    obs, _ = env.reset()
    assert set(obs.keys()) == {"policy", "image"}, f"unexpected obs keys: {obs.keys()}"
    print(f"[1] reset OK, obs keys: {list(obs.keys())}")
    print(f"    policy shape: {tuple(obs['policy'].shape)}, image shape: {tuple(obs['image'].shape)}")

    if args_cli.task == "Isaac-Reorient-Cube-Leap-Realsense-D435":
        from LEAP_Isaaclab.tasks.leap_hand_reorient.leap_hand_realsense_d435_env_cfg import (
            D435_INTRINSICS_320X240,
        )

        expected_k = torch.tensor(D435_INTRINSICS_320X240, device=args_cli.device).reshape(3, 3)
        actual_k = env.unwrapped.scene.sensors["camera"].data.intrinsic_matrices[0]
        torch.testing.assert_close(actual_k, expected_k, rtol=1e-4, atol=1e-4)
        print(f"[1a] D435 intrinsics OK: {actual_k.tolist()}")

    # framing check at reset: project the spawn pose of the object and the
    # default-pose fingertips into the camera image. computed directly from the
    # cfg camera pose (world convention) — no camera.data needed.
    from isaaclab.utils.math import quat_rotate

    cam_cfg = env.unwrapped.cfg.camera
    # camera prim lives inside each env, so its world pose includes the env origin
    cam_offset_pos = torch.tensor(cam_cfg.offset.pos, dtype=torch.float, device=args_cli.device)
    cam_pos = cam_offset_pos + env.unwrapped.scene.env_origins
    cam_quat = torch.tensor(cam_cfg.offset.rot, dtype=torch.float, device=args_cli.device)  # xyzw
    forward_w = quat_rotate(cam_quat, torch.tensor([1.0, 0.0, 0.0], device=args_cli.device))  # world conv: +X
    up_w = quat_rotate(cam_quat, torch.tensor([0.0, 0.0, 1.0], device=args_cli.device))  # world conv: +Z
    right_w = torch.cross(up_w, forward_w, dim=0)
    f_px = cam_cfg.width * cam_cfg.spawn.focal_length / cam_cfg.spawn.horizontal_aperture

    def project(points_w, cam_pos):
        p = points_w - cam_pos  # (N, 3)
        z = (p * forward_w).sum(-1)
        x = (p * right_w).sum(-1)
        y = (p * up_w).sum(-1)
        return cam_cfg.width / 2 + f_px * x / z, cam_cfg.height / 2 - f_px * y / z

    h, w = cam_cfg.height, cam_cfg.width
    cam_pos0 = cam_pos[0:1]  # camera of env 0 only
    obj_w = env.unwrapped.in_hand_pos[0:1] + env.unwrapped.scene.env_origins[0:1]
    u, v = project(obj_w, cam_pos0)
    print(f"[1b] object spawn pixel coords: u={u[0].item():.1f}, v={v[0].item():.1f} (frame {w}x{h})")
    assert 0 < u[0].item() < w and 0 < v[0].item() < h, "object spawn pose not in camera frame!"
    tips_w = env.unwrapped.fingertip_pos[0] + env.unwrapped.scene.env_origins[0:1]  # (4, 3) env 0
    u_t, v_t = project(tips_w, cam_pos0)
    for i in range(tips_w.shape[0]):
        print(f"    fingertip {i} pixel coords: u={u_t[i].item():.1f}, v={v_t[i].item():.1f}")
        assert 0 < u_t[i].item() < w and 0 < v_t[i].item() < h, f"fingertip {i} not in camera frame!"

    # step a few times so the camera renders frames
    for step in range(5):
        actions = torch.randn((args_cli.num_envs, env.unwrapped.cfg.action_space), device=args_cli.device) * 0.1
        obs, reward, terminated, truncated, info = env.step(actions)

    image_h, image_w = env.unwrapped.cfg.camera.height, env.unwrapped.cfg.camera.width
    assert obs["policy"].shape == (args_cli.num_envs, 96)
    assert obs["image"].shape == (args_cli.num_envs, 4, image_h, image_w)
    assert obs["image"].min() >= 0.0 and obs["image"].max() <= 1.0, "image not normalized to [0, 1]"
    assert obs["image"][:, :3].max() > 0.1, "rgb channels look empty (black frame)"
    assert obs["image"][:, 3].max() < 1.0, "depth is clipping at the clip distance"
    print(f"[2] obs OK: policy {tuple(obs['policy'].shape)}, image {tuple(obs['image'].shape)}, "
          f"image range [{obs['image'].min():.3f}, {obs['image'].max():.3f}]")

    # save frames for visual inspection (save_images_to_file expects (N, H, W, C))
    save_images_to_file(obs["image"][:, :3].permute(0, 2, 3, 1), "scripts/rgbd_smoke_rgb.png")
    save_images_to_file(obs["image"][:, 3:4].permute(0, 2, 3, 1), "scripts/rgbd_smoke_depth.png")
    print("[3] saved scripts/rgbd_smoke_rgb.png and scripts/rgbd_smoke_depth.png")

    # rsl-rl wrapper: obs dict becomes a TensorDict with the same group keys
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env_rl = RslRlVecEnvWrapper(env)
    obs_td, _, _, _ = env_rl.step(
        torch.randn((args_cli.num_envs, env.unwrapped.cfg.action_space), device=args_cli.device)
    )
    assert set(obs_td.keys()) == {"policy", "image"}
    assert tuple(obs_td["policy"].shape) == (args_cli.num_envs, 96)
    assert tuple(obs_td["image"].shape) == (args_cli.num_envs, 4, image_h, image_w)
    print(f"[4] rsl-rl wrapper OK: obs keys {list(obs_td.keys())}")

    # one full PPO learning iteration: builds the CNN model, runs rollouts, updates
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.num_steps_per_env = 32
    agent_cfg.max_iterations = 1
    agent_cfg.save_interval = 10000
    runner = OnPolicyRunner(env_rl, agent_cfg.to_dict(), log_dir="logs/rsl_rl/rgbd_smoke", device=args_cli.device)
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
    print("[5] one PPO iteration (CNN rollout + update) OK")

    env.close()
    print("\nSMOKE TEST PASSED ✅  (check the saved pngs: hand + cube should be centered)")


if __name__ == "__main__":
    main()
