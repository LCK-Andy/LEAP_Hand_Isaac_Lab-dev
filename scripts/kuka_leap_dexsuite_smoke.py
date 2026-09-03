"""Headless smoke test: spawn Isaac-Dexsuite-Kuka-LEAP-Reorient-v0, run a few steps.

Verifies: registration, URDF->USD conversion, articulation spawn, fingertip
contact sensors, observations dims, and 20 env steps. Use --num_envs small.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--device", type=str, default="cuda:0")
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(parser.parse_args(), headless=True)
sim_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401  (registers all tasks)
import LEAP_Isaaclab.tasks  # noqa: F401  (registers LEAP tasks incl. kuka_leap_dexsuite)

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# resolve the gym id
gym_id = "Isaac-Dexsuite-Kuka-LEAP-Reorient-v0"
print(f"== gym_id: {gym_id}")
import torch

env_cfg = parse_env_cfg(gym_id, device=args.device, num_envs=args.num_envs)
env = gym.make(gym_id, cfg=env_cfg)
env = env.unwrapped  # ManagerBasedRLEnv (drop gym wrappers)

obs, _ = env.reset()
print("== reset done; obs keys:", list(obs.keys()))
for k, v in obs.items():
    print(f"   {k}: {tuple(v.shape)}")

robot = env.scene["robot"]
print("== robot body count:", robot.num_bodies)
print("== robot joint count:", robot.num_joints)
print("== robot joint names:", robot.joint_names)
print("== robot body names (first 5):", robot.body_names[:5])

# fingertip sensors must exist
for name, sensor in env.scene.sensors.items():
    if name.endswith("_object_s"):
        print(f"== sensor {name}: count={sensor.num_sensors}")

# run some steps with a zero action
n_steps = args.steps
for i in range(n_steps):
    action = torch.zeros_like(env.action_manager.action)
    obs, rew, terminated, truncated, info = env.step(action)
    if (i + 1) % 5 == 0:
        print(
            f"   step {i+1}/{n_steps} rew[0]={rew[0].item():.3f} "
            f"term={terminated[0].item()} trunc={truncated[0].item()}"
        )

print("== SMOKE OK")
sim_app.close()
