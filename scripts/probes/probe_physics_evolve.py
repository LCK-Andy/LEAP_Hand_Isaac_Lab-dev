"""Probe: is physics stepping? step-count, obj velocity."""
import argparse
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
args_cli = parser.parse_known_args()[0]

AppLauncher(launcher_args=args_cli)

import gymnasium as gym
import LEAP_Isaaclab.tasks
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg("Isaac-Dexsuite-Kuka-LEAP-Reorient-v0", device="cuda:0")
env_cfg.scene.num_envs = 1
env_cfg.sim.use_fabric = True
env = gym.make("Isaac-Dexsuite-Kuka-LEAP-Reorient-v0", cfg=env_cfg)
env = env.unwrapped

obs, info = env.reset()
obj = env.scene["object"]
robot = env.scene["robot"]

print("[probe] step_count after reset:", env.sim._physics_step_count)
print("[probe] obj pos:", obj.data.root_pos_w[0].cpu().numpy().round(4),
      "vel:", obj.data.root_lin_vel_w[0].cpu().numpy().round(4))
print("[probe] robot vel:", robot.data.root_lin_vel_w[0].cpu().numpy().round(4))
print("[probe] obj rigid state:", obj.is_initialized, type(obj).__name__)
print("[probe] obj data device:", obj.data.root_pos_w.device)

for i in range(30):
    with torch.inference_mode():
        actions = 2 * torch.rand(env.action_space.shape, device=env.device) - 1
        env.step(actions)
    if i in (4, 29):
        print(f"[probe] step {i+1}: count={env.sim._physics_step_count} "
              f"obj_vel={obj.data.root_lin_vel_w[0].cpu().numpy().round(4)} "
              f"obj_pos={obj.data.root_pos_w[0].cpu().numpy().round(4)}")
print("[probe] done")
