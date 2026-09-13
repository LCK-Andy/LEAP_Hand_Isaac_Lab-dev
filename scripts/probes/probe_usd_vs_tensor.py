"""Probe: usd-level object pose vs physx tensor pose, GUI mode (fabric off)."""
import argparse
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
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

import isaaclab.sim as sim_utils
stage = sim_utils.SimulationContext.instance().stage
object_prim = stage.GetPrimAtPath("/World/envs/env_0/Object")
from pxr import UsdGeom
xf = UsdGeom.XformCache()
print("[probe] USD object world:", np.array(xf.GetLocalToWorldTransform(object_prim).ExtractTranslation()).round(4))
print("[probe] physx tensor obj:  ", obj.data.root_pos_w[0].cpu().numpy().round(4))

import time
for i in range(60):
    with torch.inference_mode():
        actions = 2 * torch.rand(env.action_space.shape, device=env.device) - 1
        env.step(actions)
    if i % 20 == 19:
        usd = np.array(xf.GetLocalToWorldTransform(object_prim).ExtractTranslation()).round(4)
        ten = obj.data.root_pos_w[0].cpu().numpy().round(4)
        print(f"[probe] step {i+1}: usd={usd} tensor={ten}")

import sys
print("[probe] done")
