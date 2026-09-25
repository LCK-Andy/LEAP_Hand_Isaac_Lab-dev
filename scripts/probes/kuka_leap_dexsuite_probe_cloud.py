"""Probe: points_w (what the visualization draws) vs object root.
Correct math: points_w = quat_apply(object_quat_w, points_local) + object_pos_w
"""
import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
args_cli = parser.parse_known_args()[0]

AppLauncher(launcher_args=args_cli)

import gymnasium as gym
import LEAP_Isaaclab.tasks
from LEAP_Isaaclab.tasks.kuka_leap_dexsuite.dexsuite_kuka_leap_env_cfg import DexsuiteKukaLeapReorientEnvCfg
from LEAP_Isaaclab.compat.dexsuite.mdp.utils import sample_object_point_cloud
from isaaclab_tasks.utils import resolve_task_config

env_cfg, _ = resolve_task_config("Isaac-Dexsuite-Kuka-LEAP-Reorient-v0", "")
env_cfg.scene.num_envs = 1  # probe only needs one env for the FK/cloud alignment check
env = gym.make("Isaac-Dexsuite-Kuka-LEAP-Reorient-v0", cfg=env_cfg)
env = env.unwrapped

obs, info = env.reset()

obj = env.scene["object"]
print("[probe] object name:", obj.__class__.__name__)
print("[probe] object cfg prim_path:", obj.cfg.prim_path)

# sample points exactly like the obs term does (at reset time, object at spawn pose)
points_local = sample_object_point_cloud(env.num_envs, 64, obj.cfg.prim_path, device=env.device)
print("[probe] points_local shape:", tuple(points_local.shape))
print("[probe] points_local min:", points_local.min().item(), "max:", points_local.max().item())
print("[probe] points_local mean:", points_local.mean(dim=1).cpu().numpy().round(4))

# now what the term computes each call:
from isaaclab.utils.math import quat_apply
object_pos_w = obj.data.root_pos_w.torch.unsqueeze(1).repeat(1, 64, 1)
object_quat_w = obj.data.root_quat_w.torch.unsqueeze(1).repeat(1, 64, 1)
points_w = quat_apply(object_quat_w, points_local) + object_pos_w
print("[probe] points_w mean:", points_w.mean(dim=1).cpu().numpy().round(4))
print("[probe] obj root pos:", obj.data.root_pos_w.cpu().numpy().round(4))
print("[probe] obj root quat:", obj.data.root_quat_w.cpu().numpy().round(4))
d = (points_w.mean(dim=1) - obj.data.root_pos_w).norm(dim=1)
print("[probe] |points_w.mean - obj.pos|:", d.cpu().numpy().round(4))

# the red/cyan cluster in the image is likely the goal_pose command marker.
# Its world pose is (pose_command_w), which stays fixed per episode unless resampled.
for cname in dir(env.command_manager):
    pass
cm = env.command_manager
term_info = {n: t for n in getattr(cm, "terms", [])}
try:
    t = cm.get_term("object_pose")
    cmd = t.command
    print("[probe] command term found:", type(t).__name__)
    print("[probe] pose_command_w:", t.pose_command_w.cpu().numpy().round(4))
    print("[probe] object pos   :", obj.data.root_pos_w.cpu().numpy().round(4))
    print("[probe] |goal - obj |:", (t.pose_command_w[:, :3] - obj.data.root_pos_w).norm(dim=-1).cpu().numpy().round(4))
except Exception as e:
    print("[probe] command access failed:", e)

# check the sampled points are rigid with object (fixed offset)?
# world pos of a local point p: obj_pos + R @ p. points_local are already local, so
# the mean of points_local should be ~0 (surface points around centroid).
# If points_local has a big offset, the sampling used a different world frame.
