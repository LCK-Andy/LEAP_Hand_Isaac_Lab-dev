"""Probe the bodex-converted LEAP ghost in Isaac:
  - Articulation joint_names order
  - root link name
  - after writing (grasp_root, hand_q) with the curobo convention,
    fingertip BODY positions vs curobo fk_tips (world).
This tells us the ghost-side joint order + root frame, so the DGN2 plan can be
encoded in the ghost's conventions."""
import argparse
import sys

import numpy as np
import torch

import isaaclab_tasks  # noqa: F401 (hydra)
from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

from isaaclab.app import AppLauncher
app = AppLauncher(exp_name="probe_ghost_dgn2")
sim_app = app.app

import isaaclab.sim as sim_utils
import LEAP_Isaaclab.tasks  # noqa
import gymnasium as gym
import LEAP_Isaaclab
from LEAP_Isaaclab.assets import LEAP_HAND_CFG
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_tasks.utils import resolve_task_config
from isaaclab.utils.math import convert_quat

env_cfg, _ = resolve_task_config("Isaac-Dexsuite-Kuka-LEAP-Reorient-v0", "")
env_cfg.scene.num_envs = 1

_bodex_cfg = LEAP_HAND_CFG.replace(
    spawn=LEAP_HAND_CFG.spawn.replace(
        usd_path=("/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab/source/"
                  "LEAP_Isaaclab/LEAP_Isaaclab/assets/leap_bodex_usd2/leap_right.usda"),
    ),
)
ghost_cfg = _bodex_cfg.replace(
    prim_path="/World/envs/env_0/BODex_phantom_probe",
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5),
                                               joint_pos={"tn__.*": 0.0}),
)
ghost_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
    collision_enabled=False, contact_offset=0.001, rest_offset=0.0)
ghost_cfg.spawn.articulation_props.fix_root_link = False
env_cfg.scene.bodex_phantom_probe = ghost_cfg

env = gym.make("Isaac-Dexsuite-Kuka-LEAP-Reorient-v0", cfg=env_cfg)
env.reset()
ghost = env.unwrapped.scene["bodex_phantom_probe"]
print("ghost joint_names:", list(ghost.data.joint_names))
print("ghost body_names:", list(ghost.data.body_names))
print("num_joints:", ghost.num_joints, "num_bodies:", ghost.num_bodies)

d = np.load("/home/andy/research/bodex_dexsuite/plan_leap_pose.npz")
grasp_root = d["grasp_root"]
hand_q = d["hand_q"]
fk_tips = d["fk_tips"]
print("curobo fk_tips (world):\n", np.round(fk_tips, 4))

# write curobo-convention root pose + hand_q (as-is, like the visualizer does)
rot_xyzw = convert_quat(torch.tensor(grasp_root[3:7], dtype=torch.float32),
                        to="xyzw")
root = torch.tensor([grasp_root[0], grasp_root[1], grasp_root[2],
                     rot_xyzw[0], rot_xyzw[1], rot_xyzw[2], rot_xyzw[3]],
                    dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
q = torch.tensor(hand_q, dtype=torch.float32,
                 device=env.unwrapped.device).unsqueeze(0)
ghost.write_root_pose_to_sim(root)
ghost.write_root_velocity_to_sim_index(
    root_velocity=torch.zeros((1, 6), device=env.unwrapped.device))
ghost.write_joint_state_to_sim(q, torch.zeros_like(q))
ghost.write_data_to_sim()

with torch.inference_mode():
    pos = ghost.data.body_pos_w[0].cpu().numpy()
names = list(ghost.data.body_names)
print("ghost fingertips (world):")
tip_body_ix = {}
for want in ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]:
    ix = [i for i, n in enumerate(names) if n == want]
    print(f"  {want:16s} {np.round(pos[ix], 4) if ix else 'NOT FOUND'}")
    if ix:
        tip_body_ix[want] = pos[ix[0]]
err = np.abs(np.stack([tip_body_ix[n] for n in
                       ["fingertip", "fingertip_2", "fingertip_3",
                        "thumb_fingertip"]]) - fk_tips).max()
print(f"max |ghost_tips - curobo_fk_tips| = {err * 1000:.2f} mm")
