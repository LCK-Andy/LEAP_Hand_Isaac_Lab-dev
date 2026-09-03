"""Verify the two ghost-side fixes:
  (a) root pose: curobo root -> palm frame (R_palm = R_root @ A, t_palm = t_root - R_palm @ b)
  (b) joint reorder: curobo hand_q -> ghost joint order [1,5,9,12,0,4,8,13,...]

Compare ghost fingertip bodies against curobo fk_tips (world). If max err drops
to a few mm, both fixes are correct and apply to BODex and DGN2 pipelines."""
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
app = AppLauncher(exp_name="probe_ghost_dgn2b")
sim_app = app.app

import isaaclab.sim as sim_utils
import LEAP_Isaaclab.tasks  # noqa
import gymnasium as gym
import LEAP_Isaaclab
from LEAP_Isaaclab.assets import LEAP_HAND_CFG
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_tasks.utils import resolve_task_config
from isaaclab.utils.math import convert_quat
from scipy.spatial.transform import Rotation as Rst

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
env_cfg.scene.ghost = ghost_cfg

env = gym.make("Isaac-Dexsuite-Kuka-LEAP-Reorient-v0", cfg=env_cfg)
ghost = env.unwrapped.scene["ghost"]

plan = np.load("/home/andy/research/bodex_dexsuite/plan_leap_pose.npz")
grasp_root = plan["grasp_root"]   # curobo root, pos + wxyz
hand_q = plan["hand_q"]
fk_tips = plan["fk_tips"]

# A: DGN2 palm frame -> curobo root frame (validated on all 16 single-joint FKs)
A = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=float)
b = np.array([0.0, -0.038, 0.098])
R_root = Rst.from_quat(np.roll(grasp_root[3:7], -1)).as_matrix()
t_root = grasp_root[:3]
# curobo root pose -> palm pose: p_world = (R_root@A)@p + (R_root@b + t_root)
R_palm = R_root @ A
t_palm = t_root + R_root @ b
# ghost root frame == palm frame (verified: direct DGN2 probe 0.10 mm)
R_ghost = R_palm
t_ghost = t_palm
q_ghost_xyzw = Rst.from_matrix(R_ghost).as_quat()

# ghost joint order (from probe): value for ghost slot j = hand_q index of that joint name
BODEX_ORDER = ['1', '0', '2', '3', '5', '4', '6', '7', '9', '8', '10', '11', '12', '13', '14', '15']
GHOST_ORDER = ['1', '5', '9', '12', '0', '4', '8', '13', '2', '6', '10', '14', '3', '7', '11', '15']
q_ghost = np.array([hand_q[BODEX_ORDER.index(jn)] for jn in GHOST_ORDER])

env.reset()
ghost.write_root_pose_to_sim(torch.tensor(np.r_[t_ghost, q_ghost_xyzw],
                                           dtype=torch.float32).unsqueeze(0).to(env.unwrapped.device))
ghost.write_joint_state_to_sim(
    torch.tensor(q_ghost, dtype=torch.float32).unsqueeze(0).to(env.unwrapped.device),
    torch.zeros(1, 16, device=env.unwrapped.device),
)
ghost.write_data_to_sim()
# read back directly from the written buffers (like probe1); no physics step

bn = ghost.data.body_names
tip_pos = ghost.data.body_pos_w[0]
idx = [bn.index(n) for n in ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]]
ghost_tips = tip_pos[idx].cpu().numpy()
print("[probe2] ghost fingertips (world):\n", np.round(ghost_tips, 4))
print("[probe2] curobo fk_tips (world):\n", np.round(fk_tips, 4))
print(f"[probe2] max |ghost_tips - curobo_fk_tips| = "
      f"{np.abs(ghost_tips - fk_tips).max() * 1000:.2f} mm")

env.close()
sim_app.close()
