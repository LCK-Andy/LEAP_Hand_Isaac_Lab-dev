"""Direct DGN2->ghost test: no curobo involved.
For candidate 0 of dgn2_cube_raw.npz:
  R_ghost = R_palm @ Rx(pi), t_ghost = t_palm (palm pose from DGN2)
  q_ghost[j] = qpos[joint GHOST_ORDER[j]]  (by joint name)
Compare ghost fingertips vs fk_tips_dgn2 (world, DGN2's own FK)."""
import argparse
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rst

import isaaclab_tasks  # noqa: F401 (hydra)
from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

from isaaclab.app import AppLauncher
app = AppLauncher(exp_name="probe_dgn2_direct")
sim_app = app.app

import isaaclab.sim as sim_utils
import LEAP_Isaaclab.tasks  # noqa
import gymnasium as gym
import LEAP_Isaaclab
from LEAP_Isaaclab.assets import LEAP_HAND_CFG
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_tasks.utils import resolve_task_config

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
env.reset()
ghost = env.unwrapped.scene["ghost"]
dev = env.unwrapped.device

d = np.load("/home/andy/research/bodex_dexsuite/plan_dgn2_multi.npz")
bn = list(ghost.data.body_names)
# verify 3 grasps from the plan: grasp_root is palm pose pos+wxyz,
# hand_q already in GHOST order
for gi in [0, 5, 15]:
    gr = d["grasp_root"][gi]
    R_palm = Rst.from_quat(np.roll(gr[3:7], -1)).as_matrix()
    t_palm = gr[:3]
    q_ghost = d["hand_q"][gi]
    env.reset()
    ghost.write_root_pose_to_sim(
        torch.tensor(np.r_[t_palm, np.roll(gr[3:7], -1)], dtype=torch.float32)
        .unsqueeze(0).to(dev))
    ghost.write_joint_state_to_sim(
        torch.tensor(q_ghost, dtype=torch.float32).unsqueeze(0).to(dev),
        torch.zeros(1, 16, device=dev))
    ghost.write_data_to_sim()
    pos = ghost.data.body_pos_w[0].cpu().numpy()
    ghost_tips = pos[[bn.index(n) for n in
                      ["fingertip", "fingertip_2", "fingertip_3",
                       "thumb_fingertip"]]]
    # expected: DGN2-side FK tips at the SAME qpos (recomputed from raw)
    raw = np.load("/home/andy/research/bodex_dexsuite/dgn2_cube_raw.npz")
    raw_tips = raw["fk_tips_dgn2"].reshape(-1, 4, 3)
    # find the raw candidate matching this palm pose
    raw_p = raw["translation"]
    i = np.argmin(np.linalg.norm(raw_p - t_palm, axis=1))
    print(f"[plan-check] grasp {gi} (raw {i}) max err = "
          f"{np.abs(ghost_tips - raw_tips[i]).max() * 1000:.2f} mm")

env.close()
sim_app.close()
