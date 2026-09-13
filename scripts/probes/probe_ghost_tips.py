"""Decisive probe: which root rotation makes the Isaac LEAP ghost fingertips
match the curobo FK ground truth tips for the first grasp?

We load the env + ONE LEAP ghost, and for each candidate rotation applied on
top of the (wxyz->xyzw converted) grasp_root quat, we read the ghost's
fingertip body world positions and compare against curobo's own FK output for
the same grasp (which is authoritative: it reproduces the solver's answer).
Note: ghost uses the BODex-converted USD (same URDF as the solver), so the
joint vector maps 1:1 to hand_q.
"""
import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
import sys
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(exp_name="probe_ghost_tips")
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import LEAP_Isaaclab.tasks  # noqa: F401  (registers Isaac-Dexsuite-Kuka-LEAP-Reorient-v0)
import LEAP_Isaaclab
from LEAP_Isaaclab.assets import LEAP_HAND_CFG
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.math import convert_quat, quat_mul

from LEAP_Isaaclab.tasks.kuka_leap_dexsuite.dexsuite_kuka_leap_env_cfg import KukaLeapSceneCfg
from isaaclab_tasks.utils import resolve_task_config
import gymnasium as gym

env_cfg, _ = resolve_task_config("Isaac-Dexsuite-Kuka-LEAP-Reorient-v0", "")
env_cfg.scene.num_envs = 1

# --- ghost as a scene entity (path fixed to env_0 to keep the probe simple) ---
LEAP_HAND_CFG.replace  # ensure config class loaded
# ghost uses the BODex-converted USD (same URDF/model as curobo) so the joint
# vector maps 1:1 to hand_q, and geometry matches the solver exactly.
_bodex_cfg = LEAP_HAND_CFG.replace(
    spawn=LEAP_HAND_CFG.spawn.replace(
        usd_path=(
            "/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab/source/"
            "LEAP_Isaaclab/LEAP_Isaaclab/assets/leap_bodex_usd2/leap_right.usda"
        ),
    ),
)
ghost_cfg = _bodex_cfg.replace(
    prim_path="/World/envs/env_0/BODex_phantom_probe",
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), joint_pos={"tn__.*": 0.0}),
)

# attach as scene entity
ghost_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
    collision_enabled=False, contact_offset=0.001, rest_offset=0.0)
ghost_cfg.spawn.articulation_props.fix_root_link = False
env_cfg.scene.bodex_phantom_probe = ghost_cfg

env = gym.make("Isaac-Dexsuite-Kuka-LEAP-Reorient-v0", cfg=env_cfg)
env.reset()
ghost = env.unwrapped.scene["bodex_phantom_probe"]

# --- BODex plan for the FIRST grasp ---
d = np.load("/home/andy/research/bodex_dexsuite/plan_leap_pose.npz")
grasp_root = d["grasp_root"]   # [7] pos + wxyz
hand_q = d["hand_q"]           # [16]
fk_tips = d["fk_tips"]         # (4,3) curobo ground truth

# curobo FK ground truth tips via the BODex python (compute in numpy from plan:
# tip positions were saved from curobo; we trust them as truth)

# LEAP bodex-converted USD: joint order == hand_q order (tn__0_..tn__15_ maps
# to curobo joints 0..15), so the vector is used as-is.
q_joints = torch.tensor(hand_q, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)

# candidate rotations: quat applied as R_isaac = R_grasp @ extra
R_t = torch.tensor([0.0, -0.70710678, 0.0, 0.70710678])   # Ry(-90)  (current)
R_tT = torch.tensor([0.0, 0.70710678, 0.0, 0.70710678])   # Ry(+90)
IDENT = torch.tensor([0.0, 0.0, 0.0, 1.0])
cands = {"none": IDENT, "R_t(-90)": R_t, "R_tT(+90)": R_tT}

q_base = convert_quat(torch.tensor(grasp_root[3:7], dtype=torch.float32), to="xyzw")

import itertools
def body_pos(name_exact):
    names = ghost.data.body_names
    if isinstance(name_exact, list):
        idxs = [i for i, n in enumerate(names) if n in name_exact][:3]
    else:
        idxs = [i for i, n in enumerate(names) if n == name_exact]
    if not idxs:
        return None, names
    return ghost.data.body_pos_w[0, idxs].cpu().numpy(), names

with torch.inference_mode():
    for label, q_extra in cands.items():
        rot = quat_mul(q_base, q_extra)
        root = torch.cat([torch.tensor(grasp_root[:3], dtype=torch.float32), rot]).unsqueeze(0).to(env.unwrapped.device)
        ghost.write_root_pose_to_sim(root)
        ghost.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=env.unwrapped.device))
        ghost.write_joint_state_to_sim(q_joints, torch.zeros_like(q_joints))
        ghost.write_data_to_sim()
        # read fingertip body positions
        fpos, names = body_pos(["fingertip", "fingertip_2", "fingertip_3"])
        tpos, _ = body_pos("thumb_fingertip")
        if fpos is None:
            print(label, "body lookup failed; bodies:", [n for n in names if "tip" in n][:10])
            continue
        # we expect 3 fingertips + 1 thumb
        tips_ix = np.vstack([fpos, tpos[:1]]) if tpos is not None else fpos
        err = np.linalg.norm(tips_ix - fk_tips, axis=1).mean()
        print(f"{label:12s} get_pos tip mean-dist to curobo FK: {err:.4f} m")
        print("   tips_ix:\n", np.round(tips_ix, 3))
        print("   fk_tips:\n", np.round(fk_tips, 3))

env.close()
simulation_app.close()
