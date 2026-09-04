"""Visualize the BODex-planned Kuka-LEAP grasp pose in the DexSuite scene.

Builds the exact Isaac-Dexsuite-Kuka-LEAP-Reorient scene (table, cube object,
KUKA-LEAP robot), then:
  - sets robot joint positions to the planned grasp pose (arm + LEAP hand q)
  - lets the sim settle (no controller, finger actuators hold), and
  - renders with `--viz Kit`, saving screenshot frames.

No control/policy: this only displays the target dexterous grasp pose.
Usage:
  /home/andy/env_isaaclab/bin/python scripts/visualize_dexsuite_leap_grasp.py \\
      --viz Kit [--headless]
"""
import argparse

import numpy as np
import torch
from isaaclab.app import AppLauncher
from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli

parser = argparse.ArgumentParser(description="Visualize BODex-planned Kuka-LEAP grasp pose.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--screenshot_dir", type=str, default="/home/andy/research/bodex_dexsuite/frames_leap")
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--save_every", type=int, default=50)
parser.add_argument("--plan_path", type=str, default="/home/andy/research/bodex_dexsuite/plan_leap_pose.npz")
add_launcher_args(parser)
args_cli, _ = setup_preset_cli(parser)
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.math import quat_mul, quat_apply

# import AFTER AppLauncher (pxr import safety)
import LEAP_Isaaclab.tasks  # noqa: F401

import os

args, _ = parser.parse_known_args()
os.makedirs(args.screenshot_dir, exist_ok=True)

# ------------------------------------------------------------------- plan ----
plan = np.load(args.plan_path)
full_q = plan["full_q"]  # [23]: 7 arm (iiwa7_joint_1..7) + 16 hand (a_0..a_15)
object_pose = plan["object_pose"] if "object_pose" in plan.files else np.array([-0.55, 0.1, 0.305, 1, 0, 0, 0])
print(f"[viz] full_q: {np.round(full_q[:7], 3)} (arm) | {np.round(full_q[7:], 3)} (hand)")

# ------------------------------------------------------------------- env ----
gym_id = "Isaac-Dexsuite-Kuka-LEAP-Reorient-v0"
env_cfg = parse_env_cfg(gym_id, device="cuda:0", num_envs=args.num_envs)
# extra: move object to the same pose BODex planned with (it spawns at its default,
# which is already the DexSuite object pose -> keep default; but BODex world used
# [-0.55, 0.1, 0.305] which IS the DexSuite reset pose)
env = gym.make(gym_id, cfg=env_cfg)
env = env.unwrapped

obs, _ = env.reset()

# ------------------------------------------------------------------- align object ----
# The BODex plan assumes the object at [-0.55, 0.1, 0.305]; the env resets it
# to the DexSuite default (-0.55, 0.1, 0.35). Move the object to the planned
# pose so the hand-grasp geometry matches.
obj = env.scene["object"]
obj_root = obj.data.root_pose_w.clone()
obj_pos = torch.tensor(object_pose[:3], dtype=torch.float32, device=env.device)
obj_quat = torch.tensor([object_pose[3], object_pose[4], object_pose[5], object_pose[6]],
                        dtype=torch.float32, device=env.device)
obj_quat = obj_quat / obj_quat.norm()
obj_root[:, :3] = obj_pos.unsqueeze(0)
obj_root[:, 3:] = obj_quat.unsqueeze(0)
obj.write_root_pose_to_sim(obj_root)
env.sim.step()
env.sim.step()
print(f"[viz] object moved to plan pose {np.round(object_pose[:3], 3)}")

# ------------------------------------------------------------------- align arm ----
# curobo URDF vs Isaac USD frame offsets ≈ [0, 0, -0.05] (weld). Instead of
# trusting the BODex IK q, solve the arm IN the env with a small finite-
# difference IK so the hand base lands exactly on the planned grasp root.
robot = env.scene["robot"]
q = torch.tensor(full_q, dtype=torch.float32, device=env.device).unsqueeze(0)

# hand-base target (URDF 'base' == env body 'palm_lower')
targ_w = torch.tensor(plan["grasp_root"][:3], dtype=torch.float32, device=env.device).unsqueeze(0)

def apply_q(qv):
    robot.set_joint_position_target(qv)
    env.step(torch.zeros(1, 23, device=env.device))

def hand_pose(qv):
    apply_q(qv)
    return robot.data.body_pos_w[0, robot.data.body_names.index("palm_lower")].clone()

# finite-difference Jacobian of palm pos wrt the 7 arm joints
arm_ids = list(range(7))
p0 = hand_pose(q)
# measure the FIXED frame offset between the curobo hand root and the env body
offset0 = p0 - targ_w[0]
print(f"[viz] hand-root frame offset: {offset0.detach().cpu().numpy().round(3)}")
# Offset the curobo target by the measured frame difference: the IK below will
# drive the env's palm_lower to this corrected target (hand root in USD).
targ_c = targ_w + offset0.unsqueeze(0)

J = torch.zeros(3, 7, device=env.device)
dq = 0.02
for i in arm_ids:
    qd = q.clone(); qd[0, i] += dq
    pd = hand_pose(qd)
    J[:, i] = (pd - p0) / dq

# solve arm: damped least squares, few iterations (strong damping for stability)
q_arm = q.clone()
for it in range(25):
    p = hand_pose(q_arm)
    e = targ_c - p
    if e.norm() < 0.015:
        print(f"[viz] arm solve converged iter {it}, err {e.norm().item()*1000:.1f} mm")
        break
    J = torch.zeros(3, 7, device=env.device)
    for i in arm_ids:
        qd = q_arm.clone(); qd[0, i] += dq
        pd = hand_pose(qd)
        J[:, i] = (pd - p) / dq
    JtJ = J @ J.T
    JtJ_reg = JtJ + 0.25 * torch.eye(3, device=env.device)
    e_col = e.reshape(3, 1)
    dq_arm = J.T @ torch.linalg.solve(JtJ_reg, e_col)
    q_arm[0, :7] += 0.6 * dq_arm.reshape(-1)

q = q_arm
p_final = hand_pose(q)
print(f"[viz] final palm {p_final.detach().cpu().numpy()} vs target {targ_w[0].detach().cpu().numpy()}")
print(f"[viz] final arm q {q[0, :7].detach().cpu().numpy().round(3)}")

# ------------------------------------------------------------------- settle ----
# set robot to plan pose: arm joints + LEAP hand joints (joint names from asset)
robot = env.scene["robot"]
import re

# map full_q onto joint names in asset order
joint_names = robot.data.joint_names  # ["iiwa7_joint_1", ..., "a_0", ...]
q = torch.tensor(full_q, dtype=torch.float32, device=env.device).unsqueeze(0)
robot.set_joint_position_target(q)

# hold the object upright on table (it should already be; keep physics on)
n_steps = args.max_steps

# Hold the pose by re-issuing the joint targets every step: with the default
# ImplicitActuator the arm droops under gravity. Re-applying targets keeps the
# hand at the planned position. Also disable object gravity for stability.
for i in range(n_steps):
    env.step(torch.zeros(1, 23, device=env.device))
    robot.set_joint_position_target(q)
    if i == 0:
        # after 1 step the actuators have taken effect; print poses
        robot = env.scene["robot"]
        palm_candidates = ["palm_lower", "base", "hand_base_link"]
        palm_name = next((n for n in palm_candidates if n in robot.data.body_names), None)
        palm_pos = robot.data.body_pos_w[:, robot.data.body_names.index(palm_name)].cpu().numpy()[0]
        palm_quat = robot.data.body_quat_w[:, robot.data.body_names.index(palm_name)].cpu().numpy()[0]
        obj = env.scene["object"] if isinstance(env.scene, dict) else env.scene["object"]
        print(f"[viz] palm pos:  {np.round(palm_pos, 3)}")
        print(f"[viz] palm quat: {np.round(palm_quat, 3)}")
        try:
            obj_pos = obj.data.root_pos_w.cpu().numpy()[0]
            print(f"[viz] object pos: {np.round(obj_pos, 3)}")
        except Exception:  # noqa: BLE001
            print("[viz] object position unavailable")
        # finger tip positions for spot-check
        for tip_name in ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]:
            tip_pos = robot.data.body_pos_w[:, robot.data.body_names.index(tip_name)].cpu().numpy()[0]
            print(f"[viz]   tip {tip_name}: {np.round(tip_pos, 3)}")

print(f"[viz] ran {n_steps} steps; frames dir: {args.screenshot_dir}")
print("[viz] done (close the Kit window to exit)")
env.close()
sim_app.close()
