"""Unified probe: sim body SE(3) vs DRO FK per-link SE(3), in ONE process.

For q=zero and q=close(grasp0):
  - sim: spawn hand, write root + joints (write_joint_position_to_sim every step), read body poses
  - DRO: HandModel FK (same python process, CPU)
Report per-link: translation diff (mm), rotation diff (deg), and the fitted
root-frame correction T_corr = T_base_sim @ inv(T_palm_dro) at zero.
"""
import argparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from LEAP_Isaaclab.assets.leap import LEAP_HAND_CFG

import sys
sys.path.insert(0, '/home/andy/Documents/DRO-Grasp')
from utils.hand_model import create_hand_model

DRO_JOINT_ORDER = ['1', '0', '2', '3', '5', '4', '6', '7', '9', '8', '10', '11', '12', '13', '14', '15']
BODY_MAP = {'palm_lower': 'base'}


@configclass
class PScene(InteractiveSceneCfg):
    env_spacing = 2.0
    hand: ArticulationCfg = LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/leap_hand")


def mat_angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


sim_cfg = SimulationCfg(dt=1.0 / 120.0, gravity=(0.0, 0.0, 0.0), device="cuda:0")
sim = sim_utils.SimulationContext(sim_cfg)
scene = InteractiveScene(PScene(num_envs=1))
sim.reset()
scene.reset()
hand = scene["hand"]
jn = hand.joint_names
bn = hand.body_names
dof_map = [jn.index(f"a_{m}") for m in DRO_JOINT_ORDER]

h = create_hand_model('leaphand', torch.device('cpu'))
links = list(h.links_pc.keys())
print("DRO links:", links)
print("sim bodies:", bn)

d = np.load('/home/andy/Documents/DRO-Grasp/tmp/isaac_eval/ycb_potted_meat_can.npz')
qs = {'zero': np.zeros(22), 'close': np.asarray(d['predict_q'][0])}

for qname, q in qs.items():
    # DRO FK per-link world SE3
    h.update_status(torch.tensor(q, dtype=torch.float32))
    dro_T = {ln: h.frame_status[ln].get_matrix()[0].numpy() for ln in links}

    # sim: root pose + joints
    pos = q[:3]
    quat = Rotation.from_euler('XYZ', q[3:6]).as_quat()
    hand.write_root_pose_to_sim(torch.tensor([[pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]],
                                            dtype=torch.float32, device=sim.device))
    jp = torch.zeros(1, hand.num_joints, device=sim.device)
    for j in range(16):
        jp[0, dof_map[j]] = float(q[6 + j])
    hand.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    for _ in range(400):
        hand.write_joint_position_to_sim(jp)
        sim.step()
    bp = hand.data.body_pos_w[0].cpu().numpy()
    bq = hand.data.body_quat_w[0].cpu().numpy()
    sim_T = {}
    for i, b in enumerate(bn):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(bq[i]).as_matrix()
        T[:3, 3] = bp[i]
        sim_T[b] = T

    print(f"== {qname} q (root={np.round(q[:6], 3)}) ==")
    for ln in links:
        b = BODY_MAP.get(ln, ln)
        if b not in sim_T:
            print(f"  {ln}: body '{b}' missing!")
            continue
        dT = sim_T[b] @ np.linalg.inv(dro_T[ln])
        dt = np.linalg.norm(dT[:3, 3]) * 1000
        dR = mat_angle(dT[:3, :3])
        print(f"  {ln:20s} dt={dt:6.1f}mm dR={dR:6.1f}deg")
    T_base_sim = sim_T['base']
    T_palm_dro = dro_T['palm_lower']
    T_corr = T_base_sim @ np.linalg.inv(T_palm_dro)
    print(f"  T_corr (base = T_corr @ palm): t={np.round(T_corr[:3,3]*1000,1)}mm dR={mat_angle(T_corr[:3,:3]):.1f}deg")
simulation_app.close()
