"""Play the D(R,O) policy in Isaac Lab: LEAP hand grasps a YCB object on the floor.

Object falls & settles on the floor -> hand approaches the predicted grasp pose ->
fingers close -> hand lifts -> object rises with it (success). Telemetry saved to
logs/play_dro/<object>_g<idx>.json.
"""
import argparse
import json
import os
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--object", type=str, default="ycb_potted_meat_can")
parser.add_argument("--grasp-idx", type=int, default=0)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from scipy.spatial.transform import Rotation as SciRot
from LEAP_Isaaclab.assets.leap import LEAP_HAND_CFG

ROOT = "/home/andy/Documents/DRO-Grasp"
NPZ = f"{ROOT}/tmp/isaac_eval/{args.object}.npz"
OUT = "/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab/logs/play_dro"
os.makedirs(OUT, exist_ok=True)
DRO_JOINT_ORDER = [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15]

# ---------- scene ----------
sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, gravity=(0.0, 0.0, -9.81), device="cuda:0"))
scene = InteractiveScene(InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False))

hand_cfg = LEAP_HAND_CFG.copy()
hand_cfg.prim_path = "/World/envs/env_0/hand"
hand_cfg.init_state.pos = (0.6, 0.0, 0.2)
for art in hand_cfg.actuators.values():
    art.stiffness = 200.0
    art.damping = 20.0
    art.effort_limit = 2.0
hand = Articulation(hand_cfg)
scene.articulations["hand"] = hand

obj_cfg = RigidObjectCfg(
    prim_path="/World/envs/env_0/object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab/source/LEAP_Isaaclab/LEAP_Isaaclab/assets/ycb/{args.object.replace('ycb_', '')}.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1000.0)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25), rot=(1.0, 0.0, 0.0, 0.0)),
)
obj = RigidObject(obj_cfg)
scene.rigid_objects["object"] = obj

# floor: heavy dynamic cuboid (kit GroundPlane breaks physx integration in this build)
floor_cfg = RigidObjectCfg(
    prim_path="/World/envs/env_0/floor",
    spawn=sim_utils.CuboidCfg(
        size=(5.0, 5.0, 0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
)
floor = RigidObject(floor_cfg)
scene.rigid_objects["floor"] = floor

sim.reset()
# apply friction 0.8 to every hand collision prim via a physics material
from isaaclab.sim.spawners.materials.physics_materials_cfg import PhysxRigidBodyMaterialCfg
from isaaclab.sim.spawners.materials import spawn_rigid_body_material
from isaaclab.sim.utils import bind_physics_material
from pxr import Usd, UsdPhysics
mat_path = spawn_rigid_body_material(
    "/World/envs/env_0/hand_friction_mat",
    PhysxRigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8, restitution=0.0),
)
n_mat = 0
import omni.usd
hand_stage = sim.stage if hasattr(sim, "stage") else omni.usd.get_context().get_stage()
for pr in hand_stage.Traverse():
    if not str(pr.GetPath()).startswith("/World/envs/env_0/hand"):
        continue
    if not UsdPhysics.CollisionAPI(pr):
        continue
    bind_physics_material(pr.GetPath(), mat_path)
    n_mat += 1
print(f"[play] hand friction 0.8 bound on {n_mat} collision prims")
# also bind 0.8 to the object (default material friction ~0.25 would halve contact mu)
n_obj = 0
for pr in hand_stage.Traverse():
    if not str(pr.GetPath()).startswith("/World/envs/env_0/object"):
        continue
    if not UsdPhysics.CollisionAPI(pr):
        continue
    try:
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI(pr).Bind(UsdShade.Material(mat_path), "physics", "weakest")
        n_obj += 1
    except Exception as e:
        print(f"[play]   obj bind skipped {pr.GetPath()}: {e}")
print(f"[play] object friction 0.8 bound on {n_obj} collision prims")
scene.reset()
sim.step()

dof_map = [hand.joint_names.index(f"a_{m}") for m in DRO_JOINT_ORDER]
n = 1

def step():
    scene.write_data_to_sim()
    sim.step()
    scene.update(1.0 / 120.0)

# ---------- 1. object settles on the floor ----------
print("[play] settling object...")
from pxr import Usd, UsdGeom
import omni.usd
_st = omni.usd.get_context().get_stage()
for _pr in _st.Traverse():
    if "ground" in str(_pr.GetPath()):
        _x = UsdGeom.Xformable(_pr)
        print(f"[play]   prim {_pr.GetPath()}: z={_x.GetLocalTransformation()[2][3]*100:.1f} cm")
for i in range(300):
    step()
    if i in (0, 30, 60, 120, 299):
        print(f"[play]   step {i}: obj z={obj.data.root_pos_w[0, 2].item()*100:.1f} cm v={obj.data.root_lin_vel_w[0, 2].item():.2f}")
obj_pos0 = obj.data.root_pos_w[0, :3].clone()
obj_quat0 = obj.data.root_quat_w[0].clone()
print(f"[play] object settled at z={obj_pos0[2].item()*100:.1f} cm")

# ---------- 2. grasp pose from the policy (object frame) ----------
d = np.load(NPZ)
q = np.asarray(d["predict_q"][args.grasp_idx])
T_obj = np.eye(4)
T_obj[:3, :3] = SciRot.from_quat(obj_quat0.cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
T_obj[:3, 3] = obj_pos0.cpu().numpy()
T_g = np.eye(4)
T_g[:3, :3] = SciRot.from_euler("XYZ", q[3:6]).as_matrix()
T_g[:3, 3] = q[:3]
T_w = T_obj @ T_g
grasp_pos = T_w[:3, 3]
# the predicted palm is ~5cm too far from the object (loose D(R,O) grasps):
# pull the palm 5cm closer along the object direction
dir_obj = grasp_pos - obj_pos0.cpu().numpy()
grasp_pos = grasp_pos - 0.05 * dir_obj / (np.linalg.norm(dir_obj) + 1e-9)
grasp_pos[2] = max(float(grasp_pos[2]), 0.03)  # keep the palm above the ground plane
# flip the wrist orientation 180 deg about X: grasp from the top (floor-viable)
R_w = T_w[:3, :3]
R_flip = R_w @ SciRot.from_euler("X", np.pi).as_matrix()
grasp_quat = SciRot.from_matrix(R_flip).as_quat()  # xyzw

joint_targets = torch.zeros(n, hand.num_joints, device=sim.device)
for j in range(16):
    joint_targets[0, dof_map[j]] = float(q[6 + j])
open_q = torch.zeros(n, hand.num_joints, device=sim.device)

def set_root(pos, quat):
    hand.write_root_pose_to_sim(torch.tensor([[float(pos[0]), float(pos[1]), float(pos[2]), float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]], dtype=torch.float32, device=sim.device))

# ---------- 3. approach from above ----------
start_pos = grasp_pos + np.array([0.0, 0.0, 0.18])
set_root(start_pos, grasp_quat)
for _ in range(30):
    hand.data.joint_pos_target[0, dof_map] = open_q[0]
    step()
print("[play] approaching grasp pose...")
for i in range(60):
    t = i / 59
    set_root((1 - t) * start_pos + t * grasp_pos, grasp_quat)
    hand.data.joint_pos_target[0, dof_map] = open_q[0]
    step()

# ---------- 4. close fingers ----------
print("[play] closing fingers...")
for i in range(90):
    t = min(1.0, i / 60)
    hand.data.joint_pos_target[0, dof_map] = (1 - t) * open_q + t * joint_targets[0]
    set_root(grasp_pos, grasp_quat)
    step()

# extra squeeze: tighten 0.5 rad beyond the predicted close (clamped to joint limits)
print("[play] squeezing...")
lim = hand.data.joint_limits
tight = joint_targets.clone()
for j in range(16):
    v = tight[0, j].item() - 0.5
    lo, hi = lim[0, j, 0].item(), lim[0, j, 1].item()
    tight[0, j] = min(max(v, lo), hi)
for i in range(60):
    t = min(1.0, i / 40)
    hand.data.joint_pos_target[0, dof_map] = (1 - t) * joint_targets + t * tight[0]
    set_root(grasp_pos, grasp_quat)
    step()
    if i in (0, 30, 59):
        jp = hand.data.joint_pos[0, dof_map].cpu().numpy()
        print(f"[play]   jpos {i}: mean={jp.mean():.3f} a0={jp[0]:.3f} a5={jp[4]:.3f} a9={jp[8]:.3f} a12={jp[12]:.3f}")
    if i in (0, 30, 59):
        print(f"[play]   squeeze {i}: obj z={obj.data.root_pos_w[0, 2].item()*100:.1f} cm")

# ---------- 5. lift ----------
print("[play] lifting...")
obj_z0 = obj.data.root_pos_w[0, 2].item()
traj = []
for i in range(150):
    z = grasp_pos[2] + 0.22 * (i / 149)
    set_root(np.array([grasp_pos[0], grasp_pos[1], z]), grasp_quat)
    hand.data.joint_pos_target[0, dof_map] = tight[0]
    step()
    if i % 5 == 0:
        traj.append([i, z, obj.data.root_pos_w[0, 2].item()])

obj_z1 = obj.data.root_pos_w[0, 2].item()
rise = obj_z1 - obj_z0
print(f"[play] object rise = {rise*100:.1f} cm -> {'SUCCESS (lifted)' if rise > 0.08 else 'FAIL (not lifted)'}")

json.dump({"object": args.object, "grasp_idx": args.grasp_idx, "rise_m": rise,
           "traj": traj, "q": q.tolist()}, open(f"{OUT}/{args.object}_g{args.grasp_idx}.json", "w"), indent=1)
print(f"[play] telemetry saved: {OUT}/{args.object}_g{args.grasp_idx}.json")
simulation_app.close()
