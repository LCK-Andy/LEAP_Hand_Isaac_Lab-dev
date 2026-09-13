"""DRO-Grasp model_leaphand grasp demo in Isaac Sim.

Loads a predicted grasp q (22 = root pos3 + euler3 + 16 joints, DRO URDF order)
plus precomputed open (outer_q) / close (inner_q) joint configs,
spawns the LEAP hand + a YCB object, executes open->close->hold, renders RGB frames.
"""
import argparse
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DRO LEAP grasp demo")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--q_file", type=str, default="/home/andy/Documents/DRO-Grasp/tmp/predict_q_potted_meat.npy")
parser.add_argument("--outer_q_file", type=str, default="/home/andy/Documents/DRO-Grasp/tmp/outer_q_potted_meat.npy")
parser.add_argument("--inner_q_file", type=str, default="/home/andy/Documents/DRO-Grasp/tmp/inner_q_potted_meat.npy")
parser.add_argument("--object_usd", type=str,
                    default="/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab/source/LEAP_Isaaclab/LEAP_Isaaclab/assets/ycb/potted_meat_can.usd")
parser.add_argument("--out_dir", type=str, default="/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab/logs/dro_demo")
parser.add_argument("--frames_close", type=int, default=240)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from LEAP_Isaaclab.assets.leap import LEAP_HAND_CFG

DRO_JOINT_NAMES = ['1', '0', '2', '3', '5', '4', '6', '7', '9', '8', '10', '11', '12', '13', '14', '15']


@configclass
class DemoSceneCfg(InteractiveSceneCfg):
    """Single-env scene: LEAP hand + one YCB object."""
    env_spacing = 2.0
    hand: ArticulationCfg = LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/leap_hand")
    obj: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/obj",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.3, 0.2), roughness=0.4, metallic=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


def main():
    q = np.load(args_cli.q_file)
    outer_q = np.load(args_cli.outer_q_file)
    inner_q = np.load(args_cli.inner_q_file)
    print("Loaded q / outer_q / inner_q")

    # root pose from DRO q (euler XYZ -> quat xyzw)
    pos = q[:3]
    quat = Rotation.from_euler('XYZ', q[3:6]).as_quat()  # xyzw
    print("root pos:", pos, "quat xyzw:", quat)

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, gravity=(0.0, 0.0, 0.0), device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = DemoSceneCfg(num_envs=1)
    scene = InteractiveScene(scene_cfg)

    # dome light for rendering (spawned directly on stage, not part of the scene)
    light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0))
    light_cfg.func("/World/Light", light_cfg)

    # camera must be created BEFORE sim.reset(): sensors initialize on PHYSICS_READY event
    cam_cfg = CameraCfg(
        prim_path="/World/Camera",
        update_period=0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=0.5, horizontal_aperture=20.955, clipping_range=(0.01, 20.0)
        ),
    )
    camera = Camera(cam_cfg)

    sim.reset()
    scene.reset()

    hand: Articulation = scene["hand"]
    obj: RigidObject = scene["obj"]

    # joint name mapping: DRO URDF order -> USD order
    usd_names = hand.joint_names
    print("USD joint names:", usd_names)
    dro_to_usd = [usd_names.index(f"a_{n}") if f"a_{n}" in usd_names else -1 for n in DRO_JOINT_NAMES]
    assert all(i >= 0 for i in dro_to_usd), f"mapping failed: {dro_to_usd}"
    print("DRO->USD joint index map:", dro_to_usd)

    print("OBJ debug: bodies:", obj.body_names, "| root_pos_w:", obj.data.root_pos_w.cpu().numpy())

    # place object at origin, hand at predicted root pose
    obj.write_root_pose_to_sim(
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device=sim.device)  # pos + quat xyzw
    )
    hand.write_root_pose_to_sim(
        torch.tensor([[pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]],
                     dtype=torch.float32, device=sim.device)
    )

    # camera
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[0.5, 0.5, 0.4]], device=sim.device),
        targets=torch.tensor([[0.0, 0.0, 0.05]], device=sim.device),
    )

    os.makedirs(args_cli.out_dir, exist_ok=True)

    def snapshot(name):
        camera.update(dt=0.0)
        rgb = camera.data.output["rgb"][0].cpu().numpy()
        from PIL import Image
        Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(args_cli.out_dir, name))
        print("saved", name, rgb.shape)

    # ---- Phase 1: open hand (outer_q), settle ----
    joint_pos = torch.zeros(1, hand.num_joints, device=sim.device)
    for i, usd_i in enumerate(dro_to_usd):
        joint_pos[0, usd_i] = float(outer_q[6 + i])
    hand.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    hand.set_joint_position_target(joint_pos)
    for _ in range(20):
        sim.step()

    snapshot("frame_1_open.png")

    # ---- Phase 2: close to inner_q, hold ----
    for i, usd_i in enumerate(dro_to_usd):
        joint_pos[0, usd_i] = float(inner_q[6 + i])
    hand.set_joint_position_target(joint_pos)
    for step in range(args_cli.frames_close):
        sim.step()
        if step in (60, 180):
            snapshot(f"frame_2_close_{step}.png")

    snapshot("frame_3_final.png")

    # object displacement check
    obj_pos = obj.data.root_pos_w[0].cpu().numpy()
    print("Object final pos:", obj_pos, "| displacement:", np.linalg.norm(obj_pos))

    simulation_app.close()


if __name__ == "__main__":
    main()
