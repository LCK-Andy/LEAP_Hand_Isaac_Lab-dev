"""
DRO-Grasp LEAP Hand grasp demo — multi-angle video recording.

Loads DRO-Grasp predicted q (root pose + 16 joints) for the LEAP hand,
plays open -> close (interpolated) -> hold in Isaac Sim, and records
videos from 3 camera angles.

Usage:
    DISPLAY=:0 ~/env_isaaclab/bin/python scripts/dro_grasp_demo_video.py --out_dir logs/dro_demo/video
"""
import argparse
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.app import AppLauncher

# ---------- DRO-Grasp URDF joint order (root xyz + rpy + 16 joints) ----------
DRO_JOINT_NAMES = [
    "1", "0", "2", "3", "5", "4", "6", "7", "9", "8", "10", "11", "12", "13", "14", "15"
]

parser = argparse.ArgumentParser(description="DRO-Grasp LEAP demo video")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--predict_q", type=str, default="/home/andy/Documents/DRO-Grasp/tmp/predict_q_potted_meat.npy")
parser.add_argument("--outer_q", type=str, default="/home/andy/Documents/DRO-Grasp/tmp/outer_q_potted_meat.npy")
parser.add_argument("--inner_q", type=str, default="/home/andy/Documents/DRO-Grasp/tmp/inner_q_potted_meat.npy")
parser.add_argument("--out_dir", type=str, default="logs/dro_demo/video")
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Sim / Omniverse imports must come AFTER AppLauncher
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from LEAP_Isaaclab.assets.leap import LEAP_HAND_CFG  # noqa: E402


@configclass
class DemoSceneCfg(InteractiveSceneCfg):
    """Single-env scene: LEAP hand + one object."""
    env_spacing = 2.0
    hand: ArticulationCfg = LEAP_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/leap_hand",
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["a_.*"],
                stiffness=50.0,
                damping=2.0,
                effort_limit=5.0,
            )
        },
    )
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


CAMERA_VIEWS = {
    "front": dict(eye=(0.55, 0.35, 0.30), target=(0.05, 0.0, 0.0)),
    "side": dict(eye=(-0.45, 0.30, 0.25), target=(0.0, 0.0, 0.0)),
    "top": dict(eye=(0.10, 0.05, 0.65), target=(0.0, 0.0, 0.0)),
}

STEP_RATE = 120  # physics steps per second
RECORD_EVERY = 2  # record one frame every N steps -> 60 fps video


def main():
    q = np.load(args_cli.predict_q)
    outer_q = np.load(args_cli.outer_q)
    inner_q = np.load(args_cli.inner_q)
    print("Loaded q / outer_q / inner_q")

    pos = q[:3]
    quat = Rotation.from_euler("XYZ", q[3:6]).as_quat()  # xyzw
    print("root pos:", pos, "quat xyzw:", quat)

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / STEP_RATE, gravity=(0.0, 0.0, 0.0), device="cuda:0"))

    scene_cfg = DemoSceneCfg(num_envs=1)
    scene = InteractiveScene(scene_cfg)

    # dome light for rendering (spawned directly on stage, not part of the scene)
    light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0))
    light_cfg.func("/World/Light", light_cfg)

    # cameras must be created BEFORE sim.reset(): sensors initialize on PHYSICS_READY event
    cameras = {}
    for i, (name, view) in enumerate(CAMERA_VIEWS.items()):
        cam_cfg = CameraCfg(
            prim_path=f"/World/Cam_{name}",
            update_period=0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=0.5, horizontal_aperture=20.955, clipping_range=(0.01, 20.0)
            ),
        )
        cameras[name] = Camera(cam_cfg)

    sim.reset()
    scene.reset()

    hand: Articulation = scene["hand"]
    obj: RigidObject = scene["obj"]
    print("OBJ debug: bodies:", obj.body_names, "| root_pos_w:", obj.data.root_pos_w.cpu().numpy())

    # joint name mapping: DRO URDF order -> USD order
    usd_names = hand.joint_names
    dro_to_usd = [usd_names.index(f"a_{n}") if f"a_{n}" in usd_names else -1 for n in DRO_JOINT_NAMES]
    assert all(i >= 0 for i in dro_to_usd), f"mapping failed: {dro_to_usd}"
    print("DRO->USD joint index map:", dro_to_usd)

    # place object at origin, hand at predicted root pose
    obj.write_root_pose_to_sim(torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device=sim.device))
    hand.write_root_pose_to_sim(
        torch.tensor([[pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]],
                     dtype=torch.float32, device=sim.device)
    )

    # point cameras at the grasp site
    for name, cam in cameras.items():
        view = CAMERA_VIEWS[name]
        eye = torch.tensor([view["eye"]], device=sim.device)
        tgt = torch.tensor([view["target"]], device=sim.device)
        cam.set_world_poses_from_view(eyes=eye, targets=tgt)

    os.makedirs(args_cli.out_dir, exist_ok=True)

    # video writers (60 fps: one frame per RECORD_EVERY physics steps)
    writers = {}
    for name in cameras:
        path = os.path.join(args_cli.out_dir, f"{name}.mp4")
        writers[name] = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), STEP_RATE // RECORD_EVERY, (1280, 720))
        print("writing", path)

    def record():
        for name, cam in cameras.items():
            cam.update(dt=0.0)
            rgb = cam.data.output["rgb"][0].cpu().numpy()
            writers[name].write(rgb.astype(np.uint8))

    def pose_for(q16):
        p = torch.zeros(1, hand.num_joints, device=sim.device)
        for i, usd_i in enumerate(dro_to_usd):
            p[0, usd_i] = float(q16[i])
        return p

    # ---- Phase 1: open hand, settle ----
    open_pos = pose_for(outer_q[6:])
    hand.write_joint_state_to_sim(open_pos, torch.zeros_like(open_pos))
    hand.set_joint_position_target(open_pos)
    for step in range(45):
        hand.write_joint_state_to_sim(open_pos, torch.zeros_like(open_pos))
        sim.step()
        if step % RECORD_EVERY == 0:
            record()

    # ---- Phase 2: smoothly close outer -> inner (kinematically locked) ----
    close_pos = pose_for(inner_q[6:])
    n_close = 120
    for i in range(n_close):
        alpha = i / (n_close - 1)
        target = (1.0 - alpha) * open_pos + alpha * close_pos
        hand.write_joint_state_to_sim(target, torch.zeros_like(target))
        hand.set_joint_position_target(target)
        sim.step()
        if i % RECORD_EVERY == 0:
            record()

    # ---- Phase 3: hold closed ----
    hand.set_joint_position_target(close_pos)
    for step in range(120):
        hand.write_joint_state_to_sim(close_pos, torch.zeros_like(close_pos))
        sim.step()
        if step % RECORD_EVERY == 0:
            record()

    for w in writers.values():
        w.release()
    print("Object final pos:", obj.data.root_pos_w.cpu().numpy()[0], "| displacement:",
          float(torch.norm(obj.data.root_pos_w[0])))


if __name__ == "__main__":
    import cv2  # imported here so AppLauncher args parse cleanly
    main()
    simulation_app.close()
