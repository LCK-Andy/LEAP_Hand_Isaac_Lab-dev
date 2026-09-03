"""Render several plan grasps from a clear 3/4 view with the Isaac camera."""
import argparse
import sys

import numpy as np

import isaaclab_tasks  # noqa: F401 (hydra)
from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

from isaaclab.app import AppLauncher
app = AppLauncher(exp_name="probe_render_multi", headless=args_cli.headless,
                  enable_cameras=getattr(args_cli, "enable_cameras", False))
sim_app = app.app

import isaaclab.sim as sim_utils
import LEAP_Isaaclab.tasks  # noqa
import gymnasium as gym
import LEAP_Isaaclab
from LEAP_Isaaclab.assets import LEAP_HAND_CFG
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils import resolve_task_config
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
    prim_path="/World/envs/env_0/BODex_phantom_render",
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5),
                                               joint_pos={"tn__.*": 0.0}),
)
ghost_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
    collision_enabled=False, contact_offset=0.001, rest_offset=0.0)
ghost_cfg.spawn.articulation_props.fix_root_link = False
env_cfg.scene.ghost = ghost_cfg
env_cfg.scene.cam = CameraCfg(
    prim_path="/World/envs/env_0/CameraSensor",
    update_period=0.0,
    height=512,
    width=512,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.1, 20.0)),
)

env = gym.make("Isaac-Dexsuite-Kuka-LEAP-Reorient-v0", cfg=env_cfg)
env.reset()
ghost = env.unwrapped.scene["ghost"]
obj = env.unwrapped.scene["object"]
cam = env.unwrapped.scene["cam"]
dev = env.unwrapped.device

import torch
import imageio.v2 as iio

d = np.load("/home/andy/research/bodex_dexsuite/plan_dgn2_multi.npz")
# pin the object at the plan pose (like the visualizer does)
o_root = torch.tensor(
    [d["object_pose"][0], d["object_pose"][1], d["object_pose"][2],
     *np.roll(d["object_pose"][3:7], -1)],
    dtype=torch.float32, device=dev).unsqueeze(0)
obj.write_root_pose_to_sim(o_root)
obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((obj.num_instances, 6), device=dev))
obj.write_data_to_sim()

OC = np.array([-0.55, 0.1, 0.305])
panels = []
for gi in [0, 3, 7, 10]:
    gr = d["grasp_root"][gi]
    t = torch.tensor(gr[:3], dtype=torch.float32, device=dev)
    q = torch.tensor(np.roll(gr[3:7], -1), dtype=torch.float32, device=dev)
    ghost.write_root_pose_to_sim(torch.cat([t, q]).unsqueeze(0))
    ghost.write_joint_state_to_sim(
        torch.tensor(d["hand_q"][gi], dtype=torch.float32, device=dev).unsqueeze(0),
        torch.zeros(1, 16, device=dev))
    ghost.write_data_to_sim()
    env.step(torch.zeros(env.action_space.shape, device=dev))
    ghost.write_root_pose_to_sim(torch.cat([t, q]).unsqueeze(0))
    ghost.write_joint_state_to_sim(
        torch.tensor(d["hand_q"][gi], dtype=torch.float32, device=dev).unsqueeze(0),
        torch.zeros(1, 16, device=dev))
    ghost.write_data_to_sim()
    # re-pin object after the physics step (free object would fall)
    obj.write_root_pose_to_sim(o_root)
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((obj.num_instances, 6), device=dev))
    obj.write_data_to_sim()
    # camera: 40 cm from object, elevated 20 deg
    cam.set_world_poses_from_view(
        np.array([OC + [0.16, 0.16, 0.20]]), np.array([OC]))
    for _ in range(3):
        cam.update(dt=0.1)
    img = cam.data.output["rgb"][0].cpu().numpy()
    iio.imwrite(f"/tmp/isaac_grasp{gi}.png", img)
    print(f"saved /tmp/isaac_grasp{gi}.png")

env.close()
sim_app.close()
