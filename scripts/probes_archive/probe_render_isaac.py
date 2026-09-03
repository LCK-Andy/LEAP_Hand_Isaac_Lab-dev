"""Definitive visual ground truth: render the ghost hand + object in Isaac
itself (same USD the GUI shows) with a camera, save PNG.

Runs: python probe_render_isaac.py --headless  (GPU render, 512px)
"""
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
app = AppLauncher(exp_name="probe_render_isaac", headless=args_cli.headless,
                  enable_cameras=getattr(args_cli, "enable_cameras", False))
sim_app = app.app

import isaaclab.sim as sim_utils
import LEAP_Isaaclab.tasks  # noqa
import gymnasium as gym
import LEAP_Isaaclab
from LEAP_Isaaclab.assets import LEAP_HAND_CFG
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import CameraCfg, Camera
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
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 20.0),
    ),
)

env = gym.make("Isaac-Dexsuite-Kuka-LEAP-Reorient-v0", cfg=env_cfg)
env.reset()
ghost = env.unwrapped.scene["ghost"]
cam = env.unwrapped.scene["cam"]
cam.set_world_poses_from_view(
    np.array([[-0.50, 0.45, 0.85]]),   # camera position (world)
    np.array([[-0.55, 0.1, 0.305]]),   # target = object center
)
dev = env.unwrapped.device

import torch
from scipy.spatial.transform import Rotation as Rst

d = np.load("/home/andy/research/bodex_dexsuite/plan_dgn2_multi.npz")
gr = d["grasp_root"][0]
t_palm = torch.tensor(gr[:3], dtype=torch.float32, device=dev)
q_xyzw = torch.tensor(np.roll(gr[3:7], -1), dtype=torch.float32, device=dev)

ghost.write_root_pose_to_sim(torch.cat([t_palm, q_xyzw]).unsqueeze(0))
ghost.write_joint_state_to_sim(
    torch.tensor(d["hand_q"][0], dtype=torch.float32, device=dev).unsqueeze(0),
    torch.zeros(1, 16, device=dev))
ghost.write_data_to_sim()

# report hand + object locations, and target the hand itself
tip_pos = ghost.data.body_pos_w[0].cpu().numpy()
bn = list(ghost.data.body_names)
print("ghost root pos:", ghost.data.root_pos_w.cpu().numpy().round(4))
print("palm_lower body:", tip_pos[bn.index("palm_lower")].round(4))
print("fingertip body:", tip_pos[bn.index("fingertip")].round(4))

import imageio.v2 as iio
import matplotlib.pyplot as plt
# palm-frame vectors in world
R_gg = Rst.from_quat(np.roll(d["grasp_root"][0][3:7], -1)).as_matrix()
palm_pos = tip_pos[bn.index("palm_lower")]
# camera placed along the palm normal (-z), looking back at the palm:
# i.e. what the palm "sees" = the object. If geometry is correct, this view
# should show the cube directly in front of the palm plate.
cam_from_palm = palm_pos + 0.3 * (-R_gg[:, 2])
for campos, target, tag in [
    (cam_from_palm, palm_pos, "palm_frontal"),
    (tip_pos[bn.index("palm_lower")] + [0.28, 0.28, 0.12], tip_pos[bn.index("palm_lower")], "palm"),
]:
    cam.set_world_poses_from_view(np.array([campos]), np.array([target]))
    for _ in range(3):
        cam.update(dt=0.1)
    img = cam.data.output["rgb"][0].cpu().numpy()
    iio.imwrite(f"/tmp/isaac_grasp0_{tag}.png", img)
    print(f"saved /tmp/isaac_grasp0_{tag}.png")

env.close()
sim_app.close()
