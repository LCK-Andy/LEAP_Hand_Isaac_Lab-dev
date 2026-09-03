"""Clean frame test: identity root pose + zero joints -> read ghost fingertips.
Compare against:
  palm frame zero-config (DGN2):   [0.0821, 0.0227, -0.0195] etc
  curobo root zero-config:         [-0.0195, -0.0606, 0.1802] etc
Whichever matches tells us the ghost root frame."""
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
app = AppLauncher(exp_name="probe_ghost_clean")
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

root = torch.tensor([[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], device=dev)
ghost.write_root_pose_to_sim(root)
ghost.write_joint_state_to_sim(torch.zeros(1, 16, device=dev),
                               torch.zeros(1, 16, device=dev))
ghost.write_data_to_sim()

bn = list(ghost.data.body_names)
pos = ghost.data.body_pos_w[0].cpu().numpy()
print("root body:", bn[0])
for want in ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip", "palm_lower"]:
    ix = [i for i, n in enumerate(bn) if n == want]
    if ix:
        print(f"  {want:16s} {np.round(pos[ix[0]] - [0, 0, 0.5], 4)}")
print("palm-frame zero-config (expect if root=palm):")
print("  fingertip        [ 0.0821  0.0227 -0.0195]")
print("  fingertip_2      [ 0.0821 -0.0228 -0.0195]")
print("  fingertip_3      [ 0.0821 -0.0682 -0.0195]")
print("  thumb_fingertip  [-0.0824  0.0767 -0.0069]")
print("curobo-root zero-config (expect if root=curobo):")
print("  fingertip        [-0.0195 -0.0606  0.1802]")
print("  fingertip_2      [-0.0195 -0.0152  0.1801]")
print("  fingertip_3      [-0.0195  0.0302  0.1801]")
print("  thumb_fingertip  [-0.0069 -0.1147  0.0157]")

env.close()
sim_app.close()
