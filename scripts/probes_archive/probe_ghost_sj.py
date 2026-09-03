"""Ghost single-joint sign test: set tn__0_=+1.0 (joint '0' = j0 = mcp->pip of
finger 1), identity root, read fingertip body local pos. Compare with the
palm-frame FK expectation [0.0706, -0.0269, -0.0195] mapped via Rx(pi):
expect [0.0706, 0.0269, 0.0195]."""
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
app = AppLauncher(exp_name="probe_ghost_sj")
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

print("joint_names:", list(ghost.data.joint_names))

root = torch.tensor([[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], device=dev)
q = torch.zeros(1, 16, device=dev)
q[0, 4] = 1.0   # tn__0_ is ghost slot 4 (probe1 order)
ghost.write_root_pose_to_sim(root)
ghost.write_joint_state_to_sim(q, torch.zeros_like(q))
ghost.write_data_to_sim()

bn = list(ghost.data.body_names)
pos = ghost.data.body_pos_w[0].cpu().numpy()
ix = bn.index("fingertip")
print("ghost fingertip local (tn__0_=+1):", np.round(pos[ix] - [0, 0, 0.5], 4))
print("expect (Rx(pi) @ palm j0=+1):       [ 0.0706  0.0269  0.0195]")
print("if sign flipped (j0=-1 palm):")
# palm j0=-1: rotate opposite -> compute from curobo q[1]=-1? not saved; just show range
env.close()
sim_app.close()
