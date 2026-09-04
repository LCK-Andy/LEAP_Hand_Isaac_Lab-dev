"""
Franka Panda arm + LEAP hand environment config.

The LEAP hand is welded to the Franka flange (single 23-DOF articulation).
Actions: 23-d joint position deltas (7 arm + 16 hand), clamped to joint limits.
Observations: 46-d (23 joint positions + 23 joint velocities).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from LEAP_Isaaclab.assets import FRANKA_LEAP_CFG


@configclass
class FrankaLeapEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    action_space = 23  # 7 arm + 16 hand
    observation_space = 46  # joint pos + joint vel
    state_space = 0

    # control
    actions_scale = 0.1  # rad per policy step (relative joint position control)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physics=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**25,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

    # robot: Franka 7-DOF arm + welded LEAP 16-DOF hand
    robot_cfg: ArticulationCfg = FRANKA_LEAP_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    arm_joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                       "panda_joint5", "panda_joint6", "panda_joint7"]
    hand_joint_names = [f"a_{i}" for i in range(16)]
    # fingertip bodies (LEAP), used for world-space reporting
    fingertip_body_names = ["fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"]

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=False)

    # reward scales
    action_penalty_scale = -0.01
    velocity_penalty_scale = -0.001
