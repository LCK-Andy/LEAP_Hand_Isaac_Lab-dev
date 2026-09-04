"""
Franka Panda arm with a LEAP hand welded to the end-effector flange.

Single 23-DOF articulation (Franka 7-DOF + LEAP 16-DOF) produced by merging
the Franka arm URDF (flange panda_link8) and the LEAP hand URDF with a fixed
weld joint. The weld maps the LEAP wrist frame onto the flange such that the
palm normal (LEAP base +X) aligns with the flange tool axis (+Z): the hand
extends along the tool direction like a wrist extension.

Built from: scripts/build_franka_leap_urdf.py
Joint order: panda_joint1..7 (arm), a_0..a_15 (LEAP fingers)
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_ASSET_DIR = Path(__file__).parent

# Default arm pose (same as isaaclab_assets FRANKA_PANDA_CFG);
# LEAP fingers start straight (0.0) = open hand.
FRANKA_LEAP_DEFAULT_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "a_.*": 0.0,
}

FRANKA_LEAP_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(_ASSET_DIR / "urdf" / "franka_leap.urdf"),
        usd_dir=str(_ASSET_DIR / "urdf"),
        usd_file_name="franka_leap.usd",
        fix_base=True,  # converter fixes the base link to the world (root prim)
        merge_fixed_joints=False,  # keep panda_link8 flange + LEAP base as rigid bodies
        force_usd_conversion=False,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / 3.141592653589793 * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),  # xyzw (scalar last)
        joint_pos=FRANKA_LEAP_DEFAULT_JOINT_POS,
    ),
    actuators={
        # arm: same gains as isaaclab_assets FRANKA_PANDA_CFG
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
            armature=1e-3,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
            armature=1e-3,
        ),
        # hand: same gains as LEAP_HAND_CFG (slightly stiff for PD tracking)
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["a_.*"],
            effort_limit_sim=0.5,
            velocity_limit_sim=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
