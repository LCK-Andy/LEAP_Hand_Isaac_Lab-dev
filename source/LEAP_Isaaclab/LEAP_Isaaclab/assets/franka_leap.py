"""
Franka Panda arm with a welded LEAP hand.

Single articulated robot (23 DOF). The policy controls joint position deltas
for the whole arm+hand. This is a minimal working environment intended as a
base for task development (grasping, in-hand manipulation, tabletop tasks).

Built from: scripts/build_franka_leap_urdf.py + scripts/build_franka_leap_usd.py
(the latter converts the URDF and patches the articulation to the fixed-base
leap_hand-style encoding: one ArticulationRootAPI on panda_link0 and no
importer world joint -- the Isaac Lab 3 URDF-import pipeline emits a floating
base nondeterministically and its root_joint breaks under env cloning).
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
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_ASSET_DIR / "franka_leap" / "franka_leap.usd"),
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
            # OFF: the converter's coarse convex hulls (palm/wrist region after
            # fixed-link merging) overlap the straight fingers at the open
            # pose, holding them ~0.3 rad out of position via self-contacts.
            # Revisit with the fine LEAP-team collision meshes.
            enabled_self_collisions=False,
            # Isaac Lab creates the world->base fixed joint at spawn (the
            # cloner-safe encoding, same mechanism as the LEAP hand asset).
            # Works because the pre-built USD carries its single articulation
            # root API on the base body.
            fix_root_link=True,
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
        # arm: strict position-control gains (400/80, cf. Isaac Lab manipulation
        # demos). The "physically accurate" 80/4 of FRANKA_PANDA_CFG assumes the
        # official USD's real masses; with this URDF's mesh-density-derived
        # masses they leave a ~0.21 rad gravity sag at the ready pose.
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=400.0,
            damping=80.0,
            armature=1e-3,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=400.0,
            damping=80.0,
            armature=1e-3,
        ),
        # hand: same gains as LEAP_HAND_CFG (slightly stiff for PD tracking)
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["a_.*"],
            effort_limit_sim=0.5,
            velocity_limit_sim=100.0,
            stiffness=3.0,
            damping=0.1,
            armature=1e-3,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
