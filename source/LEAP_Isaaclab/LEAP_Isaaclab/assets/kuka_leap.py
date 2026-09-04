"""
Kuka iiwa7 arm with a LEAP hand welded to the end-effector flange.

Single 23-DOF articulation (Kuka iiwa7 7-DOF + LEAP 16-DOF) produced by merging
the Kuka iiwa7 arm URDF (wrist link iiwa7_link_7) and the LEAP hand URDF with a
fixed weld joint. The weld maps the LEAP wrist frame onto the flange such that
the palm normal (LEAP base +X) aligns with world -Z at the default ready pose:
the hand faces down toward the object like the Kuka-Allegro mount in the
Dexsuite envs; fingers point horizontal (-0.496, 0.868, 0) in world.

Built from: scripts/build_kuka_leap_urdf.py
Joint order: iiwa7_joint_1..7 (arm), a_0..a_15 (LEAP fingers)

NOTE: a custom spawn function is used because the URDF importer nests body
prims (link Xforms under one another) and the stock `schemas.activate_contact_sensors`
stops at the first rigid body, assigning PhysxContactReportAPI only to the
outermost link. `activate_contact_sensors_nested` walks the whole subtree so
fingertip contact sensors can pick up every body.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# NOTE: `from isaaclab.sim.schemas.schemas import safe_set_attribute_on_usd_prim`
# is imported lazily inside activate_contact_sensors_nested: schemas.py does
# `from pxr import UsdPhysics` at module level, which (if imported before
# SimulationApp) breaks the PhysX extension python bindings (PhysxSchema.Tokens
# / GfVec3f converter errors when Kit starts).

_ASSET_DIR = Path(__file__).parent

# Default arm pose (same as isaaclab_assets KUKA_ALLEGRO_CFG);
# LEAP fingers start straight (0.0) = open hand.
KUKA_LEAP_DEFAULT_JOINT_POS = {
    "iiwa7_joint_1": 0.0,
    "iiwa7_joint_2": 0.0,
    "iiwa7_joint_3": 0.7854,
    "iiwa7_joint_4": 1.5708,
    "iiwa7_joint_5": -1.5708,
    "iiwa7_joint_6": -1.5708,
    "iiwa7_joint_7": 0.0,
    "a_.*": 0.0,
}


def activate_contact_sensors_nested(prim_path: str, cfg, *args, **kwargs) -> "Usd.Prim":
    """Spawn the URDF artic then attach PhysxContactReportAPI to every rigid body.

    Replacement for the stock ``spawn_from_urdf`` that handles nested body prims
    produced by the URDF importer 3.0. The cloner replicates source schemas to
    all env instances, so adding the API on the template suffices.
    """
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_urdf

    # Deferred pxr imports: schemas.py does `from pxr import UsdPhysics` at
    # module level. Top-level pxr imports before SimulationApp starts break the
    # PhysX extension python bindings (PhysxSchema.Tokens / GfVec3f).
    from isaaclab.sim.schemas.schemas import safe_set_attribute_on_usd_prim
    from pxr import UsdPhysics

    prim = spawn_from_urdf(prim_path, cfg, *args, **kwargs)

    # activate contact reporting on all rigid bodies in the spawned subtree
    threshold = kwargs.get("contact_sensor_threshold", 0.0)
    stack = [prim]
    num_added = 0
    seen = set()
    while stack:
        p = stack.pop()
        path_key = p.GetPath()
        if path_key in seen:
            continue
        seen.add(path_key)
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            applied = p.GetAppliedSchemas()
            if "PhysxContactReportAPI" not in applied:
                p.AddAppliedSchema("PhysxContactReportAPI")
            if "PhysxRigidBodyAPI" not in applied:
                p.AddAppliedSchema("PhysxRigidBodyAPI")
            safe_set_attribute_on_usd_prim(p, "physxContactReport:threshold", threshold, camel_case=False)
            safe_set_attribute_on_usd_prim(p, "physxRigidBody:sleepThreshold", 0.0, camel_case=False)
            num_added += 1
        stack.extend(p.GetChildren())
    print(f"[KukaLeap] activate_contact_sensors_nested: {num_added} rigid bodies at '{prim_path}'")
    return prim


_DEFAULT_URDF_CFG = sim_utils.UrdfFileCfg(
    asset_path=str(_ASSET_DIR / "urdf" / "kuka_leap.urdf"),
    usd_dir=str(_ASSET_DIR / "urdf"),
    usd_file_name="kuka_leap.usd",
    fix_base=True,  # converter fixes the base link to the world (root prim)
    merge_fixed_joints=False,  # keep iiwa7_link_7 flange + LEAP base as rigid bodies
    force_usd_conversion=False,
    activate_contact_sensors=False,  # handled by activate_contact_sensors_nested
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
)

KUKA_LEAP_CFG = ArticulationCfg(
    spawn=_DEFAULT_URDF_CFG.replace(
        func="LEAP_Isaaclab.assets.kuka_leap:activate_contact_sensors_nested",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),  # xyzw (scalar last)
        joint_pos=KUKA_LEAP_DEFAULT_JOINT_POS,
    ),
    actuators={
        # arm: same gains as isaaclab_assets KUKA_ALLEGRO_CFG (iiwa7 section)
        "iiwa_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["iiwa7_joint_[1-4]"],
            effort_limit_sim=300.0,
            velocity_limit_sim=2.0,
            stiffness=300.0,
            damping=45.0,
            friction=1.0,
            armature=0.01,
        ),
        "iiwa_wrist": ImplicitActuatorCfg(
            joint_names_expr=["iiwa7_joint_[5-7]"],
            effort_limit_sim=300.0,
            velocity_limit_sim=2.0,
            stiffness={"iiwa7_joint_5": 100.0, "iiwa7_joint_6": 50.0, "iiwa7_joint_7": 25.0},
            damping={"iiwa7_joint_5": 20.0, "iiwa7_joint_6": 15.0, "iiwa7_joint_7": 15.0},
            friction=1.0,
            armature=0.01,
        ),
        # hand: same gains as LEAP_HAND_CFG (slightly stiff PD for contact)
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["a_.*"],
            effort_limit_sim=0.5,
            velocity_limit_sim=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
            armature=1e-3,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
