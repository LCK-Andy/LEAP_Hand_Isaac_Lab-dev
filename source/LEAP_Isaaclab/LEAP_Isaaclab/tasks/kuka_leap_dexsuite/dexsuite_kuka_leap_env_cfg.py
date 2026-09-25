# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kuka LEAP participant env configs (Dexsuite Reorient/Lift with LEAP hand).

Based on the Isaac Lab Dexsuite Kuka-Allegro env (config/kuka_allegro/) but with
the Allegro hand replaced by a LEAP hand welded onto the Kuka iiwa7 flange
(23-DOF articulation, see assets/kuka_leap.py).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from LEAP_Isaaclab.assets import KUKA_LEAP_CFG

from LEAP_Isaaclab.compat.dexsuite import dexsuite_env_cfg as dexsuite
from LEAP_Isaaclab.compat.dexsuite import mdp as dexsuite_mdp

from . import force_utils

# Monkey-patch stock Dexsuite reward helpers so base-class terms
# (object_ee_distance, position/orientation tracking, success) use our
# multi-filter-tolerant force readers ({num_envs, num_filters, 3} matrices).
force_utils.patch_dexsuite_mdp()

from .camera_cfg import (
    BaseTiledCameraCfg,
    DuoCameraObservationsCfg,
    SingleCameraObservationsCfg,
    StateObservationCfg,
    WristTiledCameraCfg,
)
FINGERTIP_LIST = ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]
THUMB_SENSOR = "thumb_fingertip_object_s"
FINGER_SENSORS = [f"{name}_object_s" for name in FINGERTIP_LIST if name != "thumb_fingertip"]

# wrist camera used to sit on /Robot/ee_link/palm_link/Camera for Allegro.
# LEAP wrist camera position on palm_lower is the LEAP hand palm (a_0..a_15).


@configclass
class KukaLeapSceneCfg(PresetCfg):
    @configclass
    class KukaLeapSceneCfg(dexsuite.SceneCfg):
        """Kuka LEAP participant scene for Dexsuite Lifting/Reorientation"""

        robot: ArticulationCfg = KUKA_LEAP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        base_camera: CameraCfg | None = None

        wrist_camera: CameraCfg | None = None

        def __post_init__(self: dexsuite.SceneCfg):
            super().__post_init__()
            for link_name in FINGERTIP_LIST:
                setattr(
                    self,
                    f"{link_name}_object_s",
                    ContactSensorCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
                        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                        # URDF importer nests body prims; stock PhysX contact
                        # view can't resolve deeply nested leaves without an
                        # explicit '**' recursive component.
                        class_type="LEAP_Isaaclab.tasks.kuka_leap_dexsuite.contact_sensor:NestedContactSensor",
                    ),
                )

    default = KukaLeapSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    single_camera = default.replace(base_camera=BaseTiledCameraCfg())
    duo_camera = default.replace(base_camera=BaseTiledCameraCfg(), wrist_camera=WristTiledCameraCfg())


@configclass
class KukaLeapRelJointPosActionCfg:
    action = dexsuite_mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaLeapReorientRewardCfg(dexsuite.RewardsCfg):
    good_finger_contact = RewTerm(
        func=force_utils.contacts,
        weight=0.5,
        params={"threshold": 0.1, "thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
    )

    contact_count = RewTerm(
        func=force_utils.contact_count,
        weight=1.0,
        params={
            "threshold": 0.01,
            "sensor_names": FINGER_SENSORS + [THUMB_SENSOR],
        },
    )

    def __post_init__(self: dexsuite.RewardsCfg):
        super().__post_init__()
        # LEAP fingers: palm_lower + fingertip/thumb_fingertip links
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["palm_lower", "fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]
        )
        self.fingers_to_object.params["thumb_name"] = THUMB_SENSOR
        self.fingers_to_object.params["finger_names"] = FINGER_SENSORS
        self.position_tracking.params["thumb_name"] = THUMB_SENSOR
        self.position_tracking.params["finger_names"] = FINGER_SENSORS
        if self.orientation_tracking:
            self.orientation_tracking.params["thumb_name"] = THUMB_SENSOR
            self.orientation_tracking.params["finger_names"] = FINGER_SENSORS
        self.success.params["thumb_name"] = THUMB_SENSOR
        self.success.params["finger_names"] = FINGER_SENSORS


@configclass
class KukaLeapObservationCfg(PresetCfg):
    state = StateObservationCfg()
    single_camera = SingleCameraObservationsCfg()
    duo_camera = DuoCameraObservationsCfg()
    default = state


@configclass
class KukaLeapMixinCfg:
    scene: KukaLeapSceneCfg = KukaLeapSceneCfg()
    rewards: KukaLeapReorientRewardCfg = KukaLeapReorientRewardCfg()
    observations: KukaLeapObservationCfg = KukaLeapObservationCfg()
    actions: KukaLeapRelJointPosActionCfg = KukaLeapRelJointPosActionCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class DexsuiteKukaLeapReorientEnvCfg(KukaLeapMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteKukaLeapReorientEnvCfg_PLAY(KukaLeapMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


# Table geometry of the official dexsuite scene: kinematic slab, center z = 0.235 m, 0.04 m thick.
TABLE_TOP_Z = 0.235 + 0.04 / 2
CUBE_SIZE = 0.05
CUBE_REST_Z = TABLE_TOP_Z + CUBE_SIZE / 2  # cube center height when resting flat on the table


def _apply_cube_pick_overrides(env_cfg):
    """Turn the Dexsuite lift task into a table-top cube pick-up task (Lift + Lift-Play).

    Overrides to the stock Dexsuite lift task:
    * The object is a fixed 5 cm cube (stock: one random shape from a
      cuboid/sphere/capsule/cone catalog) that spawns RESTING ON the table with
      a flat orientation (stock: floats up to 0.4 m above the table with
      random roll/pitch/yaw).
    * Gravity is always on (stock: zero-g start ramped to Earth gravity by the
      ADR curriculum; the ``gravity_adr`` term is disabled here).
    * The success-colored table proxy markers and the observation point-cloud
      debug markers are removed so the scene shows only robot, table and cube.
    """
    # -- scene: fixed cube resting on the table, in every scene preset
    # (default/single_camera/duo_camera); only the picked one survives preset
    # resolution, and these mutations stay local to this task
    for scene_preset in (env_cfg.scene.default, env_cfg.scene.single_camera, env_cfg.scene.duo_camera):
        scene_preset.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CuboidCfg(
                size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, CUBE_REST_Z)),
        )

    # -- reset: cube stays on the table (small x/y jitter, drop of at most
    # 2 cm onto the surface, flat orientation, random heading)
    env_cfg.events.reset_object.params["pose_range"] = {
        "x": [-0.05, 0.05],
        "y": [-0.05, 0.05],
        "z": [0.0, 0.02],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-3.14, 3.14],
    }

    # -- gravity: always on; drop the stock zero-g ADR ramp
    env_cfg.events.variable_gravity.params["gravity_distribution_params"] = (
        (0.0, 0.0, -9.81),
        (0.0, 0.0, -9.81),
    )
    if env_cfg.curriculum is not None:
        env_cfg.curriculum.gravity_adr = None

    # -- visuals: no success-colored table proxy, no red point-cloud markers
    env_cfg.commands.object_pose.success_visualizer_cfg.markers = {}
    for obs_preset in (
        env_cfg.observations.default,
        env_cfg.observations.state,
        env_cfg.observations.single_camera,
        env_cfg.observations.duo_camera,
    ):
        obs_preset.perception.object_point_cloud.params["visualize"] = False


@configclass
class DexsuiteKukaLeapLiftEnvCfg(KukaLeapMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    """Kuka-LEAP pick-up task: grasp the cube on the table and lift it to the goal position."""

    def __post_init__(self):
        super().__post_init__()
        _apply_cube_pick_overrides(self)


@configclass
class DexsuiteKukaLeapLiftEnvCfg_PLAY(KukaLeapMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    """Evaluation variant of the Kuka-LEAP cube pick-up task (max difficulty, goal markers)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_cube_pick_overrides(self)
