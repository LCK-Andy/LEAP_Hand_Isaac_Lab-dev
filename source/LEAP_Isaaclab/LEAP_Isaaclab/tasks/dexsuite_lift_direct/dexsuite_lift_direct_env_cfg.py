"""Configuration for the direct-RL port of the official Dexsuite KUKA+Allegro Lift task.

Mirrors ``isaaclab_tasks.manager_based.manipulation.dexsuite`` (KukaAllegro
scene + Lift env cfg): same robot asset, table, goal ranges, reward weights,
ADR curriculum, observation content, and per-fingertip ContactSensors --
restructured into a DirectRLEnv.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from isaaclab_physx.physics import PhysxCfg

##
# Scene
##

FINGERTIP_LINKS = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]

OBJECT_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(0.05, 0.1, 0.1),
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=0,
        disable_gravity=False,
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
    mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
)

TABLE_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(0.8, 1.5, 0.04),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    collision_props=sim_utils.CollisionPropertiesCfg(),
)


@configclass
class DexsuiteLiftDirectSceneCfg(InteractiveSceneCfg):
    """KUKA+Allegro tabletop scene of the official dexsuite task (no sensors)."""

    num_envs: int = 1024
    env_spacing: float = 3.0
    replicate_physics: bool = True

    robot: ArticulationCfg = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=OBJECT_SPAWN_CFG,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
    )

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=TABLE_SPAWN_CFG,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.235)),
    )

    plane = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())

    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
        ),
    )

    # per-fingertip contact sensors filtered against the object (official scene;
    # only the object is ever in reach, so the single-column force matrix is the
    # net contact force per tip)
    index_link_3_object_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/index_link_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    middle_link_3_object_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/middle_link_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    ring_link_3_object_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/ring_link_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    thumb_link_3_object_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/thumb_link_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )


@configclass
class DexsuiteKukaAllegroLiftDirectEnvCfg(DirectRLEnvCfg):
    """Direct-RL Lift task: pick the object off the table and lift it to the goal pose."""

    # 50 Hz control (decimation 2 at 120 Hz physics), 6 s episodes
    decimation: int = 2
    episode_length_s: float = 6.0

    # 23 joint-position deltas (KUKA 7 + Allegro 16); relative control with scale 0.1
    action_space: int = 23
    # 157 per step x 5-step history -- see the env module for the breakdown
    observation_space: int = 785

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        # NOTE: the PhysX GPU buffer fields are defined but unread in this Isaac Lab
        # beta (same caveat as the LEAP repose tasks); kept for forward compatibility.
        physics=PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_max_rigid_patch_count=4 * 5 * 2**15,
            gpu_found_lost_pairs_capacity=2**26,
        ),
    )

    scene: DexsuiteLiftDirectSceneCfg = DexsuiteLiftDirectSceneCfg()

    fingertip_links: list[str] = FINGERTIP_LINKS

    ##
    # Actions (official RelativeJointPositionAction semantics)
    ##
    action_scale: float = 0.1
    # EMA smoothing of the applied action (flash_rl FLASH_RL_ACT_MOVING_AVERAGE knob;
    # 1.0 disables, the official dexsuite task applies no smoothing)
    act_moving_average: float = 1.0

    ##
    # Observations
    ##
    history_length: int = 5
    # observation noise upper bounds at max ADR difficulty (official ADR final values)
    joint_pos_noise: float = 0.1
    joint_vel_noise: float = 0.2
    tip_state_noise: float = 0.01
    object_quat_noise: float = 0.03

    ##
    # Contact gating (official per-fingertip force thresholds; thumb AND at
    # least one finger above the threshold counts as good contact)
    ##
    fingers_to_object_contact_threshold: float = 1.0
    good_finger_contact_threshold: float = 0.1
    contact_count_threshold: float = 0.01
    success_contact_threshold: float = 0.01

    ##
    # Goal command (robot-base frame, official ObjectUniformPoseCommand ranges)
    ##
    goal_pos_x: tuple[float, float] = (-0.7, -0.3)
    goal_pos_y: tuple[float, float] = (-0.25, 0.25)
    goal_pos_z: tuple[float, float] = (0.55, 0.95)
    goal_resample_time_range: tuple[float, float] = (2.0, 3.0)

    ##
    # Resets (official reset-mode event ranges)
    ##
    object_init_pos: tuple[float, float, float] = (-0.55, 0.1, 0.35)
    object_reset_pos_xy: float = 0.2
    object_reset_pos_z: tuple[float, float] = (0.0, 0.4)
    joint_reset_range: float = 0.5
    wrist_joint_reset_range: float = 3.0

    ##
    # Rewards (official term weights; each term is multiplied by the step dt)
    ##
    action_l2_scale: float = -0.005
    action_rate_l2_scale: float = -0.005
    fingers_to_object_scale: float = 1.0
    fingers_to_object_std: float = 0.4
    good_finger_contact_scale: float = 0.5
    contact_count_scale: float = 1.0
    position_tracking_scale: float = 2.0
    position_tracking_std: float = 0.2
    success_scale: float = 10.0
    success_pos_std: float = 0.1
    abnormal_termination_scale: float = -1.0

    ##
    # ADR curriculum (official DifficultyScheduler: +1 on episode success, -1 otherwise)
    ##
    adr_init_difficulty: int = 0
    adr_min_difficulty: int = 0
    adr_max_difficulty: int = 10

    ##
    # Terminations (official out_of_bound ranges, relative to env origins)
    ##
    object_out_of_bound_x: tuple[float, float] = (-1.5, 0.5)
    object_out_of_bound_y: tuple[float, float] = (-2.0, 2.0)
    object_out_of_bound_z: tuple[float, float] = (0.0, 2.0)


@configclass
class DexsuiteKukaAllegroLiftDirectEnvCfg_PLAY(DexsuiteKukaAllegroLiftDirectEnvCfg):
    """Eval variant: ADR pinned at max difficulty (full gravity + full obs noise)."""

    def __post_init__(self):
        super().__post_init__()
        self.adr_init_difficulty = self.adr_max_difficulty
