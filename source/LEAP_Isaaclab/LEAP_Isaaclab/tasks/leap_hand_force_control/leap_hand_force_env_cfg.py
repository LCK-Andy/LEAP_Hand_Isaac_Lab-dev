# --------------------------------------------------------
# LEAP Hand Force Control Task
# Given desired fingertip positions (on a table surface) and
# desired normal contact forces, control the LEAP hand joints.
# --------------------------------------------------------

from isaaclab_physx.physics import PhysxCfg

from LEAP_Isaaclab.assets import LEAP_HAND_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.utils.configclass import configclass


@configclass
class LeapHandForceEnvCfg(DirectRLEnvCfg):
    """LEAP hand force control on a fixed table surface."""

    # env
    decimation = 4
    episode_length_s = 16.67  # 500 policy steps at 30 Hz
    action_space = 16
    hist_len = 3
    store_cur_actions = False
    observation_space = 192  # 64 per timestep x 3 history
    state_space = 0

    # default joint positions at episode start (same open hand pose as reorient)
    default_joint_pos: list = [
        [0.000, 0.500, 0.000, 0.000,
         -0.750, 1.300, 0.000, 0.750,
         1.750, 1.500, 1.750, 1.750,
         0.000, 1.000, 0.000, 0.000],
    ]

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
            gpu_max_rigid_patch_count=2**25,
        ),
    )

    # robot (original LEAP hand, no sensor pads)
    # NOTE: init_state rot is xyzw (scalar last) in Isaac Lab 3.0
    robot_cfg: ArticulationCfg = LEAP_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # contact sensor needs the PhysX contact reporter API on bodies
    robot_cfg.spawn.activate_contact_sensors = True
    # disable self-collision: fingertips pressing the table is the only contact we care about;
    # self-collision massively inflates PhysX contact patch pairs (caused patch buffer overflow)
    robot_cfg.spawn.articulation_props.enabled_self_collisions = False
    # stiffer PD gains than the reorient default (3.0/0.1): fingertips must sustain 1-5 N of
    # normal force against the table. 50/2.0 matches the ADR ceiling and is closer to real XC330.
    robot_cfg.actuators["fingers"].stiffness = 50.0
    robot_cfg.actuators["fingers"].damping = 2.0
    actuated_joint_names = [
        'a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'a_6', 'a_7',
        'a_8', 'a_9', 'a_10', 'a_11', 'a_12', 'a_13', 'a_14', 'a_15',
    ]
    fingertip_body_names = [
        'fingertip',
        'thumb_fingertip',
        'fingertip_2',
        'fingertip_3',
    ]

    # table: small fixed kinematic slab below the fingertips (visual + collision same size;
    # centered on the default-pose fingertip landing area ~(-0.06, -0.13, 0.54) env-local)
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=CuboidCfg(
            size=(0.12, 0.12, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.5, 0.6)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.06, -0.13, 0.535), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # contact sensors: net force between each fingertip and the table only.
    # (Isaac Lab 3.0: bodies selected via leaf-name regex in prim_path; filter reporting
    # is one-to-many only, so each fingertip gets its own sensor. ".*" regex is converted
    # to "*" glob internally.)
    contact_sensor_cfgs: list = [
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{name}",
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=["/World/envs/env_.*/table"],
        )
        for name in [
            "fingertip",
            "thumb_fingertip",
            "fingertip_2",
            "fingertip_3",
        ]
    ]

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=False)

    # task parameters
    force_target_range = (1.0, 5.0)  # desired normal force per fingertip [N]
    target_pos_noise = 0.015  # +/- m around default fingertip position (reachability by construction)
    force_reward_scale = 0.5
    pos_reward_scale = 2.0
    action_penalty_scale = -0.0002
    success_tolerance = 0.15  # fraction of target force error tolerated
    success_window = 50  # last N policy steps must be within tolerance

    # control (same as reorient: relative position deltas + moving average)
    action_type = "relative"
    act_moving_average = 1.0 / 24.0
