# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on the official Isaac Lab direct in-hand manipulation task:
# https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/allegro_hand/allegro_hand_env_cfg.py
#
# The DirectRLEnv class is reused from isaaclab_tasks (InHandManipulationEnv) --
# it is robot-agnostic and fully driven by this config. Only the hand is
# substituted: LEAP hand (16 DOF, joints a_0..a_15) instead of Allegro, and the
# cube spawn is moved to the LEAP palm center (y offset -0.17 -> -0.10).
# --------------------------------------------------------

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from LEAP_Isaaclab.assets import LEAP_HAND_CFG  # isort: skip


@configclass
class ObjectCfg(PresetCfg):
    physx = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.10, 0.56), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    newton_mjwarp = ArticulationCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.10, 0.565), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
        articulation_root_prim_path="",
    )
    ovphysx = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.10, 0.56), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    default = physx


@configclass
class PhysicsCfg(PresetCfg):
    physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=80,
            nconmax=70,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    ovphysx = OvPhysxCfg()
    default = physx


@configclass
class LeapHandDirectEnvCfg(DirectRLEnvCfg):
    """Direct (DirectRLEnv) in-hand cube reorientation with the LEAP hand.

    Identical to ``Isaac-Repose-Cube-Allegro-Direct-v0`` except for the hand
    substitution and the reward. The env class is ``LeapHandDirectEnv`` (this
    repo): the official reward is restructured into the 7-term reference reward
    documented in ``leap_direct_env.py`` -- the official dense object-position
    term is disabled by default and five shaping terms are added.
    """

    # env
    decimation = 4
    episode_length_s = 10.0
    action_space = 16
    observation_space = 136  # 124 official "full" + 12 pulp contact-force channels (4 tips x 3)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physics=PhysicsCfg(),
    )
    # robot
    robot_cfg: ArticulationCfg = LEAP_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    actuated_joint_names = [f"a_{i}" for i in range(16)]
    fingertip_body_names = [
        "fingertip",
        "thumb_fingertip",
        "fingertip_2",
        "fingertip_3",
    ]

    # in-hand object
    object_cfg: ObjectCfg = ObjectCfg()
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.2, 1.2, 1.2),
            )
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True
    )
    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = 0.0  # official dense object-position term disabled; set -10.0 to restore
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.24
    # reference shaping-term scales (see leap_direct_env.py for the term mapping)
    fingertip_dist_penalty_scale = -1.0  # term 3: sum ||fingertip - object||^2
    joint_energy_penalty_scale = -0.001  # term 4: |qdot|^T |tau|
    object_away_penalty_scale = -1.0  # term 5: object pushed away from the in-hand position
    object_away_dist = 0.20  # below fall_dist so the penalty fires before termination
    object_drop_penalty_scale = 0.0  # term 6 removed: no table; dropping terminates via fall_dist
    object_drop_z_offset = -0.05  # term 6: object z relative to the hand root
    fingertip_overcurl_penalty_scale = 0.0  # term 7 removed (user, 2026-09-08); set -1.0 to restore
    fingertip_overcurl_z_offset = 0.0  # term 7: fingertip z rel hand root; 0.0 = palm-base plane
    palm_contact_dist = 0.06  # term 9: object closer than this to the hand root counts as palm-pinned [m]
    palm_contact_penalty_scale = -0.1  # term 9 RESTORED (2026-09-08): guards the palm-crush degenerate mode
    # tactile terms 8/10 -- gated on MEASURED pulp contact forces (per-tip
    # ContactSensors filtered vs the in-hand object), i.e. exactly the signal
    # the real tactile pulps will report
    fingertip_contact_scale = 0.1  # term 8: reward per pulp with contact force above the threshold (max +0.4/step)
    fingertip_contact_force = 0.1  # term 8: contact force threshold per pulp [N]
    # term 10: goal-state press -- when the repose is DONE (within
    # goal_press_tolerance of the goal rotation) AND the object is NOT
    # palm-pinned, reward each pulp PRESSED above the force threshold:
    # finishing with every pulp preloaded on the object (tactile preload).
    # The palm-pinned gate makes palm-crushing earn ZERO press bonus (plus the
    # term-9 penalty) -- a symmetric fingertip grasp strictly dominates it.
    # Separate tolerance knob on purpose: the FlashSAC hold-hook overwrites
    # success_tolerance (positional arg 14) to -1 inside the base reward.
    goal_press_scale = 0.5  # per pressed pulp per step while at the goal (max +2.0/step)
    goal_press_tolerance = 0.2  # rotation-distance gate [rad]; keep == success_tolerance
    goal_press_force = 3.0  # press force threshold per pulp [N] (calibrated: random-action contacts are multi-N -- nonzero p10 ~2 N, median ~8.5 N)
    # per-pulp PhysX ContactSensors (filtered vs the in-hand object): tactile
    # emulation for the pulp sensors to be installed on the real hand; read
    # env.fingertip_contact_forces (N, 4, 3) [N] -- no observation-space change
    fingertip_contact_sensors = True
    vel_obs_scale = 0.2
    success_tolerance = 0.2
    max_consecutive_success = 0
    success_count_threshold: int = 1
    """Minimum number of goals reached in an episode to count it as a successful episode."""
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0
