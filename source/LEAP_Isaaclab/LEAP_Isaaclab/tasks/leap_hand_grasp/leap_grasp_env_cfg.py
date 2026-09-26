"""Stage A: persistent four-pulp grasp of a fixed cube."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from ..leap_hand_repose.leap_direct_env_cfg import LeapHandDirectEnvCfg, ObjectCfg
from .grasp_logic import GraspCriteria


@configclass
class LeapGraspEnvCfg(LeapHandDirectEnvCfg):
    """Force-regulated grasp, with no random point or orientation goals."""

    observation_space = 124
    episode_length_s = 10.0
    is_finite_horizon = True
    compute_final_obs = True
    grasp: GraspCriteria = GraspCriteria()
    hold_duration = 2.0
    max_joint_delta = 0.015  # radians per 30 Hz control step, not per physics substep
    force_observation_scale = 10.0  # N per observation unit
    fall_penalty = -5.0
    unsafe_penalty = -5.0
    cube_half_extent = 0.036  # measured loaded DexCube world extent is 0.072 m at scale 1.2
    object_mass = 0.216
    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.0
    pregrasp_fixture = str(Path(__file__).with_name("pregrasp.json"))
    pulp_contact_tolerance = 0.004
    pulp_normal_alignment = 0.4
    force_contact_threshold = 0.05
    object_cfg = ObjectCfg()
    allow_unvalidated_backend = False

    def __post_init__(self):
        self.scene.num_envs = 256
        for name in ("physx", "newton_mjwarp", "ovphysx"):
            obj = getattr(self.object_cfg, name)
            obj.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=self.object_mass)
        self.sim.physics.physx.gpu_max_rigid_patch_count = 2**19
        self.sim.physics.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.use_newton_actuators = False
        self.sim.physics.newton_mjwarp.solver_cfg.use_mujoco_contacts = True
        self.sim.physics.newton_mjwarp.collision_cfg = None


@configclass
class LeapGraspEnvCfg_PLAY(LeapGraspEnvCfg):
    """Identical finite-horizon MDP, with a smaller evaluation batch."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
