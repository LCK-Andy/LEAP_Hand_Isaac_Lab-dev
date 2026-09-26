"""Stage B adds safe fixed-hand pose tracking without changing the Stage A task."""

from isaaclab.utils import configclass

from ..leap_hand_grasp.leap_grasp_env_cfg import LeapGraspEnvCfg
from .motion_logic import MotionCriteria, validate_motion_criteria


@configclass
class LeapGraspMotionEnvCfg(LeapGraspEnvCfg):
    """Small multi-axis reference motion driven only by the hand's joint actions."""

    observation_space = 148  # Exact 124-channel Stage A prefix plus 24 command features.
    motion: MotionCriteria = MotionCriteria()

    def __post_init__(self):
        super().__post_init__()
        validate_motion_criteria(self.motion)
        if (
            self.episode_length_s
            <= self.motion.settle_seconds + self.motion.ramp_seconds + self.motion.success_hold_seconds
        ):
            raise ValueError("Stage B requires an active motion window beyond settling and the reference ramp")
        if sum(x * x for x in self.motion.translation_amplitude) ** 0.5 >= self.grasp.retention_radius * 0.5:
            raise ValueError("Translation references must remain well inside the grasp retention region")


@configclass
class LeapGraspMotionEnvCfg_PLAY(LeapGraspMotionEnvCfg):
    """The identical finite-horizon MDP with fewer evaluation environments."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
