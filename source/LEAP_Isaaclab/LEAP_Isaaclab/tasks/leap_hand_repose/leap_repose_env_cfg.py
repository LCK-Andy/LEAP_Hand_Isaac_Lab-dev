# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on the official Isaac Lab in-hand reorientation task:
# https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/allegro_env_cfg.py
# --------------------------------------------------------

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import LEAP_Isaaclab.compat.inhand.inhand_env_cfg as inhand_env_cfg

from .mdp import (
    fingertip_object_dist_sq,
    fingertip_overcurl,
    joint_energy,
    object_away_from_hand,
    object_below_hand,
)

##
# Pre-defined configs
##
from LEAP_Isaaclab.assets import LEAP_HAND_CFG  # isort: skip

# LEAP fingertip bodies (same names as the direct-task config)
LEAP_FINGERTIP_BODY_NAMES = ["fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"]


@configclass
class LeapCubeEnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
    """In-hand cube reorientation with the LEAP hand.

    Identical to ``Isaac-Repose-Cube-Allegro-v0`` except that the Allegro hand is
    substituted with the 16-DOF LEAP hand (right, v1). All MDP terms (actions,
    observations, events, rewards, terminations, commands) are inherited unchanged
    from the official environment since they address the robot via regex joint names.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to LEAP hand
        self.scene.robot = LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # enable clone in fabric
        self.scene.clone_in_fabric = True
        # place the cube in the LEAP palm: the LEAP palm center sits closer to the
        # hand origin than the Allegro's (-0.19 -> -0.10 along y in the world frame)
        self.scene.object.init_state.pos = (0.0, -0.10, 0.56)

        # rewards: keep the two official task terms (success_bonus and
        # track_orientation_inv_l2, i.e. terms 1-2 of the reference reward) and
        # add the five shaping terms of the 7-term in-hand reorientation reward
        # documented in mdp/rewards.py. Weights c3..c7 chosen so each penalty
        # stays small next to the dense orientation reward; zero a weight to
        # disable a term.
        # term 3: keep the fingertips close to the object
        self.rewards.fingertip_object_dist = RewTerm(
            func=fingertip_object_dist_sq,
            weight=-1.0,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg("robot", body_names=LEAP_FINGERTIP_BODY_NAMES),
            },
        )
        # term 4: joint energy |qdot|^T |tau| (replaces the official joint_vel_l2)
        self.rewards.joint_energy = RewTerm(
            func=joint_energy,
            weight=-0.001,
            params={"robot_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        self.rewards.joint_vel_l2 = None
        # term 5: object pushed away from the hand; 0.25 m leaves a warning band
        # below the 0.3 m object_out_of_reach termination
        self.rewards.object_away = RewTerm(
            func=object_away_from_hand,
            weight=-1.0,
            params={
                "dist_threshold": 0.25,
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        # term 6: no table in this scene and dropping the object already
        # terminates the episode, so this is disabled by default (set a
        # negative weight for an extra terminal fall penalty)
        self.rewards.object_drop = RewTerm(
            func=object_below_hand,
            weight=0.0,
            params={
                "z_offset": -0.05,
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        # term 7: discourage over-curling the fingers past the object (which
        # then rests on the penultimate joints); the fingertip z offset is
        # relative to the hand root, with 0.0 = the palm-base plane
        self.rewards.fingertip_overcurl = RewTerm(
            func=fingertip_overcurl,
            weight=-1.0,
            params={
                "z_offset": 0.0,
                "robot_cfg": SceneEntityCfg("robot", body_names=LEAP_FINGERTIP_BODY_NAMES),
            },
        )


@configclass
class LeapCubeEnvCfg_PLAY(LeapCubeEnvCfg):
    """Play configuration of the LEAP cube reorientation environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        self.terminations.time_out = None
