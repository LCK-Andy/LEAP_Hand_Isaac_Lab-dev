# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Reward shaping terms for the LEAP in-hand reorientation task, following the
# 7-term reward function of dexterous in-hand reorientation work:
#
#   r_t = c1 * 1(task successful)                (official success_bonus term)
#       + c2 / (|theta_t| + eps_theta)           (official track_orientation_inv_l2 term)
#       + c3 * sum_i ||p^f_i - p^o||^2           (fingertip_object_dist_sq, below)
#       + c4 * |qdot_t|^T |tau_t|                (joint_energy, below)
#       + c5 * 1(||p^o||^2 > p_bar)              (object_away_from_hand, below)
#       + c6 * 1(object contacts the table)      (object_below_hand, below)
#       + c7 * sum_i 1(fingertip over-curled)    (fingertip_overcurl, below)
#
# with c1, c2 > 0 and c3..c7 < 0. The first two terms are inherited from the
# official in-hand MDP; this module implements the remaining five.
# --------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fingertip_object_dist_sq(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sum of squared distances between each fingertip and the object center.

    Implements ``sum_{i=1}^{G} ||p^f_i - p^o||^2`` over the fingertip bodies in
    ``robot_cfg.body_names``. With a negative weight it penalizes grasps that
    let the fingertips drift away from the object.

    Args:
        env: The environment object.
        object_cfg: The object scene entity. Default is "object".
        robot_cfg: The robot scene entity with ``body_names`` set to the fingertips.
    """
    object_pos = env.scene[object_cfg.name].data.root_pos_w.torch
    fingertip_pos = env.scene[robot_cfg.name].data.body_pos_w.torch[:, robot_cfg.body_ids]
    return torch.sum(torch.sum((fingertip_pos - object_pos.unsqueeze(1)) ** 2, dim=-1), dim=-1)


def joint_energy(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mechanical energy dissipated by the joints: ``|qdot|^T |tau|``.

    Implements ``sum_j |qdot_j| * |tau_j|`` over the joints in ``robot_cfg.joint_names``.
    With a negative weight it penalizes fast, forceful joint motion.

    Note:
        With implicit actuators PhysX does not expose the applied joint forces;
        ``applied_torque`` holds the approximate PD torque recomputed by the
        actuator model, which is exact for the LEAP hand's ImplicitActuatorCfg.

    Args:
        env: The environment object.
        robot_cfg: The robot scene entity.
    """
    data = env.scene[robot_cfg.name].data
    joint_ids = robot_cfg.joint_ids
    return torch.sum(
        torch.abs(data.joint_vel.torch[:, joint_ids]) * torch.abs(data.applied_torque.torch[:, joint_ids]), dim=-1
    )


def object_away_from_hand(
    env: ManagerBasedRLEnv,
    dist_threshold: float = 0.25,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Indicator for the object being pushed away from the hand.

    Implements ``1(||p^o - p^hand||^2 > p_bar)``: 1.0 while the object is
    farther than ``dist_threshold`` meters from the robot root, else 0.0. With
    a negative weight it penalizes losing control of the object before the
    ``object_out_of_reach`` termination fires.

    Args:
        env: The environment object.
        dist_threshold: Distance from the hand root above which the penalty is
            active. Keep below the 0.3 m out-of-reach termination threshold.
        object_cfg: The object scene entity. Default is "object".
        robot_cfg: The robot scene entity.
    """
    object_pos = env.scene[object_cfg.name].data.root_pos_w.torch
    hand_pos = env.scene[robot_cfg.name].data.root_pos_w.torch
    dist_sq = torch.sum((object_pos - hand_pos) ** 2, dim=-1)
    return (dist_sq > dist_threshold**2).float()


def object_below_hand(
    env: ManagerBasedRLEnv,
    z_offset: float = -0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Indicator for the object having dropped below the hand.

    Stand-in for the reference's object-table contact penalty: this scene has
    no table, and dropping the object already terminates the episode via
    ``object_out_of_reach``, so the term can only act as a terminal penalty.
    It is 1.0 when the object center is lower than ``z_offset`` meters below
    the robot root (world-frame z), else 0.0.

    Args:
        env: The environment object.
        z_offset: Object height relative to the hand root below which the
            indicator is active (negative, e.g. -0.05).
        object_cfg: The object scene entity. Default is "object".
        robot_cfg: The robot scene entity.
    """
    object_z = env.scene[object_cfg.name].data.root_pos_w.torch[:, 2]
    hand_z = env.scene[robot_cfg.name].data.root_pos_w.torch[:, 2]
    return (object_z - hand_z < z_offset).float()


def fingertip_overcurl(
    env: ManagerBasedRLEnv,
    z_offset: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Count of fingertips over-curled below the palm-base plane.

    Reference counterpart of the fingertip-height (penultimate-joint) penalty
    ``sum_i 1(p^f_{i,z} > p_bar_z)``. In this task's frame the LEAP palm is
    vertical: the fingers extend along the world +z axis (open hand: tips at
    +0.04..+0.11 m above the hand root; cube center at +0.07 m) and curl back
    toward the palm base, whose plane passes through the (fixed) hand root.
    Curling far enough that a fingertip drops below that plane (measured:
    tips reach -0.017..+0.009 m at the joint limits) means the finger is
    closed past the object, which rests on the penultimate joints instead of
    the fingertips.

    Args:
        env: The environment object.
        z_offset: Fingertip height relative to the hand root below which a
            fingertip counts as over-curled (0.0 = the palm-base plane).
        robot_cfg: The robot scene entity with ``body_names`` set to the fingertips.
    """
    fingertip_z = env.scene[robot_cfg.name].data.body_pos_w.torch[:, robot_cfg.body_ids, 2]
    hand_z = env.scene[robot_cfg.name].data.root_pos_w.torch[:, 2].unsqueeze(-1)
    return torch.sum((fingertip_z - hand_z < z_offset).float(), dim=-1)
