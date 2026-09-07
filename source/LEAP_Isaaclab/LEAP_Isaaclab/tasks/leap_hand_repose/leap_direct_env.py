# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Direct in-hand cube reorientation with the LEAP hand, using the 7-term
# reference reward function (see the manager-based sibling mdp/rewards.py):
#
#   r_t = c1 * 1(task successful)                (reach_goal_bonus)
#       + c2 / (|theta_t| + eps_theta)           (rot_reward_scale / rot_eps)
#       + c3 * sum_i ||p^f_i - p^o||^2           (fingertip_dist_penalty_scale)
#       + c4 * |qdot_t|^T |tau_t|                (joint_energy_penalty_scale)
#       + c5 * 1(||p^o||^2 > p_bar)              (object_away_penalty_scale)
#       + c6 * 1(object contacts the table)      (object_drop_penalty_scale)
#       + c7 * sum_i 1(fingertip over-curled)    (fingertip_overcurl_penalty_scale)
#
# Observations, actions, resets, and goal bookkeeping are inherited unchanged
# from the official shared InHandManipulationEnv; only _get_rewards differs.
# --------------------------------------------------------

from __future__ import annotations

import torch

from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import (
    InHandManipulationEnv,
    rotation_distance,
)

from .leap_direct_env_cfg import LeapHandDirectEnvCfg


class LeapHandDirectEnv(InHandManipulationEnv):
    """InHandManipulationEnv with the LEAP hand and the reference reward function.

    Identical to the official ``Isaac-Repose-Cube-LEAP-Direct-v0`` environment
    except for the reward: the official dense object-position term is disabled
    by default (``dist_reward_scale = 0``) and the five shaping terms of the
    reference reward are added. The official behavior is restored by setting
    ``dist_reward_scale = -10.0`` and zeroing the new scales in the cfg.
    """

    cfg: LeapHandDirectEnvCfg

    def _get_rewards(self) -> torch.Tensor:
        hand_root_pos = self.hand.data.root_pos_w.torch - self.scene.env_origins
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_leap_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
            # reference shaping terms
            self.fingertip_pos,
            self.hand_dof_vel,
            self.hand.data.applied_torque.torch,
            hand_root_pos,
            self.cfg.fingertip_dist_penalty_scale,
            self.cfg.joint_energy_penalty_scale,
            self.cfg.object_away_penalty_scale,
            self.cfg.object_away_dist,
            self.cfg.object_drop_penalty_scale,
            self.cfg.object_drop_z_offset,
            self.cfg.fingertip_overcurl_penalty_scale,
            self.cfg.fingertip_overcurl_z_offset,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward


@torch.jit.script
def compute_leap_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
    fingertip_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    applied_torque: torch.Tensor,
    hand_root_pos: torch.Tensor,
    fingertip_dist_penalty_scale: float,
    joint_energy_penalty_scale: float,
    object_away_penalty_scale: float,
    object_away_dist: float,
    object_drop_penalty_scale: float,
    object_drop_z_offset: float,
    fingertip_overcurl_penalty_scale: float,
    fingertip_overcurl_z_offset: float,
):
    """Official in-hand reward extended with the five reference shaping terms.

    All positions are relative to the env origins. ``fingertip_pos`` is
    (num_envs, num_fingertips, 3) and ``object_pos``/``hand_root_pos`` are
    (num_envs, 3).
    """
    goal_dist = torch.linalg.norm(object_pos - target_pos, ord=2, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale
    action_penalty = torch.sum(actions**2, dim=-1)

    # term 3: sum of squared fingertip-to-object distances
    fingertip_dist = torch.sum(torch.sum((fingertip_pos - object_pos.unsqueeze(1)) ** 2, dim=2), dim=1)
    # term 4: mechanical joint energy |qdot|^T |tau|
    joint_energy = torch.sum(torch.abs(joint_vel) * torch.abs(applied_torque), dim=-1)
    # term 5: object pushed away from the in-hand position
    object_away = (goal_dist > object_away_dist).to(action_penalty.dtype)
    # term 6: object dropped below the hand (no table in this scene)
    object_drop = ((object_pos[:, 2] - hand_root_pos[:, 2]) < object_drop_z_offset).to(action_penalty.dtype)
    # term 7: fingertips over-curled below the palm-base plane (the LEAP palm is
    # vertical: fingers extend along world +z and curl back toward the palm base
    # through the hand root; tips tucked below that plane are closed past the
    # object, which then rests on the penultimate joints -- see mdp/rewards.py)
    fingertip_overcurl = torch.sum(
        (fingertip_pos[:, :, 2] - hand_root_pos[:, 2:3] < fingertip_overcurl_z_offset).to(action_penalty.dtype), dim=1
    )

    # Total reward is: position distance + orientation alignment + action regularization
    # + the reference shaping terms + success bonus + fall penalty
    reward = (
        dist_rew
        + rot_rew
        + action_penalty * action_penalty_scale
        + fingertip_dist * fingertip_dist_penalty_scale
        + joint_energy * joint_energy_penalty_scale
        + object_away * object_away_penalty_scale
        + object_drop * object_drop_penalty_scale
        + fingertip_overcurl * fingertip_overcurl_penalty_scale
    )

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes
