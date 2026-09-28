# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Simulator-free tensor port of the *released-source* DexPBT reorientation task.

Source: IsaacGymEnvs 665c32170d84b4be66722eea405a1e08b6e7f761 (July 2023),
``tasks/allegro_kuka/{allegro_kuka_base,allegro_kuka_reorientation,allegro_kuka_utils}.py``.
This is explicitly not a reconstructed paper-110D profile. Preserve source quirks,
including mixed control, 117 observations, stale bookkeeping observation channels,
accumulated (not consecutive) near-goal steps, and the unscaled hardcoded fall test.
All quaternions are xyzw. Positions/velocities use environment axes, not robot axes.
No Isaac Lab, simulator, package registration, or filesystem imports are required.
"""

import torch
from dataclasses import dataclass, replace
from torch import Tensor
from typing import Any, Mapping

SOURCE_COMMIT = "665c32170d84b4be66722eea405a1e08b6e7f761"
PROFILE = "isaacgymenvs-665c321-released"
OBSERVATION_DIM = 117
ACTION_DIM = 23

SOURCE_REWARD_FIELDS = {
    "distanceDeltaRewScale": "distance_delta_scale",
    "liftingRewScale": "lifting_scale",
    "liftingBonus": "lifting_bonus",
    "liftingBonusThreshold": "lifting_threshold",
    "keypointScale": "keypoint_scale",
    "keypointRewScale": "keypoint_reward_scale",
    "successTolerance": "success_tolerance",
    "successSteps": "success_steps",
    "reachGoalBonus": "reach_goal_bonus",
    "kukaActionsPenaltyScale": "arm_velocity_scale",
    "allegroActionsPenaltyScale": "hand_velocity_scale",
    "maxConsecutiveSuccesses": "max_consecutive_successes",
    "episodeLength": "max_episode_length",
    "fallDistance": "fall_distance",
    "fallPenalty": "fall_penalty",
}
SOURCE_INACTIVE_FIELDS = frozenset(
    {"fallDistance", "fallPenalty", "resetRotationNoise", "stiffnessScale", "forceLimitScale"}
)


@dataclass
class RewardConfig:
    """Source coefficients; ``fall_distance`` and ``fall_penalty`` intentionally have no effect."""

    distance_delta_scale: float = 50.0
    lifting_scale: float = 20.0
    lifting_bonus: float = 300.0
    lifting_threshold: float = 0.15
    keypoint_scale: float = 1.5
    keypoint_reward_scale: float = 200.0
    success_tolerance: float = 0.075
    success_steps: int = 1
    reach_goal_bonus: float = 1000.0
    arm_velocity_scale: float = 0.003
    hand_velocity_scale: float = 0.0003
    max_consecutive_successes: int = 50
    max_episode_length: int = 600
    fall_distance: float = 0.24
    fall_penalty: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RewardConfig":
        """Read a source *reward* mapping; reject absent/unsupported source fields."""
        return reward_config_from_original(values)

    def to_source_mapping(self) -> dict[str, Any]:
        return {source: getattr(self, field) for source, field in SOURCE_REWARD_FIELDS.items()}


def reward_config_from_original(overrides: Mapping[str, Any], base: RewardConfig | None = None) -> RewardConfig:
    """Apply explicit camelCase source values without silently inventing new reward terms."""
    unknown = set(overrides) - SOURCE_REWARD_FIELDS.keys()
    if unknown:
        raise ValueError(f"Unsupported original reward fields: {sorted(unknown)}")
    return replace(base or RewardConfig(), **{SOURCE_REWARD_FIELDS[k]: v for k, v in overrides.items()})


@dataclass
class MDPState:
    """Per-environment reward memory; progress is incremented by the runtime before observations."""

    closest_fingertip_dist: Tensor
    furthest_hand_dist: Tensor
    closest_keypoint_max_dist: Tensor
    lifted_object: Tensor
    near_goal_steps: Tensor
    successes: Tensor
    progress_buf: Tensor


@dataclass
class Geometry:
    """Measured geometry, with source virtual fingertip and palm offsets already applied."""

    palm_center_pos: Tensor
    fingertip_pos_rel_palm: Tensor
    curr_fingertip_distances: Tensor
    keypoints_rel_palm: Tensor
    keypoints_rel_goal: Tensor
    keypoints_max_dist: Tensor


@dataclass
class RewardResult:
    reward: Tensor
    is_success: Tensor
    reset: Tensor
    timeout: Tensor
    terms: dict[str, Tensor]


def make_state(num_envs: int, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> MDPState:
    zeros = torch.zeros(num_envs, device=device, dtype=dtype)
    return MDPState(
        -torch.ones(num_envs, 4, device=device, dtype=dtype),
        -torch.ones_like(zeros),
        -torch.ones_like(zeros),
        zeros.to(torch.bool),
        zeros.to(torch.long),
        zeros.clone(),
        zeros.to(torch.long),
    )


def step_rewards(
    state: MDPState,
    geometry: Geometry,
    object_pos: Tensor,
    object_initial_z: Tensor,
    joint_vel: Tensor,
    cfg: RewardConfig,
) -> RewardResult:
    """Mutate source reward memory after observations; resets are deferred to the next control step.

    ``initialize_distances`` must have run first. No dt weighting, contact gates,
    implicit action penalties, or resets of near-goal counts on misses are applied.
    A terminal state must be reset before this function is called for its next step.
    """
    z_lift = 0.05 + object_pos[:, 2] - object_initial_z
    lifting = torch.clamp(z_lift, 0, 0.5)
    lifted = (z_lift > cfg.lifting_threshold) | state.lifted_object
    lift_bonus = cfg.lifting_bonus * (lifted & ~state.lifted_object)
    lifting *= ~lifted
    state.lifted_object.copy_(lifted)

    finger_deltas = state.closest_fingertip_dist - geometry.curr_fingertip_distances
    state.closest_fingertip_dist.copy_(torch.minimum(state.closest_fingertip_dist, geometry.curr_fingertip_distances))
    hand_deltas = state.furthest_hand_dist - geometry.curr_fingertip_distances[:, 0]
    state.furthest_hand_dist.copy_(torch.maximum(state.furthest_hand_dist, geometry.curr_fingertip_distances[:, 0]))
    finger_reward = torch.clamp(finger_deltas, 0, 10).sum(dim=-1) * ~lifted
    hand_penalty = torch.clamp(hand_deltas, -10, 0) * ~lifted * 4

    keypoint_deltas = state.closest_keypoint_max_dist - geometry.keypoints_max_dist
    state.closest_keypoint_max_dist.copy_(torch.minimum(state.closest_keypoint_max_dist, geometry.keypoints_max_dist))
    keypoint_reward = torch.clamp(keypoint_deltas, 0, 100) * lifted
    near_goal = geometry.keypoints_max_dist <= cfg.success_tolerance * cfg.keypoint_scale
    state.near_goal_steps.add_(near_goal)
    is_success = state.near_goal_steps >= cfg.success_steps
    state.successes.add_(is_success)

    terms = {
        "raw_fingertip_delta_rew": finger_reward.clone(),
        "raw_hand_delta_penalty": hand_penalty.clone(),
        "raw_lifting_rew": lifting.clone(),
        "raw_keypoint_rew": keypoint_reward.clone(),
        "fingertip_delta_rew": finger_reward * cfg.distance_delta_scale,
        "hand_delta_penalty": hand_penalty * (cfg.distance_delta_scale * 0),
        "lifting_rew": lifting * cfg.lifting_scale,
        "lift_bonus_rew": lift_bonus,
        "keypoint_rew": keypoint_reward * cfg.keypoint_reward_scale,
        "kuka_actions_penalty": -1 * (joint_vel[:, :7].abs().sum(dim=-1) * cfg.arm_velocity_scale),
        "allegro_actions_penalty": -1 * (joint_vel[:, 7:23].abs().sum(dim=-1) * cfg.hand_velocity_scale),
        "bonus_rew": near_goal * (cfg.reach_goal_bonus / cfg.success_steps),
    }
    reward = (
        terms["fingertip_delta_rew"]
        + terms["hand_delta_penalty"]
        + terms["lifting_rew"]
        + terms["lift_bonus_rew"]
        + terms["keypoint_rew"]
        + terms["kuka_actions_penalty"]
        + terms["allegro_actions_penalty"]
        + terms["bonus_rew"]
    )
    reset = object_pos[:, 2] < 0.1
    if cfg.max_consecutive_successes > 0:
        state.progress_buf.masked_fill_(is_success, 0)
        reset |= state.successes >= cfg.max_consecutive_successes
    reset |= state.progress_buf >= cfg.max_episode_length - 1
    reset |= geometry.curr_fingertip_distances.max(dim=-1).values > 1.5
    timeout = (state.progress_buf >= cfg.max_episode_length - 1) & reset
    return RewardResult(reward, is_success, reset, timeout, terms)


def quat_rotate_xyzw(quaternion: Tensor, vector: Tensor) -> Tensor:
    """Source quaternion-vector operation, including its unit-quaternion assumption."""
    xyz, w = quaternion[..., :3], quaternion[..., 3:4]
    return (
        vector * (2.0 * w.square() - 1.0)
        + torch.cross(xyz, vector, dim=-1) * w * 2.0
        + (xyz * (xyz * vector).sum(dim=-1, keepdim=True) * 2.0)
    )


def compute_geometry(
    object_state: Tensor,
    palm_state: Tensor,
    fingertip_states: Tensor,
    goal_pose: Tensor,
    object_scales: Tensor,
    object_base_size: float = 0.05,
    keypoint_scale: float = 1.5,
) -> Geometry:
    """Source observation geometry; fingertip order is index, middle, ring, thumb."""
    palm_offset = palm_state.new_tensor([0.0, -0.02, 0.16]).expand_as(palm_state[:, :3])
    palm_pos = palm_state[:, :3] + quat_rotate_xyzw(palm_state[:, 3:7], palm_offset)
    tip_offsets = fingertip_states.new_tensor([[0.05, 0.005, 0.0]] * 3 + [[0.06, 0.005, 0.0]])
    tip_offsets = tip_offsets.expand_as(fingertip_states[:, :, :3])
    tips = fingertip_states[:, :, :3] + quat_rotate_xyzw(fingertip_states[:, :, 3:7], tip_offsets)
    offsets = object_state.new_tensor([[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]])
    offsets = offsets.unsqueeze(0) * object_scales.unsqueeze(1) * (object_base_size * keypoint_scale / 2)
    obj_keypoints = object_state[:, None, :3] + quat_rotate_xyzw(object_state[:, None, 3:7].expand(-1, 4, -1), offsets)
    goal_keypoints = goal_pose[:, None, :3] + quat_rotate_xyzw(goal_pose[:, None, 3:7].expand(-1, 4, -1), offsets)
    rel_goal = obj_keypoints - goal_keypoints
    return Geometry(
        palm_pos,
        tips - palm_pos[:, None, :],
        torch.norm(tips - object_state[:, None, :3], dim=-1),
        obj_keypoints - palm_pos[:, None, :],
        rel_goal,
        torch.norm(rel_goal, dim=-1).max(dim=-1).values,
    )


def initialize_distances(state: MDPState, geometry: Geometry) -> None:
    """Initialize negative reset sentinels before packing observations, not after reward."""
    state.closest_fingertip_dist.copy_(
        torch.where(state.closest_fingertip_dist < 0, geometry.curr_fingertip_distances, state.closest_fingertip_dist)
    )
    state.furthest_hand_dist.copy_(
        torch.where(state.furthest_hand_dist < 0, geometry.curr_fingertip_distances[:, 0], state.furthest_hand_dist)
    )
    state.closest_keypoint_max_dist.copy_(
        torch.where(state.closest_keypoint_max_dist < 0, geometry.keypoints_max_dist, state.closest_keypoint_max_dist)
    )


def reset_state(state: MDPState, env_ids: Tensor, goal_only: bool = False) -> None:
    """Reset memory only; runtime must sample target/object/joint poses and save episode stats."""
    state.closest_keypoint_max_dist[env_ids] = -1
    state.near_goal_steps[env_ids] = 0
    if not goal_only:
        state.closest_fingertip_dist[env_ids] = -1
        state.furthest_hand_dist[env_ids] = -1
        state.lifted_object[env_ids] = False
        state.successes[env_ids] = 0
        state.progress_buf[env_ids] = 0


def pack_observations(
    joint_pos: Tensor,
    joint_vel: Tensor,
    joint_lower: Tensor,
    joint_upper: Tensor,
    palm_state: Tensor,
    object_state: Tensor,
    object_scales: Tensor,
    geometry: Geometry,
    state: MDPState,
) -> Tensor:
    """Pack 117 channels BEFORE reward mutation; current scaled reward is inserted separately.

    Source offsets: q[0:23], qdot[23:46], palm xyz[46:49], palm q/v/w[49:59],
    object q/v/w[59:69], tip-minus-palm[69:81], kp-minus-palm[81:93],
    kp-minus-goal[93:105], scales[105:108], best kp[108], best tips[109:113],
    lifted[113], log(progress/10+1)[114], log(successes+1)[115], reward*.01[116].
    Palm velocity is that of iiwa7_link_7, NOT corrected for the virtual palm offset.
    """
    n = joint_pos.shape[0]
    return torch.cat(
        (
            (2.0 * joint_pos - joint_upper - joint_lower) / (joint_upper - joint_lower),
            joint_vel,
            geometry.palm_center_pos,
            palm_state[:, 3:13],
            object_state[:, 3:13],
            geometry.fingertip_pos_rel_palm.reshape(n, 12),
            geometry.keypoints_rel_palm.reshape(n, 12),
            geometry.keypoints_rel_goal.reshape(n, 12),
            object_scales,
            state.closest_keypoint_max_dist[:, None],
            state.closest_fingertip_dist,
            state.lifted_object[:, None].to(joint_pos.dtype),
            torch.log(state.progress_buf / 10 + 1)[:, None],
            torch.log(state.successes + 1)[:, None],
            torch.zeros_like(state.successes[:, None]),
        ),
        dim=-1,
    )


def finish_observations(obs: Tensor, reward: Tensor, clamp: float = 10.0) -> Tensor:
    """Insert current raw reward *.01 and clamp in place, after all other observations."""
    obs[:, 116] = reward * 0.01
    if clamp > 0:
        obs.clamp_(-clamp, clamp)
    return obs


def tolerance_curriculum(
    last_curriculum_update: int,
    frames_since_restart: int,
    curriculum_interval: int,
    prev_episode_successes: Tensor,
    success_tolerance: float,
    initial_tolerance: float,
    target_tolerance: float,
    tolerance_curriculum_increment: float,
) -> tuple[float, int]:
    """Original cadence and >=3 mean-last-episode gate (not completed-episode window)."""
    if frames_since_restart - last_curriculum_update < curriculum_interval:
        return success_tolerance, last_curriculum_update
    if prev_episode_successes.mean() < 3.0:
        return success_tolerance, last_curriculum_update
    success_tolerance *= tolerance_curriculum_increment
    success_tolerance = min(success_tolerance, initial_tolerance)
    success_tolerance = max(success_tolerance, target_tolerance)
    return success_tolerance, frames_since_restart


def tolerance_successes_objective(
    success_tolerance: float, initial_tolerance: float, target_tolerance: float, successes: Tensor
) -> Tensor:
    """Original release objective; before final tolerance the successes multiplier is .01."""
    if initial_tolerance > target_tolerance:
        objective = (initial_tolerance - success_tolerance) / (initial_tolerance - target_tolerance)
    else:
        objective = 1.0
    if success_tolerance > target_tolerance:
        return successes * 0.01 + objective
    return successes + objective


def joint_targets(
    actions: Tensor,
    previous_targets: Tensor,
    lower: Tensor,
    upper: Tensor,
    dt: float = 0.01667,
    speed_scale: float = 10.0,
    moving_average: float = 1.0,
) -> Tensor:
    """Return source mixed targets; caller owns policy action clipping and target storage."""
    arm = previous_targets[:, :7] + speed_scale * dt * actions[:, :7]
    hand = (0.5 * (actions[:, 7:] + 1.0)) * (upper[7:] - lower[7:]) + lower[7:]
    hand = moving_average * hand + (1.0 - moving_average) * previous_targets[:, 7:]
    return torch.maximum(torch.minimum(torch.cat((arm, hand), dim=-1), upper), lower)
