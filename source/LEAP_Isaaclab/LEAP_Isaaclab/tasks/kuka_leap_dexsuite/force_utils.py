# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-force helpers that tolerate multiple filter shapes per sensor.

The stock Dexsuite helpers assume exactly ONE filter shape per sensor
(``force_matrix_w.view(num_envs, 3)``). The nested-contact sensor concretizes
the object filter per environment (``/World/envs/env_0/Object`` ...), so
``force_matrix_w`` is ``(num_instances, num_sensors, num_filter_shapes, 3)``
with ``num_filter_shapes == num_envs``. These helpers reduce over the filter
shape dimension (sum), recovering the stock force-matrix semantics.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse


def _force_mag(sensor: ContactSensor, num_envs: int) -> torch.Tensor:
    """Per-env contact force magnitude, summed over all filter shapes.

    Args:
        sensor: Contact sensor instance.
        num_envs: Number of environments.

    Returns:
        Tensor of shape (num_envs,) with per-env force magnitude.
    """
    fw = sensor.data.force_matrix_w.torch
    if fw is None:
        fw = sensor.data.net_forces_w.torch
    fw = fw.reshape(num_envs, -1, 3)
    # sum over filter shapes, then take vector norm
    force_sum = fw.sum(dim=1)
    return torch.linalg.norm(force_sum, dim=-1)


def contacts(env: ManagerBasedRLEnv, threshold: float, thumb_name: str, finger_names: list[str]) -> torch.Tensor:
    """Reward for good contact: thumb + at least one finger above threshold."""
    thumb_mag = _force_mag(env.scene.sensors[thumb_name], env.num_envs)

    any_finger_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for finger_name in finger_names:
        finger_mag = _force_mag(env.scene.sensors[finger_name], env.num_envs)
        any_finger_contact = any_finger_contact | (finger_mag > threshold)

    return (thumb_mag > threshold) & any_finger_contact


def contact_count(env: ManagerBasedRLEnv, threshold: float, sensor_names: list[str]) -> torch.Tensor:
    """Count the number of contact sensors with force above threshold."""
    count = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    for sensor_name in sensor_names:
        mag = _force_mag(env.scene.sensors[sensor_name], env.num_envs)
        count += (mag > threshold).float()

    return count


def fingers_contact_force_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base-frame contact forces from listed sensors, concatenated per env.

    Same as Dexsuite stock but tolerant of multiple filter shapes
    (``force_matrix_w`` shape (num_envs, num_filters, 3) after our nested
    sensor concretizes filters per env).
    """
    force_w = []
    for name in contact_sensor_names:
        fw = env.scene.sensors[name].data.force_matrix_w.torch
        if fw is None:
            fw = env.scene.sensors[name].data.net_forces_w.torch
        fw = fw.reshape(env.num_envs, -1, 3)
        force_w.append(fw.sum(dim=1))
    force_w = torch.stack(force_w, dim=1)

    robot = env.scene[asset_cfg.name]
    root_link_quat_w = robot.data.root_link_quat_w.torch
    forces_b = quat_apply_inverse(root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)
    return forces_b.reshape(env.num_envs, -1)


def patch_dexsuite_mdp() -> None:
    """Replace the stock Dexsuite rewards.contacts/contact_count with ours.

    The stock Dexsuite reward terms (object_ee_distance, position/orientation
    tracking, success) call ``mdp.contacts`` internally, which assumes a single
    filter shape per sensor. Our nested contact sensor has one filter per env
    (force_matrix_w is (num_envs, num_filters, 3)), so stock helpers crash.
    Swapping the module-level names makes all Dexsuite terms use our helpers.
    """
    import LEAP_Isaaclab.compat.dexsuite.mdp.rewards as _rewards

    _rewards.contacts = contacts
    _rewards.contact_count = contact_count
