"""Bounded palm-pose control of the seven arm joints; finger control stays separate."""

import math
import torch
from collections.abc import Sequence

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    apply_delta_pose,
    compute_pose_error,
    matrix_from_quat,
    quat_inv,
)

from .control import ARM_JOINT_NAMES, bounded_joint_target, bounded_pose_delta


class BoundedPalmPoseAction(DifferentialInverseKinematicsAction):
    """Capture a base-frame pose increment once, then track it at physics frequency."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not self._asset.is_fixed_base or set(self._joint_names) != set(ARM_JOINT_NAMES):
            raise ValueError("Cartesian arm control requires the fixed-base seven-joint KUKA arm")
        self._joint_names = list(ARM_JOINT_NAMES)
        self._joint_ids = [self._asset.joint_names.index(name) for name in self._joint_names]
        self._jacobi_joint_ids = [joint + self._asset.num_base_dofs for joint in self._joint_ids]
        if cfg.body_offset is not None or cfg.body_name != "palm_link":
            raise ValueError("Cartesian v1 controls the palm_link frame without an additional tool offset")
        values = (
            cfg.translation_scale,
            cfg.rotation_scale,
            cfg.max_tracking_error,
            cfg.max_position_error,
            cfg.max_rotation_error,
            *cfg.joint_target_speed,
        )
        if len(cfg.joint_target_speed) != 7 or not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Cartesian control scales and seven target-speed limits must be finite and positive")
        if cfg.controller.command_type != "pose" or not cfg.controller.use_relative_mode:
            raise ValueError("Cartesian v1 requires six-dimensional relative pose actions")
        if (
            cfg.controller.ik_method != "dls"
            or cfg.controller.joint_limit_avoidance_gain != 0.0
            or cfg.controller.orientation_weight is not None
        ):
            raise ValueError("Cartesian v1 uses full-pose DLS without an additional null-space posture bias")
        damping = cfg.controller.ik_params["lambda_val"]
        if not math.isfinite(damping) or damping <= 0.0:
            raise ValueError("Cartesian DLS damping must be finite and positive")
        self.joint_target = torch.zeros(self.num_envs, 7, device=self.device)
        self.target_limited = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.control_valid = torch.zeros_like(self.target_limited)
        self._initialized = torch.zeros_like(self.target_limited)
        self._max_step = torch.tensor(cfg.joint_target_speed, device=self.device) * env.physics_dt

    def _compute_frame_jacobian(self):
        # Keep the simulator's shared Jacobian buffer immutable.
        source = self._asset.data.body_link_jacobian_w.torch
        jacobian = source[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        rotation = matrix_from_quat(quat_inv(self._asset.data.root_quat_w.torch))
        self._jacobian_b[:, :3] = torch.bmm(rotation, jacobian[:, :3])
        self._jacobian_b[:, 3:] = torch.bmm(rotation, jacobian[:, 3:])
        return self._jacobian_b

    @staticmethod
    def _validate_state(position, quaternion, joints):
        if (
            not torch.isfinite(position).all()
            or not torch.isfinite(quaternion).all()
            or not torch.isfinite(joints).all()
            or (quaternion.norm(dim=-1) < 1.0e-6).any()
        ):
            raise ValueError("Non-finite robot state or invalid palm quaternion")

    def process_actions(self, actions: torch.Tensor):
        if not torch.isfinite(actions).all():
            raise ValueError("Palm pose actions must be finite")
        self._raw_actions[:] = actions
        self._processed_actions[:] = bounded_pose_delta(actions, self.cfg.translation_scale, self.cfg.rotation_scale)
        position, quaternion = self._compute_frame_pose()
        joints = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self._validate_state(position, quaternion, joints)
        self.joint_target[:] = torch.where(self._initialized[:, None], self.joint_target, joints)
        reference_position = torch.where(self._initialized[:, None], self._ik_controller.ee_pos_des, position)
        reference_quaternion = torch.where(self._initialized[:, None], self._ik_controller.ee_quat_des, quaternion)
        desired_position, desired_quaternion = apply_delta_pose(
            reference_position, reference_quaternion, self._processed_actions
        )
        position_error, rotation_error = compute_pose_error(
            position,
            quaternion,
            desired_position,
            desired_quaternion,
            rot_error_type="axis_angle",
        )
        position_ratio = (position_error.norm(dim=-1, keepdim=True) / self.cfg.max_position_error).clamp_min(1.0)
        rotation_ratio = (rotation_error.norm(dim=-1, keepdim=True) / self.cfg.max_rotation_error).clamp_min(1.0)
        bounded_error = torch.cat((position_error / position_ratio, rotation_error / rotation_ratio), dim=-1)
        self._ik_controller.set_command(bounded_error, position, quaternion)
        self._initialized[:] = True
        self.control_valid[:] = True
        self.target_limited[:] = (position_ratio[:, 0] > 1.0) | (rotation_ratio[:, 0] > 1.0)

    def apply_actions(self):
        position, quaternion = self._compute_frame_pose()
        joints = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self._validate_state(position, quaternion, joints)
        jacobian = self._compute_frame_jacobian()
        if not torch.isfinite(jacobian).all():
            raise ValueError("Non-finite arm Jacobian")
        desired = self._ik_controller.compute(position, quaternion, jacobian, joints)
        finite = torch.isfinite(desired).all(dim=-1)
        previous = torch.where(self._initialized[:, None], self.joint_target, joints)
        desired = torch.where((finite & self._initialized)[:, None], desired, previous)
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        target, limited = bounded_joint_target(
            desired,
            joints,
            previous,
            limits[..., 0],
            limits[..., 1],
            self._max_step,
            self.cfg.max_tracking_error,
        )
        self.joint_target[:] = target
        self.control_valid[:] = finite & self._initialized
        self.target_limited[:] |= limited
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0
        self._initialized[ids] = False
        self.control_valid[ids] = False
        self.target_limited[ids] = False
        self.joint_target[ids] = 0.0
        self._ik_controller.ee_pos_des[ids] = 0.0
        self._ik_controller.ee_quat_des[ids] = 0.0
        self._ik_controller.ee_quat_des[ids, 3] = 1.0
        self._ik_controller._command[ids] = 0.0

    def observation(self):
        position, quaternion = self._compute_frame_pose()
        joints = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self._validate_state(position, quaternion, joints)
        active = self._initialized[:, None]
        target_position = torch.where(active, self._ik_controller.ee_pos_des, position)
        target_quaternion = torch.where(active, self._ik_controller.ee_quat_des, quaternion)
        position_error, rotation_error = compute_pose_error(
            position,
            quaternion,
            target_position,
            target_quaternion,
            rot_error_type="axis_angle",
        )
        return torch.cat(
            (
                target_position,
                target_quaternion,
                torch.where(active, self.joint_target, joints),
                torch.where(active, position_error, 0.0),
                torch.where(active, rotation_error, 0.0),
                self.target_limited[:, None].float(),
                self.control_valid[:, None].float(),
            ),
            dim=-1,
        )


def palm_controller_state(env):
    """Current target, applied joint target, tracking error and control status (22-d)."""
    return env.action_manager.get_term("arm").observation()


@configclass
class BoundedPalmPoseActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = BoundedPalmPoseAction
    asset_name: str = "robot"
    joint_names: list[str] = list(ARM_JOINT_NAMES)
    body_name: str = "palm_link"
    controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )
    translation_scale: float = 0.005
    rotation_scale: float = 0.02
    joint_target_speed: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    max_tracking_error: float = 0.2
    max_position_error: float = 0.05
    max_rotation_error: float = 0.2
