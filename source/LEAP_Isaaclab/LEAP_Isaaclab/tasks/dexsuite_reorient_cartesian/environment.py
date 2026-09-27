"""Keep Cartesian checkpoints and comparison evidence separate from joint control."""

import json
import torch

from isaaclab_physx.physics import PhysxCfg

from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.environment import DexsuiteReorientPbtEnv

from .control import ACTION_DIM, CONTROL_VERSION, FINGER_JOINT_NAMES


class DexsuiteReorientCartesianEnv(DexsuiteReorientPbtEnv):
    def load_managers(self):
        super().load_managers()
        if self.action_manager.total_action_dim != ACTION_DIM:
            raise ValueError("Cartesian reorientation requires 22 actions")
        fingers = self.action_manager.get_term("fingers")
        if fingers._joint_names != list(FINGER_JOINT_NAMES):
            raise ValueError("Unexpected Cartesian task finger action order")
        if not isinstance(self.sim._physics, PhysxCfg):
            raise ValueError("Cartesian reorientation v1 is only validated for PhysX")

    def control_contract(self):
        cfg = self.cfg.actions.arm
        return {
            "version": CONTROL_VERSION,
            "action_dim": ACTION_DIM,
            "translation_scale": cfg.translation_scale,
            "rotation_scale": cfg.rotation_scale,
            "joint_target_speed": list(cfg.joint_target_speed),
            "max_tracking_error": cfg.max_tracking_error,
            "max_position_error": cfg.max_position_error,
            "max_rotation_error": cfg.max_rotation_error,
            "target_reference": "previous_target_capped_from_measured_pose",
            "damping": cfg.controller.ik_params["lambda_val"],
            "physics_dt": self.physics_dt,
            "decimation": self.cfg.decimation,
            "finger_scale": self.cfg.actions.fingers.scale,
        }

    def get_evaluation_condition(self):
        condition = super().get_evaluation_condition()
        condition["cartesian_control"] = json.dumps(self.control_contract(), sort_keys=True)
        return condition

    def get_env_state(self):
        state = super().get_env_state()
        state["cartesian_control"] = self.control_contract()
        return state

    def set_env_state(self, env_state):
        if env_state and env_state.get("cartesian_control") != self.control_contract():
            raise ValueError("Checkpoint does not match the Cartesian control contract")
        super().set_env_state(env_state)
        self.action_manager.get_term("arm").reset()

    def step(self, action):
        if not torch.isfinite(action).all():
            raise ValueError("Cartesian task actions must be finite")
        return super().step(action.clamp(-1.0, 1.0))
