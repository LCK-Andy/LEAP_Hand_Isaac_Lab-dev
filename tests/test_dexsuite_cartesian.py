"""CPU-only Cartesian action contracts; never import the task package or launch Isaac.

Pure helpers execute by file spec. Action/environment callbacks execute from their
AST classes against explicit simulator boundaries, not substitute control logic.
These checks do not establish IK convergence, contact safety, or learning quality.
"""

import ast
import copy
import importlib.util
import json
import math
import sys
import torch
import yaml
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_cartesian"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


control = load_file("cartesian_control_test", TASK / "control.py")


def test_action_and_observation_dimensions_and_explicit_joint_order():
    assert control.ACTION_DIM == 22
    assert control.CONTROLLER_OBS_DIM == 22
    assert control.OBS_DIM == 1762
    assert control.OBS_DIM - control.CONTROLLER_OBS_DIM == 1740
    assert control.ARM_JOINT_NAMES == tuple(f"iiwa7_joint_{i}" for i in range(1, 8))
    assert control.FINGER_JOINT_NAMES == tuple(
        f"{finger}_joint_{i}" for finger in ("index", "middle", "ring", "thumb") for i in range(4)
    )
    assert len(set(control.ARM_JOINT_NAMES + control.FINGER_JOINT_NAMES)) == 23


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pose_delta_preserves_subunit_commands_and_uses_separate_units(dtype):
    command = torch.tensor([[0.6, -0.8, 0.0, 0.0, 0.0, 0.5], [0.0] * 6], dtype=dtype)
    before = command.clone()
    actual = control.bounded_pose_delta(command, 0.005, 0.02)
    expected = torch.tensor([[0.003, -0.004, 0.0, 0.0, 0.0, 0.01], [0.0] * 6], dtype=dtype)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(command, before, rtol=0, atol=0)
    assert actual.dtype == dtype
    assert actual.device.type == "cpu"


def test_pose_delta_bounds_each_three_vector_by_l2_not_each_coordinate_only():
    commands = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    actual = control.bounded_pose_delta(commands, 0.005, 0.02)
    torch.testing.assert_close(actual[:, :3].norm(dim=-1), torch.tensor([0.005, 0.0]))
    torch.testing.assert_close(actual[:, 3:].norm(dim=-1), torch.tensor([0.02, 0.02]))
    assert actual[0, 1] < 0
    assert actual[0, 3] < 0
    # Saturating translation must not consume the independent rotation budget.
    torch.testing.assert_close(actual[1], torch.tensor([0.0, 0.0, 0.0, 0.02, 0.0, 0.0]))


@pytest.mark.parametrize("shape", [(6,), (2, 6), (2, 3, 6)])
def test_large_pose_commands_clip_before_l2_projection_and_preserve_batch_shape(shape):
    command = torch.tensor([1.0e30, -2.0, 0.0, 0.0, -1.0e30, 0.0]).expand(shape).clone()
    actual = control.bounded_pose_delta(command, 0.005, 0.02)
    expected = torch.tensor([0.005 / math.sqrt(2), -0.005 / math.sqrt(2), 0.0, 0.0, -0.02, 0.0])
    torch.testing.assert_close(actual, expected.expand(shape))
    assert tuple(actual.shape) == shape
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("shape", [(), (0,), (5,), (7,), (3, 5), (6, 1)])
def test_pose_delta_rejects_wrong_channel_dimensions(shape):
    with pytest.raises(ValueError, match="six channels"):
        control.bounded_pose_delta(torch.zeros(shape), 0.005, 0.02)


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("which", [0, 1])
def test_pose_delta_rejects_nonpositive_or_nonfinite_scales(bad, which):
    scales = [0.005, 0.02]
    scales[which] = bad
    with pytest.raises(ValueError, match="finite and positive"):
        control.bounded_pose_delta(torch.zeros(2, 6), *scales)


def test_joint_target_leaves_feasible_commands_unchanged_and_does_not_mutate_inputs():
    desired = torch.tensor([[0.1, -0.1], [0.4, -0.4]])
    current = torch.tensor([[0.0, 0.0], [0.3, -0.3]])
    previous = current.clone()
    lower, upper = torch.full_like(desired, -1), torch.full_like(desired, 1)
    step = torch.tensor([0.15, 0.15])
    inputs = [desired, current, previous, lower, upper, step]
    before = [value.clone() for value in inputs]
    target, limited = control.bounded_joint_target(*inputs, max_tracking_error=0.2)
    torch.testing.assert_close(target, desired, rtol=0, atol=0)
    assert limited.tolist() == [False, False]
    assert limited.dtype == torch.bool
    for actual, expected in zip(inputs, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_joint_target_enforces_hard_position_limits_per_environment_and_joint():
    desired = torch.tensor([[100.0, -100.0], [100.0, -100.0]])
    lower = torch.tensor([[-0.5, -0.2], [-0.1, -0.7]])
    upper = torch.tensor([[0.3, 0.4], [0.8, 0.2]])
    target, limited = control.bounded_joint_target(
        desired, torch.zeros_like(desired), torch.zeros_like(desired), lower, upper, torch.tensor([2.0, 2.0]), 200.0
    )
    torch.testing.assert_close(target, torch.tensor([[0.3, -0.2], [0.8, -0.7]]))
    assert limited.tolist() == [True, True]


def test_joint_target_slew_is_per_joint_and_per_environment_with_per_env_status():
    desired = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [0.01, -0.02]])
    previous = torch.tensor([[0.3, -0.3], [-0.2, 0.2], [0.0, 0.0]])
    step = torch.tensor([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
    target, limited = control.bounded_joint_target(
        desired, previous, previous, torch.full_like(desired, -2), torch.full_like(desired, 2), step, 2.0
    )
    torch.testing.assert_close(target, torch.tensor([[0.31, -0.32], [-0.23, 0.24], [0.01, -0.02]]))
    assert limited.tolist() == [True, True, False]


def test_joint_target_prioritizes_command_slew_when_measured_joint_jumps():
    target, limited = control.bounded_joint_target(
        desired=torch.tensor([[1.0, -1.0]]),
        current=torch.tensor([[0.8, -0.8]]),
        previous=torch.zeros(1, 2),
        lower=torch.full((1, 2), -2.0),
        upper=torch.full((1, 2), 2.0),
        max_step=torch.tensor([0.01, 0.02]),
        max_tracking_error=0.2,
    )
    torch.testing.assert_close(target, torch.tensor([[0.01, -0.02]]))
    assert (target.abs() < 0.6).all(), "Tracking must not jump the target outside its slew budget"
    assert limited.tolist() == [True]


def test_joint_target_applies_tracking_bound_when_slew_budget_can_satisfy_it():
    target, limited = control.bounded_joint_target(
        torch.tensor([[1.0, -1.0]]),
        torch.tensor([[0.1, -0.1]]),
        torch.zeros(1, 2),
        torch.full((1, 2), -2.0),
        torch.full((1, 2), 2.0),
        torch.ones(2),
        0.2,
    )
    torch.testing.assert_close(target, torch.tensor([[0.3, -0.3]]))
    assert limited.tolist() == [True]


def test_joint_target_repairs_out_of_bounds_previous_target_before_applying_slew():
    target, limited = control.bounded_joint_target(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.tensor([[4.0, -4.0]]),
        torch.tensor([[-0.5, -0.25]]),
        torch.tensor([[0.5, 0.25]]),
        torch.tensor([0.01, 0.02]),
        0.2,
    )
    # The hard position bound wins over slew measured from an already-invalid command.
    torch.testing.assert_close(target, torch.tensor([[0.49, -0.23]]))
    assert limited.tolist() == [True]


def test_joint_target_repeated_steps_preserve_position_and_slew_bounds():
    generator = torch.Generator().manual_seed(83)
    lower = -0.2 - torch.rand((4, 7), generator=generator)
    upper = 0.2 + torch.rand((4, 7), generator=generator)
    previous = torch.zeros(4, 7)
    max_step = torch.tensor([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035])
    for _ in range(25):
        desired = torch.randn((4, 7), generator=generator) * 1000
        current = torch.randn((4, 7), generator=generator)
        target, limited = control.bounded_joint_target(desired, current, previous, lower, upper, max_step, 0.2)
        assert torch.isfinite(target).all()
        assert (target >= lower).all() and (target <= upper).all()
        assert ((target - previous).abs() <= max_step + 1e-7).all()
        assert limited.shape == (4,)
        previous = target


def definition(path, name):
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def execute_class(path, name, scope):
    node = definition(path, name)
    namespace = dict(scope)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class IKConfigBoundary(SimpleNamespace):
    body_offset = None


def action_config(**overrides):
    config_type = execute_class(
        TASK / "actions.py",
        "BoundedPalmPoseActionCfg",
        {
            "configclass": lambda cls: cls,
            "DifferentialInverseKinematicsActionCfg": IKConfigBoundary,
            "DifferentialIKControllerCfg": lambda **kwargs: SimpleNamespace(
                joint_limit_avoidance_gain=0.0, orientation_weight=None, **kwargs
            ),
            "BoundedPalmPoseAction": object,
            "ARM_JOINT_NAMES": control.ARM_JOINT_NAMES,
        },
    )
    return config_type(**overrides)


class IKBoundary:
    """Supply solver output; do not pretend the CPU seam solves differential IK."""

    def __init__(self, count):
        self.ee_pos_des = torch.zeros(count, 3)
        self.ee_quat_des = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(count, 1)
        self._command = torch.zeros(count, 6)
        self.desired = torch.zeros(count, 7)
        self.commands = []

    def set_command(self, command, position, quaternion):
        self.commands.append((command.clone(), position.clone(), quaternion.clone()))
        self._command[:] = command
        self.ee_pos_des[:] = position + command[:, :3]
        self.ee_quat_des[:] = quaternion

    def compute(self, position, quaternion, jacobian, joints):
        self.last_jacobian = jacobian.clone()
        return self.desired.clone()


class ActionBoundary:
    """Replace simulator construction, frame readback, and the native IK solver only."""

    def __init__(self, cfg, env):
        self.cfg = cfg
        self._asset = env.asset
        self._joint_names = list(env.arm_names)
        self.num_envs = 3
        self.device = "cpu"
        self._raw_actions = torch.zeros(3, 6)
        self._processed_actions = torch.zeros(3, 6)
        self._jacobi_body_idx = 0
        self._jacobian_b = torch.zeros(3, 6, 7)
        self._ik_controller = IKBoundary(3)
        self.position = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        self.quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)

    def _compute_frame_pose(self):
        return self.position, self.quaternion


def translation_only_delta_pose(position, quaternion, delta):
    # Rotation composition is an Isaac math boundary, not reimplemented by this seam.
    assert not delta[:, 3:].any()
    return position + delta[:, :3], quaternion.clone()


def matching_rotation_pose_error(position, quaternion, target_position, target_quaternion, rot_error_type):
    assert rot_error_type == "axis_angle"
    torch.testing.assert_close(quaternion, target_quaternion)
    return target_position - position, torch.zeros_like(position)


@pytest.fixture
def action_type():
    return execute_class(
        TASK / "actions.py",
        "BoundedPalmPoseAction",
        {
            **vars(control),
            "DifferentialInverseKinematicsAction": ActionBoundary,
            "Sequence": Sequence,
            "apply_delta_pose": translation_only_delta_pose,
            "compute_pose_error": matching_rotation_pose_error,
            "quat_inv": lambda quat: quat,
            "matrix_from_quat": lambda quat: torch.eye(3).expand(quat.shape[0], 3, 3),
        },
    )


def make_action(action_type, cfg=None, **asset_overrides):
    # Deliberately put the arm after fingers and reverse its asset order.
    names = list(control.FINGER_JOINT_NAMES) + list(reversed(control.ARM_JOINT_NAMES))
    joints = torch.arange(69).reshape(3, 23).float() / 100
    limits = torch.tensor([-2.0, 2.0]).expand(3, 23, 2).clone()
    asset = SimpleNamespace(
        is_fixed_base=True,
        joint_names=names,
        num_base_dofs=0,
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=joints),
            soft_joint_pos_limits=SimpleNamespace(torch=limits),
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)),
            body_link_jacobian_w=SimpleNamespace(torch=torch.arange(3 * 6 * 23).reshape(3, 1, 6, 23).float()),
        ),
        set_joint_position_target_index=Mock(),
    )
    vars(asset).update(asset_overrides)
    env = SimpleNamespace(asset=asset, arm_names=list(reversed(control.ARM_JOINT_NAMES)), physics_dt=1 / 120)
    return action_type(cfg or action_config(), env)


def test_action_defaults_are_bounded_relative_dls_and_speed_uses_physics_dt(action_type):
    cfg = action_config()
    assert cfg.translation_scale == 0.005
    assert cfg.rotation_scale == 0.02
    assert cfg.joint_target_speed == (1.0,) * 7
    assert cfg.max_tracking_error == 0.2
    assert cfg.max_position_error == 0.05
    assert cfg.max_rotation_error == 0.2
    assert cfg.body_name == "palm_link"
    assert cfg.body_offset is None
    assert cfg.controller.command_type == "pose"
    assert cfg.controller.use_relative_mode is True
    assert cfg.controller.ik_method == "dls"
    assert cfg.controller.ik_params == {"lambda_val": 0.05}
    action = make_action(action_type, cfg)
    assert action._joint_ids == list(range(22, 15, -1))
    torch.testing.assert_close(action._max_step, torch.full((7,), 1 / 120))


@pytest.mark.parametrize(
    "overrides",
    [
        {"translation_scale": 0.0},
        {"rotation_scale": float("nan")},
        {"max_tracking_error": -0.1},
        {"max_position_error": 0.0},
        {"max_rotation_error": float("inf")},
        {"joint_target_speed": (1.0,) * 6},
        {"joint_target_speed": (1.0,) * 6 + (float("inf"),)},
        {"joint_target_speed": (1.0,) * 6 + (0.0,)},
        {"body_name": "index_tip"},
        {"body_offset": object()},
    ],
)
def test_action_rejects_unsupported_bounds_and_control_frame(action_type, overrides):
    with pytest.raises(ValueError):
        make_action(action_type, action_config(**overrides))


def test_action_rejects_floating_base_and_non_relative_or_non_dls_controller(action_type):
    with pytest.raises(ValueError, match="fixed-base"):
        make_action(action_type, is_fixed_base=False)
    for key, value in (
        ("command_type", "position"),
        ("use_relative_mode", False),
        ("ik_method", "pinv"),
        ("joint_limit_avoidance_gain", 0.1),
        ("orientation_weight", 0.5),
        ("ik_params", {"lambda_val": 0.0}),
        ("ik_params", {"lambda_val": float("nan")}),
    ):
        cfg = action_config()
        cfg.controller = copy.deepcopy(cfg.controller)
        setattr(cfg.controller, key, value)
        with pytest.raises(ValueError):
            make_action(action_type, cfg)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_action_rejects_nonfinite_commands_before_changing_state(action_type, bad):
    action = make_action(action_type)
    command = torch.zeros(3, 6)
    command[1, 5] = bad
    with pytest.raises(ValueError, match="finite"):
        action.process_actions(command)
    assert not action._initialized.any()
    assert not action._raw_actions.any()
    assert not action._ik_controller.commands


@pytest.mark.parametrize("callback", ["process_actions", "apply_actions", "observation"])
@pytest.mark.parametrize("field", ["position", "quaternion", "joints", "zero_quaternion"])
def test_action_rejects_invalid_measured_state_before_actuator_writes(action_type, callback, field):
    action = make_action(action_type)
    action.process_actions(torch.zeros(3, 6))
    previous = action.joint_target.clone()
    if field == "zero_quaternion":
        action.quaternion[1] = 0.0
    elif field == "joints":
        action._asset.data.joint_pos.torch[1, action._joint_ids[0]] = float("nan")
    else:
        getattr(action, field)[1, 0] = float("nan")
    with pytest.raises(ValueError, match="robot state|palm quaternion"):
        getattr(action, callback)(*([torch.zeros(3, 6)] if callback == "process_actions" else []))
    torch.testing.assert_close(action.joint_target, previous)
    action._asset.set_joint_position_target_index.assert_not_called()


def test_action_rejects_nonfinite_jacobian_before_solver_or_actuator_writes(action_type):
    action = make_action(action_type)
    action.process_actions(torch.zeros(3, 6))
    action._asset.data.body_link_jacobian_w.torch[1, 0, 0, action._joint_ids[0]] = float("inf")
    with pytest.raises(ValueError, match="Jacobian"):
        action.apply_actions()
    assert not hasattr(action._ik_controller, "last_jacobian")
    action._asset.set_joint_position_target_index.assert_not_called()


def test_action_captures_once_then_slews_each_physics_step_without_mutating_jacobian(action_type):
    action = make_action(action_type)
    command = torch.zeros(3, 6)
    command[:, 0] = 2.0
    action.process_actions(command)
    initial = action._asset.data.joint_pos.torch[:, action._joint_ids].clone()
    desired_pose = action._ik_controller.ee_pos_des.clone()
    source = action._asset.data.body_link_jacobian_w.torch.clone()
    torch.testing.assert_close(action.joint_target, initial)
    torch.testing.assert_close(action._processed_actions[:, 0], torch.full((3,), 0.005))
    action._ik_controller.desired[:] = initial + 1.0
    action.position += 0.001
    for tick in range(1, 4):
        action.apply_actions()
        torch.testing.assert_close(action.joint_target, initial + tick / 120)
        torch.testing.assert_close(action._ik_controller.ee_pos_des, desired_pose)
    assert len(action._ik_controller.commands) == 1
    torch.testing.assert_close(action._asset.data.body_link_jacobian_w.torch, source, rtol=0, atol=0)
    torch.testing.assert_close(action._ik_controller.last_jacobian, source[:, 0, :, list(range(22, 15, -1))])
    write = action._asset.set_joint_position_target_index.call_args.kwargs
    assert write["joint_ids"] == list(range(22, 15, -1))
    torch.testing.assert_close(write["target"], action.joint_target)
    assert action.control_valid.tolist() == [True, True, True]
    assert action.target_limited.tolist() == [True, True, True]


def test_action_accumulates_increments_caps_pose_error_and_zero_command_holds_target(action_type):
    action = make_action(action_type)
    commands = torch.zeros(3, 6)
    commands[:, 0] = torch.tensor([1.0, 0.0, -1.0])
    initial_position = action.position.clone()
    for step in range(1, 16):
        action.process_actions(commands)
        expected_delta = torch.zeros(3, 3)
        expected_delta[:, 0] = torch.tensor([1.0, 0.0, -1.0]) * min(step * 0.005, 0.05)
        torch.testing.assert_close(action._ik_controller.ee_pos_des, initial_position + expected_delta)
    assert action.target_limited.tolist() == [True, False, True]
    previous_target = action._ik_controller.ee_pos_des.clone()
    # Tracking movement must not make a zero command recenter the setpoint.
    action.position[0, 0] += 0.01
    action.position[2, 0] -= 0.01
    action.process_actions(torch.zeros(3, 6))
    torch.testing.assert_close(action._ik_controller.ee_pos_des, previous_target)
    assert action.target_limited.tolist() == [False, False, False]
    observation = action.observation()
    torch.testing.assert_close(observation[:, 14:17], torch.tensor([[0.04, 0.0, 0.0], [0.0] * 3, [-0.04, 0.0, 0.0]]))


def test_action_invalid_solver_output_holds_only_failed_environment(action_type):
    action = make_action(action_type)
    action.process_actions(torch.zeros(3, 6))
    previous = action.joint_target.clone()
    action._ik_controller.desired[:] = previous + 0.1
    action._ik_controller.desired[1, 3] = float("nan")
    action.apply_actions()
    torch.testing.assert_close(action.joint_target[1], previous[1])
    torch.testing.assert_close(action.joint_target[[0, 2]], previous[[0, 2]] + 1 / 120)
    assert torch.isfinite(action.joint_target).all()
    assert action.control_valid.tolist() == [True, False, True]
    assert action.target_limited.tolist() == [True, False, True]


@pytest.mark.parametrize("ids", [[1], torch.tensor([1]), torch.tensor([1], dtype=torch.int32)])
def test_action_indexed_reset_clears_only_selected_environment_and_reseeds_from_measurement(action_type, ids):
    action = make_action(action_type)
    command = torch.zeros(3, 6)
    command[:, 0] = 0.5
    action.process_actions(command)
    action._ik_controller.desired[:] = action.joint_target + 0.1
    action.apply_actions()
    before = action.observation().clone()
    previous = action.joint_target.clone()
    action.reset(ids)
    observation = action.observation()
    assert observation.shape == (3, 22)
    torch.testing.assert_close(observation[[0, 2]], before[[0, 2]], rtol=0, atol=0)
    torch.testing.assert_close(observation[1, :3], action.position[1])
    torch.testing.assert_close(observation[1, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(observation[1, 7:14], action._asset.data.joint_pos.torch[1, action._joint_ids])
    torch.testing.assert_close(observation[1, 14:], torch.zeros(8))
    for buffer in (action._raw_actions, action._processed_actions, action.joint_target, action._ik_controller._command):
        assert not buffer[1].any()
    torch.testing.assert_close(action._ik_controller.ee_quat_des[1], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert action._initialized.tolist() == [True, False, True]
    # The newly reset env seeds from its new joint pose; other command histories survive.
    action._asset.data.joint_pos.torch[1] += 0.2
    action.process_actions(torch.zeros(3, 6))
    torch.testing.assert_close(action.joint_target[1], action._asset.data.joint_pos.torch[1, action._joint_ids])
    torch.testing.assert_close(action.joint_target[[0, 2]], previous[[0, 2]])


def test_action_full_and_empty_resets_do_not_leave_stale_controller_commands(action_type):
    action = make_action(action_type)
    action.process_actions(torch.zeros(3, 6))
    before = action.observation().clone()
    action.reset([])
    torch.testing.assert_close(action.observation(), before, rtol=0, atol=0)
    action.reset()
    assert not action._initialized.any()
    assert not action.control_valid.any()
    assert not action.target_limited.any()
    assert not action.joint_target.any()
    torch.testing.assert_close(action._ik_controller.ee_quat_des, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1))


class PhysxBoundary:
    pass


class EnvironmentBoundary:
    """Record inherited env calls without running managers, physics, or curriculum."""

    def load_managers(self):
        self.managers_loaded = True

    @property
    def evaluation_protocol(self):
        return "training_conditions_fixed_definition_v1"

    def get_evaluation_condition(self):
        return {"gravity": -9.81, "protocol": self.evaluation_protocol}

    def get_env_state(self):
        return {"dexpbt": {"version": 1, "adr_difficulties": [0.0, 1.0, 2.0]}}

    def set_env_state(self, state):
        self.restored.append(copy.deepcopy(state))

    def step(self, action):
        self.received_actions = action.clone()
        return "native-step-result"


@pytest.fixture
def cartesian_env(action_type):
    env_type = execute_class(
        TASK / "environment.py",
        "DexsuiteReorientCartesianEnv",
        {**vars(control), "json": json, "DexsuiteReorientPbtEnv": EnvironmentBoundary, "PhysxCfg": PhysxBoundary},
    )
    env = env_type()
    arm = make_action(action_type)
    terms = {"arm": arm, "fingers": SimpleNamespace(_joint_names=list(control.FINGER_JOINT_NAMES))}
    env.action_manager = SimpleNamespace(total_action_dim=22, get_term=terms.__getitem__)
    env.cfg = SimpleNamespace(actions=SimpleNamespace(arm=arm.cfg, fingers=SimpleNamespace(scale=0.1)), decimation=2)
    env.physics_dt = 1 / 120
    env.sim = SimpleNamespace(_physics=PhysxBoundary())
    env.restored = []
    return env


def expected_contract():
    return {
        "version": "palm_delta_dls_v1",
        "action_dim": 22,
        "translation_scale": 0.005,
        "rotation_scale": 0.02,
        "joint_target_speed": [1.0] * 7,
        "max_tracking_error": 0.2,
        "max_position_error": 0.05,
        "max_rotation_error": 0.2,
        "target_reference": "previous_target_capped_from_measured_pose",
        "damping": 0.05,
        "physics_dt": 1 / 120,
        "decimation": 2,
        "finger_scale": 0.1,
    }


def test_env_records_complete_control_contract_in_checkpoint_and_comparison_condition(cartesian_env):
    env = cartesian_env
    assert env.control_contract() == expected_contract()
    state = env.get_env_state()
    assert state["cartesian_control"] == expected_contract()
    assert state["dexpbt"] == {"version": 1, "adr_difficulties": [0.0, 1.0, 2.0]}
    condition = env.get_evaluation_condition()
    assert condition["gravity"] == -9.81
    assert json.loads(condition["cartesian_control"]) == expected_contract()
    assert condition["cartesian_control"] == json.dumps(expected_contract(), sort_keys=True)
    # Serialized state must not alias a mutable controller configuration.
    state["cartesian_control"]["joint_target_speed"][0] = 99
    assert env.control_contract() == expected_contract()


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("version", "joint_control_v1"),
        ("action_dim", 23),
        ("translation_scale", 0.01),
        ("rotation_scale", 0.04),
        ("joint_target_speed", [2.0] + [1.0] * 6),
        ("max_tracking_error", 0.4),
        ("max_position_error", 0.1),
        ("max_rotation_error", 0.4),
        ("target_reference", "measured_pose"),
        ("damping", 0.1),
        ("physics_dt", 1 / 60),
        ("decimation", 4),
        ("finger_scale", 0.2),
    ],
)
def test_env_rejects_mixed_checkpoint_contract_before_restore_or_controller_reset(cartesian_env, key, replacement):
    env = cartesian_env
    arm = env.action_manager.get_term("arm")
    arm.process_actions(torch.zeros(3, 6))
    before = arm.observation().clone()
    state = env.get_env_state()
    state["cartesian_control"][key] = replacement
    original = copy.deepcopy(state)
    with pytest.raises(ValueError, match="Cartesian control contract"):
        env.set_env_state(state)
    assert env.restored == []
    assert state == original
    torch.testing.assert_close(arm.observation(), before, rtol=0, atol=0)


@pytest.mark.parametrize("state", [{"dexpbt": {"version": 1}}, {"cartesian_control": {}}, {"unrelated": True}])
def test_env_rejects_legacy_or_missing_checkpoint_contract(cartesian_env, state):
    with pytest.raises(ValueError, match="Cartesian control contract"):
        cartesian_env.set_env_state(state)
    assert cartesian_env.restored == []


def test_env_matching_checkpoint_restores_base_state_then_clears_all_controller_state(cartesian_env):
    env = cartesian_env
    arm = env.action_manager.get_term("arm")
    arm.process_actions(torch.zeros(3, 6))
    arm._ik_controller.desired[:] = arm.joint_target + 0.1
    arm.apply_actions()
    state = env.get_env_state()
    env.set_env_state(state)
    assert env.restored == [state]
    assert not arm._initialized.any()
    assert not arm.control_valid.any()
    assert not arm.target_limited.any()
    torch.testing.assert_close(arm.observation()[:, 7:14], arm._asset.data.joint_pos.torch[:, arm._joint_ids])
    torch.testing.assert_close(arm.observation()[:, 14:], torch.zeros(3, 8))


@pytest.mark.parametrize("state", [None, {}])
def test_env_empty_state_is_allowed_but_does_not_preserve_stale_targets(cartesian_env, state):
    arm = cartesian_env.action_manager.get_term("arm")
    arm.process_actions(torch.zeros(3, 6))
    cartesian_env.set_env_state(state)
    assert cartesian_env.restored == [state]
    assert not arm._initialized.any()


def test_env_validates_22_actions_ordered_fingers_and_physx_backend(cartesian_env):
    env = cartesian_env
    env.load_managers()
    assert env.managers_loaded
    env.action_manager.total_action_dim = 23
    with pytest.raises(ValueError, match="22 actions"):
        env.load_managers()
    env.action_manager.total_action_dim = 22
    fingers = env.action_manager.get_term("fingers")
    fingers._joint_names.reverse()
    with pytest.raises(ValueError, match="finger action order"):
        env.load_managers()
    fingers._joint_names.reverse()
    env.sim._physics = object()
    with pytest.raises(ValueError, match="PhysX"):
        env.load_managers()


def test_env_clips_all_22_normalized_actions_without_mutating_policy_samples(cartesian_env):
    samples = torch.linspace(-3, 3, 66).reshape(3, 22)
    before = samples.clone()
    assert cartesian_env.step(samples) == "native-step-result"
    torch.testing.assert_close(cartesian_env.received_actions, samples.clamp(-1, 1))
    torch.testing.assert_close(samples, before, rtol=0, atol=0)


@pytest.mark.parametrize("channel", [0, 5, 6, 21])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_env_rejects_nonfinite_arm_or_finger_action_before_native_step(cartesian_env, channel, bad):
    action = torch.zeros(3, 22)
    action[1, channel] = bad
    with pytest.raises(ValueError, match="finite"):
        cartesian_env.step(action)
    assert not hasattr(cartesian_env, "received_actions")


def assigned_values(class_node):
    values = {}
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign):
            values[node.target.id] = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value
    return values


def test_configuration_only_overrides_actions_and_appends_controller_observation():
    path = TASK / "env_cfg.py"
    for name, base in (
        ("DexsuiteReorientCartesianEnvCfg", "DexsuiteReorientPbtEnvCfg"),
        ("DexsuiteReorientCartesianEnvCfg_PLAY", "DexsuiteReorientPbtEnvCfg_PLAY"),
    ):
        cls = definition(path, name)
        assert [ast.unparse(node) for node in cls.bases] == [base]
        assert set(assigned_values(cls)) == {"actions", "observations"}
        assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in cls.body)
        # No post-init physics, reward, ADR, reset, or termination override.
        assert ast.unparse(assigned_values(cls)["actions"]) == "CartesianActionsCfg()"
        assert ast.unparse(assigned_values(cls)["observations"]) == "CartesianObservationsCfg()"
    actions = assigned_values(definition(path, "CartesianActionsCfg"))
    assert list(actions) == ["arm", "fingers"]
    assert ast.unparse(actions["arm"]) == "BoundedPalmPoseActionCfg()"
    finger_cfg = actions["fingers"]
    assert ast.unparse(finger_cfg.func) == "RelativeJointPositionActionCfg"
    finger_args = {keyword.arg: keyword.value for keyword in finger_cfg.keywords}
    assert ast.literal_eval(finger_args["scale"]) == 0.1
    assert ast.literal_eval(finger_args["asset_name"]) == "robot"
    assert ast.literal_eval(finger_args["preserve_order"]) is True
    assert ast.unparse(finger_args["joint_names"]) == "list(FINGER_JOINT_NAMES)"
    observations = definition(path, "CartesianObservationsCfg")
    assert [ast.unparse(node) for node in observations.bases] == ["StateObservationCfg"]
    assert set(assigned_values(observations)) == {"controller"}
    assert not any(isinstance(node, ast.FunctionDef) for node in observations.body)
    controller = next(node for node in observations.body if isinstance(node, ast.ClassDef))
    values = assigned_values(controller)
    assert ast.literal_eval(values["history_length"]) == 0
    assert ast.literal_eval(values["enable_corruption"]) is False
    assert ast.literal_eval(values["concatenate_terms"]) is True
    assert ast.unparse(values["state"]) == "ObservationTermCfg(func=palm_controller_state)"


def test_registration_is_additive_and_resolves_only_cartesian_train_play_entries():
    existing = {
        "Isaac-Dexsuite-Kuka-Allegro-Reorient-v0": object(),
        "Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0": object(),
        "Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-Play-v0": object(),
    }
    registry = dict(existing)
    gym = ModuleType("gymnasium")
    gym.registry = registry

    def register(id, **kwargs):
        assert id not in registry, f"Registration must not replace {id}"
        registry[id] = kwargs

    gym.register = register
    namespace = "LEAP_Isaaclab.tasks.dexsuite_reorient_cartesian"
    with patch.dict(sys.modules, {"gymnasium": gym}):
        load_file(namespace, TASK / "__init__.py")
    expected = {
        "Isaac-Dexsuite-Kuka-Allegro-Reorient-Cartesian-v0": "DexsuiteReorientCartesianEnvCfg",
        "Isaac-Dexsuite-Kuka-Allegro-Reorient-Cartesian-Play-v0": "DexsuiteReorientCartesianEnvCfg_PLAY",
    }
    assert registry.keys() - existing.keys() == expected.keys()
    for name, entry in existing.items():
        assert registry[name] is entry
    for name, config in expected.items():
        entry = registry[name]
        assert entry["entry_point"] == f"{namespace}.environment:DexsuiteReorientCartesianEnv"
        assert entry["disable_env_checker"] is True
        assert entry["kwargs"] == {
            "env_cfg_entry_point": f"{namespace}.env_cfg:{config}",
            "rl_games_cfg_entry_point": f"{namespace}.agents:rl_games_ppo.yaml",
        }
        assert definition(TASK / "env_cfg.py", config)
    assert (TASK / "agents/__init__.py").is_file()
    assert (TASK / "agents/rl_games_ppo.yaml").is_file()


def agent_config():
    return yaml.safe_load((TASK / "agents/rl_games_ppo.yaml").read_text())


def test_ppo_yaml_is_complete_with_four_ordered_groups_and_disabled_population():
    config = agent_config()
    params = config["params"]
    assert params["env"]["obs_groups"] == ["policy", "proprio", "perception", "controller"]
    assert params["env"]["concate_obs_groups"] is True
    assert params["env"]["clip_actions"] == 1.0
    assert params["algo"]["name"] == "a2c_continuous"
    assert params["model"]["name"] == "continuous_a2c_logstd"
    assert params["network"]["name"] == "actor_critic"
    assert params["network"]["mlp"]["units"] == [512, 256, 128]
    continuous = params["network"]["space"]["continuous"]
    assert continuous["fixed_sigma"] is True
    assert continuous["sigma_activation"] in (None, "None")
    assert math.exp(continuous["sigma_init"]["val"]) == pytest.approx(0.15)
    ppo = params["config"]
    assert ppo["ppo"] is True
    assert ppo["env_name"] == "rlgpu"
    assert ppo["reward_shaper"] == {"scale_value": 1.0}
    assert ppo["normalize_input"] is True
    assert ppo["normalize_value"] is True
    assert ppo["entropy_coef"] == 0.0
    assert ppo["exploration"] == {"initial_std": 0.15, "min_std": 0.03, "max_std": 0.3}
    for key in ("learning_rate", "gamma", "tau", "e_clip", "kl_threshold", "horizon_length", "mini_epochs"):
        assert ppo[key] > 0
    assert (ppo["num_actors"] * ppo["horizon_length"]) % ppo["minibatch_size"] == 0
    assert params["load_checkpoint"] is False
    assert config["pbt"]["enabled"] is False
    assert config["pbt"]["initial_diversity"] is False
    assert config["pbt"]["num_policies"] == 1
    assert "cartesian" in config["pbt"]["directory"]


def test_real_cpu_ppo_builds_1762_inputs_22_actions_and_uniform_bounded_exploration():
    from rl_games.algos_torch.model_builder import ModelBuilder

    exploration = load_file("cartesian_exploration_test", TASK.parent / "dexsuite_reorient_pbt/pbt/exploration.py")
    params = agent_config()["params"]
    with torch.random.fork_rng(devices=[]):
        model = (
            ModelBuilder()
            .load(params)
            .build(
                {
                    "actions_num": 22,
                    "input_shape": (1762,),
                    "num_seqs": 1,
                    "value_size": 1,
                    "normalize_input": True,
                    "normalize_value": True,
                }
            )
            .eval()
        )
        sigma = model.a2c_network.sigma
        assert sigma.shape == (22,)
        torch.testing.assert_close(sigma.detach().exp(), torch.full((22,), 0.15))

        def infer(obs):
            with torch.no_grad():
                return model({"obs": obs, "is_train": False})

        def restore(logstd):
            with torch.no_grad():
                sigma.copy_(logstd)

        algo = SimpleNamespace(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            get_action_values=infer,
            restore=restore,
        )
        # Even supplied robot joint names cannot label a 22D Cartesian action as 23 joint deltas.
        observer = exploration.ExplorationObserver(
            **params["config"]["exploration"],
            action_joint_names=control.ARM_JOINT_NAMES + control.FINGER_JOINT_NAMES,
        )
        try:
            observer.after_init(algo)
            result = algo.get_action_values(torch.zeros(2, 1762))
            assert result["actions"].shape == (2, 22)
            assert result["values"].shape == (2, 1)
            assert torch.isfinite(result["neglogpacs"]).all()
            torch.testing.assert_close(result["sigmas"], torch.full((2, 22), 0.15))
            for value, expected in ((math.log(0.001), 0.03), (math.log(10.0), 0.3)):
                algo.restore(torch.full((22,), value))
                result = algo.get_action_values(torch.zeros(2, 1762))
                torch.testing.assert_close(result["sigmas"], torch.full((2, 22), expected))
            metrics = observer.metrics()
            assert metrics["exploration/global/sampled_coordinates"] == 132
            assert not any("/arm/" in key or "/fingers/" in key for key in metrics)
        finally:
            observer.close()
