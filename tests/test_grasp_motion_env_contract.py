"""CPU lifecycle regressions for the real Stage B callbacks.

The Stage A sensor/observation seam supplies tensor-backed assets, while only
simulator construction, action dispatch, and physical reset writes are replaced.
Stage A measurement/reward/observation and Stage B callback bodies are executed
from source. These are bookkeeping tests, not contact-feasibility evidence.
"""

import ast
import importlib.util
import math
import sys
import torch
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_grasp_motion"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage_a = load_module("grasp_motion_contract_stage_a", ROOT / "tests/test_grasp_env_contract.py")
motion = load_module("grasp_motion_contract_logic", TASK / "motion_logic.py")
NativeStageA = stage_a.native_env_class()
# The original Stage A seam uses identity rotations. Exercise arbitrary object
# orientations here without changing its sensor data or production callbacks.
NativeStageA._measure_grasp.__globals__.update(
    quat_apply=motion.rotate_vectors,
    quat_apply_inverse=lambda q, v: motion.rotate_vectors(q * q.new_tensor((-1, -1, -1, 1)), v),
)


class PhysicalBoundary(NativeStageA):
    """Substitute simulator ownership/writes, not grasp or motion calculations."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.__dict__.update(stage_a.native_env().__dict__)
        self.cfg = SimpleNamespace(**vars(self.cfg), motion=cfg.motion, observation_space=148)
        self.device = "cpu"
        self._fixture_pose = torch.tensor([0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0])
        self.condition_key = "grasp_hold_v1_cpu_fixture"
        self.reset_time_outs = torch.zeros(3, dtype=torch.bool)
        self.reset_terminated = torch.zeros(3, dtype=torch.bool)
        self._measure_grasp()

    def _pre_physics_step(self, actions):
        self.actions = actions.clone()
        self._fresh_reset.zero_()
        self.extras["grasp_hold"] = {}

    def _reset_idx(self, env_ids):
        self.episode_length_buf[env_ids] = 0
        self.object.data.root_pos_w.torch[env_ids] = self._fixture_pose[:3] + self.scene.env_origins[env_ids]
        self.object.data.root_quat_w.torch[env_ids] = self._fixture_pose[3:]
        self.object.data.root_lin_vel_w.torch[env_ids] = 0.0
        self.object.data.root_ang_vel_w.torch[env_ids] = 0.0
        self.grasp_state.reset(env_ids)
        self.successes[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self._fresh_reset[env_ids] = True
        for name in ("_normal", "_friction", "_contact_positions", "_pulp_contact", "_slip_speed", "_support_force"):
            getattr(self, name)[env_ids] = 0
        self._compute_intermediate_values()


def native_motion_class():
    path = TASK / "leap_grasp_motion_env.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    scope = dict(vars(motion))
    scope.update(
        LeapGraspEnv=PhysicalBoundary,
        LeapGraspMotionEnvCfg=SimpleNamespace,
        grasp_condition_key=stage_a.logic.grasp_condition_key,
        asdict=asdict,
        replace=replace,
        math=math,
        torch=torch,
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    return scope[cls.name]


MotionEnv = native_motion_class()


def make_env(**overrides):
    cfg = motion.MotionCriteria(**overrides)
    env = MotionEnv(SimpleNamespace(motion=cfg, observation_space=148))
    env._reset_idx(torch.arange(3))
    return env


def reward_at(env, step, follow_reference=False):
    """Match DirectRLEnv ordering: action, physics result, counter, dones, reward."""
    env._pre_physics_step(torch.zeros(3, 16))
    env.episode_length_buf[:] = torch.as_tensor(step)
    if follow_reference:
        env._refresh_reference()
        env.object.data.root_pos_w.torch[:] = env.reference.position + env.scene.env_origins
        env.object.data.root_quat_w.torch[:] = env.reference.quaternion
    terminated, truncated = env._get_dones()
    env.reset_terminated[:] = terminated
    env.reset_time_outs[:] = truncated
    return env._get_rewards()


class GraspMotionEnvironmentContractTests(unittest.TestCase):
    def test_observation_appends_24_channels_without_changing_stage_a_full_force_prefix(self):
        env = make_env(randomize_phase=False)
        reward_at(env, 120)
        expected = NativeStageA._get_observations(env)["policy"]
        actual = env._get_observations()["policy"]
        self.assertEqual(tuple(actual.shape), (3, 148))
        torch.testing.assert_close(actual[:, :124], expected, rtol=0, atol=0)
        # Normal + shear, not a substituted normal-only observation.
        torch.testing.assert_close(actual[0, 77:80], torch.tensor([0.2, 0.02, 0.0]))
        torch.testing.assert_close(actual[0, 89:92], torch.tensor([0.2, 0.0, 0.0]))
        self.assertTrue(env._score.valid[0])
        self.assertFalse(env._score.valid[1:].any())
        torch.testing.assert_close(env.tracking.reward[1:], torch.zeros(2))

    def test_observation_reads_do_not_advance_pose_or_reference_filters(self):
        env = make_env(randomize_phase=False)
        env.object.data.root_pos_w.torch[:, 0] += 0.0001
        reward_at(env, 120)
        torch.testing.assert_close(env.motion_state.linear_velocity[:, 0], torch.full((3,), 0.0006))
        torch.testing.assert_close(
            env._filtered_ref_angular_velocity,
            env.reference.angular_velocity * env.cfg.motion.velocity_filter_alpha,
        )
        velocity = env.motion_state.linear_velocity.clone()
        angular = env.motion_state.angular_velocity.clone()
        acceleration = env.motion_state.angular_acceleration.clone()
        ref_angular = env._filtered_ref_angular_velocity.clone()
        steps = env.grasp_state.steps.clone()
        observations = env._get_observations()["policy"]
        for _ in range(3):
            torch.testing.assert_close(env._get_observations()["policy"], observations, rtol=0, atol=0)
        for actual, expected in (
            (env.motion_state.linear_velocity, velocity),
            (env.motion_state.angular_velocity, angular),
            (env.motion_state.angular_acceleration, acceleration),
            (env._filtered_ref_angular_velocity, ref_angular),
            (env.grasp_state.steps, steps),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Filtered velocity error is observable, rather than hidden EMA state.
        # Subtracting a nonzero env origin loses float32 precision before the
        # command's 1 / 0.0005 scaling; allow that quantization, not a filter step.
        torch.testing.assert_close(observations[:, 130], torch.full((3,), -1.2), atol=3e-4, rtol=0)

    def test_terminal_reference_precedes_asynchronous_reset_and_other_envs_are_unchanged(self):
        env = make_env()
        env._motion_phase[:] = torch.tensor([[0.3, 0.7, 1.1], [0.2, 0.4, 0.6], [0.1, 0.5, 0.9]])
        old_phase = env._motion_phase.clone()
        env.object.data.root_pos_w.torch[:, 0] += 0.001
        reward_at(env, [300, 120, 10])
        terminal = env._get_observations()["policy"]
        saved_terminal = terminal.clone()
        end_phase = old_phase[0] + 9.0 * (2.0 * math.pi) * torch.tensor(env.cfg.motion.frequency_hz)
        torch.testing.assert_close(terminal[0, 142:145], end_phase.sin())
        torch.testing.assert_close(terminal[0, 145:148], end_phase.cos())
        self.assertEqual(float(terminal[0, 123]), 0.0)
        other_velocity = env.motion_state.linear_velocity[1:].clone()
        other_ref_velocity = env._filtered_ref_angular_velocity[1:].clone()
        other_steps = env.grasp_state.steps[1:].clone()
        env._reset_idx(torch.tensor([0]))
        returned = env._get_observations()["policy"]
        self.assertEqual(env.episode_length_buf.tolist(), [0, 120, 10])
        torch.testing.assert_close(returned[1:], terminal[1:], rtol=0, atol=0)
        torch.testing.assert_close(terminal, saved_terminal, rtol=0, atol=0)
        torch.testing.assert_close(env._motion_phase[1:], old_phase[1:], rtol=0, atol=0)
        self.assertFalse(torch.equal(env._motion_phase[0], old_phase[0]))
        torch.testing.assert_close(returned[0, 142:145], env._motion_phase[0].sin())
        torch.testing.assert_close(returned[0, 145:148], env._motion_phase[0].cos())
        torch.testing.assert_close(env.motion_state.position[0], env.object_pos[0], rtol=0, atol=0)
        torch.testing.assert_close(env.motion_state.quaternion[0], env.object_rot[0], rtol=0, atol=0)
        torch.testing.assert_close(env.motion_state.linear_velocity[0], torch.zeros(3))
        torch.testing.assert_close(env.motion_state.angular_velocity[0], torch.zeros(3))
        torch.testing.assert_close(env._filtered_ref_angular_velocity[0], torch.zeros(3))
        torch.testing.assert_close(env.motion_state.linear_velocity[1:], other_velocity, rtol=0, atol=0)
        torch.testing.assert_close(env._filtered_ref_angular_velocity[1:], other_ref_velocity, rtol=0, atol=0)
        torch.testing.assert_close(env.grasp_state.steps[1:], other_steps, rtol=0, atol=0)
        self.assertFalse(env.reference.active[0])
        self.assertEqual(float(returned[0, 123]), 1.0)

    def test_active_horizon_includes_terminal_step_and_matches_fractional_start_boundary(self):
        for settle, ramp, expected_samples in ((1.0, 2.0, 211), (0.1, 0.2, 292)):
            with self.subTest(settle=settle, ramp=ramp):
                env = make_env(settle_seconds=settle, ramp_seconds=ramp, randomize_phase=False)
                for step in range(1, 301):
                    reward_at(env, step, follow_reference=True)
                self.assertEqual(env.active_horizon, expected_samples)
                self.assertEqual(int(env._active_samples[0]), expected_samples)
                self.assertEqual(int(env._tracking_steps[0]), expected_samples)
                env._reset_idx(torch.tensor([0]))
                self.assertEqual(float(env.extras["grasp_motion"]["completed_tracking_fraction"][0]), 1.0)

    def test_early_termination_uses_full_active_horizon_and_never_reports_unmeasured_zero_error(self):
        env = make_env(randomize_phase=False)
        for step in range(1, 95):
            reward_at(env, step, follow_reference=True)
        self.assertEqual(int(env._tracking_steps[0]), 5)
        env.reset_terminated[0] = True
        env._reset_idx(torch.tensor([0]))
        torch.testing.assert_close(env.extras["grasp_motion"]["completed_tracking_fraction"], torch.tensor([5 / 211]))
        self.assertIn("MotionEpisode/position_error_m", env.extras["log"])
        reward_at(env, [1, 95, 95])
        env.reset_terminated[0] = True
        env._reset_idx(torch.tensor([0]))
        self.assertEqual(float(env.extras["grasp_motion"]["completed_tracking_fraction"][0]), 0.0)
        self.assertEqual(env.extras["log"]["MotionEpisode/measured_episodes"], 0.0)
        self.assertNotIn("MotionEpisode/position_error_m", env.extras["log"])
        self.assertNotIn("MotionEpisode/rotation_error_rad", env.extras["log"])

    def test_invalid_pose_cannot_poison_coverage_or_earn_tracking_credit(self):
        env = make_env(randomize_phase=False)
        env.object.data.root_pos_w.torch[0, 0] = float("nan")
        reward = reward_at(env, 120)
        self.assertTrue(torch.isfinite(reward).all())
        self.assertFalse(env.tracking.tracking_valid[0])
        self.assertEqual(float(env.tracking.reward[0]), 0.0)
        self.assertEqual(int(env._active_samples[0]), 0)
        self.assertEqual(int(env._rejected_motion_samples[0]), 1)
        self.assertTrue(torch.isfinite(env._angular_gram).all())
        env._reset_idx(torch.tensor([0]))
        self.assertEqual(env.extras["log"]["MotionEpisode/measured_episodes"], 0.0)
        self.assertEqual(float(env.extras["log"]["MotionEpisode/rejected_samples"]), 1.0)

    def test_pose_gap_rejects_recovery_interval_instead_of_crediting_a_stale_pose_difference(self):
        env = make_env(randomize_phase=False)
        reward_at(env, 120)
        initial_samples = int(env._active_samples[0])
        # A zero quaternion is rejected by the pose estimator, but not by the
        # inherited grasp/distance termination. The environment can recover.
        env.object.data.root_quat_w.torch[0] = 0.0
        reward_at(env, 121)
        self.assertFalse(env.reset_terminated[0])
        self.assertFalse(env.motion_state.valid[0])
        env.object.data.root_pos_w.torch[0, 0] = 0.0002
        env.object.data.root_quat_w.torch[0] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        reward_at(env, 122)
        self.assertFalse(env.motion_state.valid[0], "First recovered pose must reseed, not span an invalid interval")
        self.assertEqual(int(env._active_samples[0]), initial_samples)
        self.assertEqual(int(env._rejected_motion_samples[0]), 2)
        self.assertEqual(float(env.tracking.reward[0]), 0.0)
        torch.testing.assert_close(env.motion_state.linear_velocity[0], torch.zeros(3))
        torch.testing.assert_close(env.motion_state.angular_acceleration[0], torch.zeros(3))
        # Only the next consecutive finite pair can yield an accepted derivative.
        env.object.data.root_pos_w.torch[0, 0] = 0.0003
        reward_at(env, 123)
        self.assertTrue(env.motion_state.valid[0])
        self.assertEqual(int(env._active_samples[0]), initial_samples + 1)
        torch.testing.assert_close(env.motion_state.linear_velocity[0], torch.tensor([0.0006, 0.0, 0.0]))

    def test_zero_motion_command_is_not_an_excitation_task(self):
        with self.assertRaisesRegex(ValueError, "nonzero motion command"):
            make_env(rotation_amplitude=(0.0, 0.0, 0.0), translation_amplitude=(0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
