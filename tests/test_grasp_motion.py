"""Behavioral contract for bounded object-motion references and safe tracking."""

import importlib.util
import sys
import torch
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / (
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_grasp_motion/motion_logic.py"
)
spec = importlib.util.spec_from_file_location("grasp_motion_logic", PATH)
motion = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = motion
spec.loader.exec_module(motion)


class MotionReferenceTests(unittest.TestCase):
    def test_reference_starts_at_fixture_with_zero_velocity_and_acceleration(self):
        cfg = motion.MotionCriteria()
        time = torch.tensor([0.0, cfg.settle_seconds, cfg.settle_seconds - 1e-5])
        phase = torch.tensor([[0.0, 1.0, 2.0]]).repeat(3, 1)
        ref = motion.motion_reference(time, phase, cfg)
        torch.testing.assert_close(ref.position, torch.zeros(3, 3))
        torch.testing.assert_close(ref.quaternion, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1))
        for value in (ref.linear_velocity, ref.angular_velocity, ref.linear_acceleration, ref.angular_acceleration):
            torch.testing.assert_close(value, torch.zeros(3, 3))
        self.assertFalse(ref.active.any())

    def test_reference_is_bounded_and_derivatives_match_motion_not_euler_rates(self):
        cfg = motion.MotionCriteria()
        cfg.rotation_amplitude = (0.3, 0.2, 0.25)
        t = torch.linspace(0.0, 15.0, 701, dtype=torch.float64)
        phases = torch.tensor([[0.3, -1.2, 2.0]], dtype=torch.float64).expand(len(t), -1)
        ref = motion.motion_reference(t, phases, cfg)
        self.assertTrue((ref.position.abs() <= torch.tensor(cfg.translation_amplitude) + 1e-8).all())
        torch.testing.assert_close(ref.quaternion.norm(dim=-1), torch.ones_like(t))
        h = 1e-6
        plus = motion.motion_reference(t + h, phases, cfg)
        minus = motion.motion_reference(t - h, phases, cfg)
        torch.testing.assert_close(
            (plus.position - minus.position) / (2 * h), ref.linear_velocity, atol=1e-8, rtol=1e-5
        )
        torch.testing.assert_close(
            (plus.angular_velocity - minus.angular_velocity) / (2 * h), ref.angular_acceleration, atol=1e-7, rtol=1e-5
        )
        qdot = (plus.quaternion - minus.quaternion) / (2 * h)
        conjugate = ref.quaternion.clone()
        conjugate[:, :3] *= -1
        omega = 2 * motion.multiply_quaternions(qdot, conjugate)[:, :3]
        torch.testing.assert_close(omega, ref.angular_velocity, atol=1e-7, rtol=1e-5)

    def test_invalid_motion_specification_is_rejected(self):
        for field, value in (
            ("ramp_seconds", 0.0),
            ("frequency_hz", (0.0, 0.2, 0.3)),
            ("rotation_amplitude", (-0.1, 0.1, 0.1)),
            ("position_tolerance", 0.0),
        ):
            with self.subTest(field=field):
                cfg = motion.MotionCriteria()
                setattr(cfg, field, value)
                with self.assertRaises(ValueError):
                    motion.motion_reference(torch.zeros(1), torch.zeros(1, 3), cfg)


class MotionKinematicsTests(unittest.TestCase):
    def test_pose_derived_velocity_ignores_solver_velocity_bias_and_resets_cleanly(self):
        estimate = motion.PoseMotionState(1, "cpu")
        p = torch.tensor([[0.0, 0.0, 0.6]])
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        estimate.reset(torch.tensor([0]), p, q)
        estimate.update(p, q, dt=0.1, alpha=1.0)
        torch.testing.assert_close(estimate.angular_velocity, torch.zeros(1, 3))
        q_new = torch.tensor([[0.0, 0.0, 0.0499791693, 0.9987502604]])
        estimate.update(p + torch.tensor([[0.01, 0.0, 0.0]]), q_new, dt=0.1, alpha=1.0)
        torch.testing.assert_close(estimate.linear_velocity, torch.tensor([[0.1, 0.0, 0.0]]))
        torch.testing.assert_close(estimate.angular_velocity, torch.tensor([[0.0, 0.0, 1.0]]))
        estimate.reset(torch.tensor([0]), p, q)
        torch.testing.assert_close(estimate.angular_velocity, torch.zeros(1, 3))


class MotionTrackingTests(unittest.TestCase):
    def test_motion_reward_requires_valid_grasp_and_distinguishes_stationary_object(self):
        cfg = motion.MotionCriteria()
        ref = motion.motion_reference(torch.tensor([4.0, 4.0, 4.0]), torch.zeros(3, 3), cfg)
        pos = ref.position.clone()
        quat = ref.quaternion.clone()
        linvel = ref.linear_velocity.clone()
        angvel = ref.angular_velocity.clone()
        pos[1] = 0.0
        quat[1] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        linvel[1] = 0.0
        angvel[1] = 0.0
        score = motion.score_tracking(pos, quat, linvel, angvel, ref, torch.tensor([True, True, False]), cfg)
        self.assertTrue(score.tracking_valid[0])
        self.assertFalse(score.tracking_valid[1])
        self.assertFalse(score.tracking_valid[2])
        self.assertGreater(score.reward[0], score.reward[1])
        self.assertEqual(score.reward[2].item(), 0.0)
        torch.testing.assert_close(score.position_error, torch.tensor([0.0, ref.position[1].norm(), 0.0]))

    def test_invalid_measured_motion_cannot_earn_tracking_credit(self):
        cfg = motion.MotionCriteria()
        ref = motion.motion_reference(torch.tensor([4.0]), torch.zeros(1, 3), cfg)
        position = torch.full((1, 3), float("nan"))
        score = motion.score_tracking(
            position,
            ref.quaternion,
            ref.linear_velocity,
            ref.angular_velocity,
            ref,
            torch.ones(1, dtype=torch.bool),
            cfg,
        )
        self.assertFalse(score.tracking_valid.item())
        self.assertTrue(torch.isfinite(score.reward).all())
        self.assertEqual(score.reward.item(), 0.0)

    def test_quaternion_sign_is_irrelevant_and_settling_never_counts_as_tracking(self):
        cfg = motion.MotionCriteria()
        ref = motion.motion_reference(torch.tensor([0.0, 4.0]), torch.zeros(2, 3), cfg)
        score = motion.score_tracking(
            ref.position,
            -ref.quaternion,
            ref.linear_velocity,
            ref.angular_velocity,
            ref,
            torch.ones(2, dtype=torch.bool),
            cfg,
        )
        torch.testing.assert_close(score.rotation_error, torch.zeros(2), atol=1e-6, rtol=0)
        self.assertEqual(score.tracking_valid.tolist(), [False, True])
        self.assertEqual(score.reward[0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
