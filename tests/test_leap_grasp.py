"""Behavioral tests for the simulator-independent Stage A grasp contract."""

import importlib.util
import sys
import torch
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / (
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_grasp/grasp_logic.py"
)
spec = importlib.util.spec_from_file_location("leap_grasp_logic", MODULE)
logic = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = logic
spec.loader.exec_module(logic)


class GraspControlTests(unittest.TestCase):
    def test_delta_commands_are_rate_limited_and_zero_holds_target(self):
        previous = torch.tensor([[0.2, -0.2, 0.95]])
        target = logic.advance_joint_targets(
            previous,
            torch.tensor([[100.0, 0.0, 1.0]]),
            -torch.ones_like(previous),
            torch.ones_like(previous),
            max_delta=0.1,
        )
        torch.testing.assert_close(target, torch.tensor([[0.3, -0.2, 1.0]]))
        torch.testing.assert_close(previous, torch.tensor([[0.2, -0.2, 0.95]]))


class GraspScoringTests(unittest.TestCase):
    def measurement(self, normal=2.0):
        forces = torch.zeros(1, 4, 4, 3)
        forces[..., 1] = normal
        return dict(
            normal_forces=forces,
            friction_forces=torch.zeros_like(forces),
            pulp_contact=torch.ones(1, 4, dtype=torch.bool),
            surface_distance=torch.zeros(1, 4),
            slip_speed=torch.zeros(1, 4),
            support_force=torch.zeros(1, 4),
            object_distance=torch.zeros(1),
            target_delta=torch.zeros(1, 16),
        )

    def test_four_in_band_pulps_beat_partial_contact_and_overgripping(self):
        cfg = logic.GraspCriteria()
        good = logic.score_grasp(**self.measurement(), cfg=cfg)
        partial_input = self.measurement()
        partial_input["normal_forces"][:, :, 1] = 0
        partial_input["pulp_contact"][:, 1] = False
        partial = logic.score_grasp(**partial_input, cfg=cfg)
        crushed = logic.score_grasp(**self.measurement(normal=100.0), cfg=cfg)
        self.assertTrue(good.valid.item())
        self.assertFalse(partial.valid.item())
        self.assertFalse(crushed.valid.item())
        self.assertGreater(good.reward.item(), partial.reward.item())
        self.assertGreater(good.reward.item(), crushed.reward.item())
        self.assertTrue(crushed.unsafe.item())

    def test_invalid_samples_and_unmeasured_support_never_count_as_valid(self):
        for kind in ("substep_dropout", "shear", "support", "slip", "non_pulp", "nan"):
            with self.subTest(kind=kind):
                inputs = self.measurement()
                if kind == "substep_dropout":
                    inputs["normal_forces"][:, 1, 2] = 0.0
                elif kind == "shear":
                    inputs["friction_forces"][..., 0] = 3.0
                elif kind == "support":
                    inputs["support_force"][:, 0] = 1.0
                elif kind == "slip":
                    inputs["slip_speed"][:, 0] = 0.1
                elif kind == "non_pulp":
                    inputs["pulp_contact"][:, 0] = False
                else:
                    inputs["normal_forces"][:, 0, 0, 0] = float("nan")
                score = logic.score_grasp(**inputs, cfg=logic.GraspCriteria())
                self.assertFalse(score.valid.item())
                self.assertTrue(torch.isfinite(score.reward).all())
                if kind == "nan":
                    self.assertTrue(score.unsafe.item())

    def test_total_force_not_only_normal_triggers_the_safety_ceiling(self):
        inputs = self.measurement(normal=10.0)
        inputs["friction_forces"][..., 0] = 20.0
        score = logic.score_grasp(**inputs, cfg=logic.GraspCriteria())
        self.assertTrue(score.unsafe.item())

    def test_more_approach_progress_helps_before_any_contact(self):
        far = self.measurement(normal=0.0)
        near = self.measurement(normal=0.0)
        far["pulp_contact"].zero_()
        near["pulp_contact"].zero_()
        far["surface_distance"].fill_(0.10)
        near["surface_distance"].fill_(0.01)
        self.assertGreater(
            logic.score_grasp(**near, cfg=logic.GraspCriteria()).reward.item(),
            logic.score_grasp(**far, cfg=logic.GraspCriteria()).reward.item(),
        )


class ConditionStampTests(unittest.TestCase):
    def test_physical_settings_change_comparability_and_mapping_order_does_not(self):
        baseline = {"gravity": [0, 0, -9.81], "material": {"friction": 1.0}}
        stamp = logic.grasp_condition_key(baseline)
        reordered = {"material": {"friction": 1.0}, "gravity": [0, 0, -9.81]}
        self.assertEqual(stamp, logic.grasp_condition_key(reordered))
        self.assertNotEqual(stamp, logic.grasp_condition_key({**baseline, "gravity": [0, 0, -5.0]}))
        self.assertNotEqual(stamp, logic.grasp_condition_key({**baseline, "material": {"friction": 0.5}}))


class PersistentHoldTests(unittest.TestCase):
    def test_a_brief_support_contact_removes_retention_credit_for_the_episode(self):
        tracker = logic.GraspEpisodeState(1, "cpu")
        for support in (True, False, False):
            tracker.advance(
                torch.tensor([not support]), torch.tensor([False]), 0.1, 0.2, supported=torch.tensor([support])
            )
        metrics = tracker.completed(torch.tensor([0]), torch.tensor([True]), horizon=3)
        self.assertEqual(metrics["completed_retained"].item(), 0.0)
        tracker.reset(torch.tensor([0]))
        tracker.advance(torch.tensor([True]), torch.tensor([False]), 0.1, 0.2, supported=torch.tensor([False]))
        self.assertEqual(
            tracker.completed(torch.tensor([0]), torch.tensor([True]), horizon=1)["completed_retained"].item(), 1.0
        )

    def test_hold_is_continuous_and_success_does_not_restart_it(self):
        tracker = logic.GraspEpisodeState(2, "cpu")
        for _ in range(3):
            tracker.advance(
                torch.tensor([True, False]),
                torch.tensor([False, False]),
                dt=0.1,
                hold_seconds=0.3,
            )
        torch.testing.assert_close(tracker.hold_seconds, torch.tensor([0.3, 0.0]))
        self.assertEqual(tracker.success.tolist(), [True, False])
        tracker.advance(
            torch.tensor([True, True]),
            torch.tensor([False, False]),
            dt=0.1,
            hold_seconds=0.3,
        )
        torch.testing.assert_close(tracker.hold_seconds, torch.tensor([0.4, 0.1]))
        tracker.advance(
            torch.tensor([False, False]),
            torch.tensor([False, True]),
            dt=0.1,
            hold_seconds=0.3,
        )
        self.assertEqual(tracker.success.tolist(), [True, False])
        torch.testing.assert_close(tracker.hold_seconds, torch.zeros(2))
        metrics = tracker.completed(torch.tensor([0, 1]), torch.tensor([True, False]), horizon=10)
        torch.testing.assert_close(metrics["completed_valid_fraction"], torch.tensor([0.4, 0.1]))
        torch.testing.assert_close(metrics["completed_overload_fraction"], torch.tensor([0.0, 0.1]))
        tracker.reset(torch.tensor([0]))
        self.assertEqual(tracker.steps.tolist(), [0, 5])
        self.assertEqual(tracker.success.tolist(), [False, False])


if __name__ == "__main__":
    unittest.main()
