"""Simulator-free tests of the task's public curriculum/measurement seams."""

import importlib.util
import pathlib
import sys
import unittest

TASK = pathlib.Path(__file__).resolve().parents[1] / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt"
spec = importlib.util.spec_from_file_location("dexpbt_evaluation", TASK / "evaluation.py")
evaluation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluation
spec.loader.exec_module(evaluation)


class CurriculumTests(unittest.TestCase):
    def test_episode_weighted_windows_not_reset_batch_votes(self):
        curriculum = evaluation.SuccessToleranceCurriculum(window_episodes=4, required_windows=2)
        self.assertEqual(curriculum.tolerance, 0.2)
        curriculum.observe([True])
        curriculum.observe([False, False, False])
        self.assertEqual(curriculum.tolerance, 0.2)
        curriculum.observe([True] * 7)
        self.assertEqual(curriculum.tolerance, 0.2)
        curriculum.observe([True])
        self.assertAlmostEqual(curriculum.tolerance, 0.18)

    def test_state_restores_partial_window_and_floor(self):
        original = evaluation.SuccessToleranceCurriculum(window_episodes=2, required_windows=1)
        original.observe([True])
        restored = evaluation.SuccessToleranceCurriculum(window_episodes=2, required_windows=1)
        restored.set_state(original.get_state())
        restored.observe([True])
        self.assertAlmostEqual(restored.tolerance, 0.18)
        self.assertEqual(original.tolerance, 0.2)
        restored.observe([True] * 100)
        self.assertEqual(restored.tolerance, 0.1)

    def test_checkpoint_validation_rejects_corrupted_or_different_window(self):
        curriculum = evaluation.SuccessToleranceCurriculum(window_episodes=4)
        state = curriculum.get_state()
        state["tolerance"] = float("nan")
        with self.assertRaises(ValueError):
            curriculum.set_state(state)
        with self.assertRaises(ValueError):
            curriculum.set_state(evaluation.SuccessToleranceCurriculum(window_episodes=8).get_state())
        self.assertEqual(curriculum.tolerance, 0.2)

    def test_adapter_initializes_actual_reward_and_reads_finished_sticky_flags(self):
        from types import SimpleNamespace
        import torch

        class RewardManager:
            def __init__(self):
                self.term = SimpleNamespace(
                    params={"pos_std": 0.1}, func=SimpleNamespace(succeeded=torch.tensor([True, False, True]))
                )

            def get_term_cfg(self, name):
                return self.term

            def set_term_cfg(self, name, term):
                self.term = term

        env = SimpleNamespace(reward_manager=RewardManager(), extras={"log": {"Metrics/success_rate": 1.0}})
        evaluation.anneal_success_tolerance(env, [0], window_episodes=2, n_succ=1)
        self.assertEqual(env.reward_manager.term.params["pos_std"], 0.2)
        self.assertEqual(env._dexpbt_anneal.completed_episodes, 0)
        env._dexpbt_in_step = True
        env.extras.clear()
        evaluation.anneal_success_tolerance(env, [0, 2], window_episodes=2, n_succ=1)
        self.assertAlmostEqual(env.reward_manager.term.params["pos_std"], 0.18)
        self.assertEqual(env._dexpbt_anneal.completed_episodes, 2)


class EpisodeMetricsTests(unittest.TestCase):
    def test_condition_and_predicate_share_fixed_protocol_thresholds(self):
        import torch

        condition = evaluation.metric_condition(0.0, 0.2)
        self.assertEqual(evaluation.FIXED_POSITION_TOLERANCE, 0.1)
        self.assertEqual(evaluation.FIXED_ROTATION_TOLERANCE, 0.5)
        self.assertEqual(evaluation.FIXED_CONTACT_THRESHOLD, 0.01)
        self.assertEqual(condition["position_tolerance"], evaluation.FIXED_POSITION_TOLERANCE)
        self.assertEqual(condition["rotation_tolerance"], evaluation.FIXED_ROTATION_TOLERANCE)
        self.assertEqual(condition["contact_threshold"], evaluation.FIXED_CONTACT_THRESHOLD)
        metrics = evaluation.CompletedEpisodeMetrics(4, "cpu")
        metrics.observe(
            torch.tensor([condition["position_tolerance"], 0.0, 0.0, 0.099], dtype=torch.float64),
            torch.tensor([0.0, condition["rotation_tolerance"], 0.0, 0.499], dtype=torch.float64),
            torch.tensor(
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [condition["contact_threshold"], 1.0, 0.0, 0.0],
                    [0.011, 0.011, 0.0, 0.0],
                ],
                dtype=torch.float64,
            ),
            condition,
        )
        for env_id, expected in enumerate([0, 0, 0, 1]):
            self.assertEqual(metrics.complete([env_id])["success_count"], expected)

    def test_completed_episode_stats_ignore_unfinished_and_require_simultaneous_contact(self):
        import torch

        metrics = evaluation.CompletedEpisodeMetrics(2, "cpu")
        condition = evaluation.metric_condition(0.0, 0.2)
        metrics.observe(
            torch.tensor([0.05, 0.2]),
            torch.tensor([0.2, 1.0]),
            torch.tensor([[0.02, 0.02, 0.0, 0.0], [0.02, 0.0, 0.0, 0.0]]),
            condition,
        )
        metrics.observe(
            torch.tensor([0.3, 0.05]),
            torch.tensor([0.8, 0.2]),
            torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.02, 0.0, 0.0]]),
            condition,
        )
        report = metrics.complete([1])
        self.assertEqual(report["completed_episodes"], 1)
        self.assertEqual(report["success_count"], 0)
        self.assertAlmostEqual(report["position_error_sum"], 0.125)
        self.assertEqual(report["episode_length_sum"], 2)
        report = metrics.complete([0])
        self.assertEqual(report["success_count"], 1)
        self.assertEqual(report["good_contact_fraction_sum"], 0.5)
        self.assertEqual(metrics.complete([0])["completed_episodes"], 0)

    def test_scheduler_progress_with_identical_actual_parameters_remains_comparable(self):
        import torch

        actual = '{"gravity_adr": [[0, 0, 0], [0, 0, 0]], "noise": 0.0}'
        first = evaluation.metric_condition(0.0, 0.2, adr_parameters=actual)
        second = evaluation.metric_condition(1 / (512 * 10), 0.2, adr_parameters=actual)
        metrics = evaluation.CompletedEpisodeMetrics(2, "cpu")
        for condition in (first, second):
            metrics.observe(torch.zeros(2), torch.zeros(2), torch.ones((2, 4)), condition)
        report = metrics.complete([0, 1])
        self.assertEqual(report["mixed_condition_episodes"], 0)
        self.assertEqual(report["groups"][0]["completed_episodes"], 2)
        self.assertAlmostEqual(report["adr_difficulty_sum"], 1 / (512 * 10))
        canonical = evaluation.comparison_condition(first)
        self.assertEqual(canonical, evaluation.comparison_condition(second))
        self.assertEqual(report["groups"][0]["condition"], canonical)
        self.assertNotIn("adr_fraction", canonical)
        self.assertIn("adr_fraction", first)

    def test_unknown_actual_parameters_keep_fraction_in_comparison_identity(self):
        for actual in (None, "", "{}", "not-json", {}):
            with self.subTest(actual=actual):
                first = evaluation.metric_condition(0.0, 0.2, adr_parameters=actual)
                second = evaluation.metric_condition(0.1, 0.2, adr_parameters=actual)
                self.assertNotEqual(evaluation.comparison_condition(first), evaluation.comparison_condition(second))

    def test_actual_gravity_or_noise_change_still_rejects_episode(self):
        import torch

        for changed in ('{"gravity": -9.81, "noise": 0}', '{"gravity": 0, "noise": 0.01}'):
            with self.subTest(changed=changed):
                metrics = evaluation.CompletedEpisodeMetrics(1, "cpu")
                for actual in ('{"gravity": 0, "noise": 0}', changed):
                    condition = evaluation.metric_condition(0.0, 0.2, adr_parameters=actual)
                    metrics.observe(torch.zeros(1), torch.zeros(1), torch.ones((1, 4)), condition)
                report = metrics.complete([0])
                self.assertEqual(report["mixed_condition_episodes"], 1)
                self.assertEqual(report["groups"], [])

    def test_changed_difficulty_excluded_from_comparable_groups_and_discard_clears_partial(self):
        import torch

        metrics = evaluation.CompletedEpisodeMetrics(2, "cpu")
        pos, rot = torch.tensor([0.05, 0.05]), torch.tensor([0.1, 0.1])
        force = torch.full((2, 4), 1.0)
        metrics.observe(pos, rot, force, evaluation.metric_condition(0.0, 0.2))
        metrics.observe(pos, rot, force, evaluation.metric_condition(0.1, 0.2))
        report = metrics.complete([0])
        self.assertEqual(report["success_count"], 1)
        self.assertEqual(report["mixed_condition_episodes"], 1)
        self.assertEqual(report["groups"], [])
        self.assertAlmostEqual(report["adr_difficulty_sum"], 0.05)
        metrics.discard()
        self.assertEqual(metrics.complete([1])["completed_episodes"], 0)

    def test_exact_boundaries_are_not_success_and_groups_preserve_sufficient_stats(self):
        import torch

        metrics = evaluation.CompletedEpisodeMetrics(3, "cpu")
        metrics.observe(
            torch.tensor([0.1, 0.01, 0.01]),
            torch.tensor([0.1, 0.5, 0.01]),
            torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.01, 1.0, 0.0, 0.0]]),
            evaluation.metric_condition(0.0, 0.2),
        )
        report = metrics.complete([0, 1, 2], "heldout_fixed_definition_v1")
        self.assertEqual(report["success_count"], 0)
        self.assertEqual(report["groups"][0]["completed_episodes"], 3)
        self.assertEqual(report["protocol"], "heldout_fixed_definition_v1")
        self.assertAlmostEqual(report["position_error_sum"], 0.12)


if __name__ == "__main__":
    unittest.main()
