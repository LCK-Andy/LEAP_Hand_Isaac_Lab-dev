"""CPU-only standalone trainer contract, persistence, and accounting tests."""

import unittest
from pathlib import Path


class StandaloneArgumentsTests(unittest.TestCase):
    def test_recommended_defaults_round_replay_up_without_population_options(self):
        from scripts.tdmpc2.standalone import build_parser, validate_args

        parser = build_parser()
        args = validate_args(parser, parser.parse_args(["--workspace", "/tmp/standalone-test"]))
        self.assertEqual(args.num_envs, 256)
        self.assertEqual(args.seed, 1)
        self.assertEqual(args.replay_capacity, 1_000_192)
        self.assertEqual(args.seed_steps, 1800)
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.updates_per_transition, 1.0)
        self.assertEqual(
            (args.num_samples, args.num_elites, args.num_pi_trajs, args.planner_iterations), (512, 64, 24, 6)
        )
        self.assertTrue(args.compile)
        self.assertEqual(args.max_steps, 10_000_000)
        self.assertFalse(any(name in vars(args) for name in ("population", "member", "disable_selection")))
        self.assertIsInstance(args.workspace, Path)


class StandaloneCheckpointTests(unittest.TestCase):
    def test_full_resume_keeps_replay_rng_debt_and_rejects_foreign_contract(self):
        import tempfile
        import torch

        from scripts.tdmpc2.replay import SequenceReplay
        from scripts.tdmpc2.schedule import UpdateSchedule
        from scripts.tdmpc2.standalone import compact_replay, load_checkpoint, restore_replay, save_checkpoint

        replay = SequenceReplay(32, 2, 3, 1, 3, 2, sample_device="cpu", seed=42)
        for i in range(4):
            replay.add(
                torch.full((2, 3), float(i)),
                torch.zeros(2, 1),
                torch.ones(2),
                torch.full((2, 3), float(i + 1)),
                torch.zeros(2),
                torch.zeros(2),
            )
        schedule = UpdateSchedule(6, 1)
        self.assertEqual(schedule.collect(4, ready=True), 0)
        self.assertEqual(schedule.collect(4, ready=True), 8)
        compact = compact_replay(replay)
        self.assertEqual(compact["buffers"]["obs"].shape, (4, 2, 3))
        self.assertEqual(compact["buffers"]["obs"].untyped_storage().nbytes(), 4 * 2 * 3 * 4)
        contract = {"version": 2, "critic_sha256": "corrected"}
        payload = {
            "agent": {"update_count": 1},
            "replay": compact,
            "trainer": {"frame": 8, "updates": 1, "pending_updates": 7, "schedule": schedule.state_dict()},
            "environment": {"curriculum": 0.2},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(Path(directory), contract, payload)
            loaded = load_checkpoint(path, contract)
            self.assertEqual(loaded["trainer"]["pending_updates"], 7)
            self.assertEqual(loaded["environment"], {"curriculum": 0.2})
            restored = SequenceReplay(32, 2, 3, 1, 3, 2, sample_device="cpu", seed=9)
            restore_replay(restored, loaded["replay"])
            for expected, actual in zip(replay.sample()[:-1], restored.sample()[:-1]):
                torch.testing.assert_close(expected, actual)
            with self.assertRaisesRegex(ValueError, "contract"):
                load_checkpoint(path, {"version": 2, "critic_sha256": "faulty"})
            foreign = Path(directory) / "foreign.pt"
            torch.save({"pbt": {}, "training_state": {}}, foreign)
            with self.assertRaisesRegex(ValueError, "standalone"):
                load_checkpoint(foreign, contract)
            save_checkpoint(Path(directory), contract, payload)
            self.assertTrue((Path(directory) / "previous.pt").is_file())
            self.assertEqual(len(list(Path(directory).glob("*.pt"))), 3)  # includes the foreign fixture


class StandaloneAdditionalCheckpointTests(unittest.TestCase):
    def test_wrapped_replay_retains_ring_and_does_not_bridge_simulator_restart(self):
        import torch

        from scripts.tdmpc2.replay import SequenceReplay
        from scripts.tdmpc2.standalone import compact_replay, restore_replay

        replay = SequenceReplay(8, 2, 1, 1, 3, 8, sample_device="cpu")
        for i in range(6):
            replay.add(
                torch.full((2, 1), float(i)),
                torch.zeros(2, 1),
                torch.ones(2),
                torch.full((2, 1), float(i + 1)),
                torch.zeros(2),
                torch.zeros(2),
            )
        restored = SequenceReplay(8, 2, 1, 1, 3, 8, sample_device="cpu")
        restore_replay(restored, compact_replay(replay))
        for expected, actual in zip(replay.sample()[:-1], restored.sample()[:-1]):
            torch.testing.assert_close(expected, actual)
        restored.start_new_stream()
        restored.add(
            torch.full((2, 1), 100.0),
            torch.zeros(2, 1),
            torch.ones(2),
            torch.full((2, 1), 101.0),
            torch.zeros(2),
            torch.zeros(2),
        )
        self.assertTrue((restored.sample()[0] < 100).all())

    def test_failed_save_retains_previous_latest_and_removes_partial_file(self):
        import tempfile
        from unittest.mock import patch

        from scripts.tdmpc2.standalone import load_checkpoint, save_checkpoint

        state = {
            "agent": {},
            "replay": {},
            "environment": {},
            "trainer": {"frame": 8, "updates": 1, "pending_updates": 7},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(Path(directory), {"critic": "fixed"}, state)
            with patch("torch.save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    save_checkpoint(Path(directory), {"critic": "fixed"}, state)
            self.assertEqual(load_checkpoint(path, {"critic": "fixed"})["trainer"]["updates"], 1)
            self.assertFalse((Path(directory) / "latest.pt.tmp").exists())


class StandaloneAccountingTests(unittest.TestCase):
    def test_warmup_debt_resumes_even_at_collection_budget(self):
        from scripts.tdmpc2.standalone import TrainingProgress

        progress = TrainingProgress(1800, 1.0)
        for _ in range(8):
            progress.collect(256, ready=True)
        self.assertEqual((progress.frame, progress.pending_updates), (2048, 2048))
        progress.updated()
        restored = TrainingProgress(1800, 1.0)
        restored.load_state_dict(progress.state_dict())
        self.assertTrue(restored.has_work(2048))
        for _ in range(2047):
            restored.updated()
        self.assertFalse(restored.has_work(2048))
        self.assertEqual(restored.updates, 2048)

    def test_episode_weighted_totals_and_nonfinite_metrics_fail_fast(self):
        import torch

        from scripts.tdmpc2.standalone import EpisodeMetrics, finite_metrics

        metrics = EpisodeMetrics(2, "cpu")
        metrics.observe(
            torch.tensor([2.0, 10.0]),
            torch.tensor([True, False]),
            {"dexpbt": {"completed_episodes": 1, "success_count": 1}},
        )
        metrics.observe(
            torch.tensor([4.0, 20.0]),
            torch.tensor([True, True]),
            {"dexpbt": {"completed_episodes": 2, "success_count": 0}},
        )
        values = metrics.values()
        self.assertEqual(values["episode/completed"], 3)
        self.assertEqual(values["episode/return"], 12.0)
        self.assertAlmostEqual(values["episode/length"], 4 / 3)
        self.assertEqual(values["episode/success_count"], 1)
        self.assertAlmostEqual(values["episode/success_rate"], 1 / 3)
        with self.assertRaisesRegex(FloatingPointError, "value_loss"):
            finite_metrics({"value_loss": torch.tensor(float("nan"))})
        resumed = EpisodeMetrics(2, "cpu")
        resumed.load_state_dict(metrics.state_dict())
        self.assertEqual(resumed.values(), values)


class StandaloneOwnershipTests(unittest.TestCase):
    def test_workspace_rejects_second_writer_and_fresh_overwrite(self):
        import tempfile

        from scripts.tdmpc2.standalone import WorkspaceOwner

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with WorkspaceOwner(root, resume=False):
                with self.assertRaisesRegex(RuntimeError, "owns"):
                    with WorkspaceOwner(root, resume=True):
                        pass
                (root / "status.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "fresh"):
                with WorkspaceOwner(root, resume=False):
                    pass
            with WorkspaceOwner(root, resume=True):
                pass

    def test_entrypoint_import_does_not_launch_simulator_or_import_population_engine(self):
        import ast
        import importlib
        import sys

        entry = importlib.import_module("scripts.tdmpc2.train")
        self.assertTrue(callable(entry.main))
        self.assertNotIn("isaaclab.app", sys.modules)
        for path in (Path(entry.__file__), Path(entry.__file__).with_name("standalone.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                self.assertFalse(
                    any("pbt" in name or "train_member" in name or name.endswith("runtime") for name in names), names
                )


if __name__ == "__main__":
    unittest.main()
