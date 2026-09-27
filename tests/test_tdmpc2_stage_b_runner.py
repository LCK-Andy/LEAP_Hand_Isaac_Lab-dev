"""Stage B CLI and runner selection without importing Isaac Lab."""

import ast
import unittest
from pathlib import Path


class StageBArgumentsTests(unittest.TestCase):
    def parse(self, *tokens):
        from scripts.tdmpc2.standalone import build_parser, validate_args

        parser = build_parser()
        return validate_args(parser, parser.parse_args(["--workspace", "/tmp/stage-b-tdmpc2", *tokens]))

    def test_stage_b_accepts_native_task_and_uses_five_episode_seed_budget(self):
        args = self.parse("--task", "Isaac-Grasp-Motion-Cube-LEAP-Direct-v0")
        self.assertEqual(args.seed_steps, 1500)
        self.assertEqual(args.num_envs, 256)
        self.assertEqual(args.updates_per_transition, 1.0)
        self.assertTrue(args.compile)

    def test_explicit_seed_budget_wins_and_dexsuite_default_is_preserved(self):
        self.assertEqual(self.parse().seed_steps, 1800)
        self.assertEqual(
            self.parse("--task", "Isaac-Grasp-Motion-Cube-LEAP-Direct-v0", "--seed-steps", "77").seed_steps, 77
        )

    def test_stage_a_accepts_native_task_with_same_seed_budget(self):
        args = self.parse("--task", "Isaac-Grasp-Cube-LEAP-Direct-v0")
        self.assertEqual(args.seed_steps, 1500)
        self.assertEqual(args.num_envs, 256)
        self.assertEqual(args.updates_per_transition, 1.0)
        self.assertTrue(args.compile)

    def test_unknown_task_stays_refused(self):
        with self.assertRaises(SystemExit):
            self.parse("--task", "Isaac-Unknown-Task-v0")

    def test_stage_b_runner_import_is_simulator_and_population_free(self):
        import importlib
        import sys

        module = importlib.import_module("scripts.tdmpc2.train")
        self.assertNotIn("isaaclab.app", sys.modules)
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            self.assertFalse(any("pbt" in name or "train_member" in name for name in names))


if __name__ == "__main__":
    unittest.main()
