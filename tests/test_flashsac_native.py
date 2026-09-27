"""CPU-only contracts for the single-learner native FlashSAC entrypoint."""

import unittest

from scripts.flashsac.runtime import build_parser, validate_args


class NativeConfigTests(unittest.TestCase):
    def test_recommended_profile_and_explicit_transition_budget(self):
        parser = build_parser()
        args = validate_args(parser, parser.parse_args(["--workspace", "/tmp/flashsac-test"]))
        self.assertEqual(args.num_envs, 1024)
        self.assertEqual(args.max_steps, 50_000_896)
        self.assertEqual(args.batch_size, 2048)
        self.assertEqual(args.updates_per_vector_step, 2)
        self.assertEqual(args.warmup, 100_000)
        self.assertEqual(args.replay_capacity, 4_000_000)
        self.assertTrue(args.compile)
        self.assertTrue(args.amp)
        self.assertNotIn("pbt", vars(args))

    def test_spaces_and_terminal_successors_preserve_native_contract(self):
        import torch
        from types import SimpleNamespace

        from scripts.flashsac.runtime import native_spaces, native_transition

        observations, actions = native_spaces(4)
        self.assertEqual(observations.shape, (4, 1745))
        self.assertEqual(actions.shape, (4, 23))
        self.assertTrue((actions.high == 1).all())
        self.assertTrue((actions.low == -1).all())
        old = torch.zeros(4, 1745)
        terminal = torch.full_like(old, 7)
        action = torch.full((4, 23), 0.7)
        result = SimpleNamespace(
            next_obs=terminal,
            obs=torch.zeros_like(old),
            reward=torch.ones(4),
            terminated=torch.tensor([True, False, False, False]),
            truncated=torch.tensor([False, True, False, False]),
        )
        transition = native_transition(old, action, result)
        self.assertIs(transition["next_observation"], terminal)
        self.assertIs(transition["action"], action)
        self.assertTrue(transition["truncated"][1])

    def test_only_ready_vectors_trigger_two_updates_without_warmup_debt(self):
        from scripts.flashsac.runtime import VectorProgress

        progress = VectorProgress(1024, 2)
        self.assertEqual(progress.collect(False), 0)
        self.assertEqual(progress.collect(True), 2)
        progress.updated()
        progress.updated()
        self.assertEqual(progress.frame, 2048)
        self.assertEqual(progress.updates, 2)
        self.assertEqual(progress.collect(True), 2)


class CheckpointTests(unittest.TestCase):
    def test_atomic_two_generations_and_stale_contract_refusal(self):
        import tempfile
        from pathlib import Path

        from scripts.flashsac.runtime import load_checkpoint, save_checkpoint

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            contract = {"task": "native", "source": "hash-one"}
            state = {"agent": {}, "trainer": {"frame": 1}, "environment": {}, "episodes": {}}
            save_checkpoint(directory, contract, state)
            save_checkpoint(directory, contract, {**state, "trainer": {"frame": 2}})
            self.assertEqual(load_checkpoint(directory / "previous.pt", contract)["trainer"]["frame"], 1)
            self.assertEqual(load_checkpoint(directory / "latest.pt", contract)["trainer"]["frame"], 2)
            with self.assertRaisesRegex(ValueError, "stale"):
                save_checkpoint(directory, {"source": "hash-two"}, state)
            self.assertEqual(load_checkpoint(directory / "latest.pt", contract)["trainer"]["frame"], 2)
            (directory / "latest.pt.tmp").write_bytes(b"interrupted")
            with self.assertRaisesRegex(ValueError, "Stale"):
                save_checkpoint(directory, contract, state)

    def test_progress_resume_rejects_inconsistent_transition_count(self):
        from scripts.flashsac.runtime import VectorProgress

        progress = VectorProgress(4, 2)
        with self.assertRaises(ValueError):
            progress.load_state_dict({"frame": 5, "vector_steps": 1, "updates": 0})
        progress.load_state_dict({"frame": 4, "vector_steps": 1, "updates": 2})
        self.assertEqual(progress.collect(False), 0)
        self.assertEqual(progress.updates, 2)

    def test_entrypoint_import_does_not_load_simulator_or_learner(self):
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                "-c",
                "import scripts.flashsac.train; import sys; "
                "assert not any(n.startswith(('pxr', 'isaaclab', 'flash_rl')) for n in sys.modules); "
                "assert 'scripts.tdmpc2.agent' not in sys.modules",
            ],
            check=True,
        )


class SnapshotConfigTests(unittest.TestCase):
    def test_real_snapshot_dataclass_uses_gradient_update_schedule(self):
        try:
            from flash_rl.buffers import torch_buffer
        except ImportError:
            self.skipTest("FlashSAC is not installed")

        if getattr(torch_buffer, "N_STEP_DISCOUNT_VERSION", None) != 1:
            self.skipTest("Run with patched immutable FlashSAC snapshot on PYTHONPATH")
        from scripts.flashsac.runtime import make_agent_config

        parser = build_parser()
        args = validate_args(parser, parser.parse_args(["--workspace", "/tmp/flash-test"]))
        args.device = "cpu"
        cfg = make_agent_config(args)
        self.assertEqual(cfg.learning_rate_warmup_step, 0)
        self.assertEqual(cfg.learning_rate_decay_step, 97658)
        self.assertEqual((cfg.n_step, cfg.gamma), (3, 0.99))
        self.assertEqual((cfg.actor_num_blocks, cfg.actor_hidden_dim), (2, 128))
        self.assertEqual((cfg.critic_num_blocks, cfg.critic_hidden_dim), (2, 256))
        self.assertEqual((cfg.critic_num_bins, cfg.critic_min_v, cfg.critic_max_v), (101, -5, 5))
        self.assertEqual(cfg.critic_target_update_tau, 0.01)
        self.assertTrue(cfg.normalize_reward)
        self.assertEqual(cfg.actor_update_period, 2)

    def test_real_learner_updates_and_training_state_restore_without_replay(self):
        import copy
        import torch

        try:
            from flash_rl.buffers import torch_buffer
        except ImportError:
            self.skipTest("FlashSAC is not installed")

        if getattr(torch_buffer, "N_STEP_DISCOUNT_VERSION", None) != 1:
            self.skipTest("Run with patched immutable FlashSAC snapshot on PYTHONPATH")
        from scripts.flashsac.runtime import (
            learner_state,
            learning_diagnostics,
            learning_snapshot,
            make_agent,
            make_agent_config,
            restore_learner,
        )

        torch.set_num_threads(2)
        parser = build_parser()
        args = validate_args(
            parser,
            parser.parse_args(
                [
                    "--workspace",
                    "/tmp/flash-learning-test",
                    "--num_envs",
                    "4",
                    "--replay-capacity",
                    "64",
                    "--warmup",
                    "8",
                    "--batch-size",
                    "8",
                    "--buffer-device",
                    "cpu",
                    "--no-compile",
                    "--no-amp",
                ]
            ),
        )
        args.device = "cpu"
        agent = make_agent(args, make_agent_config(args))
        before = learning_snapshot(agent)
        for _ in range(4):
            agent.process_transition(
                {
                    "observation": torch.randn(4, 1745),
                    "next_observation": torch.randn(4, 1745),
                    "action": torch.rand(4, 23) * 2 - 1,
                    "reward": torch.ones(4),
                    "terminated": torch.zeros(4, dtype=torch.bool),
                    "truncated": torch.zeros(4, dtype=torch.bool),
                }
            )
        self.assertTrue(agent.can_start_training())
        agent.update()
        agent.update()
        diagnostics = learning_diagnostics(agent, before)
        self.assertEqual(diagnostics["learning/critic_optimizer_state_coverage"], 1)
        self.assertEqual(diagnostics["learning/actor_optimizer_state_coverage"], 1)
        saved = copy.deepcopy(learner_state(agent))
        restored = make_agent(args, make_agent_config(args))
        restore_learner(restored, saved)
        self.assertFalse(restored.can_start_training())
        self.assertEqual(restored._update_step, 2)
        self.assertEqual(restored._critic.scheduler.state_dict(), agent._critic.scheduler.state_dict())
        self.assertEqual(restored.reward_normalizer.G_r.abs().sum().item(), 0)
        torch.testing.assert_close(restored.reward_normalizer.G_rms.var, agent.reward_normalizer.G_rms.var)
        observation = {"next_observation": torch.randn(4, 1745)}
        torch.testing.assert_close(
            agent.sample_actions(1, observation, False), restored.sample_actions(1, observation, False)
        )


if __name__ == "__main__":
    unittest.main()
