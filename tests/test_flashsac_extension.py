"""Simulator-free seams for absolute-budget extension with an unchanged learner schedule."""

import copy
import hashlib
import unittest
from pathlib import Path

from scripts.flashsac import runtime

# Original downloaded 50,000,896-transition contract, not hashes of the code under test.
OLD_RUNNER_HASHES = {
    "native/scripts/flashsac/runtime.py": "65a7f5a82d3e0215bd1a814c9697b550fda9dfda62f3182151bddfc0e7175a88",
    "native/scripts/flashsac/train.py": "4d93f655a599c4a1cf0b6ff20ef394056d671deca8cce62f5ba69daba6bb9e4f",
}


def original_contract():
    return {
        "algorithm": "FlashSAC",
        "pbt": False,
        "max_steps": 50_000_896,
        "num_envs": 1024,
        "updates_per_vector_step": 2,
        "obs_dim": 1745,
        "native_action_scale": 0.1,
        "agent": {
            "actor_bc_alpha": 0.0,
            "actor_hidden_dim": 128,
            "actor_noise_zeta_max": 16,
            "actor_noise_zeta_mu": 2.0,
            "actor_num_blocks": 2,
            "actor_update_period": 2,
            "asymmetric_observation": False,
            "buffer_device_type": "cuda:0",
            "buffer_max_length": 4_000_000,
            "buffer_min_length": 100_000,
            "compile_mode": "max-autotune",
            "critic_hidden_dim": 256,
            "critic_max_v": 5.0,
            "critic_min_v": -5.0,
            "critic_num_bins": 101,
            "critic_num_blocks": 2,
            "critic_target_update_tau": 0.01,
            "device_type": "cuda:0",
            "gamma": 0.99,
            "learning_rate_decay_rate": 1.0,
            "learning_rate_decay_step": 97658,
            "learning_rate_end": 0.00015,
            "learning_rate_init": 0.0003,
            "learning_rate_peak": 0.0003,
            "learning_rate_warmup_rate": "1e-6",
            "learning_rate_warmup_step": 0,
            "load_optimizer": True,
            "load_reward_normalizer": True,
            "n_step": 3,
            "normalize_reward": True,
            "normalized_G_max": 5.0,
            "sample_batch_size": 2048,
            "seed": 0,
            "temp_initial_value": 0.01,
            "temp_target_entropy": -10.998173388667798,
            "temp_target_sigma": 0.15,
            "use_amp": True,
            "use_compile": True,
        },
        "source_sha256": {
            **OLD_RUNNER_HASHES,
            "native/scripts/tdmpc2/env.py": "f4f7fe9c6d18918f15b1a4394b0f384d005bb352c6dad99586c2664418166721",
            "flash/flash_rl/agents/flashSAC/agent.py": "c204d0beeb01c9d1ad7cced1e76f3cba8655dbb7b17d133384e168518d55bd35",
        },
    }


class ContractExtensionTests(unittest.TestCase):
    def test_known_runner_upgrade_requires_explicit_larger_budget(self):
        old = original_contract()
        current = copy.deepcopy(old)
        current["max_steps"] = 150_000_896
        for name in OLD_RUNNER_HASHES:
            current["source_sha256"][name] = hashlib.sha256(
                (Path(runtime.__file__).parent / Path(name).name).read_bytes()
            ).hexdigest()
        before = copy.deepcopy((old, current))
        runtime.validate_checkpoint_contract(old, current, allow_budget_extension=True)
        self.assertEqual((old, current), before)
        with self.assertRaisesRegex(ValueError, "contract"):
            runtime.validate_checkpoint_contract(old, current)

    def test_only_explicit_extension_allows_cudagraph_workaround(self):
        old = original_contract()
        current = copy.deepcopy(old)
        current["max_steps"] = 150_000_896
        current["agent"]["compile_mode"] = "max-autotune-no-cudagraphs"
        before = copy.deepcopy((old, current))
        runtime.validate_checkpoint_contract(old, current, allow_budget_extension=True)
        self.assertEqual((old, current), before)
        runtime.validate_checkpoint_contract(current, copy.deepcopy(current))
        with self.assertRaises(ValueError):
            runtime.validate_checkpoint_contract(old, current)
        for key, value in (("use_amp", False), ("use_compile", False), ("learning_rate_end", 0.0003)):
            drift = copy.deepcopy(current)
            drift["agent"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runtime.validate_checkpoint_contract(old, drift, allow_budget_extension=True)
        for old_mode, new_mode in (
            ("max-autotune", "default"),
            ("max-autotune", "reduce-overhead"),
            ("auto", "max-autotune-no-cudagraphs"),
            ("max-autotune-no-cudagraphs", "max-autotune"),
        ):
            source, target = copy.deepcopy(old), copy.deepcopy(current)
            source["agent"]["compile_mode"] = old_mode
            target["agent"]["compile_mode"] = new_mode
            with self.subTest(modes=(old_mode, new_mode)), self.assertRaises(ValueError):
                runtime.validate_checkpoint_contract(source, target, allow_budget_extension=True)

    def test_extension_rejects_every_other_contract_drift(self):
        old = original_contract()
        mutations = {
            "equal budget": lambda c: c.update(max_steps=50_000_896),
            "shrink": lambda c: c.update(max_steps=50_000_000),
            "architecture": lambda c: c["agent"].update(actor_hidden_dim=64),
            "learning rate": lambda c: c["agent"].update(learning_rate_end=0.0003),
            "stretched schedule": lambda c: c["agent"].update(learning_rate_decay_step=292972),
            "replay": lambda c: c["agent"].update(buffer_max_length=8_000_000),
            "native action": lambda c: c.update(native_action_scale=0.2),
            "environment count": lambda c: c.update(num_envs=512),
            "update cadence": lambda c: c.update(updates_per_vector_step=4),
            "native source": lambda c: c["source_sha256"].update({"native/scripts/tdmpc2/env.py": "drift"}),
            "Flash source": lambda c: c["source_sha256"].update({"flash/flash_rl/agents/flashSAC/agent.py": "drift"}),
            "missing source": lambda c: c["source_sha256"].pop("native/scripts/tdmpc2/env.py"),
            "extra source": lambda c: c["source_sha256"].update({"untracked.py": "drift"}),
            "extra schema key": lambda c: c.update(schedule_steps=50_000_896),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                current = copy.deepcopy(old)
                current["max_steps"] = 150_000_896
                mutate(current)
                with self.assertRaises(ValueError):
                    runtime.validate_checkpoint_contract(old, current, allow_budget_extension=True)

    def test_unknown_old_runner_cannot_use_upgrade_bridge(self):
        for name in OLD_RUNNER_HASHES:
            old = original_contract()
            old["source_sha256"][name] = "unknown-old-runner"
            current = copy.deepcopy(old)
            current["max_steps"] = 150_000_896
            for runner in OLD_RUNNER_HASHES:
                current["source_sha256"][runner] = hashlib.sha256(
                    (Path(runtime.__file__).parent / Path(name).name).read_bytes()
                ).hexdigest()
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime.validate_checkpoint_contract(old, current, allow_budget_extension=True)

    def test_upgrade_rejects_unverified_destination_runner_hash(self):
        old = original_contract()
        current = copy.deepcopy(old)
        current["max_steps"] = 150_000_896
        for name in OLD_RUNNER_HASHES:
            current["source_sha256"][name] = "0" * 64
        with self.assertRaises(ValueError):
            runtime.validate_checkpoint_contract(old, current, allow_budget_extension=True)

    def test_new_runner_allows_strict_resume_and_future_budget_only_extension(self):
        old = original_contract()
        for name in OLD_RUNNER_HASHES:
            old["source_sha256"][name] = hashlib.sha256(
                (Path(runtime.__file__).parent / Path(name).name).read_bytes()
            ).hexdigest()
        old["max_steps"] = 150_000_896
        runtime.validate_checkpoint_contract(old, copy.deepcopy(old))
        current = copy.deepcopy(old)
        current["max_steps"] = 250_000_896
        runtime.validate_checkpoint_contract(old, current, allow_budget_extension=True)
        self.assertEqual(runtime.OLD_RUNNER_SOURCE_SHA256, OLD_RUNNER_HASHES)


class ArgumentTests(unittest.TestCase):
    def test_absolute_extension_and_default_schedule(self):
        parser = runtime.build_parser()
        args = runtime.validate_args(parser, parser.parse_args(["--workspace", "/tmp/flash-extension"]))
        self.assertEqual(args.schedule_steps, 50_000_896)
        self.assertFalse(args.extend_budget)
        args = runtime.validate_args(
            parser,
            parser.parse_args(
                [
                    "--workspace",
                    "/tmp/flash-extension",
                    "--max-steps",
                    "150000896",
                    "--schedule-steps",
                    "50000896",
                    "--extend-budget",
                    "--checkpoint",
                    "/tmp/old/latest.pt",
                ]
            ),
        )
        self.assertEqual((args.max_steps, args.schedule_steps), (150_000_896, 50_000_896))
        self.assertTrue(args.extend_budget)
        self.assertEqual(
            (args.num_envs, args.replay_capacity, args.warmup, args.updates_per_vector_step),
            (1024, 4_000_000, 100_000, 2),
        )
        self.assertTrue(args.amp and args.compile)
        self.assertEqual(args.compile_mode, "auto")
        progress = runtime.VectorProgress(1024, 2)
        progress.load_state_dict({"frame": 50_000_896, "updates": 97460, "vector_steps": 48829})
        self.assertEqual(progress.collect(False), 0)  # Replay refill retains update count, never pays warmup debt.
        self.assertEqual(progress.updates, 97460)
        while progress.frame < args.max_steps:
            progress.collect(False)
        self.assertEqual(progress.frame, 150_001_664)

    def test_cudagraph_workaround_is_explicit_cli_mode(self):
        parser = runtime.build_parser()
        args = runtime.validate_args(
            parser,
            parser.parse_args(
                [
                    "--workspace",
                    "/tmp/flash-extension",
                    "--compile-mode",
                    "max-autotune-no-cudagraphs",
                ]
            ),
        )
        self.assertEqual(args.compile_mode, "max-autotune-no-cudagraphs")
        self.assertTrue(args.compile and args.amp)

    def test_invalid_extension_arguments_fail_before_launch(self):
        import contextlib
        import io

        for extra in (["--extend-budget"], ["--schedule-steps", "0"], ["--schedule-steps", "-1"]):
            parser = runtime.build_parser()
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runtime.validate_args(parser, parser.parse_args(["--workspace", "/tmp/flash-extension", *extra]))


class CheckpointExtensionTests(unittest.TestCase):
    def test_loader_extension_preserves_payload_and_new_saves_resume_strictly(self):
        import tempfile

        old = original_contract()
        current = copy.deepcopy(old)
        current["max_steps"] = 150_000_896
        for name in OLD_RUNNER_HASHES:
            current["source_sha256"][name] = hashlib.sha256(
                (Path(runtime.__file__).parent / Path(name).name).read_bytes()
            ).hexdigest()
        current["agent"]["compile_mode"] = "max-autotune-no-cudagraphs"
        state = {
            "agent": {"update_step": 97460},
            "trainer": {"frame": 50_000_896, "updates": 97460, "vector_steps": 48829},
            "environment": {"curriculum": 3},
            "episodes": {"completed": 1000},
        }
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = runtime.save_checkpoint(directory / "old", old, state)
            old_bytes = path.read_bytes()
            with self.assertRaises(ValueError):
                runtime.load_checkpoint(path, current)
            payload = runtime.load_checkpoint(path, current, allow_budget_extension=True)
            for name, value in state.items():
                self.assertEqual(payload[name], value)
            self.assertEqual(payload["contract"], old)
            self.assertEqual(path.read_bytes(), old_bytes)
            with self.assertRaises(ValueError):
                runtime.save_checkpoint(directory / "old", current, state)
            new_path = runtime.save_checkpoint(directory / "new", current, state)
            self.assertEqual(runtime.load_checkpoint(new_path, current)["contract"], current)
            with self.assertRaises(ValueError):
                runtime.load_checkpoint(new_path, current, allow_budget_extension=True)

    def test_extension_rejects_bad_counters_and_target_already_reached(self):
        import tempfile

        old = original_contract()
        old["max_steps"] = 150_000_896
        current = copy.deepcopy(old)
        current["max_steps"] = 150_001_000
        valid = {"frame": 150_001_664, "updates": 97460, "vector_steps": 146486}
        cases = [
            (valid, 97460),
            ({**valid, "frame": 50_000_897}, 97460),
            ({**valid, "frame": 50_000_896, "vector_steps": 48829}, 97459),
            ({**valid, "updates": -1}, -1),
        ]
        with tempfile.TemporaryDirectory() as temp:
            for index, (trainer, updates) in enumerate(cases):
                state = {"agent": {"update_step": updates}, "trainer": trainer, "environment": {}, "episodes": {}}
                path = runtime.save_checkpoint(Path(temp) / str(index), old, state)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    runtime.load_checkpoint(path, current, allow_budget_extension=True)


class SnapshotScheduleTests(unittest.TestCase):
    def setUp(self):
        try:
            from flash_rl.buffers import torch_buffer
        except ImportError:
            self.skipTest("Run with patched immutable FlashSAC snapshot on PYTHONPATH")
        if getattr(torch_buffer, "N_STEP_DISCOUNT_VERSION", None) != 1:
            self.skipTest("Run with patched immutable FlashSAC snapshot on PYTHONPATH")

    def test_extension_uses_original_gradient_budget_not_new_collection_target(self):
        import dataclasses

        parser = runtime.build_parser()
        base = runtime.validate_args(parser, parser.parse_args(["--workspace", "/tmp/old-flash"]))
        base.device = "cuda:0"
        extended = copy.deepcopy(base)
        extended.max_steps = 150_000_896
        old_cfg = runtime.make_agent_config(base)
        new_cfg = runtime.make_agent_config(extended)
        self.assertEqual(dataclasses.asdict(new_cfg), dataclasses.asdict(old_cfg))
        from flash_rl.agents.flashSAC.agent import _resolve_compile_mode

        resolved = dataclasses.asdict(new_cfg)
        resolved["compile_mode"] = _resolve_compile_mode(new_cfg.compile_mode)
        import numpy as np

        resolved["temp_target_entropy"] = float(23 * 0.5 * np.log(2.0 * np.pi * np.e * new_cfg.temp_target_sigma**2))
        self.assertEqual(resolved, original_contract()["agent"])
        self.assertEqual((new_cfg.learning_rate_warmup_step, new_cfg.learning_rate_decay_step), (0, 97658))
        self.assertEqual(
            (new_cfg.learning_rate_init, new_cfg.learning_rate_peak, new_cfg.learning_rate_end),
            (0.0003, 0.0003, 0.00015),
        )

    def test_actual_learner_restore_preserves_per_network_lr_and_next_step_closure(self):
        import dataclasses
        import torch
        import warnings

        torch.set_num_threads(2)
        parser = runtime.build_parser()
        args = runtime.validate_args(parser, parser.parse_args(["--workspace", "/tmp/flash-lr"]))
        args.device = "cuda:0"
        cfg = runtime.make_agent_config(args)
        # Small CPU allocation only; retain the production 97,658-step scheduler closure.
        cfg = dataclasses.replace(
            cfg,
            device_type="cpu",
            buffer_device_type="cpu",
            buffer_max_length=64,
            buffer_min_length=8,
            sample_batch_size=8,
            use_compile=False,
            use_amp=False,
        )
        args.num_envs = 4
        original = runtime.make_agent(args, cfg)
        for name, steps in (("critic", 97460), ("actor", 48730), ("temperature", 48730)):
            net = getattr(original, "_" + name)
            net.optimizer.zero_grad()
            sum(p.square().sum() for p in net.network.parameters()).backward()
            net.optimizer.step()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                net.scheduler.step(steps)
        original._update_step = 97460
        saved = copy.deepcopy(runtime.learner_state(original))
        args.max_steps = 150_000_896
        args.num_envs = 1024
        args.compile_mode = "max-autotune-no-cudagraphs"
        extension_cfg = runtime.make_agent_config(args)
        self.assertEqual(extension_cfg.compile_mode, "max-autotune-no-cudagraphs")
        extension_cfg = dataclasses.replace(
            extension_cfg,
            device_type="cpu",
            buffer_device_type="cpu",
            buffer_max_length=64,
            buffer_min_length=8,
            sample_batch_size=8,
            use_compile=False,
            use_amp=False,
        )
        args.num_envs = 4
        restored = runtime.make_agent(args, extension_cfg)
        runtime.restore_learner(restored, saved)
        self.assertEqual(restored._update_step, 97460)
        self.assertFalse(restored.can_start_training())
        self.assertEqual(len(restored._replay_buffer), 0)
        for name, expected_lr in (
            ("critic", 0.00015000152141605472),
            ("actor", 0.000225238856912418),
            ("temperature", 0.000225238856912418),
        ):
            old_net, new_net = getattr(original, "_" + name), getattr(restored, "_" + name)
            self.assertEqual(new_net.update_step, old_net.update_step)
            self.assertEqual(new_net.scheduler.state_dict(), old_net.scheduler.state_dict())
            self.assertEqual(new_net.optimizer.param_groups[0]["lr"], old_net.optimizer.param_groups[0]["lr"])
            self.assertAlmostEqual(new_net.optimizer.param_groups[0]["lr"], expected_lr, places=12)
            for old_p, new_p in zip(old_net.network.parameters(), new_net.network.parameters()):
                torch.testing.assert_close(old_p, new_p, rtol=0, atol=0)
            old_net.optimizer.step()
            new_net.optimizer.step()
            old_net.scheduler.step()
            new_net.scheduler.step()
            self.assertEqual(new_net.optimizer.param_groups[0]["lr"], old_net.optimizer.param_groups[0]["lr"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                new_net.scheduler.step(97658)
                self.assertEqual(new_net.optimizer.param_groups[0]["lr"], 0.00015)
                new_net.scheduler.step(292972)
                self.assertEqual(new_net.optimizer.param_groups[0]["lr"], 0.00015)


if __name__ == "__main__":
    unittest.main()
