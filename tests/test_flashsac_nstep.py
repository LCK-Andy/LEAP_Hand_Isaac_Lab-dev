"""Real FlashSAC replay/update contracts, in subprocesses using an isolated source copy.

Run with env_isaaclab/bin/python -m unittest discover -s tests -p test_flashsac_nstep.py.
No installed modules or source-clone files are patched by these tests.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("FLASHSAC_TEST_SOURCE", str(Path.home() / "Documents/projects/FlashSAC")))

PRELUDE = """
import gymnasium as gym
import numpy as np
import torch
from flash_rl.buffers.torch_buffer import TorchUniformBuffer

torch.set_num_threads(1)
torch.manual_seed(42)

def buffer(max_length=64, n_step=3):
    return TorchUniformBuffer(gym.spaces.Box(-1000., 1000., (2,), dtype=np.float32),
        gym.spaces.Box(-1., 1., (1,), dtype=np.float32), n_step, .99,
        max_length, 1, 8, 'cpu')

def transition(step):
    # Three lanes: continuing; immediate timeout; second-position termination.
    return dict(observation=torch.full((3, 2), float(step)), action=torch.zeros(3, 1),
        reward=torch.tensor([1., 2., 3.]) + 100 * (step > 0) * torch.tensor([0., 1., 0.]),
        terminated=torch.tensor([False, False, step == 1]),
        truncated=torch.tensor([False, step == 0, False]),
        next_observation=torch.tensor([[step + 1., step + 1.],
            [99., 99.] if step == 0 else [step + 1., step + 1.], [step + 1., step + 1.]]))
"""


@unittest.skipUnless(SOURCE.is_dir(), "Local FlashSAC source required (or set FLASHSAC_TEST_SOURCE)")
class FlashSACNstepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="flashsac-nstep-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.snapshot = Path(cls.temporary.name) / "FlashSAC"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/flashsac/prepare_source.py"),
                "--source",
                str(SOURCE),
                "--destination",
                str(cls.snapshot),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def run_source(self, code):
        env = dict(os.environ, PYTHONPATH=str(self.snapshot), TORCHDYNAMO_DISABLE="1", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [sys.executable, "-c", PRELUDE + textwrap.dedent(code)],
            cwd=self.temporary.name,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_actual_horizon_stops_at_timeout_and_termination(self):
        self.run_source("""
            replay = buffer()
            for step in range(3):
                replay.add(transition(step))
            batch = replay.sample(np.arange(3))
            # Reproduce the original critic fallback, then require corrected targets.
            discount = batch.get('discount', torch.full((3,), .99**3))
            torch.testing.assert_close(discount, torch.tensor([.970299, .99, .9801]))
            assert discount.shape == (3,) and discount.dtype == torch.float32
            torch.testing.assert_close(batch['reward'], torch.tensor([2.9701, 2., 5.97]))
            torch.testing.assert_close(batch['next_observation'], torch.tensor([[3., 3.], [99., 99.], [2., 2.]]))
            torch.testing.assert_close(batch['terminated'], torch.tensor([0., 0., 1.]))
            torch.testing.assert_close(batch['truncated'], torch.tensor([0., 1., 0.]))
        """)

    def test_categorical_targets_accept_per_transition_discounts(self):
        self.run_source("""
            from flash_rl.agents.flashSAC.update import _compute_categorical_td_target
            # Point mass at value 2 on support [0, 1, 2, 3, 4].
            logp = torch.tensor([[float('-inf'), float('-inf'), 0., float('-inf'), float('-inf')]]).repeat(3, 1)
            target = _compute_categorical_td_target(logp, torch.zeros(3), torch.tensor([0., 0., 1.]),
                torch.zeros(3), torch.tensor([.99, .9801, .970299]), 5, 0., 4.)
            torch.testing.assert_close(target, torch.tensor([
                [0., .02, .98, 0., 0.], [0., .0398, .9602, 0., 0.], [1., 0., 0., 0., 0.]]))
            scalar = _compute_categorical_td_target(logp, torch.zeros(3), torch.zeros(3),
                torch.zeros(3), .970299, 5, 0., 4.)
            torch.testing.assert_close(scalar, torch.tensor([[0., .059402, .940598, 0., 0.]]).repeat(3, 1))
        """)

    def test_replay_roundtrip_is_cropped_and_legacy_replay_fails_closed(self):
        self.run_source("""
            replay = buffer(max_length=100000)
            for step in range(3):
                replay.add(transition(step))
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / 'replay.pt')
                replay.save(path)
                assert Path(path).stat().st_size < 20000, 'Partial replay serialized unused capacity'
                other = buffer(max_length=100000)
                other.load(path)
                for key, value in replay.sample(np.arange(3)).items():
                    torch.testing.assert_close(value, other.sample(np.arange(3))[key])
                assert other.sample()['discount'].shape == (8,)
                old = torch.load(path, weights_only=True)
                del old['discount']
                torch.save(old, path)
                try:
                    other.load(path)
                except ValueError as error:
                    assert 'missing discount' in str(error)
                else:
                    raise AssertionError('Legacy replay must not silently reconstruct timeout horizons')
        """)

    def test_later_windows_do_not_reuse_aggregated_rewards_or_cross_reset(self):
        self.run_source("""
            replay = buffer()
            for step in range(5):
                item = transition(step)
                replay.add(item)
                # Replay must copy tensors rather than retain caller-owned storage.
                item['reward'].fill_(-999)
                item['next_observation'].fill_(-999)
            batch = replay.sample(np.arange(9))
            torch.testing.assert_close(batch['discount'], torch.tensor([
                .970299, .99, .9801, .970299, .970299, .99, .970299, .970299, .970299]))
            torch.testing.assert_close(batch['reward'], torch.tensor([
                2.9701, 2., 5.97, 2.9701, 302.9502, 3., 2.9701, 302.9502, 8.9103]))
            torch.testing.assert_close(batch['next_observation'][[1, 2, 5, 8]],
                torch.tensor([[99., 99.], [2., 2.], [2., 2.], [5., 5.]]))
            # Explicit timeout in position two must retain that step's final observation.
            replay = buffer()
            for step in range(3):
                item = transition(step)
                item['terminated'].zero_()
                item['truncated'] = torch.tensor([step == 1, False, False])
                replay.add(item)
            batch = replay.sample(np.array([0]))
            torch.testing.assert_close(batch['discount'], torch.tensor([.9801]))
            torch.testing.assert_close(batch['reward'], torch.tensor([1.99]))
            torch.testing.assert_close(batch['next_observation'], torch.tensor([[2., 2.]]))
            torch.testing.assert_close(batch['truncated'], torch.ones(1))
        """)

    def test_real_flashsac_agent_sampling_and_finite_actor_critic_updates(self):
        self.run_source("""
            import dataclasses
            import math
            from pathlib import Path
            import yaml
            import flash_rl
            from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig
            source = Path(flash_rl.__file__).resolve().parents[1]
            values = yaml.safe_load((source / 'configs/agent/flashSAC.yaml').read_text())
            values.pop('agent_type')
            values.update(seed=42, device_type='cpu', buffer_device_type='cpu', buffer_max_length=64,
                buffer_min_length=1, sample_batch_size=8, actor_num_blocks=1, actor_hidden_dim=16,
                critic_num_blocks=1, critic_hidden_dim=16, critic_num_bins=11, critic_min_v=-5.,
                critic_max_v=5., gamma=.99, n_step=3, learning_rate_warmup_step=0,
                learning_rate_init=3e-4, learning_rate_peak=3e-4, learning_rate_end=1.5e-4,
                learning_rate_decay_step=100, use_compile=False, use_amp=False, temp_target_entropy=0.)
            config = FlashSACConfig(**values)
            # Same networks, samples and RNG; only replay discount differs. This
            # catches dropping the new key at the actual critic-update boundary.
            from flash_rl.agents.flashSAC.update import update_critic
            losses = []
            for use_discount in (False, True):
                torch.manual_seed(10)
                probe = FlashSACAgent(gym.spaces.Box(-1000., 1000., (2,), dtype=np.float32),
                    gym.spaces.Box(-1., 1., (1,), dtype=np.float32), {}, config)
                for step in range(3):
                    probe.process_transition(transition(step))
                batch = probe._replay_buffer.sample(np.arange(3))
                batch['actor_next_observation'] = batch['next_observation']
                if not use_discount:
                    del batch['discount']
                loss = update_critic(probe._actor, probe._critic, probe._target_critic, probe._temperature,
                    batch, -5., 5., 11, .1, 3, torch.device('cpu'), False, None)['critic/loss']
                losses.append(float(loss.detach()))
            assert abs(losses[0] - losses[1]) > 1e-5, losses
            for normalize in (False, True):
                agent = FlashSACAgent(gym.spaces.Box(-1000., 1000., (2,), dtype=np.float32),
                    gym.spaces.Box(-1., 1., (1,), dtype=np.float32), {},
                    dataclasses.replace(config, normalize_reward=normalize))
                for step in range(5):
                    agent.process_transition(transition(step))
                assert agent.can_start_training()
                for training in (False, True):
                    actions = agent.sample_actions(5, {'next_observation': torch.ones(3, 2)}, training)
                    assert actions.shape == (3, 1) and np.isfinite(actions).all()
                    assert np.abs(actions).max() <= 1.
                actor_before = [p.detach().clone() for p in agent._actor.network.parameters()]
                critic_before = [p.detach().clone() for p in agent._critic.network.parameters()]
                info = agent.update()
                assert 'actor/loss' in info and 'critic/loss' in info
                assert all(math.isfinite(value) for value in info.values()), info
                for previous, network in ((actor_before, agent._actor), (critic_before, agent._critic)):
                    assert any(not torch.equal(before, after) for before, after in zip(previous, network.network.parameters()))
                    gradients = [p.grad for p in network.network.parameters() if p.grad is not None]
                    assert gradients and all(torch.isfinite(g).all() for g in gradients)
                # Discount survives the agent's reward-normalization path unmodified.
                torch.testing.assert_close(agent._replay_buffer.sample(np.arange(3))['discount'],
                    torch.tensor([.970299, .99, .9801]))
        """)

    def test_preparation_is_idempotent_and_refuses_overwriting_changed_snapshot(self):
        command = [
            sys.executable,
            str(ROOT / "scripts/flashsac/prepare_source.py"),
            "--source",
            str(SOURCE),
            "--destination",
            str(self.snapshot),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "FlashSAC"
            changed_command = command[:-1] + [str(destination)]
            subprocess.run(changed_command, capture_output=True, check=True)
            changed_file = destination / "flash_rl/types.py"
            changed_file.write_text("# foreign change\n")
            result = subprocess.run(changed_command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing overwrite", result.stderr)
            self.assertEqual(changed_file.read_text(), "# foreign change\n")

    def test_snapshot_provenance_and_exact_hash_patch_guard(self):
        import hashlib
        import importlib.util
        import json

        manifest = json.loads((self.snapshot / "source_manifest.json").read_text())
        self.assertTrue(manifest["upstream_is_ancestor"])
        self.assertEqual(manifest["official_upstream_pin"], "87edc9061150ae9e962dd84e6544e27a1554b3ab")
        self.assertEqual(manifest["patch_version"], 2)
        for name, expected in manifest["tracked_worktree_sha256"].items():
            self.assertEqual(hashlib.sha256((SOURCE / name).read_bytes()).hexdigest(), expected)
        for name, expected in manifest["snapshot_files_sha256"].items():
            self.assertEqual(hashlib.sha256((self.snapshot / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(
            manifest["patch_files"],
            [
                "flash_rl/agents/flashSAC/update.py",
                "flash_rl/agents/utils/reward_normalization.py",
                "flash_rl/buffers/torch_buffer.py",
            ],
        )
        spec = importlib.util.spec_from_file_location(
            "flashsac_prepare_source", ROOT / "scripts/flashsac/prepare_source.py"
        )
        prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare)
        files = {name: (SOURCE / name).read_bytes() for name in prepare.BASE_HASHES}
        files["flash_rl/buffers/torch_buffer.py"] += b"\n# unsupported change\n"
        with self.assertRaisesRegex(ValueError, "Unsupported source hash"):
            prepare.patched_files(files)

    def test_reward_normalizer_records_terminal_return_then_clears_episode_carry(self):
        self.run_source("""
            from flash_rl.agents.utils.reward_normalization import RewardNormalizer
            for boundary in ('terminated', 'truncated'):
                normalizer = RewardNormalizer(.99, 5., True, torch.device('cpu'))
                for index, reward in enumerate((10., 1., 2.)):
                    done = index == 1
                    normalizer.update_reward_stats(torch.tensor([reward]),
                        torch.tensor([done and boundary == 'terminated']),
                        torch.tensor([done and boundary == 'truncated']))
                    if done:
                        torch.testing.assert_close(normalizer.G_r, torch.zeros(1))
                        torch.testing.assert_close(normalizer.G_rms.mean, torch.tensor([10.45]))
                        torch.testing.assert_close(normalizer.G_r_max, torch.tensor([10.9]))
                torch.testing.assert_close(normalizer.G_r, torch.tensor([2.]))
                torch.testing.assert_close(normalizer.G_rms.mean, torch.tensor([7.6333333333]))
                torch.testing.assert_close(normalizer.G_r_max, torch.tensor([10.9]))
                assert normalizer.G_rms.count == 3
        """)

    def test_tar_publication_does_not_overwrite_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source.tar.gz"
            archive.write_bytes(b"owned elsewhere")
            command = [
                sys.executable,
                str(ROOT / "scripts/flashsac/prepare_source.py"),
                "--source",
                str(SOURCE),
                "--destination",
                str(self.snapshot),
                "--tar",
                str(archive),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(archive.read_bytes(), b"owned elsewhere")


if __name__ == "__main__":
    unittest.main()
