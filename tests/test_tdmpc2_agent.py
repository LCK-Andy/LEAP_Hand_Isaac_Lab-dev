"""Simulator-free public agent contract checks for the pinned TD-MPC2 adapter."""

import torch
import unittest

from scripts.tdmpc2.agent import Agent
from scripts.tdmpc2.config import TDMPC2Config


def small_config(**kwargs):
    values = dict(
        obs_dim=7,
        action_dim=3,
        num_envs=3,
        plan_chunk_size=2,
        batch_size=2,
        enc_dim=16,
        mlp_dim=16,
        latent_dim=16,
        num_q=2,
        num_samples=8,
        num_elites=3,
        num_pi_trajs=2,
        iterations=1,
    )
    values.update(kwargs)
    return TDMPC2Config(**values)


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_batched_actions_are_finite_bounded_and_on_input_device(self):
        agent = Agent(small_config(), device="cpu")
        actions = agent.act(torch.zeros(3, 7), torch.ones(3, dtype=torch.bool))
        self.assertEqual(actions.shape, (3, 3))
        self.assertEqual(actions.device.type, "cpu")
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue((actions.abs() <= 1).all())

    def test_update_and_full_restore_preserve_next_update(self):
        cfg = small_config()

        class Replay:
            def sample(self):
                return (
                    torch.randn(cfg.horizon + 1, cfg.batch_size, cfg.obs_dim),
                    torch.rand(cfg.horizon, cfg.batch_size, cfg.action_dim) * 2 - 1,
                    torch.randn(cfg.horizon, cfg.batch_size, 1),
                    torch.zeros(cfg.horizon, cfg.batch_size, 1),
                    None,
                )

        agent = Agent(cfg, device="cpu")
        metrics = agent.update(Replay())
        self.assertTrue(all(torch.isfinite(v) for v in metrics.values()))
        agent.act(torch.zeros(3, 7), torch.ones(3, dtype=torch.bool))
        checkpoint = agent.state_dict()
        expected = agent.update(Replay())
        restored = Agent(cfg, device="cpu")
        restored.load_state_dict(checkpoint)
        actual = restored.update(Replay())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(restored.update_count, 2)
        self.assertGreater(len(restored.core.optim.state), 0)
        torch.testing.assert_close(restored.core.scale.value, agent.core.scale.value)
        restored.load_state_dict(checkpoint, reset_planner=True)
        self.assertEqual(restored.prev_mean.count_nonzero(), 0)

    def test_batch_one_matches_official_main_planner(self):
        agent = Agent(small_config(num_envs=1, plan_chunk_size=1), device="cpu")
        obs = torch.randn(1, 7)
        for first in (True, False):
            torch.manual_seed(117)
            expected = agent.core.act(obs[0], t0=first)
            torch.manual_seed(117)
            actual = agent.act(obs, torch.tensor([first]))[0]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_async_reset_discards_only_reset_rows_history(self):
        agent = Agent(small_config(), device="cpu")
        obs = torch.randn(3, 7)
        agent.act(obs, torch.ones(3, dtype=torch.bool))
        checkpoint = agent.state_dict()
        mask = torch.tensor([False, True, False])
        agent.load_state_dict(checkpoint)
        expected = agent.act(obs, mask)
        # Arbitrarily corrupt ONLY the reset row's old planner state.
        checkpoint["prev_mean"][1].fill_(100)
        agent.load_state_dict(checkpoint)
        actual = agent.act(obs, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        agent.load_state_dict(checkpoint)
        unreset = agent.act(obs, torch.zeros(3, dtype=torch.bool))
        torch.testing.assert_close(unreset[[0, 2]], expected[[0, 2]], rtol=0, atol=0)

    def test_terminal_transition_updates_termination_head(self):
        cfg = small_config()

        class Replay:
            def sample(self):
                terminated = torch.zeros(cfg.horizon, cfg.batch_size, 1)
                terminated[-1] = 1
                return (
                    torch.randn(cfg.horizon + 1, cfg.batch_size, cfg.obs_dim),
                    torch.zeros(cfg.horizon, cfg.batch_size, cfg.action_dim),
                    torch.ones(cfg.horizon, cfg.batch_size, 1),
                    terminated,
                    None,
                )

        agent = Agent(cfg, device="cpu")
        metrics = agent.update(Replay())
        self.assertGreater(float(metrics["termination_loss"]), 0)
        self.assertEqual(float(metrics["termination_rate"]), 1)
        self.assertTrue(all(torch.isfinite(v) for v in metrics.values()))

    def test_default_is_official_5m_and_high_dimensional_iteration_heuristic(self):
        cfg = TDMPC2Config()
        self.assertEqual((cfg.enc_dim, cfg.mlp_dim, cfg.latent_dim, cfg.num_q), (256, 512, 512, 5))
        self.assertEqual((cfg.horizon, cfg.num_samples, cfg.num_elites, cfg.num_pi_trajs), (3, 512, 64, 24))
        smoke = small_config(action_dim=23)
        agent = Agent(smoke, device="cpu")
        self.assertEqual(agent.core.cfg.iterations, 3)
        self.assertEqual(smoke.iterations, 1)

    def test_live_hparams_update_both_optimizers_and_validate_atomically(self):
        agent = Agent(small_config(), device="cpu")
        agent.set_hparams({"lr": 0.0007, "entropy_coef": 0.0002, "grad_clip_norm": 15})
        self.assertEqual(agent.core.optim.param_groups[0]["lr"], 0.0007 * 0.3)
        self.assertEqual(agent.core.optim.param_groups[1]["lr"], 0.0007)
        self.assertEqual(agent.core.pi_optim.param_groups[0]["lr"], 0.0007)
        self.assertEqual(agent.core.cfg.entropy_coef, 0.0002)
        with self.assertRaises(ValueError):
            agent.set_hparams({"lr": 0.0008, "entropy_coef": float("nan")})
        self.assertEqual(agent.cfg.lr, 0.0007)


if __name__ == "__main__":
    unittest.main()
