"""Public DexPBT reset acceptance at a simulator boundary, without launching Kit."""

import ast
import copy
import importlib.util
import json
import math
import sys
import torch
from pathlib import Path
from types import SimpleNamespace

TASK = Path(__file__).resolve().parents[1] / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt"
spec = importlib.util.spec_from_file_location("dexpbt_reset_evaluation", TASK / "evaluation.py")
evaluation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluation
spec.loader.exec_module(evaluation)


class SimulatorBoundary:
    """Stand-in simulator: ordered indexed writes and draws, exposed by public reset."""

    def reset(self, seed=None, env_ids=None):
        if seed is not None:
            torch.manual_seed(seed)
        if env_ids is None:
            env_ids = torch.arange(4, dtype=torch.int32)
        self._reset_idx(env_ids)
        return self.positions.clone(), self.extras

    def step(self, action):
        # The simulator seam supplies one terminal transition, then resets it
        # immediately, as the inherited manager-based step does.
        self.lengths += 1
        self._dexpbt_metrics.observe(
            torch.zeros(4),
            torch.zeros(4),
            torch.ones((4, 4)),
            evaluation.metric_condition(0.0, 0.2),
        )
        done = torch.tensor([False, True, False, False])
        self._reset_idx(done.nonzero().flatten().int())
        return (
            self.positions.clone(),
            action.sum(dim=1),
            done,
            torch.zeros(4, dtype=torch.bool),
            self.extras,
        )

    def _reset_idx(self, env_ids):
        self.positions[env_ids] = torch.rand((len(env_ids), 3))
        self.lengths[env_ids] = 0
        self.extras["log"] = {"reset_count": len(env_ids)}


def make_env():
    path = TASK / "environment.py"
    tree = ast.parse(path.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = {
        "ManagerBasedRLEnv": SimulatorBoundary,
        "torch": torch,
        "copy": copy,
        "json": json,
        "math": math,
        **vars(evaluation),
    }
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    env = namespace["DexsuiteReorientPbtEnv"]()
    env.device = "cpu"
    env.positions = torch.arange(12).reshape(4, 3).float()
    env.lengths = torch.full((4,), 7)
    env.extras = {}
    env._dexpbt_fixed_adr = None
    env._dexpbt_in_step = False
    env._dexpbt_episode_eligible = torch.zeros(4, dtype=torch.bool)
    env._dexpbt_metrics = evaluation.CompletedEpisodeMetrics(4, "cpu")
    env.reward_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(params={"pos_std": 0.2}))
    env._dexpbt_metrics.observe(
        torch.ones(4),
        torch.ones(4),
        torch.zeros((4, 4)),
        evaluation.metric_condition(0.0, 0.2),
    )
    return env


def test_public_reset_preserves_selected_order_rng_and_unselected_episodes():
    """Changing index representation must not change reset draws, state, or episode eligibility."""
    results = []
    for ids in (torch.tensor([3, 1], dtype=torch.int32), torch.tensor([3, 1]), [3, 1]):
        env = make_env()
        positions, info = env.reset(seed=45, env_ids=ids)
        results.append((positions, torch.get_rng_state()))
        torch.testing.assert_close(positions[[0, 2]], torch.tensor([[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]]))
        assert env.lengths.tolist() == [7, 0, 7, 0]
        assert env._dexpbt_episode_eligible.tolist() == [False, True, False, True]
        assert info["dexpbt"]["completed_episodes"] == 0
        assert info["log"] == {"reset_count": 2}
        assert env._dexpbt_metrics.complete([3, 1])["completed_episodes"] == 0
        assert env._dexpbt_metrics.complete([0, 2])["completed_episodes"] == 2
    for actual in results[1:]:
        for left, right in zip(actual, results[0]):
            assert torch.equal(left, right)


def test_public_step_completes_only_terminal_eligible_episode_without_delayed_reset():
    env = make_env()
    env.reset(seed=45)
    before = env.positions.clone()
    observations, rewards, done, truncated, info = env.step(torch.ones((4, 23)))
    assert done.tolist() == [False, True, False, False]
    assert not truncated.any()
    assert rewards.tolist() == [23.0, 23.0, 23.0, 23.0]
    assert env.lengths.tolist() == [1, 0, 1, 1]
    assert not torch.equal(observations[1], before[1])
    assert torch.equal(observations[[0, 2, 3]], before[[0, 2, 3]])
    assert info["dexpbt"]["completed_episodes"] == 1
    assert info["dexpbt"]["episode_length_sum"] == 1
    assert info["dexpbt"]["success_count"] == 1
    assert info["dexpbt"]["mixed_condition_episodes"] == 0
    assert not env._dexpbt_in_step
    assert env._dexpbt_metrics.complete([0, 2, 3])["completed_episodes"] == 3


def test_public_full_reset_does_not_publish_unfinished_episodes():
    env = make_env()
    _, info = env.reset(seed=45)
    assert info["dexpbt"]["completed_episodes"] == 0
    assert env.lengths.tolist() == [0, 0, 0, 0]
    assert env._dexpbt_episode_eligible.all()
    assert env._dexpbt_metrics.complete([0, 1, 2, 3])["completed_episodes"] == 0
