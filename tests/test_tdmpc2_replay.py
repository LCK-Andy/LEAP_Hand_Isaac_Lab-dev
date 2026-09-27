"""Simulator-free behavioral contracts for the vector TD-MPC2 replay."""

import importlib.util
import torch
from pathlib import Path

import pytest

REPLAY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tdmpc2" / "replay.py"


def replay_class():
    spec = importlib.util.spec_from_file_location("tdmpc2_replay_under_test", REPLAY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SequenceReplay


def make_replay(**kwargs):
    options = dict(
        capacity=12, num_envs=2, obs_dim=2, action_dim=1, horizon=3, batch_size=16, sample_device="cpu", seed=7
    )
    options.update(kwargs)
    return replay_class()(**options)


def add_step(replay, step, terminated=(False, False), truncated=(False, False), condition=None):
    obs = torch.tensor([[step, 10 + step], [100 + step, 110 + step]], dtype=torch.float64)
    action = torch.tensor([[step], [100 + step]], dtype=torch.float64)
    reward = torch.tensor([step + 0.5, 100 + step + 0.5], dtype=torch.float64)
    replay.add(obs, action, reward, obs + 1, torch.tensor(terminated), torch.tensor(truncated), condition=condition)


def test_fixed_horizon_tuple_contains_full_same_environment_sequences():
    replay = make_replay()
    assert len(replay) == 0
    assert not replay.ready
    assert not replay.can_sample()
    with pytest.raises(RuntimeError, match="sequence"):
        replay.sample()
    add_step(replay, 0)
    add_step(replay, 1)
    assert not replay.ready
    add_step(replay, 2)
    assert len(replay) == 6
    assert replay.ready and replay.can_sample()
    obs, action, reward, terminated, task = replay.sample()
    assert obs.shape == (4, 16, 2)
    assert action.shape == (3, 16, 1)
    assert reward.shape == terminated.shape == (3, 16, 1)
    assert task is None
    for tensor in (obs, action, reward, terminated):
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
    assert set(obs[0, :, 0].tolist()) == {0, 100}
    for column in range(16):
        base = obs[0, column, 0].item()
        assert obs[:, column, 0].tolist() == [base, base + 1, base + 2, base + 3]
        assert action[:, column, 0].tolist() == [base, base + 1, base + 2]
        assert reward[:, column, 0].tolist() == [base + 0.5, base + 1.5, base + 2.5]
    assert not terminated.any()
    assert replay.sample(batch_size=1)[0].shape == (4, 1, 2)


def test_terminal_and_timeout_end_sequences_but_only_terminal_disables_bootstrap():
    replay = make_replay(capacity=20)
    add_step(replay, 0)
    add_step(replay, 1)
    replay.add(
        torch.tensor([[2, 12], [102, 112]]),
        torch.tensor([[2], [102]]),
        torch.tensor([2.5, 102.5]),
        torch.tensor([[999, 999], [888, 888]]),
        torch.tensor([True, False]),
        torch.tensor([False, True]),
    )
    # Returned acting observations after auto-reset belong only to subsequent adds.
    add_step(replay, 3)
    add_step(replay, 4)
    obs, action, reward, terminal, _ = replay.sample(200)
    assert set(obs[0, :, 0].tolist()) == {0, 100}
    for column in range(200):
        is_env_zero = obs[0, column, 0] == 0
        assert obs[-1, column, 0] == (999 if is_env_zero else 888)
        assert terminal[:, column, 0].tolist() == [0, 0, 1 if is_env_zero else 0]
    add_step(replay, 5)
    assert set(replay.sample(200)[0][0, :, 0].tolist()) == {0, 3, 100, 103}


def test_async_short_episodes_are_excluded_and_counted_explicitly():
    replay = make_replay(capacity=20)
    add_step(replay, 0, terminated=(True, False))
    add_step(replay, 1)
    add_step(replay, 2, truncated=(True, False))
    add_step(replay, 3, terminated=(False, True))
    add_step(replay, 4)
    add_step(replay, 5)
    assert replay.stats["short_episodes"] == 2
    assert replay.stats["short_episode_transitions"] == 3
    assert replay.stats["eligible_sequences"] == 3
    assert set(replay.sample(200)[0][0, :, 0].tolist()) == {3, 100, 101}


@pytest.mark.parametrize("restart_method", ["clear_pending", "start_new_stream"])
def test_restart_retains_complete_old_sequences_without_continuing_partial_ones(restart_method):
    replay = make_replay(capacity=20)
    for step in range(4):
        add_step(replay, step)
    getattr(replay, restart_method)()
    add_step(replay, 50)
    add_step(replay, 51)
    assert len(replay) == 12
    assert set(replay.sample(200)[0][0, :, 0].tolist()) == {0, 1, 100, 101}
    add_step(replay, 52)
    assert set(replay.sample(200)[0][0, :, 0].tolist()) == {0, 1, 50, 100, 101, 150}
    # A restart is not an MDP terminal and must not zero any bootstraps.
    assert not replay.sample(200)[3].any()


def test_full_checkpoint_restores_wrapped_contents_pending_continuity_and_private_rng(tmp_path):
    replay = make_replay(capacity=13)
    for step in range(9):
        add_step(replay, step, truncated=(step == 4, False))
    assert replay.capacity == len(replay) == 12
    assert set(replay.sample(500)[0][0, :, 0].tolist()) == {5, 6, 103, 104, 105, 106}
    path = tmp_path / "replay.pt"
    torch.save(replay.state_dict(), path)
    restored = make_replay(capacity=12, seed=999)
    restored.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    assert restored.stats == replay.stats
    for step in range(9, 12):
        add_step(replay, step)
        add_step(restored, step)
        torch.manual_seed(step)  # Policy/global reseeding must not affect replay draws.
        expected = replay.sample()
        torch.manual_seed(step + 100)
        actual = restored.sample()
        for left, right in zip(expected[:-1], actual[:-1]):
            assert torch.equal(left, right)
    restored.clear_pending()
    add_step(restored, 90)
    assert set(restored.sample(500)[0][0, :, 0].tolist()) == {7, 8, 9, 107, 108, 109}


def test_condition_provenance_is_per_transition_bounded_and_checkpointed():
    replay = make_replay(capacity=6)
    add_step(replay, 0, condition="gravity=0")
    add_step(replay, 1, condition="gravity=0")
    add_step(replay, 2, condition="gravity=-9.81")
    state = replay.state_dict()
    ids = state["buffers"]["condition_ids"]
    assert ids.shape == (3, 2)
    labels = [[state["conditions"][int(value)] for value in row] for row in ids]
    assert labels == [["gravity=0"] * 2, ["gravity=0"] * 2, ["gravity=-9.81"] * 2]
    # Condition changes are provenance, not fake episode boundaries.
    assert replay.stats["eligible_sequences"] == 2
    restored = make_replay(capacity=6)
    restored.load_state_dict(state)
    add_step(restored, 3, condition="new")
    assert replay.state_dict()["conditions"] == state["conditions"]
    add_step(restored, 4, condition="new")
    add_step(restored, 5, condition="new")
    assert list(restored.state_dict()["conditions"].values()) == ["new"]
    # Arbitrarily many physical stamps must not make the metadata unbounded.
    for step in range(6, 30):
        add_step(restored, step, condition=f"stamp-{step}")
    assert len(restored.state_dict()["conditions"]) == 3


@pytest.mark.parametrize(
    "options",
    [
        {"capacity": 0},
        {"num_envs": 0},
        {"obs_dim": 0},
        {"action_dim": 0},
        {"horizon": 0},
        {"batch_size": 0},
        {"capacity": 5},
        {"capacity": 12.5},
        {"device": "cuda:0"},
    ],
)
def test_invalid_or_non_cpu_storage_configuration_fails_before_allocation(options):
    with pytest.raises(ValueError):
        make_replay(**options)


def test_checkpoint_rejects_incompatible_layout_or_incomplete_buffers_without_mutation():
    replay = make_replay()
    for step in range(3):
        add_step(replay, step)
    state = replay.state_dict()
    incompatible = make_replay(horizon=2)
    with pytest.raises(ValueError, match="horizon"):
        incompatible.load_state_dict(state)
    assert len(incompatible) == 0
    state["buffers"] = dict(state["buffers"])
    del state["buffers"]["next_obs"]
    target = make_replay()
    with pytest.raises(ValueError, match="buffers"):
        target.load_state_dict(state)
    assert len(target) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("obs", torch.zeros(4)),
        ("action", torch.zeros(2, 2)),
        ("reward", torch.zeros(3)),
        ("next_obs", torch.zeros(4)),
        ("terminated", torch.tensor([0, 2])),
        ("truncated", torch.tensor([0.0, float("nan")])),
        ("condition", {"gravity": 0}),
    ],
)
def test_invalid_add_leaves_complete_existing_replay_unchanged(field, value):
    replay = make_replay(capacity=6)
    for step in range(3):
        add_step(replay, step)
    before = replay.sample(100)
    args = dict(
        obs=torch.zeros(2, 2),
        action=torch.zeros(2, 1),
        reward=torch.zeros(2),
        next_obs=torch.zeros(2, 2),
        terminated=torch.zeros(2),
        truncated=torch.zeros(2),
        condition=None,
    )
    args[field] = value
    with pytest.raises(ValueError):
        replay.add(**args)
    assert len(replay) == 6
    assert replay.stats["eligible_sequences"] == 2
    assert set(replay.sample(100)[0][0, :, 0].tolist()) == set(before[0][0, :, 0].tolist())


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_sample_requires_positive_integer_batch_size(batch_size):
    replay = make_replay()
    for step in range(3):
        add_step(replay, step)
    with pytest.raises(ValueError, match="batch_size"):
        replay.sample(batch_size)


def test_random_async_rollouts_match_independent_episode_history_through_many_wraps():
    import random

    rng = random.Random(51)
    replay = make_replay(capacity=21, num_envs=3, obs_dim=1, horizon=3)
    history = [[] for _ in range(3)]
    for step in range(100):
        obs = torch.tensor([[step + env * 1000] for env in range(3)])
        terminated = [rng.random() < 0.15 for _ in range(3)]
        truncated = [rng.random() < 0.15 for _ in range(3)]
        next_obs = obs + 1
        for env in range(3):
            if terminated[env] or truncated[env]:
                next_obs[env] = -step - env * 1000 - 1
            history[env].append((int(obs[env]), int(next_obs[env]), terminated[env], truncated[env]))
            history[env] = history[env][-7:]
        replay.add(obs, obs, obs[:, 0], next_obs, torch.tensor(terminated), torch.tensor(truncated))
        expected = {}
        for episode_history in history:
            for start in range(len(episode_history) - 2):
                window = episode_history[start : start + 3]
                if any(item[2] or item[3] for item in window[:-1]):
                    continue
                expected[window[0][0]] = window
        assert replay.stats["eligible_sequences"] == len(expected)
        assert replay.ready == bool(expected)
        if expected:
            observations, actions, rewards, terminals, _ = replay.sample(128)
            for column in range(128):
                window = expected[int(observations[0, column, 0])]
                assert observations[:, column, 0].tolist() == [item[0] for item in window] + [window[-1][1]]
                assert actions[:, column, 0].tolist() == rewards[:, column, 0].tolist() == [item[0] for item in window]
                assert terminals[:, column, 0].tolist() == [int(item[2]) for item in window]


def test_real_observation_width_vector_collection_owns_unpinned_float32_storage():
    replay = make_replay(capacity=768, num_envs=256, obs_dim=1745, action_dim=23, batch_size=4)
    obs = torch.zeros(256, 1745, requires_grad=True)
    action = torch.zeros(256, 23)
    for step in range(3):
        with torch.no_grad():
            obs.fill_(step)
        replay.add(obs, action, torch.zeros(256, 1), obs + 1, torch.zeros(256, 1), torch.zeros(256, 1))
    with torch.no_grad():
        obs.fill_(999)
    sampled = replay.sample()[0]
    assert sampled[:, 0, 0].tolist() == [0, 1, 2, 3]
    assert sampled.shape == (4, 4, 1745)
    for tensor in replay.state_dict()["buffers"].values():
        assert tensor.device.type == "cpu"
        assert not tensor.is_pinned()
        assert not tensor.requires_grad
    # Entire capacity=131072 storage scales to under 1.73 GiB at these dimensions.
    stored_bytes = sum(t.numel() * t.element_size() for t in replay.state_dict()["buffers"].values())
    assert stored_bytes / replay.capacity * 131072 < 1.73 * 1024**3
