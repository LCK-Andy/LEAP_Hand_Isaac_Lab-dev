"""Simulator-free contract tests for the native Dexsuite TD-MPC2 boundary."""

import importlib.util
import sys
import torch
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "tdmpc2_env_under_test", Path(__file__).resolve().parents[1] / "scripts/tdmpc2/env.py"
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
DexsuiteVectorEnv = module.DexsuiteVectorEnv


class History:
    """EA's public oldest-to-newest history/backfill behavior, without Isaac."""

    def __init__(self, num_envs, device):
        self._buffer = None
        self._num_pushes = torch.zeros(num_envs, dtype=torch.long, device=device)

    @property
    def buffer(self):
        return self._buffer

    def append(self, value):
        if self._buffer is None:
            self._buffer = value[:, None].repeat(1, 5, 1)
        else:
            self._buffer = torch.cat((self._buffer[:, 1:], value[:, None]), dim=1)
            first = self._num_pushes == 0
            self._buffer[first] = value[first, None]
        self._num_pushes += 1

    def reset(self, ids):
        self._num_pushes[ids] = 0
        if self._buffer is not None:
            self._buffer[ids] = 0


def observation(env, width, offset):
    return (env.value[:, None] + offset).expand(-1, width).clone()


class ObservationManager:
    def __init__(self, env):
        self.env = env
        self._obs_buffer = None
        self._group_obs_term_history_buffer = {}
        self._group_obs_term_cfgs = {}
        self._group_obs_class_term_cfgs = {}
        self._group_obs_class_instances = []
        self.group_obs_dim = {"policy": (170,), "proprio": (615,), "perception": (960,)}
        for group, terms in {
            "policy": [(4, 0), (30, 10)],
            "proprio": [(123, 20)],
            "perception": [(192, 30)],
        }.items():
            self._group_obs_term_history_buffer[group] = {}
            self._group_obs_term_cfgs[group] = []
            self._group_obs_class_term_cfgs[group] = []
            for index, (width, offset) in enumerate(terms):
                self._group_obs_term_history_buffer[group][str(index)] = History(env.num_envs, env.device)
                self._group_obs_term_cfgs[group].append(
                    SimpleNamespace(func=observation, params={"width": width, "offset": offset}, modifiers=None)
                )

    def compute(self, update_history=False):
        result = {}
        for group, histories in self._group_obs_term_history_buffer.items():
            values = []
            for history, cfg in zip(histories.values(), self._group_obs_term_cfgs[group]):
                fresh = cfg.func(self.env, **cfg.params)
                if self.env.noise:
                    fresh += torch.rand_like(fresh)
                if update_history:
                    history.append(fresh)
                # EA compute(False) evaluates fresh data but returns the old stack.
                values.append(history.buffer.reshape(self.env.num_envs, -1))
            result[group] = torch.cat(values, dim=-1)
        self._obs_buffer = result
        return result

    def reset(self, ids):
        for histories in self._group_obs_term_history_buffer.values():
            for history in histories.values():
                history.reset(ids)


class NativeEnv:
    num_envs = 3
    device = "cpu"

    def __init__(self, noise=False, device="cpu"):
        self.device = device
        self.unwrapped = self
        self.cfg = SimpleNamespace(compute_final_obs=False)
        self.action_manager = SimpleNamespace(total_action_dim=23)
        self.recorder_manager = SimpleNamespace(active_terms=[])
        self.value = torch.zeros(self.num_envs, device=device)
        self.noise = noise
        self.observation_manager = ObservationManager(self)
        self.terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.truncated = torch.zeros_like(self.terminated)
        self.extras = {"dexpbt": {"completed_episodes": 7}, "log": {"native": 1.0}}
        self.closed = False
        self.state = {"dexpbt": {"curriculum": 0.2}}
        self.skip_reset = False

    def _reset_idx(self, ids):
        self.value[ids] = 0
        self.observation_manager.reset(ids)

    def reset(self, seed=None):
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.observation_manager.compute(update_history=True), self.extras

    def step(self, action):
        self.last_action = action
        self.value += 1
        ids = (self.terminated | self.truncated).nonzero().flatten()
        if len(ids) and not self.skip_reset:
            if self.cfg.compute_final_obs:
                self.extras["final_obs"] = self.observation_manager.compute()
            self._reset_idx(ids)
        obs = self.observation_manager.compute(update_history=True)
        return obs, self.value.clone() + 2, self.terminated.clone(), self.truncated.clone(), self.extras

    def get_env_state(self):
        return self.state

    def set_env_state(self, state):
        self.state = state

    def close(self):
        self.closed = True


def test_reset_concatenates_native_groups_and_step_clamps_without_rescaling():
    native = NativeEnv()
    env = DexsuiteVectorEnv(native)
    obs = env.reset(seed=42)
    assert native.seed == 42
    assert (env.num_envs, env.obs_dim, env.action_dim) == (3, 1745, 23)
    assert torch.device(env.device) == torch.device("cpu")
    assert obs.shape == (3, 1745)
    torch.testing.assert_close(
        obs[0, [0, 19, 20, 169, 170, 784, 785, 1744]], torch.tensor([0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0])
    )
    action = torch.full((3, 23), 0.5)
    action[0, :2] = torch.tensor([-2.0, 2.0])
    result = env.step(action)
    torch.testing.assert_close(native.last_action[0, :3], torch.tensor([-1.0, 1.0, 0.5]))
    assert action[0, 0] == -2  # Never mutate the planner's action.
    assert result.infos is native.extras
    assert result.infos["dexpbt"]["completed_episodes"] == 7
    torch.testing.assert_close(result.reward, torch.full((3,), 3.0))
    torch.testing.assert_close(result.next_obs, result.obs)
    assert not result.terminated.any() and not result.truncated.any()


def test_terminal_replay_shifts_each_native_history_once_and_reset_backfills():
    native = NativeEnv()
    env = DexsuiteVectorEnv(native)
    env.reset()
    for _ in range(4):
        env.step(torch.zeros(3, 23))
    native.terminated[0] = True
    native.truncated[1] = True
    result = env.step(torch.zeros(3, 23))
    assert result.terminated.tolist() == [True, False, False]
    assert result.truncated.tolist() == [False, True, False]
    # Term-major layout, not five concatenated whole-group frames.
    for row in (0, 1, 2):
        torch.testing.assert_close(
            result.next_obs[row, :20].reshape(5, 4)[:, 0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        torch.testing.assert_close(
            result.next_obs[row, 20:170].reshape(5, 30)[:, 0], torch.tensor([11.0, 12.0, 13.0, 14.0, 15.0])
        )
        torch.testing.assert_close(
            result.next_obs[row, 170:785].reshape(5, 123)[:, 0], torch.tensor([21.0, 22.0, 23.0, 24.0, 25.0])
        )
        torch.testing.assert_close(
            result.next_obs[row, 785:].reshape(5, 192)[:, 0], torch.tensor([31.0, 32.0, 33.0, 34.0, 35.0])
        )
    torch.testing.assert_close(result.obs[:2, :20], torch.zeros(2, 20))
    torch.testing.assert_close(result.obs[2], result.next_obs[2])
    native.terminated.zero_()
    native.truncated.zero_()
    following = env.step(torch.zeros(3, 23))
    torch.testing.assert_close(following.obs[0, :20].reshape(5, 4)[:, 0], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(following.obs[2, :20].reshape(5, 4)[:, 0], torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_terminal_capture_preserves_acting_noise_and_future_rng_trajectory(device):
    def rollout(wrapped):
        native = NativeEnv(noise=True, device=device)
        env = DexsuiteVectorEnv(native) if wrapped else native
        env.reset(seed=123)
        outputs = []
        for index in range(8):
            native.terminated[0] = index in (1, 5)
            native.truncated[1] = index == 3
            result = env.step(torch.zeros(3, 23, device=device))
            outputs.append(result.obs if wrapped else torch.cat(list(result[0].values()), dim=-1))
        return outputs, torch.rand(10, device=device)

    baseline, baseline_rng = rollout(False)
    adapted, adapted_rng = rollout(True)
    for actual, expected in zip(adapted, baseline):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(adapted_rng, baseline_rng, rtol=0, atol=0)


def test_missing_terminal_capture_fails_instead_of_using_reset_or_stale_info():
    native = NativeEnv()
    env = DexsuiteVectorEnv(native)
    env.reset()
    native.skip_reset = True
    native.truncated[1] = True
    native.extras["final_obs"] = {"policy": torch.zeros(3, 170)}
    with pytest.raises(RuntimeError, match="terminal observation"):
        env.step(torch.zeros(3, 23))


def test_curriculum_state_and_close_forward_to_unwrapped_task():
    native = NativeEnv()
    env = DexsuiteVectorEnv(SimpleNamespace(unwrapped=native))
    assert env.get_state() is native.state
    donor = {"dexpbt": {"curriculum": 0.1}}
    env.set_state(donor)
    assert native.state is donor
    env.close()
    assert native.closed


@pytest.mark.parametrize("unsupported", ["native_final_obs", "recorders", "stateful_modifier", "stateful_term"])
def test_refuses_unvalidated_observation_side_effects(unsupported):
    native = NativeEnv()
    if unsupported == "native_final_obs":
        native.cfg.compute_final_obs = True
    elif unsupported == "recorders":
        native.recorder_manager.active_terms = ["recorder"]
    elif unsupported == "stateful_modifier":
        native.observation_manager._group_obs_class_instances = [object()]
    else:
        native.observation_manager._group_obs_class_term_cfgs["policy"] = [SimpleNamespace(func=object())]
    with pytest.raises(ValueError):
        DexsuiteVectorEnv(native)


def test_action_shape_is_checked_before_native_step():
    env = DexsuiteVectorEnv(NativeEnv())
    env.reset()
    with pytest.raises(ValueError, match="action"):
        env.step(torch.zeros(3, 22))


class object_point_cloud_b:
    __module__ = "LEAP_Isaaclab.compat.dexsuite.mdp.observations"

    def __init__(self):
        self.points_w = torch.zeros(3, 192)
        self.visualizations = 0
        self.fail = False

    def __call__(self, env, width, offset, visualize=True):
        self.points_w = observation(env, width, offset)
        if visualize:
            self.visualizations += 1
        if self.fail:
            torch.rand(3)
            raise RuntimeError("point-cloud fault")
        return self.points_w


def test_native_point_cloud_preview_has_no_visualization_side_effects():
    native = NativeEnv()
    term = native.observation_manager._group_obs_term_cfgs["perception"][0]
    term.func = object_point_cloud_b()
    native.observation_manager._group_obs_class_term_cfgs["perception"] = [term]
    env = DexsuiteVectorEnv(native)
    env.reset()
    native.terminated[0] = True
    result = env.step(torch.zeros(3, 23))
    assert result.next_obs[0, -1] == 31
    assert result.obs[0, -1] == 30
    assert term.func.visualizations == 2  # reset and acting compute only
    assert "visualize" not in term.params


def test_failed_terminal_preview_restores_native_state_and_rng():
    native = NativeEnv(noise=True)
    manager = native.observation_manager
    term = manager._group_obs_term_cfgs["perception"][0]
    term.func = object_point_cloud_b()
    manager._group_obs_class_term_cfgs["perception"] = [term]
    env = DexsuiteVectorEnv(native)
    env.reset(seed=71)
    old_points = term.func.points_w
    old_params = term.params
    old_cache = manager._obs_buffer
    old_histories = manager._group_obs_term_history_buffer
    old_rng = torch.get_rng_state()
    term.func.fail = True
    native.truncated[0] = True
    with pytest.raises(RuntimeError, match="point-cloud fault"):
        env.step(torch.zeros(3, 23))
    assert term.func.points_w is old_points
    assert term.params is old_params
    assert manager._obs_buffer is old_cache
    assert manager._group_obs_term_history_buffer is old_histories
    assert "_reset_idx" not in vars(native)
    torch.testing.assert_close(torch.get_rng_state(), old_rng)
    term.func.fail = False
    obs = env.reset()
    torch.testing.assert_close(obs[:, 0], obs[:, 4])
