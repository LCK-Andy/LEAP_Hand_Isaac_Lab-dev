"""Simulator-free Stage A TD-MPC2 contracts; native grasp accounting, no motion stand-ins."""

import importlib.util
import sys
import torch
from pathlib import Path
from types import SimpleNamespace

import pytest

# Read the native tensor-only state machine without executing task registration/Kit imports.
_PATH = Path(__file__).resolve().parents[1] / (
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_grasp/grasp_logic.py"
)
_SPEC = importlib.util.spec_from_file_location("tdmpc2_test_native_grasp_logic", _PATH)
logic = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = logic
_SPEC.loader.exec_module(logic)


def stage_a_cfg():
    return SimpleNamespace(
        seed=0,
        observation_space=124,
        action_space=16,
        episode_length_s=10.0,
        is_finite_horizon=True,
        compute_final_obs=True,
        decimation=4,
        max_joint_delta=0.015,
        object_mass=0.216,
        cube_half_extent=0.036,
        hold_duration=2.0,
        fingertip_contact_sensors=True,
        allow_unvalidated_backend=False,
        observation_noise_model=None,
        action_noise_model=None,
        sim=SimpleNamespace(dt=1 / 120, physics=type("PhysxCfg", (), {})(), use_newton_actuators=False),
        robot_cfg=SimpleNamespace(
            spawn=SimpleNamespace(
                articulation_props=[
                    SimpleNamespace(solver_position_iteration_count=8, solver_velocity_iteration_count=0)
                ]
            )
        ),
    )


class NativeGrasp:
    """EA same-step reset ABI double using the real native persistent grasp tracker."""

    num_envs = 3
    device = "cpu"
    max_episode_length = 300
    step_dt = 1 / 30
    grasp_hold_contract_version = 1
    condition_key = "grasp_hold_v1_test"
    _physics_backend = "physx"

    def __init__(self):
        self.unwrapped = self
        self.cfg = stage_a_cfg()
        self.value = torch.zeros(3, 124)
        self.targets = torch.zeros(3, 16)
        self.grasp_state = logic.GraspEpisodeState(3, "cpu")
        self.valid = torch.zeros(3, dtype=torch.bool)
        self.supported = torch.zeros(3, dtype=torch.bool)
        self.overload = torch.zeros(3, dtype=torch.bool)
        self.terminated = torch.zeros(3, dtype=torch.bool)
        self.truncated = torch.zeros(3, dtype=torch.bool)
        self.extras = {}
        self.skip_final = False
        self.skip_reset = False
        self.mask = None
        self.fail_reset = False
        self.fail_step = False
        self.closed = False
        self.steps_applied = 0

    def _reset_idx(self, ids):
        if self.fail_reset:
            raise RuntimeError("native reset failed")
        retained = self.truncated[ids] & ~self.terminated[ids]
        self.extras["grasp_hold"] = self.grasp_state.completed(ids, retained, self.max_episode_length)
        self.extras["grasp_hold"]["condition_key"] = self.condition_key
        self.value[ids] = 0
        self.targets[ids] = 0
        self.grasp_state.reset(ids)

    def reset(self, seed=None):
        self.seed = seed
        self._reset_idx(torch.arange(3))
        return {"policy": self.value}, self.extras

    def step(self, action):
        if self.fail_step:
            raise RuntimeError("native step failed")
        self.steps_applied += 1
        self.last_action = action.clone()
        self.targets = logic.advance_joint_targets(
            self.targets, action, torch.full_like(action, -10), torch.full_like(action, 10), 0.015
        )
        self.value += 1
        self.grasp_state.advance(self.valid, self.overload, self.step_dt, 2.0, self.supported)
        self.extras["grasp_hold"] = {}
        ids = (self.terminated | self.truncated).nonzero().flatten()
        if len(ids):
            if not self.skip_final:
                self.extras["final_obs"] = {"policy": self.value}  # Aliases live storage, like EA.
                if self.mask is not None:
                    self.extras["_final_obs"] = self.mask
            if not self.skip_reset:
                self._reset_idx(ids)
        return {"policy": self.value}, torch.ones(3), self.terminated, self.truncated, self.extras

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        assert state is None

    def close(self):
        self.closed = True


def test_native_two_second_success_is_not_retention_or_occupancy_and_survives_drop():
    from scripts.tdmpc2.grasp import GraspEpisodeMetrics, GraspVectorEnv

    native = NativeGrasp()
    env = GraspVectorEnv(native)
    metrics = GraspEpisodeMetrics(3, "cpu")
    env.reset()
    # First episode retains but repeatedly breaks the hold. Others hold for 2s;
    # the second then drops, while the third finishes support-free at timeout.
    for step in range(300):
        native.valid[:] = torch.tensor([step % 2 == 0, step < 60, True])
        native.terminated[:] = False
        native.truncated[:] = False
        if step == 299:
            native.terminated[1] = True
            native.truncated[[0, 2]] = True
        result = env.step(torch.zeros(3, 16))
        metrics.observe(result.reward, result.terminated | result.truncated, result.infos)
    assert not native.grasp_state.success.any()  # Native reset cleared sticky state.
    assert result.infos["grasp_hold"]["completed_grasp_success"].tolist() == [False, True, True]
    values = metrics.values()
    assert values["episode/completed"] == 3
    assert values["episode/success_count"] == 2
    assert values["episode/success_rate"] == pytest.approx(2 / 3)
    assert values["grasp_hold/retained"] == pytest.approx(2 / 3)
    assert values["grasp_hold/valid_fraction"] == pytest.approx(1.7 / 3)
    assert values["grasp_hold/safe_success"] == pytest.approx(1 / 3)
    assert values["episode/length"] == 300
    assert "grasp_motion" not in result.infos


@pytest.mark.parametrize(
    "fault", ["reset_nan", "terminal_nan", "reward_shape", "done_shape", "done_dtype", "success_shape", "success_nan"]
)
def test_malformed_native_tensors_are_rejected_and_capture_is_restored(fault):
    from scripts.tdmpc2.grasp import GraspVectorEnv

    native = NativeGrasp()
    env = GraspVectorEnv(native)
    if fault == "reset_nan":
        native.reset = lambda seed=None: ({"policy": torch.full((3, 124), float("nan"))}, {})
        with pytest.raises((ValueError, FloatingPointError), match="finite"):
            env.reset()
        return
    env.reset()
    native.truncated[0] = True
    if fault == "terminal_nan":
        native.value[0, 0] = float("nan")
    elif fault.startswith("success"):
        native.grasp_state.success = torch.zeros((3, 1)) if fault == "success_shape" else torch.full((3,), float("nan"))
        # Leave this deliberately malformed native field untouched until terminal capture.
        native.grasp_state.advance = lambda *args: native.grasp_state.steps.add_(1)
    else:
        step = native.step

        def malformed_step(action):
            observations, reward, terminated, truncated, infos = step(action)
            if fault == "reward_shape":
                reward = torch.ones(3, 1)
            elif fault == "done_shape":
                terminated = terminated[:, None]
            else:
                terminated = terminated.to(dtype=torch.int64)
            return observations, reward, terminated, truncated, infos

        native.step = malformed_step
    with pytest.raises((ValueError, FloatingPointError), match="finite|shape|vector|boolean|success"):
        env.step(torch.zeros(3, 16))
    assert "_reset_idx" not in vars(native)


def report(retained=(1,), valid=(0.8,), overload=(0.0,), success=(True,), condition="grasp_hold_v1_test"):
    return {
        "grasp_hold": {
            "condition_key": condition,
            "completed_retained": torch.tensor(retained),
            "completed_valid_fraction": torch.tensor(valid),
            "completed_overload_fraction": torch.tensor(overload),
            "completed_grasp_success": torch.tensor(success),
        }
    }


def test_stale_completed_report_cannot_substitute_for_new_terminal_metrics():
    from scripts.tdmpc2.grasp import GraspEpisodeMetrics, GraspVectorEnv

    native = NativeGrasp()
    env = GraspVectorEnv(native)
    env.reset()
    stale = report()["grasp_hold"]
    native.extras["grasp_hold"] = stale

    def ending_without_report(action):
        native.grasp_state.steps += 1
        native.extras["final_obs"] = {"policy": native.value}
        native._reset_idx(torch.tensor([0]))
        return {"policy": native.value}, torch.ones(3), native.terminated, native.truncated, native.extras

    native.step = ending_without_report
    native._reset_idx = lambda ids: native.grasp_state.reset(ids)
    native.truncated[0] = True
    result = env.step(torch.zeros(3, 16))
    metrics = GraspEpisodeMetrics(3, "cpu")
    with pytest.raises(ValueError, match="report"):
        metrics.observe(result.reward, result.terminated, result.infos)
    assert metrics.values()["episode/completed"] == 0


def test_configuration_is_import_safe_and_changes_only_seed_and_capture():
    import ast
    import inspect
    from copy import deepcopy

    from scripts.tdmpc2 import grasp

    cfg = stage_a_cfg()
    cfg.compute_final_obs = False
    before = deepcopy(vars(cfg))
    physics = cfg.sim.physics
    assert grasp.configure_grasp(cfg, 71) is cfg
    assert cfg.seed == 71 and cfg.compute_final_obs is True
    assert cfg.sim.physics is physics
    # Namespace equality except the identity-only backend sentinel.
    before["seed"] = 71
    before["compute_final_obs"] = True
    before["sim"].physics = physics
    assert vars(cfg) == before
    native = NativeGrasp()
    env = grasp.GraspVectorEnv(native)
    assert env.get_state() is None
    env.set_state(None)
    with pytest.raises(ValueError, match="state"):
        env.set_state({"curriculum": 0.1})
    native.get_env_state = lambda: {"changed": True}
    with pytest.raises(ValueError, match="state"):
        env.get_state()
    metadata = env.contract_metadata()
    assert metadata == {
        "observation_groups": ["policy"],
        "episode_length": 300,
        "finite_horizon_terminal": True,
        "physics_dt": 1 / 120,
        "decimation": 4,
        "solver_position_iterations": 8,
        "solver_velocity_iterations": 0,
        "max_joint_delta": 0.015,
        "condition_key": "grasp_hold_v1_test",
    }
    native.condition_key = "grasp_motion_v1_wrong"
    with pytest.raises(ValueError, match="condition"):
        env.condition()
    env.close()
    assert native.closed
    for node in ast.walk(ast.parse(inspect.getsource(grasp))):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith(("isaac", "pxr", "LEAP_Isaaclab")) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("isaac", "pxr", "LEAP_Isaaclab"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("observation_space", 148),
        ("action_space", 23),
        ("episode_length_s", 20),
        ("is_finite_horizon", False),
        ("decimation", 2),
        ("max_joint_delta", 0.03),
        ("hold_duration", 0.1),
        ("cube_half_extent", 0.03),
        ("object_mass", 0.1),
        ("fingertip_contact_sensors", False),
        ("allow_unvalidated_backend", True),
        ("action_noise_model", object()),
        ("observation_noise_model", object()),
    ],
)
def test_changed_native_config_is_refused(field, value):
    from scripts.tdmpc2.grasp import configure_grasp

    cfg = stage_a_cfg()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match="Stage A"):
        configure_grasp(cfg, 1)
    assert cfg.seed == 0


@pytest.mark.parametrize("fault", ["backend", "substeps", "solver", "runtime", "horizon", "version", "capture"])
def test_changed_native_runtime_or_force_history_contract_is_refused(fault):
    from scripts.tdmpc2.grasp import GraspVectorEnv

    native = NativeGrasp()
    if fault == "backend":
        native.cfg.sim.physics = type("NewtonCfg", (), {})()
    elif fault == "substeps":
        native.cfg.sim.use_newton_actuators = True
    elif fault == "solver":
        native.cfg.robot_cfg.spawn.articulation_props[0].solver_velocity_iteration_count = 1
    elif fault == "runtime":
        native._physics_backend = "ovphysx"
    elif fault == "horizon":
        native.max_episode_length = 299
    elif fault == "version":
        native.grasp_hold_contract_version = 2
    else:
        native.cfg.compute_final_obs = False
    with pytest.raises(ValueError, match="Stage A"):
        GraspVectorEnv(native)


def test_native_channels_one_action_application_and_finite_terminal_replay_successor():
    from scripts.tdmpc2.grasp import GraspVectorEnv
    from scripts.tdmpc2.replay import SequenceReplay

    native = NativeGrasp()
    env = GraspVectorEnv(native)
    replay = SequenceReplay(30, 3, 124, 16, 1, 8, sample_device="cpu")
    obs = env.reset(seed=19)
    native.value[:] = torch.arange(124)  # Every native channel (including full forces) is preserved.
    native.terminated[0] = True
    native.truncated[1] = True
    native.mask = torch.tensor([True, True, False])
    action = torch.full((3, 16), 2.0)
    action[:, 0] = -2.0
    result = env.step(action)
    assert native.steps_applied == 1
    assert native.seed == 19
    assert action[0, 0] == -2 and action[0, 1] == 2
    torch.testing.assert_close(native.last_action[0, :2], torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(native.targets[2, :2], torch.tensor([-0.015, 0.015]))
    torch.testing.assert_close(result.next_obs, torch.arange(1, 125).float().expand(3, -1))
    assert result.obs[:2].count_nonzero() == 0
    assert result.terminated.tolist() == [True, True, False]
    assert result.truncated.tolist() == [False, True, False]
    assert result.infos["tdmpc2_grasp"]["native_terminated"].tolist() == [True, False, False]
    assert result.infos["tdmpc2_grasp"]["native_truncated"].tolist() == [False, True, False]
    replay.add(obs, action.clamp(-1, 1), result.reward, result.next_obs, result.terminated, result.truncated)
    sampled_obs, _, _, terminal, _ = replay.sample()
    assert terminal.any()  # Finite timeout is stored as terminal, not a bootstrap-eligible cut.
    assert sampled_obs[-1, :, 0].eq(1).all()
    native.value.fill_(99)
    native.terminated.zero_()
    native.truncated.zero_()
    assert result.next_obs[1, 0] == 1 and result.obs[1, 0] == 0
    assert result.truncated[1]
    assert result.infos["tdmpc2_grasp"]["native_terminated"][0]
    assert "_reset_idx" not in vars(native)


@pytest.mark.parametrize("fault", ["missing_final", "missing_reset", "bad_mask", "step_error", "reset_error"])
@pytest.mark.parametrize("override", [False, True])
def test_capture_is_restored_on_all_errors_and_stale_final_is_refused(fault, override):
    from scripts.tdmpc2.grasp import GraspVectorEnv

    native = NativeGrasp()
    env = GraspVectorEnv(native)
    env.reset()
    if override:
        native._reset_idx = native._reset_idx
    original = native._reset_idx
    native.truncated[0] = True
    native.extras["final_obs"] = {"policy": torch.full((3, 124), 999.0)}
    if fault == "missing_final":
        native.skip_final = True
    elif fault == "missing_reset":
        native.skip_reset = True
    elif fault == "bad_mask":
        native.mask = torch.tensor([False, True, False])
    elif fault == "step_error":
        native.fail_step = True
    else:
        native.fail_reset = True
    with pytest.raises(RuntimeError):
        env.step(torch.zeros(3, 16))
    assert ("_reset_idx" in vars(native)) is override
    assert native._reset_idx == original


@pytest.mark.parametrize("action", [torch.zeros(3, 23), torch.full((3, 16), float("nan"))])
def test_bad_action_never_reaches_native_task(action):
    from scripts.tdmpc2.grasp import GraspVectorEnv

    native = NativeGrasp()
    env = GraspVectorEnv(native)
    with pytest.raises(ValueError):
        env.step(action)
    assert native.steps_applied == 0
    assert "_reset_idx" not in vars(native)


@pytest.mark.parametrize("fault", ["count", "missing_success", "nan", "range", "boolean", "condition"])
def test_bad_native_report_rejection_is_atomic(fault):
    from scripts.tdmpc2.grasp import GraspEpisodeMetrics

    metrics = GraspEpisodeMetrics(2, "cpu")
    infos = report()
    if fault == "count":
        infos["grasp_hold"]["completed_retained"] = torch.tensor([1, 1])
    elif fault == "missing_success":
        del infos["grasp_hold"]["completed_grasp_success"]
    elif fault == "nan":
        infos["grasp_hold"]["completed_valid_fraction"][0] = float("nan")
    elif fault == "range":
        infos["grasp_hold"]["completed_overload_fraction"][0] = 1.5
    elif fault == "boolean":
        infos["grasp_hold"]["completed_grasp_success"] = torch.tensor([0.5])
    else:
        infos["grasp_hold"]["condition_key"] = "grasp_motion_v1_other"
    before = metrics.state_dict()
    with pytest.raises(ValueError, match="report"):
        metrics.observe(torch.ones(2), torch.tensor([True, False]), infos)
    assert metrics.state_dict() == before
    metrics.observe(torch.ones(2), torch.tensor([True, False]), report())
    assert metrics.values()["episode/return"] == 1
    assert metrics.values()["episode/length"] == 1
    with pytest.raises(ValueError, match="conditions"):
        metrics.observe(torch.ones(2), torch.tensor([True, False]), report(condition="grasp_hold_v1_other"))


def test_support_free_retention_is_native_not_raw_timeout():
    from scripts.tdmpc2.grasp import GraspEpisodeMetrics, GraspVectorEnv

    native = NativeGrasp()
    env = GraspVectorEnv(native)
    env.reset()
    native.supported[0] = True
    native.truncated[0] = True
    result = env.step(torch.zeros(3, 16))
    assert result.truncated[0]
    assert result.infos["grasp_hold"]["completed_retained"].tolist() == [0.0]
    metrics = GraspEpisodeMetrics(3, "cpu")
    metrics.observe(result.reward, result.terminated, result.infos)
    assert metrics.values()["grasp_hold/retained"] == 0
    assert metrics.values()["episode/success_count"] == 0


def test_checkpoint_keeps_recent100_and_full_statistics_but_resets_partial_episodes():
    from scripts.tdmpc2.grasp import GraspEpisodeMetrics

    metrics = GraspEpisodeMetrics(2, "cpu")
    for index in range(105):
        metrics.observe(
            torch.tensor([float(index), 99.0]),
            torch.tensor([True, False]),
            report(retained=(index >= 5,), success=(index >= 5,)),
        )
    before = metrics.values()
    assert before["episode/completed"] == 105
    assert before["episode/return"] == 52
    assert before["episode/success_rate"] == pytest.approx(100 / 105)
    assert before["recent/completed"] == 100
    assert before["recent/success_rate"] == 1
    assert before["recent/return"] == 54.5
    snapshot = metrics.state_dict()
    metrics.observe(torch.ones(2), torch.zeros(2, dtype=torch.bool), {"grasp_hold": {"stale": True}})
    assert metrics.values() == before
    restored = GraspEpisodeMetrics(2, "cpu")
    restored.load_state_dict(snapshot)
    assert restored.values() == before
    restored.observe(torch.tensor([1.0, 4.0]), torch.tensor([False, True]), report(success=(False,)))
    assert restored.values()["episode/return"] == pytest.approx((5460 + 4) / 106)
    assert restored.values()["episode/length"] == 1
    assert snapshot["totals"]["completed"] == 105
    assert len(snapshot["recent"]) == 100
    with pytest.raises(ValueError, match="checkpoint"):
        GraspEpisodeMetrics(2, "cpu", recent_window=3).load_state_dict(snapshot)
    snapshot["totals"]["success_count"] = 1000
    current = restored.state_dict()
    with pytest.raises(ValueError, match="checkpoint"):
        restored.load_state_dict(snapshot)
    assert restored.state_dict() == current


def test_agent124_update_and_full_restore_keep_next_learner_update():
    from scripts.tdmpc2.agent import Agent
    from scripts.tdmpc2.config import TDMPC2Config
    from scripts.tdmpc2.grasp import GraspVectorEnv

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        env = GraspVectorEnv(NativeGrasp())
        cfg = TDMPC2Config(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            num_envs=env.num_envs,
            episode_length=300,
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
        agent = Agent(cfg, device="cpu")
        action = agent.act(env.reset(), torch.ones(3, dtype=torch.bool))
        assert agent.cfg.obs_dim == 124 and agent.cfg.action_dim == 16
        assert action.shape == (3, 16) and torch.isfinite(action).all() and (action.abs() <= 1).all()
        assert env.step(action).obs.shape == (3, 124)

        class Replay:
            def sample(self):
                return (
                    torch.randn(cfg.horizon + 1, 2, 124),
                    torch.randn(cfg.horizon, 2, 16).clamp(-1, 1),
                    torch.randn(cfg.horizon, 2, 1),
                    torch.ones(cfg.horizon, 2, 1),
                    None,
                )

        agent.update(Replay())
        state = agent.state_dict()
        expected = agent.update(Replay())
        restored = Agent(cfg, device="cpu")
        restored.load_state_dict(state, reset_planner=True)
        actual = restored.update(Replay())
        assert restored.prev_mean.count_nonzero() == 0
        assert restored.update_count == 2
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)
