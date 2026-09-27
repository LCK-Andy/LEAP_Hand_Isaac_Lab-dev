"""Simulator-free public-boundary tests for Stage B TD-MPC2 collection and metrics."""

import torch
from types import SimpleNamespace

import pytest


def stage_b_cfg():
    return SimpleNamespace(
        seed=0,
        observation_space=148,
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
        motion=SimpleNamespace(
            rotation_amplitude=(0.001745329252,) * 3,
            translation_amplitude=(0.0,) * 3,
            frequency_hz=(0.2, 0.3, 0.4),
            settle_seconds=1.0,
            ramp_seconds=2.0,
            position_tolerance=0.0005,
            rotation_tolerance=0.0007,
            linear_velocity_tolerance=0.002,
            angular_velocity_tolerance=0.002,
            velocity_filter_alpha=0.2,
            tracking_reward_scale=3.0,
            success_hold_seconds=1.0,
            randomize_phase=True,
        ),
    )


class NativeGraspMotion:
    """Model EA's same-step reset/final-observation ABI, not Isaac physics."""

    num_envs = 3
    device = "cpu"
    max_episode_length = 300
    step_dt = 1 / 30
    grasp_motion_contract_version = 1
    grasp_hold_contract_version = 1
    condition_key = "grasp_motion_v1_test"
    _physics_backend = "physx"

    def __init__(self):
        self.unwrapped = self
        self.cfg = stage_b_cfg()
        self.value = torch.zeros(3, 148)
        self._tracking_success = torch.tensor([False, True, False])
        self.grasp_state = SimpleNamespace(steps=torch.zeros(3, dtype=torch.long))
        self.terminated = torch.zeros(3, dtype=torch.bool)
        self.truncated = torch.zeros(3, dtype=torch.bool)
        self.extras = {}
        self.closed = False
        self.skip_final = False
        self.mask = None
        self.fail_reset = False

    def _reset_idx(self, ids):
        if self.fail_reset:
            raise RuntimeError("native reset failed")
        self.value[ids] = 0
        self.grasp_state.steps[ids] = 0
        self._tracking_success[ids] = False

    def reset(self, seed=None):
        self.seed = seed
        self._reset_idx(torch.arange(3))
        return {"policy": self.value}, self.extras

    def step(self, action):
        self.last_action = action.clone()
        self.value += 1
        self.grasp_state.steps += 1
        self.extras["grasp_motion"] = {}
        self.extras["grasp_hold"] = {}
        ids = (self.terminated | self.truncated).nonzero().flatten()
        if len(ids):
            if not self.skip_final:
                # Deliberately alias live storage: the adapter must clone BEFORE reset.
                self.extras["final_obs"] = {"policy": self.value}
                if self.mask is not None:
                    self.extras["_final_obs"] = self.mask
            self.extras["grasp_motion"] = {
                "condition_key": self.condition_key,
                "completed_tracking_fraction": torch.full((len(ids),), 0.6),
                "completed_retained": (~self.terminated[ids]).float(),
                "completed_overload_fraction": torch.zeros(len(ids)),
            }
            self.extras["grasp_hold"] = {
                "condition_key": self.condition_key,
                "completed_valid_fraction": torch.full((len(ids),), 0.8),
                "completed_retained": (~self.terminated[ids]).float(),
                "completed_overload_fraction": torch.zeros(len(ids)),
            }
            self._reset_idx(ids)
        return {"policy": self.value}, torch.ones(3), self.terminated, self.truncated, self.extras

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        if state is not None:
            raise ValueError("Native Stage B has no curriculum state")

    def close(self):
        self.closed = True


def test_same_step_full_batch_final_observation_and_finite_horizon_terminal():
    from scripts.tdmpc2.grasp_motion import GraspMotionVectorEnv

    native = NativeGraspMotion()
    env = GraspMotionVectorEnv(native)
    env.reset(seed=7)
    native._tracking_success[1] = True
    native.terminated[0] = True
    native.truncated[1] = True
    native.mask = torch.tensor([True, True, False])
    action = torch.full((3, 16), 0.5)
    action[0, :2] = torch.tensor([-2.0, 2.0])
    result = env.step(action)
    assert native.seed == 7
    assert (env.obs_dim, env.action_dim, env.num_envs) == (148, 16, 3)
    torch.testing.assert_close(native.last_action[0, :3], torch.tensor([-1.0, 1.0, 0.5]))
    assert action[0, 0] == -2
    torch.testing.assert_close(result.next_obs, torch.ones(3, 148))
    torch.testing.assert_close(result.obs[:2], torch.zeros(2, 148))
    assert result.terminated.tolist() == [True, True, False]
    assert result.truncated.tolist() == [False, True, False]
    assert result.infos["tdmpc2_grasp_motion"]["native_terminated"].tolist() == [True, False, False]
    assert result.infos["grasp_motion"]["completed_tracking_success"].tolist() == [False, True]
    assert "_reset_idx" not in vars(native)
    native.value.fill_(9)
    assert result.next_obs[0, 0] == 1
    assert result.obs[0, 0] == 0


def test_configuration_state_and_condition_are_native_and_import_safe():
    import ast
    import inspect

    from scripts.tdmpc2 import grasp_motion

    cfg = stage_b_cfg()
    cfg.compute_final_obs = False
    assert grasp_motion.configure_grasp_motion(cfg, 13) is cfg
    assert cfg.seed == 13 and cfg.compute_final_obs
    native = NativeGraspMotion()
    env = grasp_motion.GraspMotionVectorEnv(native)
    assert env.get_state() is None
    env.set_state(None)
    with pytest.raises(ValueError, match="state"):
        env.set_state({"curriculum": 0.5})
    assert env.condition() == native.condition_key
    metadata = env.contract_metadata()
    assert metadata["observation_groups"] == ["policy"]
    assert metadata["solver_position_iterations"] == 8
    assert metadata["episode_length"] == 300
    assert metadata["finite_horizon_terminal"] is True
    env.close()
    assert native.closed
    for node in ast.walk(ast.parse(inspect.getsource(grasp_motion))):
        if isinstance(node, ast.Import):
            assert all(not name.name.startswith(("isaac", "LEAP_Isaaclab", "pxr")) for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("isaac", "LEAP_Isaaclab", "pxr"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("observation_space", 124),
        ("action_space", 23),
        ("is_finite_horizon", False),
        ("episode_length_s", 12.0),
        ("decimation", 2),
        ("max_joint_delta", 0.03),
        ("object_mass", 0.1),
        ("fingertip_contact_sensors", False),
    ],
)
def test_changed_stage_b_contract_is_rejected(field, value):
    from scripts.tdmpc2.grasp_motion import GraspMotionVectorEnv

    native = NativeGraspMotion()
    setattr(native.cfg, field, value)
    with pytest.raises(ValueError, match="Stage B"):
        GraspMotionVectorEnv(native)


def test_changed_motion_or_backend_is_rejected():
    from scripts.tdmpc2.grasp_motion import configure_grasp_motion

    cfg = stage_b_cfg()
    cfg.motion.success_hold_seconds = 0.1
    with pytest.raises(ValueError, match="success_hold_seconds"):
        configure_grasp_motion(cfg, 1)
    cfg = stage_b_cfg()
    cfg.sim.physics = type("NewtonCfg", (), {})()
    with pytest.raises(ValueError, match="PhysX"):
        configure_grasp_motion(cfg, 1)


@pytest.mark.parametrize("mask", [None, torch.tensor([True, False, False])])
def test_native_final_obs_without_mask_and_incomplete_mask_failure(mask):
    from scripts.tdmpc2.grasp_motion import GraspMotionVectorEnv

    native = NativeGraspMotion()
    env = GraspMotionVectorEnv(native)
    env.reset()
    native.truncated[1] = True
    native.mask = mask
    if mask is None:
        assert env.step(torch.zeros(3, 16)).next_obs[1, 0] == 1
    else:
        with pytest.raises(RuntimeError, match="mask"):
            env.step(torch.zeros(3, 16))
    assert "_reset_idx" not in vars(native)


def test_stale_final_observations_and_logs_never_substitute_for_fresh_endings():
    from scripts.tdmpc2.grasp_motion import GraspMotionVectorEnv

    native = NativeGraspMotion()
    env = GraspMotionVectorEnv(native)
    env.reset()
    native.extras.update(final_obs={"policy": torch.full((3, 148), 999.0)}, _final_obs=torch.ones(3, dtype=torch.bool))
    result = env.step(torch.zeros(3, 16))
    assert "final_obs" not in result.infos and "_final_obs" not in result.infos
    assert result.infos["grasp_motion"] == {}
    native.skip_final = True
    native.truncated[0] = True
    native.extras["final_obs"] = {"policy": torch.full((3, 148), 999.0)}
    with pytest.raises(RuntimeError, match="fresh"):
        env.step(torch.zeros(3, 16))
    assert "_reset_idx" not in vars(native)


def test_failed_step_restores_existing_reset_override_and_missing_capture_fails():
    from scripts.tdmpc2.grasp_motion import GraspMotionVectorEnv

    native = NativeGraspMotion()
    env = GraspMotionVectorEnv(native)
    env.reset()
    native._reset_idx = native._reset_idx
    original = native._reset_idx
    native.truncated[0] = True
    native.fail_reset = True
    with pytest.raises(RuntimeError, match="native reset failed"):
        env.step(torch.zeros(3, 16))
    assert native._reset_idx is original


def test_reset_starts_clean_observation_stream_and_replay_does_not_bridge():
    from scripts.tdmpc2.grasp_motion import GraspMotionVectorEnv
    from scripts.tdmpc2.replay import SequenceReplay

    env = GraspMotionVectorEnv(NativeGraspMotion())
    replay = SequenceReplay(30, 3, 148, 16, 3, 16, sample_device="cpu")
    obs = env.reset()
    for _ in range(4):
        result = env.step(torch.zeros(3, 16))
        replay.add(obs, torch.zeros(3, 16), result.reward, result.next_obs, result.terminated, result.truncated)
        obs = result.obs
    saved = obs.clone()
    replay.start_new_stream()
    obs = env.reset()
    assert obs.max() == 0 and saved.min() == 4
    result = env.step(torch.zeros(3, 16))
    replay.add(obs, torch.zeros(3, 16), result.reward, result.next_obs, result.terminated, result.truncated)
    sampled = replay.sample()[0]
    assert torch.all(sampled[1:] - sampled[:-1] == 1)


def report(tracking, retained, valid, overload, success):
    return {
        "grasp_motion": {
            "condition_key": "grasp_motion_v1_test",
            "completed_tracking_fraction": torch.tensor(tracking),
            "completed_retained": torch.tensor(retained),
            "completed_overload_fraction": torch.tensor(overload),
            "completed_tracking_success": torch.tensor(success),
        },
        "grasp_hold": {
            "condition_key": "grasp_motion_v1_test",
            "completed_valid_fraction": torch.tensor(valid),
            "completed_retained": torch.tensor(retained),
            "completed_overload_fraction": torch.tensor(overload),
        },
    }


def test_episode_metrics_use_native_sticky_success_and_episode_weighting():
    from scripts.tdmpc2.grasp_motion import GraspMotionEpisodeMetrics

    metrics = GraspMotionEpisodeMetrics(2, "cpu", recent_window=2)
    # High occupancy is NOT a continuous one-second hold; successful hold can precede dropping.
    metrics.observe(
        torch.tensor([2.0, 10.0]),
        torch.tensor([True, False]),
        report([0.9], [0], [0.95], [0.0], [False]),
    )
    metrics.observe(
        torch.tensor([4.0, 20.0]),
        torch.tensor([True, True]),
        report([0.2, 0.4], [1, 0], [0.6, 0.8], [0.1, 0.2], [True, True]),
    )
    values = metrics.values()
    assert metrics.totals["completed"] == 3 and metrics.totals["success_count"] == 2
    assert values["episode/return"] == 12.0
    assert values["episode/length"] == pytest.approx(4 / 3)
    assert values["episode/success_rate"] == pytest.approx(2 / 3)
    assert values["grasp_motion/tracking_fraction"] == pytest.approx(0.5)
    assert values["grasp_motion/retained"] == pytest.approx(1 / 3)
    assert values["grasp_motion/valid_fraction"] == pytest.approx(2.35 / 3)
    assert values["grasp_motion/overload_fraction"] == pytest.approx(0.1)
    # Average products per episode, not product of average retention and tracking.
    assert values["grasp_motion/retention_weighted_tracking"] == pytest.approx(0.2 / 3)
    assert values["recent/success_rate"] == 1.0
    assert values["recent/tracking_fraction"] == pytest.approx(0.3)
    assert values["recent/return"] == 17.0


@pytest.mark.parametrize("fault", ["count", "missing_success", "nonfinite", "range", "retention", "condition"])
def test_native_metric_arrays_are_complete_consistent_and_rejection_is_atomic(fault):
    from scripts.tdmpc2.grasp_motion import GraspMotionEpisodeMetrics

    metrics = GraspMotionEpisodeMetrics(2, "cpu")
    infos = report([0.5], [1], [0.8], [0.0], [True])
    if fault == "count":
        infos["grasp_motion"]["completed_tracking_fraction"] = torch.tensor([0.1, 0.2])
    elif fault == "missing_success":
        del infos["grasp_motion"]["completed_tracking_success"]
    elif fault == "nonfinite":
        infos["grasp_hold"]["completed_valid_fraction"][0] = float("nan")
    elif fault == "range":
        infos["grasp_motion"]["completed_tracking_fraction"][0] = 1.5
    elif fault == "retention":
        infos["grasp_hold"]["completed_retained"][0] = 0
    else:
        infos["grasp_hold"]["condition_key"] = "different"
    before = metrics.values()
    with pytest.raises(ValueError, match="report"):
        metrics.observe(torch.ones(2), torch.tensor([True, False]), infos)
    assert metrics.values() == before
    # A rejected step must not accrue reward/length for later observations.
    metrics.observe(torch.ones(2), torch.tensor([True, False]), report([0.5], [1], [0.8], [0], [True]))
    assert metrics.values()["episode/return"] == 1
    assert metrics.values()["episode/length"] == 1


def test_metric_resume_preserves_recent_and_totals_but_discards_partial_stream():
    from scripts.tdmpc2.grasp_motion import GraspMotionEpisodeMetrics

    metrics = GraspMotionEpisodeMetrics(2, "cpu", recent_window=2)
    metrics.observe(torch.tensor([2.0, 10.0]), torch.tensor([True, False]), report([0.5], [1], [0.8], [0], [True]))
    before = metrics.values()
    state = metrics.state_dict()
    # Ignore stale success/logs on a step without completed episodes.
    metrics.observe(torch.ones(2), torch.zeros(2, dtype=torch.bool), {"grasp_motion": {"invalid": "stale"}})
    assert metrics.values() == before
    restored = GraspMotionEpisodeMetrics(2, "cpu", recent_window=2)
    restored.load_state_dict(state)
    assert restored.values() == before
    restored.observe(torch.tensor([1.0, 4.0]), torch.tensor([False, True]), report([0.1], [0], [0.2], [0], [False]))
    assert restored.values()["episode/return"] == 3
    assert restored.values()["episode/length"] == 1
    assert restored.values()["episode/success_count"] == 1
    # Snapshot does not alias mutable metric totals or recent entries.
    assert state["totals"]["completed"] == 1
    with pytest.raises(ValueError, match="checkpoint"):
        GraspMotionEpisodeMetrics(2, "cpu", recent_window=3).load_state_dict(state)
    state["totals"]["success_count"] = 9
    with pytest.raises(ValueError, match="checkpoint"):
        restored.load_state_dict(state)


def test_adapter_metrics_integration_captures_success_before_native_reset():
    from scripts.tdmpc2.grasp_motion import GraspMotionEpisodeMetrics, GraspMotionVectorEnv

    native = NativeGraspMotion()
    env = GraspMotionVectorEnv(native)
    metrics = GraspMotionEpisodeMetrics(3, "cpu")
    env.reset()
    native.truncated[1] = True
    native._tracking_success[1] = True
    result = env.step(torch.zeros(3, 16))
    assert not native._tracking_success.any()
    metrics.observe(result.reward, result.terminated | result.truncated, result.infos)
    assert metrics.values()["episode/success_rate"] == 1
