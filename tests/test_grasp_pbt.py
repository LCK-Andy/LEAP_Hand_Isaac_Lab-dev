"""Simulator-free Stage A registration, completed-episode selection, and restart contracts."""

import gymnasium as gym
import importlib.util
import sys
import torch
import types
import yaml
from pathlib import Path

import pytest
from test_dexpbt_pbt import FakeAlgo, config, load_module

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks"
GRASP = TASKS / "leap_hand_grasp"
TASK = "Isaac-Grasp-Cube-LEAP-Direct-v0"
PLAY = "Isaac-Grasp-Cube-LEAP-Direct-Play-v0"


def test_new_registration_preserves_existing_contact_ids(monkeypatch):
    # Import the actual registration initializers without the simulator-loading
    # root initializer; Gym's registry is the public interface under test.
    registry = {}
    monkeypatch.setattr(gym.envs.registration, "registry", registry)
    monkeypatch.setattr(gym, "registry", registry)
    for name, path in (("LEAP_Isaaclab", TASKS.parent), ("LEAP_Isaaclab.tasks", TASKS)):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)

    def register(package):
        name = f"LEAP_Isaaclab.tasks.{package}"
        spec = importlib.util.spec_from_file_location(name, TASKS / package / "__init__.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)

    register("leap_hand_repose")
    contact = {key: value for key, value in registry.items() if key.startswith("Isaac-Contact-")}
    assert len(contact) == 4
    register("leap_hand_grasp")
    assert {key: value for key, value in registry.items() if key.startswith("Isaac-Contact-")} == contact
    for task, config_name in ((TASK, "LeapGraspEnvCfg"), (PLAY, "LeapGraspEnvCfg_PLAY")):
        spec = gym.spec(task)
        assert spec.entry_point == "LEAP_Isaaclab.tasks.leap_hand_grasp.leap_grasp_env:LeapGraspEnv"
        assert spec.disable_env_checker
        assert spec.kwargs == {
            "env_cfg_entry_point": f"LEAP_Isaaclab.tasks.leap_hand_grasp.leap_grasp_env_cfg:{config_name}",
            "rl_games_cfg_entry_point": "LEAP_Isaaclab.tasks.leap_hand_grasp.agents:rl_games_ppo_grasp_cfg.yaml",
        }


def test_ppo_configuration_preserves_force_range_and_bounds_delta_exploration():
    agent = yaml.safe_load((GRASP / "agents/rl_games_ppo_grasp_cfg.yaml").read_text())
    params = agent["params"]
    assert params["algo"]["name"] == "a2c_continuous"
    assert params["model"]["name"] == "continuous_a2c_logstd"
    assert params["env"] == {
        "clip_observations": 100.0,
        "clip_actions": 1.0,
        "obs_groups": {"obs": ["policy"], "states": ["policy"]},
    }
    continuous = params["network"]["space"]["continuous"]
    assert continuous["fixed_sigma"] is True
    assert continuous["sigma_activation"] == "None"
    config = params["config"]
    assert config["name"] == "leap_grasp"
    assert config["entropy_coef"] == 0
    assert config["exploration"] == {"initial_std": 0.15, "min_std": 0.03, "max_std": 0.3}
    assert config["ppo"] and config["normalize_input"]
    assert (256 * config["horizon_length"]) % config["minibatch_size"] == 0
    assert agent["pbt"]["objective"] == "grasp_hold"
    assert agent["pbt"]["enabled"] is False  # local PPO is safe by default; PBT is opt-in
    assert set(agent["pbt"]["mutation"]) == {
        "agent.params.config.learning_rate",
        "agent.params.config.grad_norm",
        "agent.params.config.kl_threshold",
    }


def episode_info(valid=(1.0, 0.5, 1.0), retained=(1, 0, 1), overload=(0.0, 0.0, 0.1), condition="nominal-v1"):
    return {
        "grasp_hold": {
            "condition_key": condition,
            "completed_valid_fraction": torch.tensor(valid),
            "completed_retained": torch.tensor(retained),
            "completed_overload_fraction": torch.tensor(overload),
        }
    }


def window_cfg(**overrides):
    return types.SimpleNamespace(
        **dict(
            {
                "policy_idx": 0,
                "min_completed_episodes": 3,
                "max_score_age_seconds": 1800,
                "max_frame_lag": 200,
                "threshold_abs": 0.01,
                "threshold_std": 1.0,
            },
            **overrides,
        )
    )


def test_completed_window_excludes_unsafe_credit_and_weights_episodes():
    grasp = load_module("grasp")
    window = grasp.GraspHoldWindow()
    window.add({"episode": {"reward": 999999}}, 10, now=10)
    window.add({"grasp_hold": {}}, 10, now=10)
    assert not window.ready(window_cfg(), 10, now=10)
    window.add(episode_info(), 20, now=10)
    stats = window.ready(window_cfg(), 20, now=11)["grasp_hold"]
    assert stats["completed_episodes"] == 3
    assert stats["condition_key"] == "nominal-v1"
    assert stats["success_rate"] == 1 / 3
    assert stats["progress"] == 0.5
    assert stats["overload_rate"] == 1 / 3
    assert abs(stats["variance"] - 1 / 3) < 1e-7
    assert stats["progress_variance"] == 0.25
    # An unequal reset batch adds one episode, not one vote for the batch mean.
    window.add(episode_info((1.0,), (1,), (0.0,)), 30, now=12)
    stats = window.ready(window_cfg(), 30, now=13)["grasp_hold"]
    assert stats["completed_episodes"] == 4
    assert stats["success_rate"] == 0.5
    assert stats["progress"] == 0.625


def test_window_requires_fresh_sufficient_same_condition_episodes():
    window = load_module("grasp").GraspHoldWindow()
    cfg = window_cfg()
    window.add(episode_info((1.0,), (1,), (0.0,)), 10, now=10)
    assert not window.ready(cfg, 10, now=11)
    window.add(episode_info(condition="changed-horizon"), 20, now=12)
    stats = window.ready(cfg, 20, now=13)["grasp_hold"]
    assert stats["completed_episodes"] == 3
    assert stats["condition_key"] == "changed-horizon"
    assert not window.ready(cfg, 20, now=2000)
    window.add(episode_info(), 30, now=2001)
    assert not window.ready(cfg, 231, now=2002)
    window.add(episode_info(), 240, now=2003)
    assert window.ready(cfg, 240, now=2004)
    window.consume()
    assert not window.ready(cfg, 240, now=2004)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("condition_key", ""),
        ("completed_valid_fraction", torch.tensor([float("nan"), 0.0, 0.0])),
        ("completed_valid_fraction", torch.tensor([1.1, 0.0, 0.0])),
        ("completed_valid_fraction", torch.zeros(3, 1)),
        ("completed_valid_fraction", torch.zeros(2)),
        ("completed_valid_fraction", [0.0, 0.0, 0.0]),
        ("completed_retained", torch.tensor([0.5, 0.0, 0.0])),
        ("completed_overload_fraction", torch.tensor([-0.1, 0.0, 0.0])),
    ],
)
def test_invalid_completed_metrics_fail_without_poisoning_window(key, value):
    window = load_module("grasp").GraspHoldWindow()
    window.add(episode_info(), 20, now=10)
    before = window.ready(window_cfg(), 20, now=11)
    malformed = episode_info()
    malformed["grasp_hold"][key] = value
    with pytest.raises(ValueError, match="grasp_hold"):
        window.add(malformed, 30, now=12)
    assert window.ready(window_cfg(), 30, now=13) == before


def record(retained=0.0, valid=0.2, overload=0.0, condition="nominal-v1", frame=100, created=100):
    return {
        "schema_version": 2,
        "created_at": created,
        "frame": frame,
        "evaluations": {
            "grasp_hold": {
                "condition_key": condition,
                "completed_episodes": 1000,
                "success_rate": retained,
                "variance": 0.1,
                "progress": valid,
                "progress_variance": 0.1,
                "overload_rate": overload,
                "created_at": created,
                "end_frame": frame - 16,
            }
        },
    }


def test_selector_prefers_safe_retention_then_valid4_never_raw_reward():
    select = load_module("grasp").select_grasp_donor
    ours = record(retained=0.3, valid=0.8)
    assert select({1: record(retained=0.5, valid=0.1)}, ours, window_cfg(), now=101) == 1
    assert select({1: record(retained=0.2, valid=1.0)}, ours, window_cfg(), now=101) is None
    assert select({1: record(retained=0.3, valid=0.95)}, ours, window_cfg(), now=101) == 1
    assert select({1: record(retained=0.31, valid=1.0)}, ours, window_cfg(), now=101) is None
    assert select({1: record(retained=0.5, overload=0.01)}, ours, window_cfg(), now=101) is None
    better = record(retained=0.5, overload=0.01)
    assert select({1: better}, record(retained=0.3, overload=0.02), window_cfg(), now=101) == 1
    reward_only = record(retained=0.3, valid=0.8)
    reward_only["true_objective"] = 1e12
    assert select({1: reward_only}, ours, window_cfg(), now=101) is None


def test_support_free_retention_outranks_supported_full_horizon_occupancy():
    grasp = load_module("grasp")
    cfg = window_cfg()

    def publication(info):
        window = grasp.GraspHoldWindow()
        window.add(info, 100, now=100)
        return {
            "schema_version": 2,
            "created_at": 100,
            "frame": 100,
            "evaluations": window.ready(cfg, 100, now=101),
        }

    # The env owns support_seen bookkeeping: even one supported step clears
    # completed_retained. Raw full-horizon survival is diagnostic, not selection.
    supported_info = episode_info((0.95,) * 128, (0,) * 128, (0.0,) * 128)
    supported_info["log"] = {"GraspEpisode/raw_retained": 1.0}
    supported = publication(supported_info)
    support_free = publication(episode_info((0.6,) * 128, (1,) * 96 + (0,) * 32, (0.0,) * 128))
    assert grasp.select_grasp_donor({1: supported}, support_free, cfg, now=101) is None
    assert grasp.select_grasp_donor({1: support_free}, supported, cfg, now=101) == 1


def test_selector_requires_matching_conditions_samples_and_freshness():
    select = load_module("grasp").select_grasp_donor
    ours = record()
    assert select({1: record(valid=0.8, frame=9000000)}, ours, window_cfg(), now=101) == 1
    assert select({1: record(valid=0.8, condition="other")}, ours, window_cfg(), now=101) is None
    assert select({1: record(valid=0.8, created=-2000)}, ours, window_cfg(), now=101) is None
    for field, value in (("completed_episodes", 1), ("end_frame", -2000), ("progress_variance", 1000)):
        donor = record(valid=0.8)
        donor["evaluations"]["grasp_hold"][field] = value
        assert select({1: donor}, ours, window_cfg(), now=101) is None


def test_grasp_objective_restarts_full_state_with_recipient_identity(tmp_path, monkeypatch):
    pbt = load_module("pbt")
    runtime = load_module("runtime")
    executed = []
    monkeypatch.setattr(__import__("os"), "execv", lambda exe, argv: executed.append((exe, argv)))
    for member, retained in ((0, 1), (1, 0)):
        settings = config(member)
        settings["pbt"].update(objective="grasp_hold", min_completed_episodes=4)
        algo = FakeAlgo(tmp_path, settings, "DONOR" if member == 0 else "RECIPIENT")
        algo.frame, algo.epoch_num = 200 + 100 * member, 2 + member
        args = types.SimpleNamespace(
            original_argv=["/repo/train.py", "--task", TASK, f"--seed={42 + member}", "--headless"]
        )
        observer = pbt.PbtAlgoObserver(settings, args)
        observer.after_init(algo)
        observer.process_infos(episode_info((0.8,) * 4, (retained,) * 4, (0.0,) * 4), [])
        observer.after_print_stats(algo.frame, algo.epoch_num, 1.0)
    assert len(executed) == 1
    executable, argv = executed[0]
    assert executable == sys.executable
    assert TASK in argv and "--seed=43" in argv and "--headless" in argv
    checkpoint = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--checkpoint="))
    metadata_path = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--pbt-resume="))
    metadata = runtime.resume_metadata(checkpoint, metadata_path)
    assert metadata["recipient_frame"] == 300 and metadata["donor_frame"] == 200
    assert metadata["policy_idx"] == 1 and metadata["donor_policy_idx"] == 0
    assert metadata["pbt_rng_state"]
    restored = FakeAlgo(tmp_path, settings, "FRESH")
    args.pbt_resume_metadata = metadata
    resumed = pbt.PbtAlgoObserver(settings, args)
    resumed.after_init(restored)
    restored.restore(checkpoint)
    assert restored.model["weights"] == "DONOR"
    assert restored.optimizer.param_groups[0]["momentum"] == "DONOR"
    assert (restored.frame, restored.epoch_num) == (300, 3)
    assert not resumed.window.ready(resumed.cfg, restored.frame)


def test_member_contract_accepts_grasp_backends_with_versioned_metrics_and_state_hooks():
    validate = load_module("grasp").validate_grasp_environment
    env = types.SimpleNamespace(
        _physics_backend="physx",
        grasp_hold_contract_version=1,
        get_env_state=lambda: None,
        set_env_state=lambda state: None,
    )
    for backend in ("physx", "newton"):
        env._physics_backend = backend
        validate(TASK, env)
    for task in (PLAY, "Isaac-Contact-Cube-LEAP-Direct-v0"):
        with pytest.raises(ValueError, match="grasp_hold"):
            validate(task, env)
    env._physics_backend = "ovphysx"
    with pytest.raises(ValueError, match="backend"):
        validate(TASK, env)
    env._physics_backend = "physx"
    env.grasp_hold_contract_version = 0
    with pytest.raises(ValueError, match="contract"):
        validate(TASK, env)
    env.grasp_hold_contract_version = 1
    env.set_env_state = None
    with pytest.raises(ValueError, match="checkpoint"):
        validate(TASK, env)
