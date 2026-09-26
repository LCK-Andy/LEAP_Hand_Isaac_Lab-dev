"""Stage B registration and reward-independent motion selection, without simulation."""

import gymnasium as gym
import importlib.util
import sys
import torch
import types
import yaml
from pathlib import Path

import pytest
from test_dexpbt_pbt import FakeAlgo, config, load_module
from test_grasp_pbt import window_cfg

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks"
TASK = "Isaac-Grasp-Motion-Cube-LEAP-Direct-v0"
PLAY = "Isaac-Grasp-Motion-Cube-LEAP-Direct-Play-v0"


def test_motion_registration_is_additive_and_uses_bounded_ppo(monkeypatch):
    registry = {}
    monkeypatch.setattr(gym.envs.registration, "registry", registry)
    monkeypatch.setattr(gym, "registry", registry)
    for name, path in (("LEAP_Isaaclab", TASKS.parent), ("LEAP_Isaaclab.tasks", TASKS)):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    for package in ("leap_hand_grasp", "leap_hand_grasp_motion"):
        name = f"LEAP_Isaaclab.tasks.{package}"
        spec = importlib.util.spec_from_file_location(name, TASKS / package / "__init__.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        if package == "leap_hand_grasp":
            stage_a = dict(registry)
    assert all(registry[key] is value for key, value in stage_a.items())
    for task, cfg in ((TASK, "LeapGraspMotionEnvCfg"), (PLAY, "LeapGraspMotionEnvCfg_PLAY")):
        spec = gym.spec(task)
        assert spec.entry_point == "LEAP_Isaaclab.tasks.leap_hand_grasp_motion.leap_grasp_motion_env:LeapGraspMotionEnv"
        assert spec.kwargs["env_cfg_entry_point"].endswith(f"leap_grasp_motion_env_cfg:{cfg}")
        assert spec.kwargs["rl_games_cfg_entry_point"].endswith("agents:rl_games_ppo_grasp_motion_cfg.yaml")
    agent = yaml.safe_load((TASKS / "leap_hand_grasp_motion/agents/rl_games_ppo_grasp_motion_cfg.yaml").read_text())
    cfg = agent["params"]["config"]
    assert cfg["name"] == "leap_grasp_motion"
    assert agent["pbt"]["objective"] == "grasp_motion" and not agent["pbt"]["enabled"]
    assert agent["params"]["network"]["mlp"]["units"] == [256, 128, 64]
    assert agent["params"]["network"]["separate"] is False
    assert cfg["exploration"] == {"initial_std": 0.15, "min_std": 0.03, "max_std": 0.3}
    assert cfg["entropy_coef"] == 0 and cfg["normalize_input"] and cfg["normalize_value"]
    assert cfg["torch_compile"] is False


def motion_info(tracking=(1.0, 0.5, 1.0), retained=(1, 0, 1), overload=(0.0, 0.0, 0.1), condition="motion-v1"):
    return {
        "grasp_motion": {
            "condition_key": condition,
            "completed_tracking_fraction": torch.tensor(tracking),
            "completed_retained": torch.tensor(retained),
            "completed_overload_fraction": torch.tensor(overload),
        }
    }


def test_motion_window_ignores_stage_a_and_scores_safe_tracking_first():
    motion = load_module("grasp_motion")
    window = motion.GraspMotionWindow()
    from test_grasp_pbt import episode_info

    window.add(episode_info(), 10, now=10)
    assert not window.ready(window_cfg(), 10, now=11)
    window.add(motion_info(), 20, now=10)
    stats = window.ready(window_cfg(), 20, now=11)["grasp_motion"]
    assert stats["completed_episodes"] == 3
    assert stats["success_rate"] == 0.5  # tracking, NOT retention or shaped return
    assert stats["progress"] == 1 / 3  # safe whole-episode retention tie-break
    assert stats["overload_rate"] == 1 / 3
    assert stats["variance"] == 0.25
    assert stats["condition_key"] == "motion-v1"
    window.add(motion_info((1.0,), (1,), (0.0,), "changed"), 30, now=12)
    assert not window.ready(window_cfg(), 30, now=13)
    with pytest.raises(ValueError, match="grasp_motion"):
        window.add(motion_info((float("nan"),), (1,), (0.0,)), 40, now=14)


def motion_record(tracking, retained=1.0, overload=0.0, condition="motion-v1"):
    return {
        "schema_version": 2,
        "frame": 100,
        "created_at": 100,
        "evaluations": {
            "grasp_motion": {
                "condition_key": condition,
                "completed_episodes": 1000,
                "success_rate": tracking,
                "variance": 0.1,
                "progress": retained,
                "progress_variance": 0.1,
                "overload_rate": overload,
                "created_at": 100,
                "end_frame": 100,
            }
        },
    }


def test_motion_donor_ranks_tracking_with_retention_guard_and_never_stage_a_reward():
    select = load_module("grasp_motion").select_motion_donor
    ours = motion_record(0.5, 0.8)
    assert select({1: motion_record(0.8, 0.8)}, ours, window_cfg(), now=101) == 1
    assert select({1: motion_record(0.8, 0.7)}, ours, window_cfg(), now=101) is None
    assert select({1: motion_record(0.8, 0.8, 0.01)}, ours, window_cfg(), now=101) is None
    assert select({1: motion_record(0.5, 0.95)}, ours, window_cfg(), now=101) == 1
    assert select({1: motion_record(0.51, 1.0)}, ours, window_cfg(), now=101) is None
    assert select({1: motion_record(0.8, condition="other")}, ours, window_cfg(), now=101) is None
    assert select({1: motion_record(0.8)}, ours, window_cfg(), now=3000) is None
    stage_a = {"schema_version": 2, "true_objective": 9999, "evaluations": {"grasp_hold": {}}}
    assert select({1: stage_a}, ours, window_cfg(), now=101) is None


def test_motion_objective_publication_and_restart_are_separate(tmp_path, monkeypatch):
    pbt = load_module("pbt")
    executed = []
    monkeypatch.setattr(__import__("os"), "execv", lambda exe, argv: executed.append(argv))
    for member, tracking in ((0, 1.0), (1, 0.2)):
        settings = config(member)
        settings["pbt"].update(objective="grasp_motion", min_completed_episodes=4)
        algo = FakeAlgo(tmp_path, settings, "MOTION")
        algo.frame, algo.epoch_num = 200 + 100 * member, 2 + member
        args = types.SimpleNamespace(original_argv=["train.py", "--task", TASK, "--warm-start", "stage_a.pth"])
        observer = pbt.PbtAlgoObserver(settings, args)
        observer.after_init(algo)
        observer.process_infos(motion_info((tracking,) * 4, (1,) * 4, (0.0,) * 4), [])
        observer.after_print_stats(algo.frame, algo.epoch_num, 1.0)
    assert len(executed) == 1
    assert TASK in executed[0]
    assert not any("warm-start" in arg or "stage_a" in arg for arg in executed[0])


def test_motion_validator_requires_own_task_and_measurement_contract():
    validate = load_module("grasp_motion").validate_motion_environment
    env = types.SimpleNamespace(
        _physics_backend="physx",
        grasp_motion_contract_version=1,
        get_env_state=lambda: None,
        set_env_state=lambda value: None,
    )
    validate(TASK, env)
    for task in (PLAY, "Isaac-Grasp-Cube-LEAP-Direct-v0"):
        with pytest.raises(ValueError, match="grasp_motion"):
            validate(task, env)
    env.grasp_motion_contract_version = 0
    with pytest.raises(ValueError, match="contract"):
        validate(TASK, env)
