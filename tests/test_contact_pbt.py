"""Simulator-free tests of actual native geometry/hold code and PBT measurements."""

import ast
import importlib.util
import itertools
import torch
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks"
spec = importlib.util.spec_from_file_location("contact_measurement", TASKS / "dexsuite_reorient_pbt/pbt/contact.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def native_class():
    # Execute the real methods with only simulator asset access stubbed; never
    # duplicate the distance indexing or timer implementation in the test.
    tree = ast.parse((TASKS / "leap_hand_repose/leap_contact_direct_env.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in {"_compute_intermediate_values", "_get_rewards"}
    ]
    reward = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_contact_rewards")
    reward.decorator_list = []
    base = type("Base", (), {"_compute_intermediate_values": lambda self: None})
    scope = {"torch": torch, "LeapHandDirectEnv": base, "quat_apply": lambda q, v: v}
    exec(compile(ast.Module(body=[reward, cls], type_ignores=[]), "native_contact", "exec"), scope)
    return scope[cls.name]()


def native_env():
    env = native_class()
    env.num_envs = 2
    env.object_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    env.object_pos = torch.zeros(2, 3)
    env.contact_points_local = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]]
    ).repeat(2, 1, 1)
    env.fingertip_pos = env.contact_points_local[:, [2, 0, 3, 1]].clone()
    env.fingertip_contact_forces = torch.ones(2, 4, 3)
    env._contact_force_perm = torch.arange(4)
    env._contact_perms = torch.tensor(list(itertools.permutations(range(4))))
    env._ensure_contact_state = lambda: None
    env._update_contact_point_markers = lambda: None
    env.cfg = SimpleNamespace(
        touch_distance=0.015,
        touch_force=0.1,
        point_dist_reward_scale=-0.05,
        touch_bonus_scale=0.5,
        action_penalty_scale=-0.0002,
        joint_energy_penalty_scale=-0.001,
        object_away_penalty_scale=-1.0,
        object_away_dist=0.2,
        reach_goal_bonus=250.0,
        av_factor=0.1,
        cube_pull_scale=-0.25,
        fall_penalty=-50.0,
        alive_bonus_scale=0.1,
        alive_distance=0.1,
    )
    env.hold_timer = torch.zeros(2)
    env.hold_target = torch.full((2,), 0.1)
    env.step_dt = 1 / 30
    env.reset_buf = torch.zeros(2, dtype=torch.bool)
    env.reset_terminated = env.reset_buf.clone()
    env.reset_goal_buf = torch.zeros(2)
    env.successes = torch.zeros(2)
    env.consecutive_successes = torch.zeros(1)
    env.actions = torch.zeros(2, 16)
    env.hand_dof_vel = torch.zeros(2, 16)
    env.hand = SimpleNamespace(data=SimpleNamespace(applied_torque=SimpleNamespace(torch=torch.zeros(2, 16))))
    env.in_hand_pos = torch.zeros(2, 3)
    env.extras = {}
    env.resampled = []

    def resample(ids):
        env.resampled.extend(ids.tolist())
        env.hold_timer[ids] = 0
        env.reset_goal_buf[ids] = 0

    env._reset_target_pose = resample
    return env


def test_native_assignment_uses_each_point_column_and_force_pairing():
    env = native_env()
    env._compute_intermediate_values()
    assert torch.equal(env.matched_dist, torch.zeros(2, 4))
    assert env.touched.sum().item() == 8
    env.fingertip_contact_forces[:, 0] = 0  # tip 0 belongs to point 2, NOT point 0
    env._compute_intermediate_values()
    assert env.touched.tolist() == [[1, 1, 0, 1], [1, 1, 0, 1]]
    env.fingertip_contact_forces.zero_()
    env._compute_intermediate_values()
    assert env.touched.sum() == 0  # hovering cannot earn progress


def test_native_partial_reset_geometry_does_not_advance_other_hold():
    env = native_env()
    env._compute_intermediate_values()
    env._get_rewards()
    assert torch.allclose(env.hold_timer, torch.full((2,), 1 / 30))
    # A subset reset invokes the same ALL-env geometry refresh. Repeated refresh
    # must neither advance env 1's hold nor confirm a success.
    env._reset_target_pose(torch.tensor([0]))
    for _ in range(5):
        env._compute_intermediate_values()
    assert env.hold_timer[0] == 0
    assert abs(env.hold_timer[1].item() - 1 / 30) < 1e-6
    assert env.successes.sum() == 0
    env._get_rewards()
    assert env.successes.sum() == 0
    env._get_rewards()
    assert env.successes.tolist() == [0, 1]
    env._compute_intermediate_values()
    env.fingertip_contact_forces.zero_()
    env._compute_intermediate_values()
    env._get_rewards()
    assert env.hold_timer.sum() == 0


def test_measurement_captures_native_held_completion_before_reset_once():
    env = native_env()
    env.device = "cpu"
    env._physics_backend = "physx"
    env.max_episode_length = 300
    env._reset_idx = lambda ids: env.successes.index_fill_(0, ids, 0)
    module.attach_measurement(env)
    env._reset_idx(torch.tensor([0, 1]))
    assert "contact_episodes" not in env.extras
    for _ in range(3):
        env._compute_intermediate_values()
        env._get_rewards()
    env.reset_buf[:] = True
    env._reset_idx(torch.tensor([0, 1]))
    n, goals, squares, progress, _ = env.extras["contact_episodes"]
    assert (n, goals, squares) == (2, 2, 2)
    assert abs(progress - 0.02) < 1e-6
    retained = dict(env.extras)
    env._get_rewards()
    assert "contact_episodes" not in env.extras
    assert "Contact/held_goals_per_completed_episode" not in env.extras
    assert retained["Contact/held_goals_per_completed_episode"] == 1.0
    assert not any(key.startswith("Contact/") for key in env.extras.get("log", {}))


def test_goal_resample_refreshes_observation_vectors_without_done_or_hold_advance():
    env = native_env()
    original_reset = env._reset_target_pose

    def resample(ids):
        original_reset(ids)
        env.contact_points_local[ids] += 0.3

    env._reset_target_pose = resample
    for _ in range(3):
        env._compute_intermediate_values()
        env._get_rewards()
    assert env.successes.tolist() == [1, 1]
    assert not env.reset_buf.any()
    assert env.hold_timer.sum() == 0
    expected = env.fingertip_pos.unsqueeze(2) - env.contact_points_local.unsqueeze(1)
    assert torch.allclose(env.tip_point_deltas, expected)
    assert env.touched.sum() == 0


def cfg():
    return SimpleNamespace(
        min_completed_episodes=512,
        max_score_age_seconds=1800,
        max_frame_lag=2000000,
        policy_idx=0,
        threshold_abs=0.001,
        contact_progress_threshold_abs=0.00001,
        threshold_std=1.0,
    )


def record(goals=0.0, progress=0.1, frame=1000000, created=100.0):
    return {
        "schema_version": 2,
        "frame": frame,
        "created_at": created,
        "evaluations": {
            "native_contact": {
                "completed_episodes": 10000,
                "goals_per_episode": goals,
                "variance": 0.02,
                "progress": progress,
                "progress_variance": 0.02,
                "created_at": created,
                "end_frame": frame - 16000,
            }
        },
    }


def test_selection_success_priority_zero_only_tie_break_and_freshness():
    def choose(donor, own=None):
        return module.select_contact_donor({1: donor}, record() if own is None else own, cfg(), now=101)

    assert choose(record(progress=0.11, frame=9000000)) == 1
    assert choose(record(goals=0.01, progress=0)) == 1
    assert choose(record(progress=1), record(goals=0.01)) is None
    assert choose(record(goals=0.01, progress=1), record(goals=0.01)) is None
    assert choose(record(progress=0.1001)) is None
    assert choose(record(progress=0.9, created=-2000)) is None
    stale = record(progress=0.9)
    stale["evaluations"]["native_contact"]["end_frame"] = -9000000
    assert choose(stale) is None
    noisy = record(progress=0.11)
    noisy["evaluations"]["native_contact"]["progress_variance"] = 100
    assert choose(noisy) is None


def test_contact_forced_restart_transaction(tmp_path, monkeypatch):
    from test_dexpbt_pbt import FakeAlgo, config, load_module

    pbt = load_module("pbt")
    executed = []
    monkeypatch.setattr(__import__("os"), "execv", lambda exe, argv: executed.append(argv))
    for member, progress in ((0, 2.0), (1, 0.0)):
        settings = config(member)
        settings["pbt"].update(objective="contact_goals", threshold_abs=0.001, min_completed_episodes=4)
        algo = FakeAlgo(tmp_path, settings, "DONOR" if member == 0 else "RECIPIENT")
        algo.frame, algo.epoch_num = 200 + 100 * member, 2 + member
        observer = pbt.PbtAlgoObserver(settings, SimpleNamespace(original_argv=["/repo/train.py"]))
        observer.after_init(algo)
        observer.process_infos({"contact_episodes": (4, 0, 0, progress, progress * progress / 4)}, [])
        observer.after_print_stats(algo.frame, algo.epoch_num, 1.0)
    assert len(executed) == 1
    metadata_arg = next(arg for arg in executed[0] if arg.startswith("--pbt-resume="))
    import yaml

    metadata = yaml.safe_load(Path(metadata_arg.split("=", 1)[1]).read_text())
    assert metadata["recipient_frame"] == 300
    assert metadata["donor_frame"] == 200
    assert metadata["donor_policy_idx"] == 0
    assert metadata["pbt_rng_state"]


def test_window_variance_min_samples_expiration_and_consumption():
    window = module.ContactWindow()
    window.add({"contact_episodes": (512, 1024, 2048, 128, 32)}, 100, now=100)
    stats = window.ready(cfg(), 100, now=101)["native_contact"]
    assert stats["goals_per_episode"] == 2
    assert stats["progress"] == 0.25
    assert stats["variance"] == stats["progress_variance"] == 0
    window.consume()
    assert not window.ready(cfg(), 100, now=101)
    window.add({"contact_episodes": (1, 0, 0, 0, 0)}, 100, now=100)
    assert not window.ready(cfg(), 100, now=101)
    window.add({"contact_episodes": (512, 0, 0, 0, 0)}, 100, now=100)
    assert not window.ready(cfg(), 100, now=2001)
