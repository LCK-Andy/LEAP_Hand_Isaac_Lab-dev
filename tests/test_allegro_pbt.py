"""Simulator-free native Allegro measurement and asynchronous selection tests."""

import importlib.util
import torch
from pathlib import Path
from types import SimpleNamespace

PATH = (
    Path(__file__).resolve().parents[1]
    / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt/pbt/allegro.py"
)
spec = importlib.util.spec_from_file_location("allegro_measurement", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cfg():
    return SimpleNamespace(
        min_completed_episodes=128,
        max_score_age_seconds=1800,
        max_frame_lag=2_000_000,
        policy_idx=0,
        threshold_abs=0.001,
        threshold_std=1.0,
    )


def record(mean, frame=1000000, created=100):
    return {
        "schema_version": 2,
        "created_at": created,
        "frame": frame,
        "evaluations": {
            "official_allegro": {
                "goals_per_episode": mean,
                "variance": 0.02,
                "completed_episodes": 10000,
                "created_at": created,
                "end_frame": frame - 16000,
            }
        },
    }


def test_allegro_forced_restart_transaction(tmp_path, monkeypatch):
    from test_dexpbt_pbt import FakeAlgo, config, load_module

    pbt = load_module("pbt")
    executed = []
    monkeypatch.setattr(__import__("os"), "execv", lambda exe, argv: executed.append(argv))
    for member, total in ((0, 40), (1, 0)):
        settings = config(member)
        settings["pbt"].update(objective="allegro_goals", threshold_abs=0.001, min_completed_episodes=4)
        algo = FakeAlgo(tmp_path, settings, "DONOR" if member == 0 else "RECIPIENT")
        algo.frame, algo.epoch_num = 200 + 100 * member, 2 + member
        observer = pbt.PbtAlgoObserver(settings, SimpleNamespace(original_argv=["/repo/train.py"]))
        observer.after_init(algo)
        observer.process_infos({"allegro_episodes": (4, total, total * total / 4)}, [])
        observer.after_print_stats(algo.frame, algo.epoch_num, 1.0)
    assert len(executed) == 1
    metadata_arg = next(arg for arg in executed[0] if arg.startswith("--pbt-resume="))
    import yaml

    metadata = yaml.safe_load(Path(metadata_arg.split("=", 1)[1]).read_text())
    assert metadata["recipient_frame"] == 300
    assert metadata["donor_frame"] == 200
    assert metadata["donor_policy_idx"] == 0


def test_count_window_not_binomial():
    window = module.GoalWindow()
    window.add({"allegro_episodes": (128, 512, 2048)}, 100, now=100)
    stats = window.ready(cfg(), 100, now=100)["official_allegro"]
    assert stats["goals_per_episode"] == 4
    assert stats["variance"] == 0
    window.consume()
    assert not window.ready(cfg(), 100, now=100)


def test_small_reachable_difference_and_async_peer():
    assert module.select_goal_donor({1: record(0.009, 9000000)}, record(0.001), cfg(), now=101) == 1


def test_stale_tie_and_noisy_peer_rejected():
    for other in (record(0.001), record(0.009, created=-2000), record(0.002)):
        assert module.select_goal_donor({1: other}, record(0.001), cfg(), now=101) is None


def test_measurement_keeps_native_rewards_resets_and_counts_once():
    class Env:
        extras = {}
        reset_buf = torch.tensor([False, True])
        successes = torch.tensor([9.0, 3.0])

        def _get_rewards(self):
            return torch.tensor([5.0, 7.0])

        def _reset_idx(self, ids):
            self.successes[ids] = 0

    env = Env()
    module.attach_measurement(env)
    assert env._get_rewards().tolist() == [5, 7]
    env._reset_idx(torch.tensor([0, 1]))
    assert env.extras["allegro_episodes"] == (1, 3.0, 9.0)
    assert env.successes.tolist() == [0, 0]
    env._get_rewards()
    assert "allegro_episodes" not in env.extras
    assert env.get_env_state() is None
