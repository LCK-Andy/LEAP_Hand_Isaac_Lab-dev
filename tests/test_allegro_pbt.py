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


def test_allegro_defaults_do_not_change_other_objectives():
    from test_dexpbt_pbt import load_module

    config = load_module("pbt_cfg").PbtCfg
    allegro = config(objective="allegro_goals")
    assert (allegro.threshold_std, allegro.mutation_rate, allegro.change_range) == (2.0, 0.33, (1.1, 1.5))
    for objective in ("dexpbt", "contact_goals"):
        other = config(objective=objective)
        assert (other.threshold_std, other.mutation_rate, other.change_range) == (0.2, 0.25, (1.1, 2.0))


def test_conservative_gate_reports_counts_gap_and_standard_error():
    from test_dexpbt_pbt import load_module

    settings = load_module("pbt_cfg").PbtCfg(objective="allegro_goals", donor_maturity_rollouts=0)
    settings.threshold_abs = 0.001
    own, other = record(1.0), record(1.003)
    diagnostics = []
    assert module.select_goal_donor({1: other}, own, settings, now=101, diagnostics=diagnostics) is None
    assert diagnostics[0]["reason"] == "margin"
    assert diagnostics[0]["combined_se"] == 0.002
    assert diagnostics[0]["recipient_count"] == diagnostics[0]["donor_count"] == 10000
    settings.threshold_std = 1.0
    assert module.select_goal_donor({1: other}, own, settings, now=101) == 1


def test_two_windows_require_new_nonoverlapping_donor_and_recipient_evidence():
    from test_dexpbt_pbt import load_module

    settings = load_module("pbt_cfg").PbtCfg(objective="allegro_goals", evidence_windows=2)
    evidence = {}
    own, donor = record(1), record(2, 9000000)
    assert module.select_goal_donor({1: donor}, own, settings, now=101, evidence=evidence) is None
    newer = record(1, 1100000)
    newer["evaluations"]["official_allegro"]["start_frame"] = 1000001
    assert module.select_goal_donor({1: donor}, newer, settings, now=102, evidence=evidence) is None
    fresh = record(2, 9100000)
    fresh["evaluations"]["official_allegro"]["start_frame"] = 9000001
    assert module.select_goal_donor({1: fresh}, newer, settings, now=103, evidence=evidence) == 1
    fresh["last_exploit_frame"] = 9050000
    assert module.select_goal_donor({1: fresh}, newer, settings, now=104, evidence=evidence) is None


def test_two_window_confirmation_does_not_extend_original_sample_freshness():
    from test_dexpbt_pbt import load_module

    settings = load_module("pbt_cfg").PbtCfg(objective="allegro_goals", evidence_windows=2)
    evidence = {}
    assert module.select_goal_donor({1: record(2)}, record(1), settings, now=1899, evidence=evidence) is None
    own, donor = record(1, 2000000, created=1901), record(2, 2000000, created=1901)
    for sample in (own, donor):
        sample["evaluations"]["official_allegro"]["start_frame"] = 1000001
    assert module.select_goal_donor({1: donor}, own, settings, now=1901, evidence=evidence) is None


def test_weakest_role_uses_only_fresh_windows_and_donor_own_maturity():
    from test_dexpbt_pbt import load_module

    settings = load_module("pbt_cfg").PbtCfg(
        objective="allegro_goals", recipient_role="weakest", donor_maturity_rollouts=64, warmup_steps=3000000
    )
    donor = record(3, 9000000)
    donor.update(rollout_frames=16000, last_exploit_frame=8000000)
    assert module.select_goal_donor({1: donor}, record(1), settings, now=101) is None
    donor["last_exploit_frame"] = 7976000  # exactly 64 donor rollouts
    assert module.select_goal_donor({1: donor}, record(1), settings, now=101) == 1
    assert module.select_goal_donor({1: donor, 2: record(0)}, record(1), settings, now=101) is None
    assert module.select_goal_donor({1: donor, 2: record(0, created=-2000)}, record(1), settings, now=101) == 1
    assert module.select_goal_donor({1: donor}, record(3), settings, now=101) is None
    donor["last_exploit_frame"] = 0  # cold-start donor uses its training age
    assert module.select_goal_donor({1: donor}, record(1), settings, now=101) == 1


def test_observer_two_window_evidence_survives_resume_and_cooldown(tmp_path, monkeypatch):
    from test_dexpbt_pbt import FakeAlgo, config, load_module

    pbt = load_module("pbt")
    runtime = load_module("runtime")
    executed = []
    monkeypatch.setattr(pbt.os, "execv", lambda *args: executed.append(args))

    def observer(member, metadata=None):
        settings = config(member)
        settings["pbt"].update(objective="allegro_goals", evidence_windows=2, adaptation_steps=200)
        args = SimpleNamespace(original_argv=["train.py"], pbt_resume_metadata=metadata)
        obs = pbt.PbtAlgoObserver(settings, args)
        algo = FakeAlgo(tmp_path, settings, f"member-{member}")
        obs.after_init(algo)
        if metadata:
            algo.restore(metadata["checkpoint"])
        return obs, algo

    def publish(obs, algo, frame, goals):
        algo.frame, algo.epoch_num = frame, frame // 100
        obs.process_infos({"allegro_episodes": (4, goals * 4, goals * goals * 4)}, [])
        obs.after_print_stats(frame - 100, algo.epoch_num, 0)

    donor, da = observer(0)
    recipient, ra = observer(1)
    publish(donor, da, 100, 10)
    publish(recipient, ra, 100, 0)  # recipient cooldown prevents evidence collection
    metadata = runtime.resume_metadata(recipient.curr_policy_dir + "/000000000001.pth")
    assert not metadata["selection_evidence"]
    publish(recipient, ra, 200, 0)
    metadata = runtime.resume_metadata(recipient.curr_policy_dir + "/000000000002.pth")
    assert metadata["selection_evidence"] and not executed
    resumed, algo = observer(1, metadata)
    publish(resumed, algo, 300, 0)  # old donor cannot count twice after resume
    assert not executed
    publish(donor, da, 400, 10)
    publish(resumed, algo, 400, 0)
    assert len(executed) == 1
    staged = runtime.resume_metadata(resumed.curr_policy_dir + "/restart/checkpoint.pth")
    assert not staged["selection_evidence"]
    assert staged["lineage"] == "member-1@400" and staged["donor_lineage"] == "member-0"
    assert staged["checkpoint_copy_seconds"] >= 0


def test_launcher_isolates_pilots_and_pairs_seed_budget_gpu(tmp_path):
    import json
    import subprocess
    import sys

    script = PATH.parents[6] / "scripts/launch/allegro_pbt_experiment.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "--arms",
            "baseline",
            "selective",
            "copy-only",
            "gentler",
            "diversity",
            "cadence",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    plans = json.loads(result.stdout)
    by_arm = {p["arm"]: p for p in plans if p["member"] == 0}

    def knobs(plan):
        return {a.split("=", 1)[0]: a.split("=", 1)[1] for a in plan["argv"] if a.startswith("pbt.")}

    baseline = knobs(by_arm["baseline"])
    for arm, key in (
        ("selective", "threshold_std"),
        ("gentler", "mutation_rate"),
        ("diversity", "recipient_role"),
        ("cadence", "interval_steps"),
    ):
        changes = {k for k, v in knobs(by_arm[arm]).items() if baseline[k] != v}
        assert changes == {"pbt.directory", "pbt." + key}
    assert knobs(by_arm["copy-only"])["pbt.threshold_std"] == "2.0"
    assert knobs(by_arm["copy-only"])["pbt.copy_only"] == "true"
    assert {p["steps"] for p in plans} == {33554432}
    for member in range(4):
        paired = [p for p in plans if p["member"] == member]
        assert len({(p["seed"], p["gpu"]) for p in paired}) == 1
    assert all("params.config.max_frames=33554432" in p["argv"] for p in plans)


def test_final_evaluation_plan_checks_exact_budget_and_heldout_seeds(tmp_path):
    import json
    import subprocess
    import sys
    import yaml

    checkpoint = tmp_path / "final.pth"
    torch.save({"frame": 1600, "epoch": 100, "model": {}}, checkpoint)
    checkpoint.with_suffix(".yaml").write_text(yaml.safe_dump({"terminal": True, "frame": 1600, "epoch": 100}))
    manifest = tmp_path / "experiment.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "arm": "baseline",
                    "member": 0,
                    "seed": 42,
                    "gpu": 0,
                    "steps": 1600,
                    "epochs": 100,
                    "final_checkpoint": str(checkpoint),
                }
            ]
        )
    )
    helper = PATH.parents[6] / "scripts/rl_games/evaluate_allegro_population.py"
    command = [sys.executable, str(helper), "--manifest", str(manifest), "--split", "test"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    jobs = json.loads(result.stdout)
    assert len(jobs) == 2 and {j["seed"] for j in jobs} == {2042, 2043}
    assert all(j["checkpoint"] == str(checkpoint) for j in jobs)
    leaked = subprocess.run(command + "--test-seeds 42".split(), capture_output=True, text=True)
    assert leaked.returncode != 0 and "disjoint" in leaked.stderr
    torch.save({"frame": 1616, "epoch": 101, "model": {}}, checkpoint)
    wrong = subprocess.run(command, capture_output=True, text=True)
    assert wrong.returncode != 0 and "budget" in wrong.stderr


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
