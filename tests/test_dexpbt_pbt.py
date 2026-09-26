"""Lightweight public-boundary PBT tests; no simulator or task imports."""

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PBT = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt/pbt"


def load_module(name):
    root = "_dexpbt_test"
    package = root + ".pbt"
    for module_name, path in ((root, PBT.parent), (package, PBT)):
        if module_name not in sys.modules:
            module = types.ModuleType(module_name)
            module.__path__ = [str(path)]
            sys.modules[module_name] = module
    spec = importlib.util.spec_from_file_location(f"{package}.{name}", PBT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_roundtrip_preserves_identity_and_types():
    runtime = load_module("runtime")
    original = [
        "/repo/train.py",
        "--task",
        "task",
        "--seed=44",
        "--headless",
        "--checkpoint",
        "old.pth",
        "pbt.policy_idx=2",
        "pbt.num_policies=4",
        "pbt.directory=/tmp/a b",
        "env.foo=3",
        "agent.params.config.learning_rate=1e-3",
        "params.config.entropy_coef=0.01",
    ]
    command = runtime.restart_command(
        original, "python-real", "/tmp/new.pth", "/tmp/new.yaml", {"agent.params.config.learning_rate": 5e-5}
    )
    assert command[:2] == ["python-real", "/repo/train.py"]
    assert "old.pth" not in command
    assert "--seed=44" in command and "pbt.directory=/tmp/a b" in command and "env.foo=3" in command
    assert command.count("pbt.policy_idx=2") == 1
    pbt, params, kept = runtime.split_overrides(command[2:])
    assert pbt["policy_idx"] == 2 and pbt["num_policies"] == 4
    cfg = {"params": {"config": {"learning_rate": 0.01, "entropy_coef": 0.1}}}
    runtime.apply_overrides(cfg, params)
    assert cfg["params"]["config"] == {"learning_rate": 5e-5, "entropy_coef": 0.01}
    assert "config" not in cfg
    assert "env.foo=3" in kept
    assert runtime.parse_value("1e-4") == 0.0001
    assert runtime.parse_value("false") is False
    assert runtime.parse_value("[1e-4, 0.1]") == [0.0001, 0.1]
    resolved = runtime.resolved_seed_argv(["train.py", "--seed", "-1", "params.seed=-1", "--headless"], 123)
    assert resolved == ["train.py", "--headless", "--seed=123"]


def config(member=0):
    prefix = "agent.params.config."
    return {
        "pbt": {
            "enabled": True,
            "policy_idx": member,
            "num_policies": 2,
            "interval_steps": 100,
            "min_completed_episodes": 4,
            "threshold_abs": 0.001,
            "threshold_std": 0.2,
            "mutation": {prefix + "learning_rate": "mutate_float", prefix + "kl_threshold": "mutate_float"},
            "mutation_bounds": {prefix + "learning_rate": [1e-5, 0.01], prefix + "kl_threshold": [0.001, 0.1]},
        },
        "params": {
            "seed": 42,
            "config": {"learning_rate": 0.001, "kl_threshold": 0.01, "entropy_coef": 0.0, "grad_norm": 1.0},
        },
    }


def episode_info(success=0, position=0.7, count=4, adr=0.0):
    return {
        "dexpbt": {
            "schema_version": 1,
            "protocol": "training_conditions_fixed_definition_v1",
            "groups": [
                {
                    "condition": {
                        "adr_fraction": adr,
                        "curriculum_tolerance": 0.2,
                        "position_tolerance": 0.1,
                        "rotation_tolerance": 0.5,
                        "contact_threshold": 0.01,
                        "adr_parameters": "fixed",
                    },
                    "completed_episodes": count,
                    "success_count": success,
                    "position_error_sum": count * position,
                    "rotation_error_sum": count * 0.1,
                    "good_contact_fraction_sum": count,
                }
            ],
        }
    }


def test_windows_require_completed_samples_freshness_and_conditions():
    utils = load_module("pbt_utils")
    cfg = load_module("pbt_cfg").PbtCfg(**config()["pbt"])
    window = utils.EpisodeWindow()
    window.add({"episode": {"Episode_Reward/success": 100}}, 0, now=10)
    assert not window.ready(cfg, 0, now=10)
    window.add(episode_info(count=2), 10, now=10)
    assert not window.ready(cfg, 10, now=11)
    window.add(episode_info(count=2), 20, now=12)
    ready = window.ready(cfg, 20, now=13)
    assert next(iter(ready.values()))["completed_episodes"] == 4
    window.add(episode_info(count=1, adr=0.5), 20, now=12)
    assert len(window.ready(cfg, 20, now=13)) == 1
    assert not window.ready(cfg, 20, now=2000)
    window.add(episode_info(count=4), 30, now=2001)
    assert window.ready(cfg, 30, now=2001)
    window.consume()
    assert not window.ready(cfg, 30, now=2001)


def test_slow_publication_keeps_only_recent_completed_batches():
    utils = load_module("pbt_utils")
    cfg = load_module("pbt_cfg").PbtCfg(min_completed_episodes=128)
    window = utils.EpisodeWindow()
    window.add(episode_info(success=128, position=9.0, count=128), 0, now=0)
    window.add(episode_info(success=16, position=0.5, count=64), 450_000, now=1000)
    window.add(episode_info(success=32, position=0.25, count=64), 900_000, now=2000)
    # A million frames at 450 FPS is longer than max_score_age_seconds.
    ready = window.ready(cfg, 1_000_000, now=2222)
    assert len(ready) == 1
    stats = next(iter(ready.values()))
    assert stats["completed_episodes"] == 128
    assert stats["success_count"] == 48 and stats["success_rate"] == 0.375
    assert stats["position_error_sum"] == 48.0
    assert stats["good_contact_fraction_sum"] == 128.0
    assert stats["created_at"] == 1000
    assert stats["start_frame"] == 450_000 and stats["end_frame"] == 900_000
    assert not window.ready(cfg, 1_000_000, now=2900)  # only 64 fresh episodes remain
    assert not window.ready(cfg, 1_000_000, now=3801)


def test_window_prunes_on_add_and_honors_exact_age_and_frame_boundaries():
    utils = load_module("pbt_utils")
    cfg = load_module("pbt_cfg").PbtCfg(min_completed_episodes=4, max_score_age_seconds=10, max_frame_lag=100)
    window = utils.EpisodeWindow(cfg)
    window.add(episode_info(count=2), 0, now=0)
    window.add(episode_info(count=2), 100, now=10)
    assert next(iter(window.ready(cfg, 100, now=10).values()))["completed_episodes"] == 4
    # Pruning happens even on empty infos, not just when ready() is called.
    window.add({}, 101, now=10.01)
    cfg.max_score_age_seconds, cfg.max_frame_lag = 100, 1000
    assert not window.ready(cfg, 101, now=10.01)  # widening cannot resurrect discarded evidence
    window.add(episode_info(count=2), 101, now=10.01)
    stats = next(iter(window.ready(cfg, 101, now=10.01).values()))
    assert stats["created_at"] == 10 and stats["start_frame"] == 100
    window.consume()
    for frame, now in ((101, 10), (100, 10.01)):
        cfg.max_score_age_seconds, cfg.max_frame_lag = 10, 100
        window.add(episode_info(count=4), 0, now=0)
        assert not window.ready(cfg, frame, now=now)
        window.consume()


class FakeAlgo:
    """Public RL-Games state boundary: emulate donor weights/optimizer/env normalizer restore."""

    def __init__(self, root, cfg, label):
        self.train_dir = str(root)
        self.config = dict(cfg["params"]["config"])
        self.last_lr = self.learning_rate = self.config["learning_rate"]
        self.kl_threshold = self.config["kl_threshold"]
        self.entropy_coef = self.config["entropy_coef"]
        self.grad_norm = self.config["grad_norm"]
        self.optimizer = types.SimpleNamespace(param_groups=[{"lr": self.last_lr, "momentum": label}])
        self.scheduler = types.SimpleNamespace(kl_threshold=self.kl_threshold)
        self.frame, self.epoch_num = 0, 0
        self.experiment_name = label
        self.model = {"weights": label, "running_mean_std": [1.0, 2.0]}
        self.env_state = {"curriculum": 0.18, "adr": 0.5}
        self.writer = types.SimpleNamespace(flush=lambda: None, close=lambda: None)

    def get_full_state_weights(self):
        return {
            "model": self.model,
            "optimizer": self.optimizer.param_groups,
            "frame": self.frame,
            "epoch": self.epoch_num,
            "env_state": self.env_state,
        }

    def restore(self, path):
        import torch

        weights = torch.load(path, weights_only=False)
        self.model = weights["model"]
        self.optimizer.param_groups = weights["optimizer"]
        self.frame, self.epoch_num = weights["frame"], weights["epoch"]
        self.env_state = weights["env_state"]


def test_publications_keep_rolling_evidence_without_refreshing_old_dates(tmp_path, monkeypatch):
    pbt = load_module("pbt")
    runtime = load_module("runtime")
    settings = config()
    settings["pbt"].update(selection_enabled=False, max_score_age_seconds=10, max_frame_lag=1000)
    algo = FakeAlgo(tmp_path, settings, "ROLLING")
    clock = [1.0]
    monkeypatch.setattr(pbt.time, "time", lambda: clock[0])
    observer = pbt.PbtAlgoObserver(settings, types.SimpleNamespace(original_argv=["train.py"]))
    observer.after_init(algo)
    algo.frame = 90
    observer.process_infos(episode_info(count=4, success=1), [])
    for frame, now, count in ((100, 2, 0), (200, 3, 2), (300, 12, 2)):
        clock[0] = now
        algo.frame, algo.epoch_num = frame, frame // 100
        if count:
            observer.process_infos(episode_info(count=count), [])
        observer.after_print_stats(frame, algo.epoch_num, 0)
    records = [runtime.resume_metadata(tmp_path / f"pbt_workspace/000/{index:012d}.pth") for index in (1, 2, 3)]
    stats = [next(iter(record["evaluations"].values())) for record in records]
    assert [item["completed_episodes"] for item in stats] == [4, 6, 4]
    assert [item["created_at"] for item in stats] == [1, 1, 3]
    assert [item["end_frame"] for item in stats] == [90, 200, 300]
    assert stats[0]["success_count"] == 1 and stats[2]["success_count"] == 0
    observer.window.consume()
    assert not observer.window.ready(observer.cfg, 300, now=12)


def test_forced_tournament_restart_roundtrip(tmp_path, monkeypatch):
    # rl_games itself is CPU importable. No private observer methods, no Isaac dependencies.
    pbt = load_module("pbt")
    runtime = load_module("runtime")
    donor_cfg, recipient_cfg = config(0), config(1)
    donor_args = types.SimpleNamespace(original_argv=["/repo/train.py", "pbt.policy_idx=0"])
    recipient_args = types.SimpleNamespace(
        original_argv=[
            "/repo/train.py",
            "--task",
            "fixed-task",
            "--seed=43",
            "--headless",
            "--num_envs=16",
            "pbt.policy_idx=1",
            "pbt.num_policies=2",
            "pbt.directory=/tmp/workspace",
            "pbt.interval_steps=100",
            "env.extra=retained",
            "--checkpoint=obsolete.pth",
        ]
    )
    donor = FakeAlgo(tmp_path, donor_cfg, "DONOR")
    donor.frame, donor.epoch_num = 200, 2
    donor.last_lr = 0.0004  # adaptive scheduler changed LR since construction
    donor.optimizer.param_groups[0]["lr"] = 0.0004
    donor_observer = pbt.PbtAlgoObserver(donor_cfg, donor_args)
    donor_observer.after_init(donor)
    donor_observer.process_infos(episode_info(success=4, position=0.02), [0, 1, 2, 3])
    donor_observer.after_print_stats(200, 2, 1.0)

    recipient = FakeAlgo(tmp_path, recipient_cfg, "RECIPIENT")
    recipient.frame, recipient.epoch_num = 300, 3
    recipient_observer = pbt.PbtAlgoObserver(recipient_cfg, recipient_args)
    recipient_observer.after_init(recipient)
    recipient_observer.process_infos(episode_info(), [0, 1, 2, 3])
    executed = []
    monkeypatch.setattr(__import__("os"), "execv", lambda exe, argv: executed.append((exe, argv)))
    recipient_observer.after_print_stats(300, 3, 2.0)
    assert len(executed) == 1
    executable, argv = executed[0]
    assert executable == sys.executable and argv[0] == executable
    assert "--checkpoint=obsolete.pth" not in argv
    assert "--seed=43" in argv and "pbt.policy_idx=1" in argv and "env.extra=retained" in argv
    checkpoint = next(a.split("=", 1)[1] for a in argv if a.startswith("--checkpoint="))
    metadata_path = next(a.split("=", 1)[1] for a in argv if a.startswith("--pbt-resume="))
    metadata = runtime.resume_metadata(checkpoint, metadata_path)
    assert metadata["recipient_frame"] == 300 and metadata["donor_frame"] == 200
    assert metadata["policy_idx"] == 1 and metadata["donor_policy_idx"] == 0
    assert not metadata["evaluations"] and metadata["objective_valid"] is False
    assert not recipient_observer.window.ready(recipient_observer.cfg, 300)
    assert metadata["params"] != {"agent.params.config.learning_rate": 0.0004, "agent.params.config.kl_threshold": 0.01}
    pbt_overrides, params, _ = runtime.split_overrides(argv[2:])
    restored_cfg = config(1)
    runtime.apply_overrides(restored_cfg, metadata["params"])
    runtime.apply_overrides(restored_cfg, params)
    restored_args = types.SimpleNamespace(original_argv=argv[1:], pbt_resume_metadata=metadata)
    restored = FakeAlgo(tmp_path, restored_cfg, "FRESH")
    observer = pbt.PbtAlgoObserver(restored_cfg, restored_args)
    observer.after_init(restored)
    restored.restore(checkpoint)
    assert restored.model == donor.model and restored.env_state == donor.env_state
    assert restored.optimizer.param_groups[0]["momentum"] == "DONOR"
    assert restored.frame == 300 and restored.epoch_num == 3
    assert restored.last_lr == restored.optimizer.param_groups[0]["lr"]
    assert restored.last_lr == metadata["params"]["agent.params.config.learning_rate"]
    assert restored.scheduler.kl_threshold == metadata["params"]["agent.params.config.kl_threshold"]
    # Resume under evolved params: save on the next recipient interval, even with no resets.
    restored.frame, restored.epoch_num = 400, 4
    observer.after_print_stats(400, 4, 3.0)
    saved = runtime.resume_metadata(str(tmp_path / "pbt_workspace/001/000000000004.pth"))
    assert saved["frame"] == 400 and saved["params"] == metadata["params"]
    assert len(executed) == 1  # a stale score must never cause a second exploit


def test_success_priority_and_condition_gating():
    utils = load_module("pbt_utils")
    cfg = load_module("pbt_cfg").PbtCfg(**config()["pbt"])

    def record(success, pos, adr=0.0):
        window = utils.EpisodeWindow()
        window.add(episode_info(success=success, position=pos, adr=adr), 100, now=10)
        return {"schema_version": 2, "created_at": 10, "frame": 100, "evaluations": window.ready(cfg, 100, now=10)}

    ours = record(1, 2.0)
    assert utils.select_donor({1: record(0, 0.0)}, ours, cfg, now=11) is None
    assert utils.select_donor({1: record(4, 0.0, adr=0.5)}, ours, cfg, now=11) is None
    assert utils.select_donor({1: record(4, 0.0)}, ours, cfg, now=11) == 1
    assert utils.select_donor({1: record(4, 0.0)}, ours, cfg, now=5000) is None
    assert utils.select_donor({1: record(0, 0.0)}, record(0, 2.0), cfg, now=11) == 1


def test_selection_diagnostics_distinguish_missing_ineligible_and_thresholds():
    utils = load_module("pbt_utils")
    cfg = load_module("pbt_cfg").PbtCfg(**config()["pbt"])

    def record(position=0.7, adr=0.0):
        window = utils.EpisodeWindow(cfg)
        window.add(episode_info(position=position, adr=adr), 100, now=10)
        evaluations = window.ready(cfg, 100, now=10)
        for group in evaluations.values():
            group.pop("batches", None)  # Aggregate-only legacy companions still obey their original timestamps.
        return {"schema_version": 2, "created_at": 10, "frame": 100, "evaluations": evaluations}

    own = record()
    cases = [
        ({1: None}, "no_donor_records", None),
        ({1: {**record(), "created_at": -2000}}, "no_eligible_donors", "record_expired"),
        ({1: {**record(), "frame": 3_000_000}}, "no_eligible_donors", "record_frame_mismatch"),
        ({1: record(adr=0.5)}, "no_comparable_evidence", "physical_condition_mismatch"),
        ({1: {**record(), "evaluations": {}}}, "no_comparable_evidence", "missing_evidence"),
        ({1: record()}, "no_sufficient_improvement", None),
    ]
    for records, reason, rejection in cases:
        diagnostics = {}
        assert utils.select_donor(records, own, cfg, now=11, diagnostics=diagnostics) is None
        assert diagnostics["reason"] == reason
        if rejection:
            assert diagnostics["rejections"][rejection] == 1
    diagnostics = {}
    assert utils.select_donor({1: record(position=0.0)}, own, cfg, now=11, diagnostics=diagnostics) == 1
    assert diagnostics["reason"] == "selected" and diagnostics["comparable_pairs"] == 1
    assert diagnostics["donor"] == 1
    for group_change, reason in (
        ({"created_at": -2000}, "own_expired_evidence"),
        ({"start_frame": -3_000_000, "end_frame": -3_000_000}, "own_evidence_frame_mismatch"),
        ({"completed_episodes": 1, "good_contact_fraction_sum": 1}, "own_insufficient_fresh_episodes"),
    ):
        stale = record()
        next(iter(stale["evaluations"].values())).update(group_change)
        assert utils.select_donor({1: record(position=0.0)}, stale, cfg, now=11, diagnostics=diagnostics) is None
        assert diagnostics["reason"] == reason
    own["evaluations"] = {}
    assert utils.select_donor({1: record()}, own, cfg, now=11, diagnostics=diagnostics) is None
    assert diagnostics["reason"] == "own_missing_evidence"


def test_donor_boundary_reprunes_published_batches_not_entire_old_aggregate(tmp_path):
    utils = load_module("pbt_utils")
    runtime = load_module("runtime")
    settings = config()
    settings["pbt"].update(max_score_age_seconds=10, max_frame_lag=100)
    cfg = load_module("pbt_cfg").PbtCfg(**settings["pbt"])
    algo = FakeAlgo(tmp_path, settings, "published")
    records = {}
    for member, success in ((0, 0), (1, 4)):
        window = utils.EpisodeWindow(cfg)
        window.add(episode_info(count=4, success=success), 100, now=0)
        window.add(episode_info(count=4, success=success), 110, now=5)
        algo.frame = 110
        record = utils.save_pbt_checkpoint(
            tmp_path / str(member),
            0.0,
            1,
            algo,
            {},
            policy_idx=member,
            created_at=10,
            evaluations=window.ready(cfg, 110, now=10),
        )
        records[member] = runtime.resume_metadata(record["checkpoint"])
        assert next(iter(records[member]["evaluations"].values()))["created_at"] == 0
    diagnostics = {}
    assert utils.select_donor(records, records[0], cfg, now=11, diagnostics=diagnostics) == 1
    # Age/frame cutoffs prune the old batch but leave the second one exactly on the boundary.
    records[0]["frame"] = 210
    assert utils.select_donor(records, records[0], cfg, now=15, diagnostics=diagnostics) == 1
    assert utils.select_donor(records, records[0], cfg, now=15.01, diagnostics=diagnostics) is None
    assert diagnostics["reason"] == "own_expired_evidence"
    assert next(iter(records[1]["evaluations"].values()))["created_at"] == 0  # never rewrite published evidence


def test_donor_comparison_recomputes_success_after_old_batches_expire():
    utils = load_module("pbt_utils")
    cfg = load_module("pbt_cfg").PbtCfg(**{**config()["pbt"], "max_score_age_seconds": 10})
    records = {}
    for member, old_success, new_success in ((0, 0, 1), (1, 4, 0)):
        window = utils.EpisodeWindow(cfg)
        window.add(episode_info(count=4, success=old_success), 100, now=0)
        window.add(episode_info(count=4, success=new_success), 110, now=5)
        records[member] = {
            "schema_version": 2,
            "created_at": 10,
            "frame": 110,
            "evaluations": window.ready(cfg, 110, now=10),
        }
    assert utils.select_donor(records, records[0], cfg, now=10) == 1
    diagnostics = {}
    assert utils.select_donor(records, records[0], cfg, now=11, diagnostics=diagnostics) is None
    assert diagnostics["reason"] == "no_sufficient_improvement"
    # Remaining samples in distinct physical conditions never combine to meet minimum count.
    window = utils.EpisodeWindow(cfg)
    window.add(episode_info(count=2, adr=0.0), 110, now=5)
    window.add(episode_info(count=2, adr=0.5), 110, now=5)
    assert not window.ready(cfg, 110, now=5)
    assert window.diagnostics["fresh_episodes"] == 4
    assert window.diagnostics["largest_condition_episodes"] == 2


def test_publication_diagnostics_report_invalid_evidence_and_selection_gates(tmp_path, monkeypatch, capsys):
    import json

    pbt = load_module("pbt")
    runtime = load_module("runtime")
    clock = [0.0]
    monkeypatch.setattr(pbt.time, "time", lambda: clock[0])
    cases = (
        ("missing", 0, 1, {}, False, "own_missing_evidence"),
        ("expired", 4, 2000, {}, False, "own_expired_evidence"),
        ("frame_expired", 4, 1, {"max_frame_lag": 10}, False, "own_evidence_frame_mismatch"),
        ("insufficient", 2, 1, {}, False, "own_insufficient_fresh_episodes"),
        ("zero_success", 4, 1, {}, True, "no_donor_records"),
        ("warmup", 4, 1, {"warmup_steps": 200}, True, "warmup"),
        ("adaptation", 4, 1, {"adaptation_steps": 200}, True, "adaptation"),
        ("disabled", 4, 1, {"selection_enabled": False}, True, "selection_disabled"),
    )
    for name, count, now, overrides, valid, reason in cases:
        settings = config()
        settings["pbt"].update(overrides)
        clock[0] = 0.0
        algo = FakeAlgo(tmp_path / name, settings, name)
        scalars = {}
        algo.writer.add_scalar = lambda key, value, frame: scalars.update({key: (value, frame)})
        observer = pbt.PbtAlgoObserver(settings, types.SimpleNamespace(original_argv=["train.py"]))
        observer.after_init(algo)
        if count:
            observer.process_infos(episode_info(count=count), [])
        clock[0] = now
        # No per-step or below-cadence selection logs.
        algo.frame = 99
        observer.after_print_stats(99, 1, 0)
        assert not capsys.readouterr().out
        algo.frame, algo.epoch_num = 100, 1
        observer.after_print_stats(100, 1, 0)
        record = runtime.resume_metadata(tmp_path / name / "pbt_workspace/000/000000000001.pth")
        assert record["objective_valid"] is valid
        assert record["selection_diagnostics"]["reason"] == reason
        assert scalars["pbt/objective_valid"] == (int(valid), 100)
        assert scalars[f"pbt/selection_reason/{reason}"] == (1, 100)
        if not valid:
            assert record["true_objective"] == 0.0
        else:
            assert record["true_objective"] > 0  # zero success is still a valid progress fallback
        logs = capsys.readouterr().out.splitlines()
        report = [
            json.loads(line.removeprefix("[pbt-selection] ")) for line in logs if line.startswith("[pbt-selection]")
        ]
        assert len(report) == 1 and report[0]["reason"] == reason
        observer.after_print_stats(100, 1, 0)
        assert not capsys.readouterr().out


def test_native_play_steps_stamps_actual_completion_frames_and_restore_offsets(tmp_path):
    import torch

    from rl_games.common.a2c_common import A2CBase

    pbt = load_module("pbt")
    settings = config()
    settings["pbt"].update(min_completed_episodes=1, max_frame_lag=31)
    algo = FakeAlgo(tmp_path, settings, "per-step-frames")
    algo.frame, algo.horizon_length, algo.num_actors, algo.num_agents = 100, 32, 1, 1
    algo.batch_size = 32
    algo.use_action_masks = algo.has_central_value = algo.value_bootstrap = False
    algo.update_list, algo.tensor_list = [], []
    algo.obs = {"obs": torch.zeros(1, 1)}
    algo.dones = torch.zeros(1)
    algo.current_rewards = torch.zeros(1, 1)
    algo.current_shaped_rewards = torch.zeros(1, 1)
    algo.current_lengths = torch.zeros(1)
    meter = types.SimpleNamespace(update=lambda _: None)
    algo.game_rewards = algo.game_shaped_rewards = algo.game_lengths = meter
    algo.get_action_values = lambda _: {"actions": torch.zeros(1, 1)}
    algo.get_values = lambda _: torch.zeros(1, 1)
    algo.rewards_shaper = lambda values: values
    buffer = {"dones": torch.zeros(32, 1), "values": torch.zeros(32, 1, 1), "rewards": torch.zeros(32, 1, 1)}
    algo.experience_buffer = types.SimpleNamespace(
        tensor_dict=buffer, update_data=lambda *_: None, get_transformed_list=lambda *_: {}
    )
    algo.discount_values = lambda *_: torch.zeros(32, 1, 1)
    step = [0]

    def env_step(_):
        step[0] += 1
        info = episode_info(count=1) if step[0] % 32 == 0 else {}
        return algo.obs, torch.zeros(1, 1), torch.zeros(1), info

    algo.env_step = env_step
    observer = pbt.PbtAlgoObserver(settings, types.SimpleNamespace(original_argv=["train.py"]))
    observer.after_init(algo)
    algo.algo_observer = observer
    A2CBase.play_steps(algo)
    assert algo.frame == 100  # native loop advances this only after PPO
    observer.after_steps()
    algo.frame = 132
    ready = observer.window.ready(observer.cfg, algo.frame)
    assert ready
    assert next(iter(ready.values()))["start_frame"] == 132
    # Recipient restore must discard both policy evidence and an unfinished rollout offset.
    checkpoint = tmp_path / "recipient.pth"
    torch.save(algo.get_full_state_weights(), checkpoint)
    observer.process_infos({}, [])
    observer.process_infos({}, [])
    algo.restore(checkpoint)
    observer.process_infos(episode_info(count=1), [])
    stats = next(iter(observer.window.ready(observer.cfg, 133).values()))
    assert stats["start_frame"] == 133 and stats["end_frame"] == 133
    # Batched actors count environment transitions, not one frame per callback.
    algo.frame, algo.num_actors = 200, 4
    observer.window.consume()
    observer.process_infos({}, [])
    observer.process_infos(episode_info(count=1), [])
    stats = next(iter(observer.window.ready(observer.cfg, 208).values()))
    assert stats["start_frame"] == 208


def test_in_process_restore_discards_pre_restore_completed_evidence(tmp_path):
    import torch

    pbt = load_module("pbt")
    settings = config()
    algo = FakeAlgo(tmp_path, settings, "reset-evidence")
    checkpoint = tmp_path / "restore.pth"
    torch.save(algo.get_full_state_weights(), checkpoint)
    observer = pbt.PbtAlgoObserver(settings, types.SimpleNamespace(original_argv=["train.py"]))
    observer.after_init(algo)
    observer.process_infos(episode_info(count=4), [])
    assert observer.window.ready(observer.cfg, 0)
    algo.restore(checkpoint)
    assert not observer.window.ready(observer.cfg, 0)
    assert observer.window.diagnostics["reason"] == "missing_evidence"


def test_selection_rejects_future_and_malformed_record_or_evidence_metadata():
    import copy

    utils = load_module("pbt_utils")
    cfg = load_module("pbt_cfg").PbtCfg(**config()["pbt"])
    window = utils.EpisodeWindow(cfg)
    window.add(episode_info(count=4), 100, now=10)
    own = {"schema_version": 2, "created_at": 10, "frame": 100, "evaluations": window.ready(cfg, 100, now=10)}
    for change, reason in (
        ({"created_at": 12}, "record_future_timestamp"),
        ({"created_at": float("nan")}, "record_invalid_metadata"),
        ({"created_at": "not-a-time"}, "record_invalid_metadata"),
        ({"frame": None}, "record_invalid_metadata"),
    ):
        donor = {**own, **change}
        diagnostics = {}
        assert not utils.eligible(donor, cfg, 100, now=11)
        assert utils.select_donor({1: donor}, own, cfg, now=11, diagnostics=diagnostics) is None
        assert diagnostics["reason"] == "no_eligible_donors" and diagnostics["rejections"][reason] == 1
    for stamp, reason in (
        (12, "evidence_future_timestamp"),
        (float("nan"), "invalid_evidence"),
        (None, "invalid_evidence"),
    ):
        donor = copy.deepcopy(own)
        stats = next(iter(donor["evaluations"].values()))
        stats["batches"][0]["created_at"] = stamp
        diagnostics = {}
        assert utils.select_donor({1: donor}, own, cfg, now=11, diagnostics=diagnostics) is None
        assert diagnostics["rejections"][reason] == 1


def test_selection_rejects_invalid_statistics_and_container_shapes():
    import copy

    utils = load_module("pbt_utils")
    cfg = load_module("pbt_cfg").PbtCfg(**config()["pbt"])
    window = utils.EpisodeWindow(cfg)
    window.add(episode_info(count=4), 100, now=10)
    own = {"schema_version": 2, "created_at": 10, "frame": 100, "evaluations": window.ready(cfg, 100, now=10)}
    key = next(iter(own["evaluations"]))
    for legacy in (False, True):
        for field, value in (
            ("success_count", 8),
            ("success_count", -1),
            ("position_error_sum", -4),
            ("rotation_error_sum", float("inf")),
            ("good_contact_fraction_sum", 5),
            ("completed_episodes", 1.5),
            ("position_error_sum", None),
        ):
            donor = copy.deepcopy(own)
            group = donor["evaluations"][key]
            if legacy:
                group.pop("batches")
                group[field] = value
            else:
                group["batches"][0][field] = value
            diagnostics = {}
            assert utils.select_donor({1: donor}, own, cfg, now=11, diagnostics=diagnostics) is None
            assert diagnostics["rejections"]["invalid_evidence"] == 1
        for field, value in (("success_rate", 2), ("success_rate", float("nan")), ("progress", -1)):
            donor = copy.deepcopy(own)
            donor["evaluations"][key].pop("batches")
            donor["evaluations"][key][field] = value
            diagnostics = {}
            assert utils.select_donor({1: donor}, own, cfg, now=11, diagnostics=diagnostics) is None
            assert diagnostics["rejections"]["invalid_evidence"] == 1
    for evaluations in ([1], None, {key: [1]}, {key: {"batches": [1]}}, {key: {"batches": {"bad": 1}}}):
        donor = {**own, "evaluations": evaluations}
        diagnostics = {}
        assert utils.select_donor({1: donor}, own, cfg, now=11, diagnostics=diagnostics) is None
        assert diagnostics["rejections"]["invalid_evidence"] == 1
        assert utils.select_donor({1: own}, donor, cfg, now=11, diagnostics=diagnostics) is None
        assert diagnostics["reason"] == "own_invalid_evidence"


def test_bounded_mutation_and_explicit_initial_diversity():
    pbt = load_module("pbt")
    baseline, other = config(0), config(1)
    original = dict(baseline["params"]["config"])
    pbt.initialize_population(baseline)
    assert baseline["params"]["config"] == original
    values = pbt.initialize_population(other)
    assert other["params"]["config"] != original
    mutation = load_module("mutation")
    for _ in range(100):
        before = values
        values = mutation.mutate(values, other["pbt"]["mutation"], 0.0, (1.1, 2.0), other["pbt"]["mutation_bounds"])
        assert values != before
        assert 1e-5 <= values["agent.params.config.learning_rate"] <= 0.01
        assert 0.001 <= values["agent.params.config.kl_threshold"] <= 0.1
    resumed = config(1)
    pbt.initialize_population(resumed, resumed=True)
    assert resumed["params"]["config"] == original


def test_compiled_checkpoint_restore_preserves_full_state(tmp_path):
    import torch

    runtime = load_module("runtime")
    cfg = config()
    donor = FakeAlgo(tmp_path, cfg, "COMPILED")
    donor.model = {"_orig_mod.weight": torch.tensor([7.0]), "_orig_mod.running_mean_std": torch.tensor([2.0])}
    checkpoint = tmp_path / "compiled.pth"
    torch.save(donor.get_full_state_weights(), checkpoint)
    target = FakeAlgo(tmp_path, cfg, "EAGER")
    runtime.install_restore(target, {"agent.params.config.learning_rate": 0.0002})
    target.restore(checkpoint)
    assert set(target.model) == {"weight", "running_mean_std"}
    assert target.model["weight"].item() == 7.0
    assert target.env_state == donor.env_state
    assert target.optimizer.param_groups[0]["momentum"] == "COMPILED"
    assert target.optimizer.param_groups[0]["lr"] == 0.0002


def test_discount_mutation_uses_conservative_registered_rule():
    import random

    mutation = load_module("mutation")
    result = mutation.mutate(
        {"discount": 0.99},
        {"discount": "mutate_discount"},
        1.0,
        (2.0, 3.0),
        {"discount": [0.9, 0.999]},
        rng=random.Random(42),
    )
    # The discount rule deliberately uses 1.1..1.2 on (1-discount), not generic 2..3.
    assert 0.988 <= result["discount"] <= 0.992
    assert result["discount"] != 0.99


def test_disabled_pbt_restores_but_never_publishes(tmp_path):
    import torch

    pbt = load_module("pbt")
    cfg = config()
    donor = FakeAlgo(tmp_path, cfg, "DONOR")
    donor.model = {"_orig_mod.weight": torch.tensor([3.0])}
    checkpoint = tmp_path / "source.pth"
    torch.save(donor.get_full_state_weights(), checkpoint)
    cfg["pbt"]["enabled"] = False
    cfg["params"]["config"]["learning_rate"] = 0.000123
    algo = FakeAlgo(tmp_path, cfg, "NEW")
    args = types.SimpleNamespace(
        original_argv=["train.py"], pbt_resume_metadata={"frame": 0, "recipient_frame": 500, "recipient_epoch": 5}
    )
    observer = pbt.PbtAlgoObserver(cfg, args)
    observer.after_init(algo)
    algo.restore(checkpoint)
    assert algo.frame == 500 and algo.epoch_num == 5
    assert algo.optimizer.param_groups[0]["lr"] == 0.000123
    assert "weight" in algo.model
    observer.process_infos(episode_info(success=4), [0])
    observer.after_print_stats(500, 5, 1.0)
    assert not (tmp_path / "pbt_workspace").exists()


def test_applied_adr_identity_overrides_only_informational_fraction():
    utils = load_module("pbt_utils")
    cfg = load_module("pbt_cfg").PbtCfg(**config()["pbt"])
    window = utils.EpisodeWindow()
    for adr in (0.0, 0.01):
        info = episode_info(count=2, adr=adr)
        info["dexpbt"]["groups"][0]["condition"]["adr_parameters"] = '{"gravity": 0.0}'
        window.add(info, 10, now=10)
    assert len(window.ready(cfg, 10, now=10)) == 1
    changed = episode_info(count=4, adr=0.01)
    changed["dexpbt"]["groups"][0]["condition"]["adr_parameters"] = '{"gravity": -1.0}'
    window.add(changed, 10, now=10)
    assert len(window.ready(cfg, 10, now=10)) == 2


def test_checkpoint_retention_without_eligible_episodes(tmp_path):
    pbt = load_module("pbt")
    cfg = config()
    algo = FakeAlgo(tmp_path, cfg, "RETENTION")
    observer = pbt.PbtAlgoObserver(cfg, types.SimpleNamespace(original_argv=["train.py"]))
    observer.after_init(algo)
    for index in range(1, 56):
        algo.frame, algo.epoch_num = index * 100, index
        observer.after_print_stats(algo.frame, index, 1.0)
    directory = tmp_path / "pbt_workspace/000"
    assert len(list(directory.glob("*.pth"))) == 50
    assert len(list(directory.glob("*.yaml"))) == 50
    assert not (directory / "000000000001.pth").exists()


def test_terminal_callback_publishes_without_exploit_even_off_cadence(tmp_path, monkeypatch):
    import pytest

    pbt = load_module("pbt")
    runtime = load_module("runtime")
    monkeypatch.setattr(pbt.os, "execv", lambda *_: pytest.fail("terminal exploit"))
    for member, frame, success in ((0, 100, 4), (1, 150, 0)):
        settings = config(member)
        algo = FakeAlgo(tmp_path, settings, "terminal")
        algo.max_epochs, algo.max_frames = -1, 150
        observer = pbt.PbtAlgoObserver(settings, types.SimpleNamespace(original_argv=["train.py"]))
        observer.after_init(algo)
        algo.frame, algo.epoch_num = frame, 2
        observer.process_infos(episode_info(success=success), [])
        # Native callback argument lags the actual post-update frame.
        observer.after_print_stats(frame - 50, 2, 0)
    final = runtime.resume_metadata(str(tmp_path / "pbt_workspace/001/final.pth"))
    assert final["frame"] == 150 and final["terminal"]
    assert final["evaluations"]


def test_installed_rl_games_loop_terminal_and_completed_resume(tmp_path):
    import torch

    from rl_games.algos_torch.a2c_continuous import A2CAgent

    pbt = load_module("pbt")
    runtime = load_module("runtime")
    for limit in ("max_epochs", "max_frames"):
        settings = config()
        algo = FakeAlgo(tmp_path / limit, settings, "native-loop")
        algo.max_epochs = 1 if limit == "max_epochs" else -1
        algo.max_frames = 100 if limit == "max_frames" else -1
        algo.batch_size_envs = 100
        algo.num_agents = algo.value_size = 1
        algo.multi_gpu = algo.print_stats = False
        algo.global_rank = 0
        algo.nn_dir = str(tmp_path)
        algo.config["name"] = "terminal"
        algo.dataset = types.SimpleNamespace(update_values_dict=lambda _: None)
        algo.diagnostics = types.SimpleNamespace(epoch=lambda *a, **kw: None)
        algo.game_rewards = types.SimpleNamespace(current_size=0)
        updates, resets, saves = [], [], []
        algo.init_tensors = lambda: resets.append("init")
        algo.env_reset = lambda: resets.append("reset")

        def update_epoch():
            algo.epoch_num += 1
            return algo.epoch_num

        def train_epoch():
            updates.append(algo.epoch_num)
            return (0.1, 0.1, 0.1, 0.3, [], [], [], [], [], 0.001, 1.0)

        algo.update_epoch = update_epoch
        algo.train_epoch = train_epoch
        algo.save = lambda path: saves.append(path)
        algo.train = types.MethodType(A2CAgent.train, algo)
        observer = pbt.PbtAlgoObserver(settings, types.SimpleNamespace(original_argv=["train.py"]))
        observer.after_init(algo)
        algo.write_stats = lambda *a: observer.after_print_stats(a[10], a[1], a[0])
        algo.train()
        assert updates == [1] and resets == ["init", "reset"] and len(saves) == 1
        final_path = Path(observer.curr_policy_dir) / "final.pth"
        assert torch.load(final_path, weights_only=False)["frame"] == 100
        final_bytes = final_path.read_bytes()
        # Restoring a completed publication and entering the real public train seam
        # must not reach native init_tensors, env_reset, update_epoch or train_epoch.
        algo.restore(final_path)
        algo.train()
        assert updates == [1] and resets == ["init", "reset"] and len(saves) == 1
        assert final_path.read_bytes() == final_bytes
        assert runtime.resume_metadata(final_path)["terminal"]


def test_population_forwarding_and_companion_resume_in_launcher(tmp_path):
    import os
    import subprocess
    import yaml

    root = tmp_path / "with spaces"
    for member in range(2):
        directory = root / f"logs/rl_games/dexpbt_reorient/pbt_workspace/{member:03d}"
        directory.mkdir(parents=True)
        checkpoint = directory / "000001.pth"
        checkpoint.write_bytes(b"fake")
        checkpoint.with_suffix(".yaml").write_text(yaml.safe_dump({"frame": 100, "checkpoint": str(checkpoint)}))
    env = {**os.environ, "POP": "2", "DRY_RUN": "1", "BASE": str(root), "RESUME": "1", "PYTHON": sys.executable}
    run = subprocess.run(
        ["bash", str(ROOT / "scripts/launch/dexpbt_reorient.sh")], env=env, check=True, capture_output=True, text=True
    )
    import shlex

    commands = [shlex.split(line) for line in run.stdout.splitlines()]
    assert len(commands) == 2
    for index, command in enumerate(commands):
        assert f"pbt.policy_idx={index}" in command and "pbt.num_policies=2" in command
        assert f"pbt.directory={root}" in command
        assert "--pbt-resume" in command and "--checkpoint" in command
    fresh = subprocess.run(
        ["bash", str(ROOT / "scripts/launch/dexpbt_reorient.sh")],
        env={**env, "RESUME": "0"},
        capture_output=True,
        text=True,
    )
    assert fresh.returncode == 2 and "populated workspace" in fresh.stderr
