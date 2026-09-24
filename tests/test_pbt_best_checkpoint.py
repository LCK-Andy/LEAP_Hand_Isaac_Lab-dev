"""Simulator-free regression of the RL-Games best-save lifecycle and transactions."""

import importlib.util
import json
import math
import torch
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).resolve().parents[1]
    / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt/pbt/best_checkpoint.py"
)
spec = importlib.util.spec_from_file_location("pbt_best_checkpoint", MODULE)
best = importlib.util.module_from_spec(spec)
spec.loader.exec_module(best)


class Meter:
    def __init__(self):
        self.clear()

    def clear(self):
        self.current_size = 0
        self.score = 0.0

    def get_mean(self):
        return [self.score]

    def update(self, score):
        self.score = score
        self.current_size = 1


class Algo:
    """Shape/order of a2c_continuous save/restore and A2CBase continuous train."""

    def __init__(self, nn_dir, name="allegro_hand"):
        self.nn_dir = str(nn_dir)
        Path(nn_dir).mkdir(parents=True, exist_ok=True)
        self.config = {"name": name}
        self.game_rewards = Meter()
        self.game_shaped_rewards = Meter()
        self.game_lengths = Meter()
        self.last_mean_rewards = -1e9
        self.mean_rewards = -1e9
        self.save_best_after = 0
        self.frame = self.epoch_num = 0
        self.policy = "initial"
        self.saved = []
        self.fail_save = False
        self.fail_restore = False

    def save(self, filename):
        path = Path(str(filename) + ".pth")  # rl_games.torch_ext.save_checkpoint contract
        self.saved.append(path)
        if self.fail_save:
            path.write_bytes(b"partial checkpoint")
            raise OSError("save failed")
        torch.save(
            {
                "model": {"policy": self.policy},
                "last_mean_rewards": self.last_mean_rewards,
                "frame": self.frame,
                "epoch": self.epoch_num,
            },
            path,
        )

    def restore(self, filename):
        if self.fail_restore:
            raise OSError("restore failed")
        state = load(filename)
        self.policy = state["model"]["policy"]
        self.last_mean_rewards = state["last_mean_rewards"]
        self.frame, self.epoch_num = state["frame"], state["epoch"]

    def train(self, observer, scores, after_best=None):
        # The real upstream reset is AFTER Runner.restore. Fixing only restore
        # loses the high-water mark, even on a simple manual continuation.
        self.last_mean_rewards = -1e9
        for score in scores:
            self.frame += 100
            self.epoch_num += 1
            if score is not None:
                self.game_rewards.update(score)
            # write_stats calls observers before self.mean_rewards is updated.
            observer.after_print_stats(self.frame - 100, self.epoch_num, 0)
            if after_best:
                after_best()
            if self.game_rewards.current_size:
                self.mean_rewards = self.game_rewards.get_mean()[0]
                self.save(Path(self.nn_dir) / f"last_{self.config['name']}_ep_{self.epoch_num}")
                if self.mean_rewards > self.last_mean_rewards and self.epoch_num >= self.save_best_after:
                    self.last_mean_rewards = self.mean_rewards
                    self.save(Path(self.nn_dir) / self.config["name"])
        return self.last_mean_rewards


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def attach(algo, metadata=None, history_dir=None):
    observer = best.BestCheckpointObserver(metadata, history_dir)
    observer.after_init(algo)
    return observer


def test_weaker_exploit_retains_historical_then_promotes(tmp_path):
    algo = Algo(tmp_path / "first" / "nn")
    history = tmp_path / "workspace" / "000" / "best"
    original = attach(algo, history_dir=history)
    algo.train(original, [100])
    historical_bytes = (history / "historical_best.pth").read_bytes()
    donor = Algo(tmp_path / "donor")
    donor.policy = "donor"
    donor.last_mean_rewards = 9000  # inherited donor gate is NOT this member's historical best
    donor.save(tmp_path / "donor_checkpoint")
    restarted = Algo(tmp_path / "new_timestamp" / "nn")
    observer = attach(restarted, {"recipient_frame": 100, "last_exploit_frame": 100}, history)
    restarted.restore(tmp_path / "donor_checkpoint.pth")
    assert observer.branch_id == 1
    restarted.train(observer, [10, 20])
    assert observer.branch_best_score == 20
    assert observer.record["historical_best"]["score"] == 100
    assert (history / "historical_best.pth").read_bytes() == historical_bytes
    assert load(Path(restarted.nn_dir) / "allegro_hand.pth")["model"]["policy"] == "initial"
    assert load(history / "branches/branch_000001_best.pth")["model"]["policy"] == "donor"
    assert load(history / "branches/branch_000000_best.pth")["last_mean_rewards"] == 100
    restarted.policy = "improved donor"
    restarted.train(observer, [110])
    assert observer.record["historical_best"]["score"] == 110
    assert load(history / "historical_best.pth")["model"]["policy"] == "improved donor"


def test_manual_resume_same_branch_and_upstream_train_reset(tmp_path):
    algo = Algo(tmp_path)
    observer = attach(algo, {"last_exploit_frame": 100, "recipient_frame": 100})
    algo.train(observer, [40])
    checkpoint = tmp_path / "last_allegro_hand_ep_1.pth"
    resumed = Algo(tmp_path)
    continuation = attach(resumed, {"last_exploit_frame": 100, "frame": 200})
    resumed.restore(checkpoint)
    resumed.train(continuation, [5, 30])
    assert continuation.branch_id == observer.branch_id
    assert continuation.branch_best_score == 40
    assert continuation.record["historical_best"]["score"] == 40
    assert resumed.last_mean_rewards == 40
    assert len(continuation.record["branches"]) == 1
    resumed.train(continuation, [50])
    assert continuation.branch_best_score == 50
    # Re-exec from the SAME staged exploit is a continuation, not another id.
    again = attach(Algo(tmp_path), {"last_exploit_frame": 100, "recipient_frame": 100})
    assert again.branch_id == 0
    newer = attach(Algo(tmp_path), {"last_exploit_frame": 400, "recipient_frame": 400})
    assert newer.branch_id == 1
    assert attach(Algo(tmp_path)).branch_id == 1
    with pytest.raises(ValueError, match="Stale"):
        attach(Algo(tmp_path), {"last_exploit_frame": 100})


def test_reward_raw_not_stale_shaped_or_pbt_and_precedes_exec(tmp_path):
    algo = Algo(tmp_path)
    observer = attach(algo)
    algo.mean_rewards = 10000
    algo.game_shaped_rewards.update(20000)
    observer.process_infos({"true_objective": 30000}, [])

    def pbt_callback():
        # A subsequent PBT observer may exec here, before upstream saves.
        assert observer.record["historical_best"]["score"] == 7
        assert load(tmp_path / "historical_best.pth")["last_mean_rewards"] == 7

    algo.train(observer, [7], pbt_callback)
    assert len(list((tmp_path / ".best_checkpoints").glob("*.pth"))) == 1
    assert (tmp_path / "last_allegro_hand_ep_1.pth").is_file()


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_nonfinite_and_empty_never_promote(tmp_path, score):
    algo = Algo(tmp_path)
    observer = attach(algo)
    algo.train(observer, [None, score])
    assert observer.record["historical_best"] is None
    assert not (tmp_path / "historical_best.pth").exists()
    algo.train(observer, [-2e9])  # no implicit -1e9 historical floor
    assert observer.branch_best_score == -2e9


def test_restore_clears_cold_donor_metrics_and_composes(tmp_path):
    algo = Algo(tmp_path)
    order = []
    original_restore = algo.restore

    def exploration_restore(filename):
        order.append("exploration")
        original_restore(filename)

    algo.restore = exploration_restore
    observer = attach(algo)
    algo.train(observer, [100])
    wrapped = algo.restore

    def pbt_restore(filename):
        order.append("pbt")
        wrapped(filename)
        algo.frame = 456  # recipient accounting runs after inner wrappers

    algo.restore = pbt_restore
    algo.game_rewards.update(10000)
    algo.restore(tmp_path / "historical_best.pth")
    assert order == ["pbt", "exploration"]
    assert algo.frame == 456
    assert algo.game_rewards.current_size == 0
    assert algo.game_shaped_rewards.current_size == 0
    algo.train(observer, [None, 1])
    assert observer.record["historical_best"]["score"] == 100
    algo.fail_restore = True
    algo.game_rewards.update(12)
    with pytest.raises(OSError):
        algo.restore(tmp_path / "historical_best.pth")
    assert algo.game_rewards.current_size == 1  # no successful-restore side effects


def test_save_best_after_and_independent_instances(tmp_path):
    algo = Algo(tmp_path / "a")
    untouched = Algo(tmp_path / "b")
    observer = attach(algo)
    algo.save_best_after = 2
    algo.train(observer, [50])
    assert observer.record["historical_best"] is None
    algo.train(observer, [10])
    assert observer.record["historical_best"]["score"] == 10
    assert untouched.save.__func__ is Algo.save


@pytest.mark.parametrize("name", ["allegro_hand", "kuka_allegro_reorient", "custom.experiment"])
def test_config_names_conventional_alias_and_periodic_paths(tmp_path, name):
    algo = Algo(tmp_path, name)
    observer = attach(algo)
    algo.train(observer, [100])
    original = (tmp_path / (name + ".pth")).read_bytes()
    algo.game_rewards.update(1)
    algo.save(tmp_path / name)
    algo.save(tmp_path / (name + ".pth"))
    assert (tmp_path / (name + ".pth")).read_bytes() == original
    assert not (tmp_path / (name + ".pth.pth")).exists()
    assert (tmp_path / f"last_{name}_ep_1.pth").exists()


@pytest.mark.parametrize("name", ["../escape", "a/b", "", ".", "..", "historical_best"])
def test_unsafe_config_name_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="config.name"):
        attach(Algo(tmp_path, name))


def test_legacy_explicit_best_only_not_resume_donor(tmp_path):
    algo = Algo(tmp_path)
    algo.policy = "legacy winner"
    algo.last_mean_rewards = 80
    algo.save(tmp_path / "allegro_hand")
    observer = attach(algo, {"true_objective": 100000, "last_exploit_frame": 1})
    assert observer.record["historical_best"]["score"] == 80
    assert observer.record["historical_best"]["source"] == "legacy_conventional_best"
    assert observer.branch_best_score == -math.inf
    algo.train(observer, [2])
    assert load(tmp_path / "allegro_hand.pth")["model"]["policy"] == "legacy winner"


@pytest.mark.parametrize("score", [None, math.nan, math.inf, -1e9])
def test_legacy_unknown_score_not_invented(tmp_path, score):
    torch.save({"model": {}, "last_mean_rewards": score}, tmp_path / "allegro_hand.pth")
    observer = attach(Algo(tmp_path), {"true_objective": 100})
    assert observer.record["historical_best"] is None
    assert (tmp_path / "allegro_hand.pth").is_file()


def test_checkpoint_write_failure_keeps_record_and_old_bytes_then_retry(tmp_path):
    algo = Algo(tmp_path)
    observer = attach(algo)
    algo.train(observer, [100])
    record = (tmp_path / "best_history.json").read_bytes()
    checkpoint = (tmp_path / "historical_best.pth").read_bytes()
    algo.fail_save = True
    with pytest.raises(OSError, match="save failed"):
        algo.train(observer, [110])
    assert (tmp_path / "best_history.json").read_bytes() == record
    assert (tmp_path / "historical_best.pth").read_bytes() == checkpoint
    assert observer.branch_best_score == 100
    algo.fail_save = False
    algo.train(observer, [110])
    assert observer.branch_best_score == 110


def test_json_commit_failure_is_retryable_without_corrupting_winner(tmp_path, monkeypatch):
    algo = Algo(tmp_path)
    observer = attach(algo)
    algo.train(observer, [100])
    original_replace = best.os.replace
    committed = (tmp_path / "best_history.json").read_bytes()

    def fail_record(source, target):
        if Path(target).name == "best_history.json":
            raise OSError("commit failed")
        return original_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(best.os, "replace", fail_record)
        with pytest.raises(OSError, match="commit failed"):
            algo.train(observer, [110])
    assert (tmp_path / "best_history.json").read_bytes() == committed
    assert observer.branch_best_score == 100
    recovered = attach(Algo(tmp_path))
    assert recovered.branch_best_score == 100
    recovered.algo.train(recovered, [110])
    assert recovered.branch_best_score == 110


def test_committed_promotion_alias_failure_repairs_on_reopen(tmp_path, monkeypatch):
    algo = Algo(tmp_path)
    observer = attach(algo)
    algo.train(observer, [100])
    original_alias = observer._alias

    def fail_new_alias(path, entry, **kwargs):
        if entry and entry["score"] == 110:
            raise OSError("alias interrupted")
        return original_alias(path, entry, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(observer, "_alias", fail_new_alias)
        with pytest.raises(OSError, match="alias interrupted"):
            algo.train(observer, [110])
    record = json.loads((tmp_path / "best_history.json").read_text())
    assert record["historical_best"]["score"] == 110
    assert load(tmp_path / "historical_best.pth")["last_mean_rewards"] == 100
    recovered = attach(Algo(tmp_path))
    assert recovered.branch_best_score == 110
    assert load(tmp_path / "historical_best.pth")["last_mean_rewards"] == 110


def test_reject_mismatched_config_and_broken_commit(tmp_path):
    algo = Algo(tmp_path)
    observer = attach(algo)
    algo.train(observer, [100])
    with pytest.raises(ValueError, match="different config"):
        attach(Algo(tmp_path, "other_name"))
    (tmp_path / observer.record["historical_best"]["checkpoint"]).unlink()
    with pytest.raises(ValueError, match="missing committed"):
        attach(Algo(tmp_path))


def test_invalid_exploit_identity(tmp_path):
    with pytest.raises(ValueError, match="recipient_frame"):
        attach(Algo(tmp_path), {"recipient_frame": 200, "last_exploit_frame": 100})


def test_integration_orders_best_before_pbt():
    script = (MODULE.parents[6] / "scripts/rl_games/train_dexpbt_member.py").read_text()
    assert "--legacy-best-checkpoint" in script
    assert script.index(
        "BestCheckpointObserver(metadata, history_dir, args_cli.legacy_best_checkpoint)"
    ) < script.index("observers.append(PbtAlgoObserver")


def test_generation_collection_keeps_winners_not_every_improvement(tmp_path):
    history = tmp_path / "best"
    first = Algo(tmp_path / "timestamp_1" / "nn")
    observer = attach(first, history_dir=history)
    first.train(observer, range(1, 21))
    assert len(list(observer.generations.glob("*.pth"))) == 1
    second = Algo(tmp_path / "timestamp_2" / "nn")
    successor = attach(second, {"last_exploit_frame": 2000, "recipient_frame": 2000}, history)
    second.train(successor, range(1, 11))
    assert len(list(observer.generations.glob("*.pth"))) == 2
    assert load(history / "branches/branch_000000_best.pth")["last_mean_rewards"] == 20
    second.train(successor, [30, 40])
    assert len(list(observer.generations.glob("*.pth"))) == 2
    assert load(history / "branches/branch_000001_best.pth")["last_mean_rewards"] == 40
    # Even old timestamped aliases survive collection of their original generation.
    assert load(Path(first.nn_dir) / "allegro_hand.pth")["last_mean_rewards"] == 40
    assert load(Path(first.nn_dir) / "historical_best.pth")["last_mean_rewards"] == 40


def test_collection_only_deletes_owned_unreferenced_regular_generations(tmp_path):
    algo = Algo(tmp_path)
    observer = attach(algo)
    algo.train(observer, [10])
    owned_orphan = observer.generations / ("branch_000009-" + "a" * 32 + ".pth")
    owned_orphan.write_bytes(b"interrupted save")
    unowned = observer.generations / "my_checkpoint.pth"
    unowned.write_bytes(b"user file")
    outside = tmp_path / "outside.pth"
    outside.write_bytes(b"outside file")
    symlink = observer.generations / ("legacy-" + "b" * 32 + ".pth")
    symlink.symlink_to(outside)
    directory = observer.generations / ("branch_000001-" + "c" * 32 + ".pth")
    directory.mkdir()
    attach(Algo(tmp_path))
    assert not owned_orphan.exists()
    assert unowned.exists() and symlink.is_symlink() and outside.exists() and directory.is_dir()
    assert load(tmp_path / "historical_best.pth")["last_mean_rewards"] == 10


def test_collection_waits_for_alias_repair_and_retries_on_resume(tmp_path, monkeypatch):
    algo = Algo(tmp_path)
    observer = attach(algo)
    algo.train(observer, [10])
    original_alias = observer._alias

    def fail_alias(path, entry, **kwargs):
        if entry and entry["score"] == 20:
            raise OSError("alias failed")
        return original_alias(path, entry, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(observer, "_alias", fail_alias)
        with pytest.raises(OSError):
            algo.train(observer, [20])
    assert len(list(observer.generations.glob("*.pth"))) == 2
    assert load(tmp_path / "historical_best.pth")["last_mean_rewards"] == 10
    attach(Algo(tmp_path))
    assert len(list(observer.generations.glob("*.pth"))) == 1
    assert load(tmp_path / "historical_best.pth")["last_mean_rewards"] == 20


def test_explicit_legacy_best_from_old_timestamp_and_existing_history_wins(tmp_path):
    old = Algo(tmp_path / "old" / "nn")
    old.last_mean_rewards = 80
    old.save(Path(old.nn_dir) / "allegro_hand")
    legacy = Path(old.nn_dir) / "allegro_hand.pth"
    current = Algo(tmp_path / "new" / "nn")
    observer = best.BestCheckpointObserver(history_dir=tmp_path / "best", legacy_best_path=legacy)
    observer.after_init(current)
    current.train(observer, [5])
    assert observer.record["historical_best"]["score"] == 80
    assert observer.record["historical_best"]["legacy_path"] == str(legacy)
    assert "unknown" in observer.record["historical_best"]["coverage"]
    # Once migrated, obsolete/missing CLI migration input is not re-imported.
    legacy.unlink()
    resumed = best.BestCheckpointObserver(history_dir=tmp_path / "best", legacy_best_path=legacy)
    resumed.after_init(Algo(tmp_path / "third" / "nn"))
    assert resumed.record["historical_best"]["score"] == 80


def test_explicit_missing_legacy_path_fails_clearly(tmp_path):
    observer = best.BestCheckpointObserver(legacy_best_path=tmp_path / "missing.pth")
    with pytest.raises(FileNotFoundError):
        observer.after_init(Algo(tmp_path))
