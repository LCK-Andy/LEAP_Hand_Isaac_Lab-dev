"""Member-local raw episodic reward bests, independent of PBT selection scores.

``<history_dir>/best_history.json`` is the atomic commit marker. It references immutable
checkpoint generations in ``.best_checkpoints/``; uncommitted generations are
harmless. ``historical_best.pth`` and the conventional ``<config.name>.pth`` are
repairable aliases to the historical winner, NEVER to a weaker donor branch.
``branches/branch_XXXXXX_best.pth`` names each branch's own winner. Resume repairs
aliases from the committed JSON, including interrupted multi-file promotions.
After commit AND alias repair, only unreferenced helper-owned generations are
removed: retain each branch winner plus the historical winner, not every improving
epoch. Log-directory aliases point through the stable historical alias, so old
log directories remain usable after collection. One training process owns a member.

RL-Games 1.6.x resets last_mean_rewards inside train(), AFTER restore. Its
continuous trainer calls after_print_stats from write_stats, BEFORE periodic,
best, and final saves. We publish there before a later PBT observer can exec,
and intercept only the exact conventional best save. Other save paths retain
the upstream lifecycle. Comparisons use game_rewards.get_mean()[0] (unshaped
completed-episode rolling mean), not mean_rewards (stale here), shaped reward,
PBT objective, or a checkpoint's inherited donor high-water mark.
"""

import copy
import json
import math
import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path
from uuid import uuid4

from rl_games.common.algo_observer import AlgoObserver


class BestCheckpointObserver(AlgoObserver):
    """Install after exploration and before PBT (which may re-exec in its callback).

    Args:
        metadata: Recipient's companion PBT metadata, not donor checkpoint state.
            last_exploit_frame identifies a branch; recipient_frame validates a
            staged exploit. Reopening the same token continues the same branch.
            Missing metadata means ordinary continuation of this member's ledger.
        history_dir: Stable recipient-owned directory across timestamped log runs.
            Defaults to algo.nn_dir for standalone use. Never use a donor's directory.
        legacy_best_path: Explicit recipient conventional-best checkpoint for migration
            from an older log directory. Only used when no committed history exists;
            never infer this path from staged donor metadata or a latest checkpoint.
    """

    def __init__(self, metadata=None, history_dir=None, legacy_best_path=None):
        super().__init__()
        self.metadata = metadata or {}
        self.history_dir = history_dir
        self.legacy_best_path = Path(legacy_best_path) if legacy_best_path else None

    def after_init(self, algo):
        self.algo = algo
        self.nn_dir = Path(algo.nn_dir).resolve()
        self.root = Path(self.history_dir).resolve() if self.history_dir else self.nn_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.nn_dir.mkdir(parents=True, exist_ok=True)
        name = algo.config["name"]
        if not isinstance(name, str) or not name or Path(name).name != name or name in {".", "..", "historical_best"}:
            raise ValueError("Best-checkpoint config.name must be a non-reserved filename")
        self.conventional = self.nn_dir / (name + ".pth")
        self.record_path = self.root / "best_history.json"
        self.generations = self.root / ".best_checkpoints"
        self.generations.mkdir(exist_ok=True)
        self._save = algo.save
        self._restore = algo.restore
        if self.record_path.exists():
            self.record = self._read_record()
            if self.record["config_name"] != name:
                raise ValueError("Best history belongs to a different config.name")
        else:
            self.record = {"schema_version": 1, "config_name": name, "branches": {}, "historical_best": None}
            self._import_legacy()
        self._select_branch()
        self._repair_aliases()
        self._collect_generations()
        algo.save = self._save_routed
        algo.restore = self._restore_clean
        print(
            f"[pbt-best] branch={self.branch_id}; {self.conventional.name} -> historical_best.pth; "
            f"record={self.record_path}",
            flush=True,
        )

    def _read_record(self):
        with self.record_path.open() as stream:
            record = json.load(stream)
        if record.get("schema_version") != 1:
            raise ValueError("Unsupported best-history schema")
        entries = [record["historical_best"], *(b["best"] for b in record["branches"].values())]
        for entry in entries:
            if entry is not None:
                if not math.isfinite(float(entry["score"])) or not (self.root / entry["checkpoint"]).is_file():
                    raise ValueError("Invalid best-history score or missing committed checkpoint")
        return record

    def _commit(self, record):
        """Commit last; callers never mutate the live record before this succeeds."""
        fd, temporary = tempfile.mkstemp(prefix=".best_history-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.record_path)
            self.record = record
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _import_legacy(self):
        """Only the explicit conventional best file is evidence of legacy best weights.

        An arbitrary resume/PBT/latest checkpoint's last_mean_rewards describes
        earlier weights, NOT the saved policy. Never pair those weights or a PBT
        true_objective with a historical reward. Missing/sentinel legacy scores
        remain unknown; preserve the old conventional file until a measured best.
        """
        source = self.legacy_best_path or self.conventional
        if not source.is_file():
            if self.legacy_best_path:
                raise FileNotFoundError(source)
            return
        import torch

        weights = torch.load(source, map_location="cpu", weights_only=False)
        score = weights.get("last_mean_rewards")
        try:
            score = float(score)
        except (TypeError, ValueError):
            return
        if not math.isfinite(score) or score == -1e9 or "model" not in weights:
            return
        destination = self.generations / f"legacy-{uuid4().hex}.pth"
        shutil.copyfile(source, destination)
        self._sync_file(destination)
        self.record["historical_best"] = {
            "score": score,
            "checkpoint": str(destination.relative_to(self.root)),
            "branch_id": None,
            "frame": int(weights.get("frame", 0)),
            "epoch": int(weights.get("epoch", 0)),
            "source": "legacy_conventional_best",
            "legacy_path": str(source.resolve()),
            "coverage": "recoverable_file_only; earlier overwritten maxima are unknown",
        }

    def _select_branch(self):
        token = int(self.metadata.get("last_exploit_frame", 0))
        if token < 0:
            raise ValueError("last_exploit_frame must be nonnegative")
        recipient = self.metadata.get("recipient_frame")
        if recipient is not None and int(recipient) != token:
            raise ValueError("Staged exploit recipient_frame must equal last_exploit_frame")
        record = copy.deepcopy(self.record)
        current = record.get("current_branch")
        if current is not None:
            previous = record["branches"][str(current)]["last_exploit_frame"]
            if "last_exploit_frame" not in self.metadata:
                token = previous
            if token < previous:
                raise ValueError("Stale pre-exploit resume metadata cannot continue a newer policy branch")
            if token == previous:
                self.branch_id = current
                return
        self.branch_id = 0 if current is None else max(map(int, record["branches"])) + 1
        record["current_branch"] = self.branch_id
        record["branches"][str(self.branch_id)] = {"last_exploit_frame": token, "best": None}
        self._commit(record)

    @staticmethod
    def _sync_file(path):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())

    def _alias(self, path, entry, target=None):
        if entry is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        target = target or self.root / entry["checkpoint"]
        relative_target = os.path.relpath(target, path.parent)
        if path.is_symlink() and os.readlink(path) == relative_target:
            return
        temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
        try:
            temporary.symlink_to(relative_target)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _repair_aliases(self):
        historical = self.record["historical_best"]
        stable = self.root / "historical_best.pth"
        self._alias(stable, historical)
        if self.nn_dir != self.root:
            self._alias(self.nn_dir / "historical_best.pth", historical, target=stable)
        self._alias(self.conventional, historical, target=stable)
        for branch_id, branch in self.record["branches"].items():
            self._alias(self.root / "branches" / f"branch_{int(branch_id):06d}_best.pth", branch["best"])

    def _collect_generations(self):
        """Delete only our unreferenced regular files AFTER commit and alias repair.

        A failed commit/repair never reaches collection. Reopening repairs aliases
        first and then also collects orphan generations from an interrupted save.
        Permission failures leave harmless extra files for the next collection.
        """
        entries = [self.record["historical_best"], *(b["best"] for b in self.record["branches"].values())]
        retained = {self.root / entry["checkpoint"] for entry in entries if entry is not None}
        owned = re.compile(r"(?:branch_[0-9]{6,}|legacy)-[0-9a-f]{32}\.pth")
        for path in self.generations.iterdir():
            if path in retained or not owned.fullmatch(path.name) or path.is_symlink() or not path.is_file():
                continue
            try:
                path.unlink()
            except OSError as exc:
                warnings.warn(f"Could not collect obsolete best checkpoint {path}: {exc}", RuntimeWarning, stacklevel=2)

    @property
    def branch_best_score(self):
        best = self.record["branches"][str(self.branch_id)]["best"]
        return best["score"] if best else -math.inf

    def _restore_clean(self, *args, **kwargs):
        result = self._restore(*args, **kwargs)
        # Donor/old-policy episode windows are not measurements of this branch.
        # train.init_tensors + env_reset reset episode accumulators before rollouts.
        for name in ("game_rewards", "game_shaped_rewards", "game_lengths"):
            getattr(self.algo, name).clear()
        self.algo.last_mean_rewards = self.branch_best_score
        return result

    def _publish(self, epoch):
        self._repair_aliases()  # complete a prior committed but interrupted promotion
        meter = self.algo.game_rewards
        if meter.current_size <= 0 or epoch < self.algo.save_best_after:
            return
        score = float(meter.get_mean()[0])
        if not math.isfinite(score) or score <= self.branch_best_score:
            return
        destination = self.generations / f"branch_{self.branch_id:06d}-{uuid4().hex}.pth"
        old_gate = self.algo.last_mean_rewards
        try:
            self.algo.last_mean_rewards = score
            # RL-Games appends .pth; the private generation is not visible until
            # the JSON commit, so failed/partial saves cannot replace old bests.
            self._save(str(destination.with_suffix("")))
            self._sync_file(destination)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        finally:
            self.algo.last_mean_rewards = old_gate
        entry = {
            "score": score,
            "checkpoint": str(destination.relative_to(self.root)),
            "branch_id": self.branch_id,
            "frame": int(self.algo.frame),
            "epoch": int(epoch),
            "source": "raw_episode_reward_mean",
        }
        record = copy.deepcopy(self.record)
        record["branches"][str(self.branch_id)]["best"] = entry
        historical = record["historical_best"]
        if historical is None or score > historical["score"]:
            record["historical_best"] = entry
        self._commit(record)
        self._repair_aliases()
        self._collect_generations()

    def after_print_stats(self, frame, epoch_num, total_time):
        # Repair the train-entry reset. Keep the pre-promotion branch gate for
        # upstream's score_to_win/save branch; routed save then restores it.
        self.algo.last_mean_rewards = self.branch_best_score
        self._publish(epoch_num)

    def _save_routed(self, filename, *args, **kwargs):
        # Use lexical absolute paths: resolving the conventional symlink would
        # compare its target, not the requested upstream save destination.
        requested = Path(os.path.abspath(os.fspath(filename)))
        if requested not in (self.conventional, self.conventional.with_suffix("")):
            return self._save(filename, *args, **kwargs)
        try:
            self._publish(self.algo.epoch_num)
        finally:
            self.algo.last_mean_rewards = self.branch_best_score

    def after_clear_stats(self):
        self.algo.last_mean_rewards = self.branch_best_score
