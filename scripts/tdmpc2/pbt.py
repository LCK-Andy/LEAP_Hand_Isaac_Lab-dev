"""Trainer-independent TD-MPC2 population state exchange using async-pbt 0.1.0.

Checkpoints are trusted local torch payloads containing ``training_state`` (full
agent, replay and trainer state), ``env_state``, and ``pbt`` metadata. The YAML
companion is the commit marker; a staged restart companion overrides donor PBT
metadata. Main explicitly restores a RestartDecision, applies its hyperparameters,
and resets all environment episodes. Learner/replay random streams transfer with
the donor; recipient training-budget counters and the dedicated PBT RNG do not.
"""

import hashlib
import json
import math
import os
import pickle
import time
import torch
import uuid
import yaml
from dataclasses import dataclass
from pathlib import Path

from async_pbt.mutation import mutate
from async_pbt.pbt_cfg import PbtCfg
from async_pbt.pbt_utils import (
    EpisodeWindow,
    atomic_yaml,
    load_pbt_ckpts,
    recipient_rng,
    rng_state,
    select_donor,
    stage_restart,
)

MUTATION_BOUNDS = {"lr": (1e-5, 1e-3), "entropy_coef": (1e-5, 1e-3), "grad_clip_norm": (1.0, 100.0)}


def default_pbt_cfg(**overrides) -> PbtCfg:
    """TD-MPC2 defaults; never borrow PPO mutation names or parameter bounds."""
    values = dict(
        enabled=True,
        objective="dexpbt",
        interval_steps=1_000_000,
        warmup_steps=5_000_000,
        adaptation_steps=3_000_000,
        mutation_rate=0.25,
        change_range=(1.1, 1.5),
        mutation={key: "mutate_float" for key in MUTATION_BOUNDS},
        mutation_bounds=dict(MUTATION_BOUNDS),
    )
    values.update(overrides)
    return PbtCfg(**values)


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _fingerprint(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_checkpoint(path: Path | str) -> dict:
    """Load a trusted, committed snapshot, including recipient restart metadata."""
    path = Path(path)
    with path.with_suffix(".yaml").open() as stream:
        metadata = yaml.safe_load(stream)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    original = payload["pbt"]
    if any(metadata.get(key) != original.get(key) for key in ("snapshot_id", "algorithm", "contract_fingerprint")):
        raise ValueError("Checkpoint and commit marker identity mismatch")
    if metadata.get("algorithm") != "tdmpc2" or metadata.get("schema_version") != 2:
        raise ValueError("Not a TD-MPC2 schema-2 checkpoint")
    payload["pbt"] = metadata
    return payload


@dataclass
class RestartDecision:
    training_state: dict
    env_state: dict
    hparams: dict
    frame: int
    updates: int
    metadata: dict


class PopulationMember:
    """Own publication cadence and PBT evidence, not simulator or optimizer logic.

    One process owns each member directory. The caller's contract must enumerate
    every immutable learner/planner/replay/environment setting (including task,
    observation/action definitions and population size). Exact physical Dexsuite
    condition stamps remain part of each completed-episode evaluation.
    """

    def __init__(self, workspace: Path, cfg: PbtCfg, seed: int, hparams: dict, contract: dict, *, log_callback=None):
        if cfg.objective != "dexpbt":
            raise ValueError("TD-MPC2 population requires the dexpbt completed-episode objective")
        if not set(cfg.mutation).issubset(MUTATION_BOUNDS) or any(
            method != "mutate_float" for method in cfg.mutation.values()
        ):
            raise ValueError("Only TD-MPC2 scalar hyperparameters may mutate")
        if not cfg.copy_only and not cfg.mutation:
            raise ValueError("TD-MPC2 PBT requires explicit bounded mutation targets")
        for key in cfg.mutation:
            low, high = cfg.mutation_bounds[key]
            allowed_low, allowed_high = MUTATION_BOUNDS[key]
            if not allowed_low <= low < high <= allowed_high:
                raise ValueError(f"TD-MPC2 mutation bounds invalid for {key}")
        self.cfg = cfg
        self.seed = seed
        self.hparams = dict(hparams)
        if not self._valid_hparams(self.hparams):
            raise ValueError("Invalid TD-MPC2 hyperparameter values")
        self.contract = _canonical(contract)
        self.contract_fingerprint = _fingerprint(self.contract)
        self.workspace = Path(workspace).resolve() / "tdmpc2"
        self.policy_dir = self.workspace / f"{cfg.policy_idx:03d}"
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self.rng = recipient_rng(seed, cfg.policy_idx)
        self.window = EpisodeWindow(cfg, condition=self._condition)
        self.last_publish_frame = 0
        self.last_exploit_frame = 0
        self.frame = 0
        self.updates = 0
        self.log_callback = log_callback

    def _condition(self, condition):
        result = _canonical(condition)
        result["tdmpc2_contract"] = self.contract_fingerprint
        return result

    def observe(self, infos: dict, frame: int):
        if frame < self.frame:
            raise ValueError("Recipient frame cannot move backwards")
        self.frame = int(frame)
        self.window.add(infos, frame)

    def should_publish(self, frame: int) -> bool:
        """Crossed interval boundaries count even if 256 envs cannot divide 1M."""
        return frame // self.cfg.interval_steps > self.last_publish_frame // self.cfg.interval_steps

    def publish(self, training_state: dict, frame: int, updates: int, env_state: dict, final=False) -> dict:
        if frame < self.frame or updates < self.updates:
            raise ValueError("Recipient budget counters cannot move backwards")
        if not {"agent", "replay", "trainer"}.issubset(training_state):
            raise ValueError("Publication requires full agent, replay, and trainer state")
        self.frame, self.updates = int(frame), int(updates)
        evaluations = self.window.ready(self.cfg, frame)
        # Numeric names preserve async-pbt's population discovery; nanoseconds
        # plus UUID snapshot identity prevent overwriting an already committed file.
        stem = f"{time.time_ns():020d}"
        checkpoint = self.policy_dir / f"{stem}.pth"
        companion = checkpoint.with_suffix(".yaml")
        record = {
            "schema_version": 2,
            "algorithm": "tdmpc2",
            "snapshot_id": uuid.uuid4().hex,
            "contract": self.contract,
            "contract_fingerprint": self.contract_fingerprint,
            "policy_idx": self.cfg.policy_idx,
            "frame": self.frame,
            "epoch": self.updates,
            "updates": self.updates,
            "created_at": time.time(),
            "params": dict(self.hparams),
            "trainer_state": _canonical(training_state["trainer"]),
            "checkpoint": str(checkpoint),
            "pbt_checkpoint": str(companion),
            "last_publish_frame": self.frame,
            "last_exploit_frame": self.last_exploit_frame,
            "pbt_rng_state": rng_state(self.rng),
            "evaluations": evaluations,
            "evaluation_diagnostics": dict(self.window.diagnostics),
            "objective_valid": bool(evaluations),
            "true_objective": max((g["success_rate"] for g in evaluations.values()), default=0.0),
            "final": bool(final),
        }
        temporary = checkpoint.with_suffix(".pth.tmp")
        try:
            with temporary.open("wb") as stream:
                torch.save({"training_state": training_state, "env_state": env_state, "pbt": record}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, checkpoint)
            atomic_yaml(companion, record)
        finally:
            temporary.unlink(missing_ok=True)
        self.last_publish_frame = self.frame
        self._prune_archives()
        self._log("publication", record)
        return record

    def _prune_archives(self):
        owned = []
        for path in sorted(self.policy_dir.glob("[0-9]*.yaml"), reverse=True):
            try:
                with path.open() as stream:
                    record = yaml.safe_load(stream)
                if (
                    self._compatible(record)
                    and record.get("policy_idx") == self.cfg.policy_idx
                    and record.get("checkpoint") == str(path.with_suffix(".pth"))
                    and record.get("pbt_checkpoint") == str(path)
                    and record.get("snapshot_id")
                ):
                    owned.append(path)
            except (OSError, ValueError, TypeError, yaml.YAMLError):
                continue
        for path in owned[2:]:
            # Remove the discovery marker first; a concurrent already-selected
            # donor copy either holds an open inode or fails and skips replacement.
            path.unlink(missing_ok=True)
            path.with_suffix(".pth").unlink(missing_ok=True)

    def _log(self, event, detail):
        if self.log_callback is not None:
            self.log_callback({"event": event, "policy_idx": self.cfg.policy_idx, "frame": self.frame, **detail})

    def _valid_hparams(self, params):
        return (
            isinstance(params, dict)
            and set(params) == set(MUTATION_BOUNDS)
            and all(
                type(params[key]) in (int, float) and math.isfinite(params[key]) and low <= params[key] <= high
                for key, (low, high) in MUTATION_BOUNDS.items()
            )
        )

    def _compatible(self, record):
        return (
            isinstance(record, dict)
            and record.get("schema_version") == 2
            and record.get("algorithm") == "tdmpc2"
            and record.get("contract_fingerprint") == self.contract_fingerprint
            and record.get("contract") == self.contract
            and self._valid_hparams(record.get("params"))
        )

    def restore_metadata(self, record: dict):
        """Restore recipient PBT bookkeeping only; discard pre-reset evidence."""
        if not self._compatible(record) or record.get("policy_idx") != self.cfg.policy_idx:
            raise ValueError("Incompatible TD-MPC2 recipient metadata")
        self.rng = recipient_rng(self.seed, self.cfg.policy_idx, record)
        self.hparams = dict(record["params"])
        self.frame = int(record.get("recipient_frame", record["frame"]))
        self.updates = int(record.get("recipient_epoch", record["epoch"]))
        self.last_publish_frame = int(record.get("last_publish_frame", self.frame))
        self.last_exploit_frame = int(record.get("last_exploit_frame", 0))
        self.window.consume()

    def maybe_exploit(self, record: dict) -> RestartDecision | None:
        """Stage a selected full snapshot locally, then return explicit restore data.

        Donor copy races/corruption skip replacement. No shared donor archive is
        loaded directly: pruning cannot remove the staged recipient-owned file.
        A returned decision must be applied before observing/publishing again.
        """
        if not self._compatible(record) or record.get("policy_idx") != self.cfg.policy_idx:
            raise ValueError("Selection requires this member's compatible publication")
        reason = None
        if record.get("final"):
            reason = "terminal_publication"
        elif not self.cfg.enabled or not self.cfg.selection_enabled:
            reason = "selection_disabled"
        elif record["frame"] < self.cfg.warmup_steps:
            reason = "warmup"
        elif self.last_exploit_frame and record["frame"] - self.last_exploit_frame < self.cfg.adaptation_steps:
            reason = "adaptation"
        if reason:
            self._log("selection", {"reason": reason})
            return None
        records = load_pbt_ckpts(self.workspace, self.cfg.policy_idx, self.cfg.num_policies, 10**30)
        compatible = {key: value for key, value in records.items() if self._compatible(value)}
        diagnostics = {}
        donor_idx = select_donor(compatible, record, self.cfg, diagnostics=diagnostics)
        diagnostics["incompatible_records"] = sum(
            value is not None and not self._compatible(value) for value in records.values()
        )
        self._log("selection", diagnostics)
        if donor_idx is None:
            return None
        donor = compatible[donor_idx]
        params = dict(donor["params"])
        # Use a speculative recipient stream: a failed stage never advances RNG.
        rng = recipient_rng(self.seed, self.cfg.policy_idx, {"pbt_rng_state": rng_state(self.rng)})
        if not self.cfg.copy_only:
            params = mutate(
                params,
                self.cfg.mutation,
                self.cfg.mutation_rate,
                self.cfg.change_range,
                self.cfg.mutation_bounds,
                rng=rng,
            )
        recipient = {**record, "pbt_rng_state": rng_state(rng)}
        try:
            checkpoint, companion = stage_restart(self.policy_dir, donor, recipient, params)
            payload = load_checkpoint(checkpoint)
            metadata = payload["pbt"]
            # stage_restart preserves generic frame/epoch from the donor, so
            # pin all explicitly recipient-owned counters before committing.
            metadata.update(
                last_publish_frame=self.last_publish_frame,
                donor_epoch=donor["epoch"],
                frame=self.frame,
                epoch=self.updates,
                updates=self.updates,
                recipient_updates=self.updates,
                recipient_trainer_state=record["trainer_state"],
                final=False,
                reset_episodes=True,
                rng_transfer_policy="donor_learner_replay_recipient_pbt",
            )
            atomic_yaml(companion, metadata)
        except (OSError, ValueError, KeyError, TypeError, EOFError, RuntimeError, pickle.UnpicklingError) as error:
            self._log("replacement_skipped", {"reason": str(error)})
            return None
        self.restore_metadata(metadata)
        self._log(
            "replacement",
            {
                "donor_policy_idx": donor_idx,
                "params": params,
                "recipient_frame": self.frame,
                "recipient_updates": self.updates,
            },
        )
        return RestartDecision(
            payload["training_state"], payload["env_state"], params, self.frame, self.updates, metadata
        )
