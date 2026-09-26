"""Atomic PBT publications and completed-episode comparison (no simulator imports)."""

import json
import math
import os
import random
import shutil
import time
import warnings
import yaml
from collections import deque
from pathlib import Path

from ..evaluation import comparison_condition
from .pbt_cfg import PbtCfg
from .runtime import effective_params

PROTOCOL = "training_conditions_fixed_definition_v1"


def rng_state(rng):
    """Encode the dedicated generator state as JSON/YAML-safe lists, never pickle."""
    return json.loads(json.dumps(rng.getstate()))


def recipient_rng(seed, policy_idx, metadata=None):
    """Construct a member-owned stream, restoring only its companion metadata."""
    rng = random.Random(f"pbt_rng_v1:{int(seed)}:{int(policy_idx)}")
    if metadata is None:
        return rng
    if "pbt_rng_state" not in metadata:
        frame = int(metadata.get("recipient_frame", metadata.get("frame", 0)))
        epoch = int(metadata.get("recipient_epoch", metadata.get("epoch", 0)))
        # A legacy restart's iteration/epoch may belong to the donor. Use recipient
        # age when available; never depend on salted hash() or the global RNG.
        if "recipient_frame" in metadata:
            epoch = int(metadata.get("recipient_epoch", 0))
        fallback = f"pbt_rng_legacy_v1:{int(seed)}:{int(policy_idx)}:{frame}:{epoch}"
        warnings.warn(
            f"Resuming legacy PBT metadata without pbt_rng_state; using deterministic {fallback}. "
            "The original PBT random stream cannot be recovered.",
            RuntimeWarning,
            stacklevel=2,
        )
        rng.seed(fallback)
        return rng

    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, (list, tuple)) else value

    try:
        state = tuples(metadata["pbt_rng_state"])
        if not isinstance(state, tuple) or len(state) != 3:
            raise ValueError("expected the three-part Random state")
        if state[2] is not None and (type(state[2]) not in (int, float) or not math.isfinite(state[2])):
            raise ValueError("invalid cached Gaussian value")
        rng.setstate(state)
    except (TypeError, ValueError, IndexError, OverflowError, RecursionError) as error:
        raise ValueError(f"Invalid PBT RNG state in recipient metadata for member {policy_idx}: {error}") from error
    return rng


def flatten_dict(data, prefix="", separator="."):
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result.update(flatten_dict(value, prefix + key + separator, separator))
        else:
            result[prefix + key] = value
    return result


def filter_params(params, keys):
    return {key: float(params[key]) for key in keys}


def atomic_yaml(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        yaml.safe_dump(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_pbt_checkpoint(workspace_dir, curr_policy_score, curr_iter, algo, params, **metadata):
    """Publish weights first, then an atomic metadata commit marker."""
    import torch

    root = Path(workspace_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / ("final.pth" if metadata.get("terminal") else f"{curr_iter:012d}.pth")
    companion = checkpoint.with_suffix(".yaml")
    temporary = checkpoint.with_suffix(".pth.tmp")
    torch.save(algo.get_full_state_weights(), temporary)
    os.replace(temporary, checkpoint)
    record = {
        "schema_version": 2,
        "iteration": int(curr_iter),
        "true_objective": float(curr_policy_score),
        "frame": int(algo.frame),
        "epoch": int(algo.epoch_num),
        "params": effective_params(algo, params),
        "checkpoint": str(checkpoint.resolve()),
        "pbt_checkpoint": str(companion.resolve()),
        "experiment_name": algo.experiment_name,
        "created_at": time.time(),
        **metadata,
    }
    atomic_yaml(companion, record)
    return record


def load_pbt_ckpts(workspace_dir, cur_policy_id, num_policies, pbt_iteration):
    """Only committed records from the explicitly configured population participate."""
    checkpoints = {}
    for policy in range(num_policies):
        checkpoints[policy] = None
        for path in sorted((Path(workspace_dir) / f"{policy:03d}").glob("[0-9]*.yaml"), reverse=True):
            try:
                if int(path.stem) > pbt_iteration:
                    continue
                with path.open() as stream:
                    record = yaml.safe_load(stream)
                if record.get("schema_version") != 2 or record.get("policy_idx") != policy:
                    continue
                if not Path(record["checkpoint"]).is_file():
                    continue
                checkpoints[policy] = record
                break
            except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
                continue
    return checkpoints


def cleanup(checkpoints, policy_dir, keep_back=20, max_yaml=50):
    # A selected donor is copied to recipient-owned files before restart, so pruning
    # never invalidates a restart already prepared. A concurrent copy failure skips exploitation.
    paths = sorted(Path(policy_dir).glob("[0-9]*.yaml"), reverse=True)
    for path in paths[max_yaml:]:
        path.with_suffix(".pth").unlink(missing_ok=True)
        path.unlink(missing_ok=True)


def stage_restart(policy_dir, donor, recipient, params):
    started = time.time()
    tick = time.monotonic()
    root = Path(policy_dir) / "restart"
    root.mkdir(exist_ok=True)
    checkpoint = root / "checkpoint.pth"
    temporary = root / "checkpoint.pth.tmp"
    shutil.copyfile(donor["checkpoint"], temporary)
    os.replace(temporary, checkpoint)
    metadata = root / "checkpoint.yaml"
    record = {
        **donor,
        "params": params,
        "checkpoint": str(checkpoint.resolve()),
        "policy_idx": recipient["policy_idx"],
        "recipient_frame": recipient["frame"],
        "recipient_epoch": recipient["epoch"],
        "last_exploit_frame": recipient["frame"],
        "donor_policy_idx": donor["policy_idx"],
        "donor_frame": donor["frame"],
        "donor_lineage": donor.get("lineage", f"member-{donor['policy_idx']}"),
        "recipient_previous_lineage": recipient.get("lineage", f"member-{recipient['policy_idx']}"),
        "lineage": f"member-{recipient['policy_idx']}@{recipient['frame']}",
        "selection_evidence": {},
        "evaluations": {},
        "true_objective": 0.0,
        "objective_valid": False,
        "evaluation_diagnostics": {"reason": "missing_evidence", "fresh_episodes": 0},
        "selection_diagnostics": {"reason": "restart_requires_fresh_evidence", "donor": None},
        "replacement_started_at": started,
        "checkpoint_copy_seconds": time.monotonic() - tick,
        "pbt_checkpoint": str(metadata.resolve()),
    }
    # Never inherit a donor stream, even when the recipient is a legacy record.
    record.pop("pbt_rng_state", None)
    if "pbt_rng_state" in recipient:
        record["pbt_rng_state"] = recipient["pbt_rng_state"]
    atomic_yaml(metadata, record)
    return str(checkpoint.resolve()), str(metadata.resolve())


def _aggregate_batches(condition, batches, cfg):
    group = {
        "condition": condition,
        "protocol": PROTOCOL,
        **{
            name: sum(batch[name] for batch in batches)
            for name in (
                "completed_episodes",
                "success_count",
                "position_error_sum",
                "rotation_error_sum",
                "good_contact_fraction_sum",
            )
        },
        "created_at": min(batch["created_at"] for batch in batches),
        "start_frame": min(batch["start_frame"] for batch in batches),
        "end_frame": max(batch["end_frame"] for batch in batches),
    }
    n = group["completed_episodes"]
    pos, rot = group["position_error_sum"] / n, group["rotation_error_sum"] / n
    # Fixed, reward-independent geometric progress; contact remains diagnostic only.
    group["success_rate"] = group["success_count"] / n
    group["progress"] = math.exp(-pos / cfg.position_error_scale) * max(0.0, 1 - rot / math.pi)
    # Keep the aggregate schema, plus sufficient evidence to age individual batches
    # at comparison time. No timestamps are refreshed by publication/deserialization.
    group["batches"] = [{k: v for k, v in batch.items() if k not in {"condition", "protocol"}} for batch in batches]
    return group


class EpisodeWindow:
    """Sufficient statistics of fresh, completed, condition-stable episodes only."""

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else PbtCfg()
        self.groups = {}
        self.pruned = {"expired_evidence": 0, "evidence_frame_mismatch": 0}
        self.diagnostics = {}

    def _prune(self, frame, now):
        # Arrival-ordered batches, not lifetime accumulators. Removal is amortized
        # O(1) per batch, including while no publication/ready call is due.
        for key, batches in list(self.groups.items()):
            while batches and not (
                0 <= now - batches[0]["created_at"] <= self.cfg.max_score_age_seconds
                and 0 <= frame - batches[0]["start_frame"] <= self.cfg.max_frame_lag
            ):
                batch = batches.popleft()
                reason = (
                    "expired_evidence"
                    if not 0 <= now - batch["created_at"] <= self.cfg.max_score_age_seconds
                    else "evidence_frame_mismatch"
                )
                self.pruned[reason] += batch["completed_episodes"]
            if not batches:
                del self.groups[key]

    def add(self, infos, frame, now=None):
        now = time.time() if now is None else now
        self._prune(frame, now)
        stats = infos.get("dexpbt", {})
        if stats.get("schema_version") != 1 or stats.get("protocol") != PROTOCOL:
            return
        for group in stats.get("groups", []):
            try:
                n = int(group["completed_episodes"])
                condition = comparison_condition(group["condition"])
                required = {"curriculum_tolerance", "position_tolerance", "rotation_tolerance", "contact_threshold"}
                if "adr_fraction" in condition or not condition.get("adr_parameters"):
                    required.add("adr_fraction")
                if not required.issubset(condition):
                    continue
                for name in required:
                    condition[name] = float(condition[name])
                if not all(math.isfinite(condition[name]) for name in required):
                    continue
                values = {
                    k: float(group[k])
                    for k in ("success_count", "position_error_sum", "rotation_error_sum", "good_contact_fraction_sum")
                }
                if n <= 0 or not all(math.isfinite(v) and v >= 0 for v in values.values()):
                    continue
                if values["success_count"] > n or values["good_contact_fraction_sum"] > n:
                    continue
                key = json.dumps(condition, sort_keys=True)
                self.groups.setdefault(key, deque()).append(
                    {
                        "condition": condition,
                        "protocol": PROTOCOL,
                        "completed_episodes": n,
                        **values,
                        "start_frame": int(frame),
                        "end_frame": int(frame),
                        "created_at": now,
                    }
                )
            except (TypeError, ValueError, KeyError):
                continue

    def ready(self, cfg, frame, now=None):
        now = time.time() if now is None else now
        self.cfg = cfg
        self._prune(frame, now)
        result = {}
        counts = []
        for key, batches in self.groups.items():
            group = _aggregate_batches(batches[0]["condition"], batches, cfg)
            counts.append(group["completed_episodes"])
            if group["completed_episodes"] >= cfg.min_completed_episodes:
                result[key] = group
        self.diagnostics = {
            "reason": (
                "ready"
                if result
                else (
                    "insufficient_fresh_episodes"
                    if counts
                    else next((reason for reason, count in self.pruned.items() if count), "missing_evidence")
                )
            ),
            "fresh_episodes": sum(counts),
            "largest_condition_episodes": max(counts, default=0),
            "ready_conditions": len(result),
            "min_completed_episodes": cfg.min_completed_episodes,
            "pruned_episodes": dict(self.pruned),
        }
        return result

    def consume(self):
        """Explicitly discard policy evidence; publications alone do not reset its age."""
        self.groups.clear()
        self.pruned = dict.fromkeys(self.pruned, 0)
        self.diagnostics = {}


def _record_rejection(record, cfg, frame, now):
    if not isinstance(record, dict) or record.get("schema_version") != 2:
        return "record_schema_mismatch"
    try:
        if not all(math.isfinite(record[key]) for key in ("created_at", "frame")):
            return "record_invalid_metadata"
        if now < record["created_at"]:
            return "record_future_timestamp"
        if now - record["created_at"] > cfg.max_score_age_seconds:
            return "record_expired"
        if abs(frame - record["frame"]) > cfg.max_frame_lag:
            return "record_frame_mismatch"
    except (KeyError, TypeError, ValueError):
        return "record_invalid_metadata"
    return None


def eligible(record, cfg, frame, now=None):
    now = time.time() if now is None else now
    return _record_rejection(record, cfg, frame, now) is None


def _fresh_evaluation(group, cfg, frame, now):
    try:
        if not isinstance(group, dict):
            return None, "invalid_evidence"
        if "batches" not in group:
            return group, _evidence_rejection(group, cfg, frame, now)
        if not isinstance(group["batches"], list) or not isinstance(group.get("condition"), dict):
            return None, "invalid_evidence"
        batches, reasons = [], []
        for batch in group["batches"]:
            reason = _evidence_rejection(batch, cfg, frame, now, check_count=False)
            if reason == "invalid_evidence":
                return None, reason
            if reason:
                reasons.append(reason)
            else:
                batches.append(batch)
        if not batches:
            return None, reasons[0] if reasons else "missing_evidence"
        fresh = _aggregate_batches(group["condition"], batches, cfg)
        return fresh, _evidence_rejection(fresh, cfg, frame, now)
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None, "invalid_evidence"


def _evidence_rejection(group, cfg, frame, now, check_count=True):
    try:
        if not isinstance(group, dict):
            return "invalid_evidence"
        keys = ("created_at", "end_frame", "completed_episodes")
        if not all(type(group[key]) in (int, float) and math.isfinite(group[key]) for key in keys):
            return "invalid_evidence"
        n = group["completed_episodes"]
        if n <= 0 or int(n) != n:
            return "invalid_evidence"
        for key in ("success_count", "position_error_sum", "rotation_error_sum", "good_contact_fraction_sum"):
            if type(group[key]) not in (int, float) or not math.isfinite(group[key]) or group[key] < 0:
                return "invalid_evidence"
        if group["success_count"] > n or group["good_contact_fraction_sum"] > n:
            return "invalid_evidence"
        if check_count and any(
            type(group[key]) not in (int, float) or not math.isfinite(group[key]) or not 0 <= group[key] <= 1
            for key in ("success_rate", "progress")
        ):
            return "invalid_evidence"
        start_frame = group.get("start_frame", group["end_frame"])
        if not math.isfinite(start_frame) or start_frame > group["end_frame"]:
            return "invalid_evidence"
        if now < group["created_at"]:
            return "evidence_future_timestamp"
        if now - group["created_at"] > cfg.max_score_age_seconds:
            return "expired_evidence"
        if any(abs(frame - value) > cfg.max_frame_lag for value in (start_frame, group["end_frame"])):
            return "evidence_frame_mismatch"
        if group["completed_episodes"] <= 0:
            return "invalid_evidence"
        if check_count and group["completed_episodes"] < cfg.min_completed_episodes:
            return "insufficient_fresh_episodes"
    except (KeyError, TypeError, ValueError):
        return "invalid_evidence"
    return None


def select_donor(records, own, cfg, now=None, diagnostics=None):
    """Success-first comparison, optionally explaining why no donor can be selected."""
    stamp = time.time() if now is None else now
    detail = {"donor_records": 0, "eligible_donors": 0, "comparable_pairs": 0, "rejections": {}}
    rejections = detail["rejections"]

    def reject(reason):
        rejections[reason] = rejections.get(reason, 0) + 1

    def finish(reason, donor=None):
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(detail, reason=reason, donor=donor)
        return donor

    ours_valid = {}
    own_reasons = []
    if not isinstance(own, dict) or not isinstance(own.get("evaluations", {}), dict):
        return finish("own_invalid_evidence")
    for key, group in own.get("evaluations", {}).items():
        group, reason = _fresh_evaluation(group, cfg, own["frame"], stamp)
        if reason:
            own_reasons.append(reason)
        else:
            ours_valid[key] = group
    if not ours_valid:
        reason = (
            own_reasons[0] if own_reasons else own.get("evaluation_diagnostics", {}).get("reason", "missing_evidence")
        )
        return finish("own_" + reason)

    candidates = []
    for policy, record in records.items():
        if policy == cfg.policy_idx or not record:
            continue
        detail["donor_records"] += 1
        reason = _record_rejection(record, cfg, own["frame"], stamp)
        if reason:
            reject(reason)
            continue
        detail["eligible_donors"] += 1
        theirs_groups = record.get("evaluations", {})
        if not isinstance(theirs_groups, dict):
            reject("invalid_evidence")
            continue
        if not theirs_groups or record.get("objective_valid") is False:
            reject(record.get("evaluation_diagnostics", {}).get("reason", "missing_evidence"))
            continue
        shared = ours_valid.keys() & theirs_groups.keys()
        if not shared:
            reject("physical_condition_mismatch")
        for key in shared:
            ours, theirs = ours_valid[key], theirs_groups[key]
            theirs, reason = _fresh_evaluation(theirs, cfg, own["frame"], stamp)
            if reason:
                reject(reason)
                continue
            detail["comparable_pairs"] += 1
            sr, dr = ours["success_rate"], theirs["success_rate"]
            if sr or dr:
                # Binomial standard error with a finite-sample floor; no soft reward can
                # overturn even a single measured success, and noisy ties never exploit.
                se = math.sqrt(
                    sum(
                        (g["success_rate"] * (1 - g["success_rate"]) + 1 / g["completed_episodes"])
                        / g["completed_episodes"]
                        for g in (ours, theirs)
                    )
                )
                improves = dr - sr > max(cfg.threshold_abs, cfg.threshold_std * se)
            else:
                improves = cfg.progress_weight * (theirs["progress"] - ours["progress"]) > cfg.threshold_abs
            if improves:
                candidates.append((dr, theirs["progress"], policy))
    if candidates:
        return finish("selected", max(candidates)[2])
    if not detail["donor_records"]:
        return finish("no_donor_records")
    if not detail["eligible_donors"]:
        return finish("no_eligible_donors")
    if not detail["comparable_pairs"]:
        return finish("no_comparable_evidence")
    return finish("no_sufficient_improvement")
