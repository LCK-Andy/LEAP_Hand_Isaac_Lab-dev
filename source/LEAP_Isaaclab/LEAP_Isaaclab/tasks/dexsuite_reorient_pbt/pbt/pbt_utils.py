"""Atomic PBT publications and completed-episode comparison (no simulator imports)."""

import json
import math
import os
import random
import shutil
import time
import warnings
import yaml
from pathlib import Path

from ..evaluation import comparison_condition
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


class EpisodeWindow:
    """Sufficient statistics of fresh, completed, condition-stable episodes only."""

    def __init__(self):
        self.groups = {}

    def add(self, infos, frame, now=None):
        now = time.time() if now is None else now
        stats = infos.get("dexpbt", {})
        if stats.get("schema_version") != 1 or stats.get("protocol") != PROTOCOL:
            return
        for group in stats.get("groups", []):
            try:
                n = int(group["completed_episodes"])
                condition = comparison_condition(group["condition"])
                required = {
                    "curriculum_tolerance",
                    "position_tolerance",
                    "rotation_tolerance",
                    "contact_threshold",
                }
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
                accumulator = self.groups.setdefault(
                    key,
                    {
                        "condition": condition,
                        "protocol": PROTOCOL,
                        "completed_episodes": 0,
                        "success_count": 0.0,
                        "position_error_sum": 0.0,
                        "rotation_error_sum": 0.0,
                        "good_contact_fraction_sum": 0.0,
                        "start_frame": int(frame),
                        "created_at": now,
                    },
                )
                accumulator["completed_episodes"] += n
                for name, value in values.items():
                    accumulator[name] += value
                accumulator["end_frame"] = int(frame)
            except (TypeError, ValueError, KeyError):
                continue

    def ready(self, cfg, frame, now=None):
        now = time.time() if now is None else now
        result = {}
        for key, group in list(self.groups.items()):
            if (
                now - group["created_at"] > cfg.max_score_age_seconds
                or frame - group["start_frame"] > cfg.max_frame_lag
            ):
                del self.groups[key]
                continue
            n = group["completed_episodes"]
            if n < cfg.min_completed_episodes:
                continue
            pos = group["position_error_sum"] / n
            rot = group["rotation_error_sum"] / n
            # Fixed, reward-independent [0,1] geometric progress. Contact can't compensate
            # for bad pose: contact occupancy is diagnostic only, never the ranking signal.
            progress = math.exp(-pos / cfg.position_error_scale) * max(0.0, 1 - rot / math.pi)
            result[key] = {**group, "success_rate": group["success_count"] / n, "progress": progress}
        return result

    def consume(self):
        self.groups.clear()


def eligible(record, cfg, frame, now=None):
    now = time.time() if now is None else now
    return (
        record
        and record.get("schema_version") == 2
        and 0 <= now - record["created_at"] <= cfg.max_score_age_seconds
        and abs(frame - record["frame"]) <= cfg.max_frame_lag
    )


def select_donor(records, own, cfg, now=None):
    """Success-first pairwise selection; zero-success fallback uses normalized pose progress."""
    candidates = []
    for policy, record in records.items():
        if policy == cfg.policy_idx or not eligible(record, cfg, own["frame"], now):
            continue
        for key, ours in own.get("evaluations", {}).items():
            theirs = record.get("evaluations", {}).get(key)
            if not theirs or min(ours["completed_episodes"], theirs["completed_episodes"]) < cfg.min_completed_episodes:
                continue
            stamp = time.time() if now is None else now
            if any(stamp - g["created_at"] > cfg.max_score_age_seconds for g in (ours, theirs)):
                continue
            if any(abs(own["frame"] - g["end_frame"]) > cfg.max_frame_lag for g in (ours, theirs)):
                continue
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
    return max(candidates)[2] if candidates else None
