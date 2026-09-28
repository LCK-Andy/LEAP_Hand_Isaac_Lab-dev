# Copyright (c) 2018-2023, NVIDIA Corporation
# Modifications Copyright (c) 2026, LCK-Andy.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Isolated adaptation of July-2023 IsaacGymEnvs pbt.py and mutation.py.

Released-source selection retains the upstream bottom-only intervention,
current/best objective distinction, relative margins, donor/self parameter
coin flip and historical <=iteration comparison. Paper Algorithm 1 is a
separate opt-in recipe (not a different physical task).

Intentional runtime changes: atomic publication; strict metric/state/config
contracts; recipient frame AND epoch accounting; recipient-owned random
stream; checkpoint at post-update boundary (upstream uses pre-update
``after_steps``); no silent restore errors. No freshness or physical-difficulty
rank gates from Dexsuite are imported. Historical records are compared at
matching iteration ceilings, including old records of crashed members.
"""

import copy
import json
import math
import os
import random
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .runtime import (
    MUTATIONS,
    SCHEMA,
    TASK,
    UPSTREAM,
    budget_reached,
    restart_command,
    validate_checkpoint,
    validate_mutable,
)


@dataclass
class PbtConfig:
    profile: str = "released_source"
    enabled: bool = True
    member: int = 0
    population: int = 8
    interval: int = 20_000_000
    initial_delay: int = 200_000_000
    adaptation: int | None = None
    mutation_rate: float | None = None
    top_fraction: float = 0.3
    bottom_fraction: float = 0.3
    threshold_std: float = 0.5
    threshold_absolute: float = 0.05
    minimum_runtime_seconds: float = 30.0

    def __post_init__(self):
        if self.profile not in {"released_source", "paper"}:
            raise ValueError("Choose released_source or paper PBT profile")
        if self.adaptation is None:
            self.adaptation = 100_000_000 if self.profile == "released_source" else 50_000_000
        if self.mutation_rate is None:
            self.mutation_rate = 0.15 if self.profile == "released_source" else 0.2
        if not 0 <= self.member < self.population:
            raise ValueError("Invalid population/member identity")
        if self.enabled and self.population < 4:
            raise ValueError("Original selection requires at least four policies; use --no-pbt for a single learner")
        if self.interval <= 0 or min(self.initial_delay, self.adaptation, self.minimum_runtime_seconds) < 0:
            raise ValueError("Invalid PBT cadence")
        if not 0 <= self.mutation_rate <= 1 or min(self.threshold_std, self.threshold_absolute) < 0:
            raise ValueError("Invalid PBT mutation or margins")
        if not 0 < self.top_fraction < 1 or not 0 < self.bottom_fraction < 1:
            raise ValueError("Invalid PBT rank fractions")
        if (
            self.enabled
            and math.ceil(self.top_fraction * self.population) + math.ceil(self.bottom_fraction * self.population)
            > self.population
        ):
            raise ValueError("Top and bottom groups overlap")


def mutate(values, rate, rng):
    """Original function-specific rules, without eval() or additional bounds."""
    validate_mutable(values)
    result = dict(values)
    for key, value in values.items():
        if rng.random() > rate:
            continue
        kind = MUTATIONS[key]
        if kind == "integer":
            result[key] = min(8, max(1, value + (1 if rng.random() < 0.5 else -1)))
            continue
        factor = rng.uniform(1.1, 1.2 if kind == "discount" else 1.5)
        target = 1.0 - value if kind == "discount" else value
        target = target / factor if rng.random() < 0.5 else target * factor
        if kind == "discount":
            target = 1.0 - target
        elif kind == "clip":
            target = max(0.01, min(0.3, target))
        result[key] = target
    validate_mutable(result)
    return result


class ObjectiveWindow:
    """Latest completed objective per environment, after every env has finished.

    The source deliberately keeps the latest per-env value across publications,
    rather than averaging a fresh disjoint batch. Restart creates a fresh window.
    """

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.latest = {}
        self.current = None
        self.best = None

    def add(self, values, done_indices):
        if len(values) != self.num_envs:
            raise ValueError("true_objective must have exactly num_envs entries")
        for index in done_indices:
            if not 0 <= index < self.num_envs:
                raise ValueError("Invalid completed environment index")
            value = float(values[index])
            if not math.isfinite(value):
                raise ValueError("Nonfinite terminal true_objective")
            self.latest[index] = value
        if len(self.latest) == self.num_envs:
            self.current = statistics.fmean(self.latest.values())
            self.best = self.current if self.best is None else max(self.best, self.current)

    def consume(self):
        self.best = None


def select(records, cfg, current, best, rng):
    """Return (weight parent, hyperparameter parent, reason), or no intervention."""
    missing = -1e9
    ranked = sorted(
        ((records[i]["objective"] if i in records else missing, i) for i in range(cfg.population)), reverse=True
    )
    top_n = math.ceil(cfg.top_fraction * cfg.population)
    bottom_n = math.ceil(cfg.bottom_fraction * cfg.population)
    top = [i for _, i in ranked[:top_n]]
    bottom = [i for _, i in ranked[-bottom_n:]]
    if cfg.member not in records:
        raise ValueError("Current member needs a published score before selection")
    available = [score for score, _ in ranked if score > missing]
    if len(available) <= max(2, cfg.population // 2) or any(i not in records for i in top):
        return None, None, "insufficient_population"
    if cfg.profile == "paper":
        if cfg.member in top:
            return None, None, "top_unchanged"
        if cfg.member in bottom:
            donor = rng.choice(top)
            return donor, donor, "bottom_copy_top"
        return cfg.member, cfg.member, "middle_self_mutation"
    if cfg.member not in bottom:
        return None, None, "outside_bottom"
    if best >= min(score for score, _ in ranked[:top_n]):
        return None, None, "best_in_interval_protection"
    candidate = rng.choice(top)
    candidate_score = records[candidate]["objective"]
    trimmed = sorted(available)[int(math.floor(0.2 * len(available))) :]
    delta = candidate_score - current
    replace = delta > cfg.threshold_std * statistics.pstdev(trimmed) and delta > cfg.threshold_absolute * abs(
        candidate_score
    )
    donor = candidate if replace else cfg.member
    param_parent = donor if rng.random() < 0.5 else cfg.member
    return donor, param_parent, "bottom_copy_top" if replace else "margin_self_mutation"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_checkpoint(path, state):
    import torch

    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def population_contract(cfg):
    return {key: value for key, value in asdict(cfg).items() if key != "member"}


def load_population(workspace, cfg, iteration, stamp):
    """Newest historical record NOT newer than recipient; never latest-only."""
    records = {}
    for member in range(cfg.population):
        directory = Path(workspace) / f"{member:03d}"
        for path in sorted(directory.glob("[0-9]*.json"), reverse=True):
            if int(path.stem) > iteration:
                continue
            record = json.loads(path.read_text())
            if record["stamp"] != stamp or record["profile"] != cfg.profile:
                raise ValueError(f"Incompatible task/training profile in population: {path}")
            if record.get("population_contract") != population_contract(cfg):
                raise ValueError(f"Incompatible PBT cadence/configuration in population: {path}")
            if record["member"] != member or record["iteration"] != int(path.stem):
                raise ValueError(f"Corrupt population identity: {path}")
            if record["objective"] is None:
                continue
            if not math.isfinite(record["objective"]):
                raise ValueError(f"Nonfinite population objective: {path}")
            if not Path(record["checkpoint"]).is_file():
                # A donor may be concurrently pruning: try its older history.
                continue
            records[member] = record
            break
    return records


def cleanup_history(directory, records):
    """Source retention: oldest participating iteration minus20; cap25, keep last10."""
    directory = Path(directory)
    oldest = min((record["iteration"] for record in records.values()), default=0)
    paths = sorted(directory.glob("[0-9]*.json"))
    remove = [path for path in paths if int(path.stem) <= oldest - 20]
    paths = [path for path in paths if path not in remove]
    while len(paths) > 25:
        candidates = paths[:-10]
        index = min(
            range(1, len(candidates) - 1), key=lambda i: int(candidates[i + 1].stem) - int(candidates[i - 1].stem)
        )
        remove.append(paths.pop(index))
    for path in remove:
        path.unlink(missing_ok=True)
        path.with_suffix(".pth").unlink(missing_ok=True)


def restart_state(donor, own, mutable, rng_state):
    """Donor training state + recipient identity/counters + selected reward/PPO settings."""
    validate_checkpoint(donor)
    validate_checkpoint(own)
    state = copy.deepcopy(donor)
    recipient = own["dexpbt_original"]
    if donor["dexpbt_original"]["stamp"] != recipient["stamp"]:
        raise ValueError("Cannot copy between incompatible experiments")
    state["frame"], state["epoch"] = own["frame"], own["epoch"]
    state["last_mean_rewards"] = -1e9  # Raw reward bests across different shaping are incomparable.
    meta = copy.deepcopy(recipient)
    meta.update(mutable=mutable, rng_state=rng_state, segment_start_frame=own["frame"])
    meta["lineage"] = [
        *donor["dexpbt_original"].get("lineage", []),
        {"recipient": recipient["member"], "frame": own["frame"], "donor": donor["dexpbt_original"]["member"]},
    ]
    state["dexpbt_original"] = meta
    validate_checkpoint(state)
    return state


class OriginalPbtObserver:
    """Public RL-Games observer, with no inheritance from the Dexsuite engine."""

    def __init__(self, cfg, workspace, mutable, stamp, num_envs, seed, argv, resume=None, agent_config=None):
        self.cfg, self.workspace = cfg, Path(workspace)
        self.mutable, self.stamp, self.argv = dict(mutable), stamp, list(argv)
        self.window = ObjectiveWindow(num_envs)
        self.rng = random.Random(seed + cfg.member)
        if resume is not None:
            self.rng.setstate(resume["rng_state"])
        self.resume = resume
        self.agent_config = copy.deepcopy(agent_config)
        self.started = time.monotonic()
        self.completed_episodes = 0
        self.completed_successes = 0.0
        self.tolerance = None
        self.mutation_count = (resume or {}).get("mutation_count", 0)
        # Upstream adaptation restarts for EVERY process, including manual resume.
        self.segment_start = (resume or {}).get("resume_frame", 0)
        self.iteration = -1
        self.directory = self.workspace / f"{cfg.member:03d}"
        self.directory.mkdir(parents=True, exist_ok=True)

    def before_init(self, base_name, config, experiment_name):
        pass

    def after_init(self, algo):
        self.algo = algo

    def after_clear_stats(self):
        pass

    def after_steps(self):
        pass

    def process_infos(self, infos, done_indices):
        if "true_objective" not in infos:
            raise ValueError("Original DexPBT requires infos['true_objective']; reward fallback is forbidden")
        values = infos["true_objective"]
        if tuple(values.shape) != (self.window.num_envs,):
            raise ValueError("true_objective must be a vector with shape (num_envs,)")
        # Transfer only completed entries rather than all simulator values each step.
        indices = done_indices.reshape(-1).tolist()
        self.window.add(values, indices)
        if indices and "successes" in infos:
            successes = infos["successes"]
            if tuple(successes.shape) != (self.window.num_envs,):
                raise ValueError("successes must have shape (num_envs,)")
            self.completed_episodes += len(indices)
            self.completed_successes += float(successes[indices].sum())
        tolerance = infos.get("scalars", {}).get("success_tolerance")
        if tolerance is not None:
            self.tolerance = float(tolerance)

    def checkpoint_state(self):
        state = self.algo.get_full_state_weights()
        state["dexpbt_original"] = {
            "schema": SCHEMA,
            "task": TASK,
            "upstream": UPSTREAM,
            "physics_parity": "experimental_modern_physx_signed_inertia_mismatch",
            "profile": self.cfg.profile,
            "pbt_config": asdict(self.cfg),
            "agent_config": self.agent_config,
            "member": self.cfg.member,
            "stamp": self.stamp,
            "mutable": dict(self.mutable),
            "rng_state": self.rng.getstate(),
            "segment_start_frame": self.segment_start,
            "mutation_count": self.mutation_count,
            "lineage": (self.resume or {}).get("lineage", []),
        }
        validate_checkpoint(state)
        return state

    def publish(self, terminal=False):
        state = self.checkpoint_state()
        name = "final" if terminal else f"{self.iteration:09d}"
        path = self.directory / f"{name}.pth"
        atomic_checkpoint(path, state)
        record = {
            "member": self.cfg.member,
            "profile": self.cfg.profile,
            "population_contract": population_contract(self.cfg),
            "stamp": self.stamp,
            "iteration": self.iteration,
            "frame": state["frame"],
            "epoch": state["epoch"],
            "objective": self.window.current,
            "mutable": self.mutable,
            "checkpoint": str(path.resolve()),
        }
        atomic_json(path.with_suffix(".json"), record)
        if self.window.current is not None:
            best_path = self.directory / "best.json"
            old = json.loads(best_path.read_text())["objective"] if best_path.exists() else -math.inf
            if self.window.current > old:
                atomic_checkpoint(self.directory / "best.pth", state)
                atomic_json(best_path, {**record, "checkpoint": str((self.directory / "best.pth").resolve())})
        return state

    def after_print_stats(self, frame, epoch_num, total_time):
        current = int(self.algo.frame)  # callback's frame is pre-update in RL-Games1.6.5
        writer = self.algo.writer
        writer.add_scalar("original/completed_episodes", self.completed_episodes, current)
        writer.add_scalar("original/mutation_count", self.mutation_count, current)
        writer.add_scalar(
            "original/member_fps",
            (current - int((self.resume or {}).get("resume_frame", 0))) / max(time.monotonic() - self.started, 1e-6),
            current,
        )
        if self.tolerance is not None:
            writer.add_scalar("original/success_tolerance", self.tolerance, current)
        if self.completed_episodes:
            writer.add_scalar(
                "original/completed_successes_mean", self.completed_successes / self.completed_episodes, current
            )
        if self.window.current is not None:
            writer.add_scalar("original/true_objective", self.window.current, current)
        iteration = current // self.cfg.interval
        terminal = budget_reached(self.algo)
        if terminal:
            self.iteration = iteration
            self.publish(terminal=True)
            return
        if self.iteration < 0:
            self.iteration = int((self.resume or {}).get("resume_frame", 0)) // self.cfg.interval
        if iteration <= self.iteration:
            return
        if time.monotonic() - self.started < self.cfg.minimum_runtime_seconds:
            return
        self.iteration = iteration
        own = self.publish()
        best = self.window.best
        self.window.consume()
        if not self.cfg.enabled:
            cleanup_history(self.directory, {0: {"iteration": 0}})
            return
        if self.window.current is None:
            return
        records = load_population(self.workspace, self.cfg, iteration, self.stamp)
        # A missing member's iteration is0 in the upstream cleanup calculation.
        cleanup_records = {i: records.get(i, {"iteration": 0}) for i in range(self.cfg.population)}
        cleanup_history(self.directory, cleanup_records)
        if current < self.cfg.initial_delay or current - self.segment_start < self.cfg.adaptation:
            return
        donor_id, parent_id, reason = select(records, self.cfg, self.window.current, best, self.rng)
        print(
            "[dexpbt-original] "
            + json.dumps({"frame": current, "member": self.cfg.member, "donor": donor_id, "reason": reason}),
            flush=True,
        )
        if donor_id is None:
            return
        import torch

        mutable = mutate(records[parent_id]["mutable"], self.cfg.mutation_rate, self.rng)
        try:
            donor = torch.load(records[donor_id]["checkpoint"], map_location="cpu", weights_only=False)
        except FileNotFoundError:
            print("[dexpbt-original] donor pruned during transaction; defer until next interval", flush=True)
            return
        restart = restart_state(donor, own, mutable, self.rng.getstate())
        restart["dexpbt_original"]["mutation_count"] = self.mutation_count + 1
        self.algo.writer.add_scalar("original/mutation_count", self.mutation_count + 1, current)
        atomic_json(
            self.directory / "last_intervention.json",
            {
                "frame": current,
                "member": self.cfg.member,
                "donor": donor_id,
                "parameter_parent": parent_id,
                "reason": reason,
                "changed_parameters": {
                    key: [self.mutable[key], value] for key, value in mutable.items() if value != self.mutable[key]
                },
            },
        )
        destination = self.directory / "restart.pth"
        atomic_checkpoint(destination, restart)
        self.algo.writer.flush()
        self.algo.writer.close()
        os.execv(sys.executable, [sys.executable, *restart_command(self.argv, destination)])


class MultiObserver:
    """Ordered composition of the documented public observer hooks only."""

    def __init__(self, observers):
        self.observers = observers

    def before_init(self, *args):
        for observer in self.observers:
            observer.before_init(*args)

    def after_init(self, *args):
        for observer in self.observers:
            observer.after_init(*args)

    def process_infos(self, *args):
        for observer in self.observers:
            observer.process_infos(*args)

    def after_steps(self):
        for observer in self.observers:
            observer.after_steps()

    def after_clear_stats(self):
        for observer in self.observers:
            observer.after_clear_stats()

    def after_print_stats(self, *args):
        for observer in self.observers:
            observer.after_print_stats(*args)
