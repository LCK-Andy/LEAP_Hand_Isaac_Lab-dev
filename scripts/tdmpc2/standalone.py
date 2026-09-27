"""Simulator-free CLI, checkpoint ownership, and metrics for one TD-MPC2 learner."""

import argparse
from pathlib import Path

TASK = "Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0"
GRASP_MOTION_TASK = "Isaac-Grasp-Motion-Cube-LEAP-Direct-v0"
GRASP_TASK = "Isaac-Grasp-Cube-LEAP-Direct-v0"


def build_parser():
    parser = argparse.ArgumentParser(description="Standalone TD-MPC2 on native Dexsuite or Stage A/B LEAP grasp")
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10_000_000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--seed-steps", type=int, help="Random-action transitions; default 5 native episode lengths")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-transition", type=float, default=1.0)
    parser.add_argument("--plan-chunk-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--num-elites", type=int, default=64)
    parser.add_argument("--num-pi-trajs", type=int, default=24)
    parser.add_argument("--planner-iterations", type=int, default=6, help="Base iterations; >=20 actions add two")
    parser.add_argument("--log-every", type=int, default=2560)
    parser.add_argument("--stop-after-vector-steps", type=int, default=0)
    parser.add_argument("--stop-after-updates", type=int, default=0, help="Safe-boundary probe; persist remaining debt")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    return parser


def validate_args(parser, args):
    import math

    if args.task not in (TASK, GRASP_MOTION_TASK, GRASP_TASK):
        parser.error("Only native Dexsuite reorientation and Stage A/B LEAP grasp tasks are supported")
    if args.seed_steps is None:
        args.seed_steps = 1800 if args.task == TASK else 1500
    for name in (
        "num_envs",
        "max_steps",
        "checkpoint_interval",
        "replay_capacity",
        "batch_size",
        "plan_chunk_size",
        "num_samples",
        "num_elites",
        "planner_iterations",
        "log_every",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if min(args.seed_steps, args.stop_after_vector_steps, args.stop_after_updates, args.seed) < 0:
        parser.error("Seed, warmup, and probe bounds must be nonnegative")
    if not math.isfinite(args.updates_per_transition) or args.updates_per_transition <= 0:
        parser.error("Update density must be positive and finite")
    if not 0 <= args.num_pi_trajs < args.num_samples or args.num_elites > args.num_samples:
        parser.error("Invalid planner samples/elites/policy trajectories")
    args.replay_capacity = ((args.replay_capacity + args.num_envs - 1) // args.num_envs) * args.num_envs
    if args.replay_capacity < 3 * args.num_envs:
        parser.error("Replay must hold a full horizon per environment")
    args.workspace = args.workspace.expanduser().resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    return args


def json_write(path, value):
    import json
    import os

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def compact_replay(replay):
    """Snapshot only initialized large rows; full rings remain zero-copy views."""
    state = replay.state_dict()
    state["compact_version"] = 1
    rows = min(state["step"], replay.slots)
    for name in ("obs", "next_obs", "action", "reward", "terminated"):
        if rows < replay.slots:
            # A slice alone serializes its entire backing storage, including uninitialized memory.
            state["buffers"][name] = state["buffers"][name][:rows].clone()
    return state


def restore_replay(replay, state):
    """Expand initialized rows directly into the destination allocation, then validate/load."""
    import torch

    if state.get("compact_version") != 1 or state.get("config") != replay.state_dict()["config"]:
        raise ValueError("Incompatible compact replay checkpoint")
    rows = min(state["step"], replay.slots)
    if not 0 <= rows <= replay.slots or state["size"] != rows * replay.num_envs:
        raise ValueError("Invalid compact replay size")
    buffers = dict(state["buffers"])
    for name in ("obs", "next_obs", "action", "reward", "terminated"):
        source, destination = buffers[name], getattr(replay, name)
        if not isinstance(source, torch.Tensor) or source.shape != (rows, *destination.shape[1:]):
            raise ValueError(f"Invalid compact replay {name} shape")
        if source.dtype != destination.dtype:
            raise ValueError(f"Invalid compact replay {name} dtype")
        if rows < replay.slots:
            destination.zero_()
            destination[:rows].copy_(source)
            buffers[name] = destination
    replay.load_state_dict({**state, "buffers": buffers})


def save_checkpoint(directory, contract, state):
    """Atomic synchronous full-state save, retaining at most two large payloads.

    Remove the older backup BEFORE writing; latest remains intact if saving fails.
    Checkpoint payload is authoritative if a crash interrupts JSON sidecar publication.
    """
    import os
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    latest, previous = directory / "latest.pt", directory / "previous.pt"
    temporary = directory / "latest.pt.tmp"
    previous.unlink(missing_ok=True)
    (directory / "previous.json").unlink(missing_ok=True)
    payload = {"format": "tdmpc2_standalone", "version": 2, "contract": contract, **state}
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if latest.exists():
            os.link(latest, previous)
            if latest.with_suffix(".json").exists():
                os.link(latest.with_suffix(".json"), directory / "previous.json")
        os.replace(temporary, latest)
        json_write(
            latest.with_suffix(".json"),
            {
                "format": payload["format"],
                "version": payload["version"],
                "contract": contract,
                "frame": state["trainer"]["frame"],
                "updates": state["trainer"]["updates"],
                "pending_updates": state["trainer"]["pending_updates"],
                "bytes": latest.stat().st_size,
                "critic": state.get("critic", {}),
                "resume_semantics": "learner/replay/RNG/accounting restored; curriculum restored; physics reset",
            },
        )
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return latest


def load_checkpoint(path, contract):
    """Load trusted local full-state files only; foreign trainer formats are refused."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "tdmpc2_standalone" or payload.get("version") != 2:
        raise ValueError("Not a supported standalone checkpoint")
    if payload.get("contract") != contract:
        raise ValueError("Checkpoint task/algorithm/critic contract differs")
    if not {"agent", "replay", "trainer", "environment"}.issubset(payload):
        raise ValueError("Incomplete standalone checkpoint")
    return payload


class WorkspaceOwner:
    """Hold an advisory exclusive lock through launcher shutdown and all publications."""

    def __init__(self, directory, resume):
        self.directory, self.resume, self.stream = directory, resume, None

    def __enter__(self):
        import fcntl

        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / "writer.lock").open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.stream.close()
            raise RuntimeError("Another process owns this standalone workspace") from exc
        if not self.resume and any(path.name != "writer.lock" for path in self.directory.iterdir()):
            self.stream.close()
            raise ValueError("Refusing a fresh start in a populated workspace; use --checkpoint")
        return self

    def __exit__(self, *args):
        self.stream.close()


class TrainingProgress:
    """Transition-indexed budget plus resumable debt, including the first warmup burst."""

    def __init__(self, seed_steps, updates_per_transition):
        from .schedule import UpdateSchedule

        self.schedule = UpdateSchedule(seed_steps, updates_per_transition)
        self.frame = self.updates = self.pending_updates = 0

    def collect(self, transitions, ready):
        self.pending_updates += self.schedule.collect(transitions, ready)
        self.frame += transitions

    def updated(self):
        if self.pending_updates <= 0:
            raise ValueError("No update debt to consume")
        self.pending_updates -= 1
        self.updates += 1

    def has_work(self, max_steps):
        return self.frame < max_steps or self.pending_updates > 0

    def state_dict(self):
        return {
            "frame": self.frame,
            "updates": self.updates,
            "pending_updates": self.pending_updates,
            "schedule": self.schedule.state_dict(),
        }

    def load_state_dict(self, state):
        for key in ("frame", "updates", "pending_updates"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"Invalid trainer {key}")
        self.schedule.load_state_dict(state["schedule"])
        if state["frame"] != self.schedule.frames:
            raise ValueError("Trainer and schedule frame accounting differ")
        for key in ("frame", "updates", "pending_updates"):
            setattr(self, key, state[key])


def finite_metrics(values):
    """Convert scalar learner diagnostics and reject nonfinite values, never hide them."""
    import math

    result = {}
    for key, value in values.items():
        if hasattr(value, "numel"):
            if value.numel() != 1:
                raise ValueError(f"Expected scalar metric {key}")
            value = value.item()
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite or nonnumeric metric {key}: {value}")
        result[key] = value
    return result


class EpisodeMetrics:
    """Lifetime episode-weighted returns and native fixed-definition success counts.

    Partial trajectories are deliberately discarded on simulator restart. Native
    measurements cover completed episodes, not a held-out evaluation distribution.
    """

    def __init__(self, num_envs, device):
        import torch

        self.returns = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self.lengths = torch.zeros_like(self.returns)
        self.totals = dict.fromkeys(("completed", "return_sum", "length_sum", "measured", "success_count"), 0)

    def observe(self, reward, done, infos):
        self.returns += reward.reshape(-1)
        self.lengths += 1
        count = int(done.sum())
        if count:
            self.totals["completed"] += count
            self.totals["return_sum"] += self.returns[done].sum().item()
            self.totals["length_sum"] += self.lengths[done].sum().item()
            self.returns[done] = self.lengths[done] = 0
        report = infos.get("dexpbt", {})
        measured, success = report.get("completed_episodes", 0), report.get("success_count", 0)
        if not 0 <= success <= measured <= count:
            raise ValueError("Native episode report does not match completed trajectories")
        self.totals["measured"] += measured
        self.totals["success_count"] += success
        finite_metrics(self.totals)

    def values(self):
        t = self.totals
        values = {
            "episode/completed": t["completed"],
            "episode/measured": t["measured"],
            "episode/success_count": t["success_count"],
        }
        if t["completed"]:
            values.update(
                {"episode/return": t["return_sum"] / t["completed"], "episode/length": t["length_sum"] / t["completed"]}
            )
        if t["measured"]:
            values["episode/success_rate"] = t["success_count"] / t["measured"]
        return values

    def state_dict(self):
        return dict(self.totals)

    def load_state_dict(self, state):
        if state.keys() != self.totals.keys():
            raise ValueError("Invalid episode accounting checkpoint")
        self.totals = finite_metrics(state)
        self.returns.zero_()
        self.lengths.zero_()
