"""Import-safe contracts for native Dexsuite FlashSAC (no simulator imports)."""

import argparse
from pathlib import Path

TASK = "Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0"


def build_parser():
    parser = argparse.ArgumentParser(description="Single FlashSAC learner on unchanged native Dexsuite; no PBT")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--num_envs", "--num-envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50_000_896, help="Total native transitions, not vector steps")
    parser.add_argument("--replay-capacity", type=int, default=4_000_000)
    parser.add_argument("--buffer-device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=100_000, help="Stored replay transitions before updates")
    parser.add_argument("--updates-per-vector-step", type=int, default=2)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compile-mode", default="auto", choices=("auto", "default", "max-autotune", "reduce-overhead")
    )
    parser.add_argument("--checkpoint", type=Path, help="Trusted native checkpoint; replay refills on resume")
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000)
    parser.add_argument("--log-every", type=int, default=100_000)
    parser.add_argument("--stop-after-vector-steps", type=int, default=0)
    return parser


def validate_args(parser, args):
    if args.task != TASK:
        parser.error("Only the unchanged native Dexsuite task is supported")
    for name in (
        "num_envs",
        "max_steps",
        "replay_capacity",
        "batch_size",
        "warmup",
        "updates_per_vector_step",
        "checkpoint_interval",
        "log_every",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.seed < 0 or args.stop_after_vector_steps < 0:
        parser.error("Seed and probe limit must be nonnegative")
    if args.replay_capacity < max(args.num_envs, args.warmup):
        parser.error("Replay capacity must accommodate a vector batch and warmup")
    if args.buffer_device != "cpu" and not args.buffer_device.startswith("cuda"):
        parser.error("buffer-device must be cpu or cuda[:index]")
    args.workspace = args.workspace.expanduser().resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    return args


def native_spaces(num_envs):
    import gymnasium as gym
    import numpy as np

    return (
        gym.spaces.Box(-np.inf, np.inf, (num_envs, 1745), dtype=np.float32),
        gym.spaces.Box(-1.0, 1.0, (num_envs, 23), dtype=np.float32),
    )


def native_transition(observation, action, result):
    """Use pre-reset terminal histories, never the next episode's acting observation."""
    return {
        "observation": observation,
        "action": action,
        "reward": result.reward,
        "next_observation": result.next_obs,
        "terminated": result.terminated,
        "truncated": result.truncated,
    }


class VectorProgress:
    """Two gradient updates per ready vector step; no transition-scaled warmup burst."""

    def __init__(self, num_envs, updates_per_vector_step):
        self.num_envs, self.updates_per_vector_step = num_envs, updates_per_vector_step
        self.frame = self.updates = self.vector_steps = 0

    def collect(self, ready):
        self.frame += self.num_envs
        self.vector_steps += 1
        return self.updates_per_vector_step if ready else 0

    def updated(self):
        self.updates += 1

    def state_dict(self):
        return {"frame": self.frame, "updates": self.updates, "vector_steps": self.vector_steps}

    def load_state_dict(self, state):
        for name in ("frame", "updates", "vector_steps"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError(f"Invalid progress {name}")
        if state["frame"] != state["vector_steps"] * self.num_envs:
            raise ValueError("Transition and vector-step accounting disagree")
        self.frame, self.updates, self.vector_steps = state["frame"], state["updates"], state["vector_steps"]


def make_agent_config(args):
    """Parse the actual snapshot YAML into its actual dataclass, resolving only known interpolations."""
    import dataclasses
    import math
    import yaml

    import flash_rl
    from flash_rl.agents.flashSAC.agent import FlashSACConfig
    from flash_rl.buffers import torch_buffer

    if getattr(torch_buffer, "N_STEP_DISCOUNT_VERSION", None) != 1:
        raise RuntimeError("FlashSAC source lacks the validated terminal-aware n-step discount patch")
    root = Path(flash_rl.__file__).resolve().parents[1]
    values = yaml.safe_load((root / "configs/agent/flashSAC.yaml").read_text())
    if values.pop("agent_type") != "flashSAC":
        raise ValueError("Expected the real FlashSAC agent config")
    gradient_budget = math.ceil(args.max_steps / args.num_envs) * args.updates_per_vector_step
    values.update(
        seed=args.seed,
        device_type=args.device,
        buffer_device_type=args.buffer_device,
        buffer_max_length=args.replay_capacity,
        buffer_min_length=args.warmup,
        sample_batch_size=args.batch_size,
        gamma=0.99,
        n_step=3,
        use_compile=args.compile,
        compile_mode=args.compile_mode,
        use_amp=args.amp,
    )
    values["learning_rate_warmup_step"] = int(float(values["learning_rate_warmup_rate"]) * gradient_budget)
    values["learning_rate_decay_step"] = int(float(values["learning_rate_decay_rate"]) * gradient_budget)
    values["critic_min_v"] = -float(values["normalized_G_max"])
    values["critic_max_v"] = float(values["normalized_G_max"])
    for key in ("learning_rate_init", "learning_rate_peak", "learning_rate_end"):
        values[key] = float(values[key])
    if set(values) != {field.name for field in dataclasses.fields(FlashSACConfig)}:
        raise ValueError("Snapshot FlashSAC YAML/dataclass contract differs")
    if any(isinstance(value, str) and "${" in value for value in values.values()):
        raise ValueError("Unresolved FlashSAC configuration interpolation")
    return FlashSACConfig(**values)


def make_agent(args, cfg):
    """Only sampling transport differs: retain upstream zeta noise, return CUDA tensors."""
    import torch

    from flash_rl.agents.flashSAC.agent import FlashSACAgent, _sample_flashsac_actions

    class NativeFlashSACAgent(FlashSACAgent):
        def sample_actions(self, interaction_step, prev_transition, training):
            with torch.no_grad():
                self._cached_noise, actions, self._cur_noise_repeat_count, self._cur_noise_repeat_n = (
                    _sample_flashsac_actions(
                        actor=self._actor,
                        noise=self._cached_noise,
                        observations=prev_transition["next_observation"],
                        temperature=1.0 if training else 0.0,
                        cur_count=self._cur_noise_repeat_count,
                        cur_n=self._cur_noise_repeat_n,
                        zeta_cdf=self._zeta_cdf,
                    )
                )
            return actions

    obs, action = native_spaces(args.num_envs)
    return NativeFlashSACAgent(obs, action, {}, cfg)


def finite_metrics(values):
    import math

    output = {}
    for name, value in values.items():
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite metric {name}: {value}")
        output[name] = value
    return output


def json_write(path, value):
    import json
    import os

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_manifest():
    """Content-address code, including imported dirty snapshot contents, not merely git HEAD."""
    import hashlib

    import flash_rl

    import LEAP_Isaaclab

    repo = Path(__file__).resolve().parents[2]
    flash = Path(flash_rl.__file__).resolve().parents[1]
    leap = Path(LEAP_Isaaclab.__file__).resolve().parent
    files = [("flash/" + str(path.relative_to(flash)), path) for path in (flash / "flash_rl").rglob("*.py")]
    files += [("flash/" + name, flash / name) for name in ("configs/agent/flashSAC.yaml", "scripts/run_isaaclab.sh")]
    files += [
        ("native/" + name, repo / name)
        for name in (
            "scripts/flashsac/train.py",
            "scripts/flashsac/runtime.py",
            "scripts/tdmpc2/env.py",
            "scripts/tdmpc2/standalone.py",
        )
    ]
    for name in ("compat/dexsuite", "tasks/dexsuite_reorient_pbt"):
        files += [("leap/" + str(path.relative_to(leap)), path) for path in (leap / name).rglob("*.py")]
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files)}


def _module(network):
    return getattr(network.network, "_orig_mod", network.network)


def learning_snapshot(agent):
    return {
        name: {key: value.detach().clone() for key, value in _module(getattr(agent, "_" + name)).named_parameters()}
        for name in ("actor", "critic", "target_critic")
    }


def learning_diagnostics(agent, before, require_change=True):
    """Fail closed on missing optimizer coverage, frozen learners, or invalid target tracking."""
    import torch

    output = {}
    for name in ("actor", "critic", "target_critic"):
        network = getattr(agent, "_" + name)
        params = dict(_module(network).named_parameters())
        for value in params.values():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite {name} parameters")
        delta = sum((value.detach() - before[name][key]).square().sum() for key, value in params.items()).sqrt().item()
        output[f"learning/{name}_weight_delta"] = delta
        if require_change and delta <= 0:
            raise RuntimeError(f"{name} parameters did not change after FlashSAC updates")
        if network.optimizer is not None:
            trainable = [value for value in params.values() if value.requires_grad]
            owned = {id(p) for group in network.optimizer.param_groups for p in group["params"]}
            if any(id(p) not in owned for p in trainable):
                raise RuntimeError(f"{name} optimizer omits trainable parameters")
            coverage = sum(bool(network.optimizer.state.get(p)) for p in trainable) / len(trainable)
            output[f"learning/{name}_optimizer_state_coverage"] = coverage
            if require_change and coverage != 1:
                raise RuntimeError(f"{name} optimizer state is incomplete: {coverage}")
    critic, target = _module(agent._critic), _module(agent._target_critic)
    output["learning/target_critic_distance"] = (
        sum((p.detach() - q.detach()).square().sum() for p, q in zip(critic.parameters(), target.parameters()))
        .sqrt()
        .item()
    )
    return finite_metrics(output)


def learner_state(agent):
    """Training state without multi-gigabyte replay; all four networks and schedulers included."""
    import numpy as np
    import random
    import torch

    networks = {}
    for name in ("actor", "critic", "target_critic", "temperature"):
        net = getattr(agent, "_" + name)
        networks[name] = {
            "model": _module(net).state_dict(),
            "optimizer": net.optimizer.state_dict() if net.optimizer else None,
            "scheduler": net.scheduler.state_dict() if net.scheduler else None,
            "update_step": net.update_step,
        }
    normalizer = agent.reward_normalizer
    normalization = (
        None
        if normalizer is None
        else {
            "G_r_max": normalizer.G_r_max,
            "mean": normalizer.G_rms.mean,
            "var": normalizer.G_rms.var,
            "count": normalizer.G_rms.count,
        }
    )
    return {
        "networks": networks,
        "update_step": agent._update_step,
        "normalizer": normalization,
        "amp": agent._grad_scaler.state_dict(),
        "noise": {
            name: getattr(agent, name)
            for name in ("_cached_noise", "_cur_noise_repeat_count", "_cur_noise_repeat_n", "_zeta_cdf")
        },
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }


def restore_learner(agent, state):
    import numpy as np
    import random
    import torch

    for name, saved in state["networks"].items():
        net = getattr(agent, "_" + name)
        _module(net).load_state_dict(saved["model"], strict=True)
        if net.optimizer:
            net.optimizer.load_state_dict(saved["optimizer"])
        if net.scheduler:
            net.scheduler.load_state_dict(saved["scheduler"])
        net.update_step = saved["update_step"]
    agent._update_step = state["update_step"]
    agent._grad_scaler.load_state_dict(state["amp"])
    if agent.reward_normalizer is not None:
        normalizer, saved = agent.reward_normalizer, state["normalizer"]
        normalizer.G_r.zero_()  # physics restarted: do not carry unfinished episode returns
        normalizer.G_r_max = saved["G_r_max"].to(agent._device)
        for name in ("mean", "var", "count"):
            setattr(normalizer.G_rms, name, saved[name].to(agent._device))
    for name, value in state["noise"].items():
        setattr(agent, name, value.to(agent._device))
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"].cpu())
    if state["rng"]["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["rng"]["cuda"]])


FORMAT = "flashsac_native_no_replay_v1"
RESUME_SEMANTICS = (
    "learner/RNG/curriculum/accounting restored; physics reset; replay and n-step queue discarded; warmup refill"
)


def load_checkpoint(path, contract):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT or payload.get("contract") != contract:
        raise ValueError("Foreign or stale FlashSAC checkpoint contract")
    if not {"agent", "trainer", "environment", "episodes"}.issubset(payload):
        raise ValueError("Incomplete FlashSAC checkpoint")
    return payload


def save_checkpoint(directory, contract, state):
    """Atomic file publication with two generations; refuse foreign/stale generations before rotation."""
    import os
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    latest, previous, temporary = (directory / name for name in ("latest.pt", "previous.pt", "latest.pt.tmp"))
    for path in (latest, previous):
        if path.exists():
            load_checkpoint(path, contract)
    if temporary.exists():
        raise ValueError("Stale checkpoint staging file; inspect it before resuming")
    payload = {"format": FORMAT, "contract": contract, "resume_semantics": RESUME_SEMANTICS, **state}
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        previous.unlink(missing_ok=True)
        if latest.exists():
            os.link(latest, previous)
        os.replace(temporary, latest)
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return latest
