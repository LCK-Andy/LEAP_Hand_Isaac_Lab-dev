"""Simulator-free CLI, resume, and effective PPO parameter boundary."""

import json
import math
import re
import yaml
from pathlib import Path

PREFIX = "agent.params.config."
SUPPORTED = {"learning_rate", "entropy_coef", "grad_norm", "kl_threshold"}


def parse_value(text):
    """Parse YAML values including scientific notation PyYAML treats as text."""
    value = yaml.safe_load(text)
    if isinstance(value, str) and re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", value):
        return float(value)
    if isinstance(value, list):
        return [parse_value(v) if isinstance(v, str) else v for v in value]
    return value


def canonical(key):
    return key.lstrip("+").removeprefix("agent.")


def split_overrides(argv):
    pbt, params, kept = {}, {}, []
    for arg in argv:
        key, sep, value = arg.partition("=")
        key = canonical(key)
        if sep and key.startswith("pbt."):
            pbt[key[4:]] = parse_value(value)
        elif sep and key.startswith("params."):
            params[key] = parse_value(value)
        else:
            kept.append(arg)
    return pbt, params, kept


def apply_overrides(config, overrides):
    for key, value in overrides.items():
        parts = canonical(key).split(".")
        node = config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def resolved_seed_argv(argv, seed):
    """Replace both argparse and Hydra seed forms, including a one-time --seed=-1."""
    result = [argv[0]]
    skip = False
    for arg in argv[1:]:
        if skip:
            skip = False
            continue
        key = arg.split("=", 1)[0]
        if key == "--seed":
            skip = "=" not in arg
            continue
        if canonical(key) == "params.seed":
            continue
        result.append(arg)
    return [*result, f"--seed={int(seed)}"]


def validate_warm_start_options(args):
    """Reject ambiguous Stage A policy initialization before launching the simulator."""
    if not getattr(args, "warm_start", None):
        return
    if args.task != "Isaac-Grasp-Motion-Cube-LEAP-Direct-v0":
        raise ValueError("--warm-start requires the Stage B grasp-motion training task")
    if any(
        getattr(args, name, None) is not None
        for name in ("checkpoint", "pbt_resume", "legacy_best_checkpoint", "sigma")
    ):
        raise ValueError("--warm-start cannot be combined with checkpoint/resume/legacy-best/sigma overrides")


def validate_warm_start_architecture(params):
    """Pin policy/layout semantics for either donor or recipient, independent of resume history."""
    network, config, env = params["network"], params["config"], params["env"]
    continuous, mlp = network["space"]["continuous"], network["mlp"]
    if (
        params["model"]["name"] != "continuous_a2c_logstd"
        or network["name"] != "actor_critic"
        or network.get("separate", False)
        or network.get("value_activation") not in (None, "None")
        or network.get("normalization") not in (None, "None")
        or network.get("central_value", False)
        or network.get("joint_obs_actions")
        or "rnn" in network
        or "cnn" in network
        or mlp["units"] != [256, 128, 64]
        or mlp["activation"] != "elu"
        or mlp.get("d2rl", False)
        or mlp.get("normalization") not in (None, "None")
        or continuous["mu_activation"] not in (None, "None")
        or continuous["sigma_activation"] not in (None, "None")
        or not continuous["fixed_sigma"]
        or not config["normalize_input"]
        or not config["normalize_value"]
        or config.get("central_value_config")
        or env.get("obs_groups") != {"obs": ["policy"], "states": ["policy"]}
        or env.get("clip_observations") != 100.0
        or env.get("clip_actions") != 1.0
    ):
        raise ValueError("--warm-start requires the unchanged Stage A policy architecture/observation layout")


def validate_warm_start_config(params):
    """Require a compatible, fresh recipient; donor manifests use architecture validation only."""
    validate_warm_start_architecture(params)
    if params.get("load_checkpoint", False) or params.get("load_path"):
        raise ValueError("--warm-start requires a fresh recipient without load_checkpoint/load_path")


def validate_warm_start_destination(history_dir, experiment_dir):
    """New objective/age requires new recipient best history, not just a timestamp."""
    for path in (Path(history_dir), Path(experiment_dir)):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"--warm-start requires fresh history and experiment directories: {path}")


def restart_command(original_argv, executable, checkpoint, metadata, params):
    """Retain every original launch choice except superseded restore/mutation tokens."""
    replaced = {canonical(key) for key in params}
    result = [executable, original_argv[0]]
    skip = False
    for arg in original_argv[1:]:
        if skip:
            skip = False
            continue
        key = arg.split("=", 1)[0]
        if key in {"--checkpoint", "--pbt-resume", "--warm-start"}:
            skip = "=" not in arg
            continue
        if canonical(key) in replaced:
            continue
        result.append(arg)
    result += [f"--checkpoint={checkpoint}", f"--pbt-resume={metadata}"]
    result += [f"{canonical(k)}={json.dumps(v)}" for k, v in sorted(params.items())]
    return result


def resume_metadata(checkpoint, explicit=None):
    path = Path(explicit) if explicit else Path(checkpoint).with_suffix(".yaml")
    if not path.exists():
        if explicit:
            raise FileNotFoundError(path)
        return None
    with path.open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("params"), dict):
        raise ValueError(f"Not PBT companion metadata: {path}")
    return data


def effective_params(algo, params):
    result = dict(params)
    for key in result:
        name = key.removeprefix(PREFIX)
        if not key.startswith(PREFIX) or name not in SUPPORTED:
            raise ValueError(f"Unsupported live PBT parameter: {key}")
        value = algo.last_lr if name == "learning_rate" else getattr(algo, name)
        result[key] = float(value)
    return result


def apply_effective_params(algo, params):
    for key, value in params.items():
        name = key.removeprefix(PREFIX)
        if not key.startswith(PREFIX) or name not in SUPPORTED or not math.isfinite(float(value)):
            raise ValueError(f"Unsupported/nonfinite live PBT parameter: {key}")
        value = float(value)
        algo.config[name] = value
        if name == "learning_rate":
            algo.last_lr = algo.learning_rate = value
            for group in algo.optimizer.param_groups:
                group["lr"] = value
        else:
            setattr(algo, name, value)
        if name == "kl_threshold" and hasattr(algo.scheduler, "kl_threshold"):
            algo.scheduler.kl_threshold = value


def budget_reached(algo):
    """Read actual post-update/restored counters, never the lagging callback frame."""
    return any(
        limit != -1 and current >= limit
        for current, limit in (
            (algo.epoch_num, getattr(algo, "max_epochs", -1)),
            (algo.frame, getattr(algo, "max_frames", -1)),
        )
    )


def install_train_guard(algo, terminal_callback):
    """Guard the public train entry AFTER Runner.restore, before native init/update.

    Native rl-games checks limits only after an update. Keep its loop and final
    saves unchanged for active runs, but never enter it for a completed resume.
    """
    train = algo.train

    def guarded(*args, **kwargs):
        if budget_reached(algo):
            terminal_callback(algo.frame, algo.epoch_num, 0.0)
            print(f"[pbt] budget_complete frame={algo.frame} epoch={algo.epoch_num}; no update", flush=True)
            return getattr(algo, "last_mean_rewards", None), algo.epoch_num
        return train(*args, **kwargs)

    algo.train = guarded


def install_restore(algo, params, metadata=None):
    """Compose around public restore: donor state first, effective settings/recipient age last."""
    restore = algo.restore

    def wrapped(checkpoint, *args, **kwargs):
        import tempfile
        import torch

        # RL-Games restores before torch.compile, but saves after compilation.
        # Adapt trusted local full checkpoints at this public boundary, not by
        # globally monkey-patching torch.load or discarding optimizer/env state.
        weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = weights.get("model", {})
        target = algo.model.state_dict() if hasattr(algo.model, "state_dict") else {}
        compiled_source = bool(model) and all(key.startswith("_orig_mod.") for key in model)
        compiled_target = bool(target) and all(key.startswith("_orig_mod.") for key in target)
        if compiled_source != compiled_target:
            weights["model"] = {
                (key.removeprefix("_orig_mod.") if compiled_source else "_orig_mod." + key): value
                for key, value in model.items()
            }
            with tempfile.NamedTemporaryFile(suffix=".pth") as temporary:
                torch.save(weights, temporary.name)
                del weights
                result = restore(temporary.name, *args, **kwargs)
        else:
            del weights
            result = restore(checkpoint, *args, **kwargs)
        apply_effective_params(algo, params)
        if metadata:
            algo.frame = int(metadata.get("recipient_frame", metadata["frame"]))
            algo.epoch_num = int(metadata.get("recipient_epoch", metadata.get("epoch", algo.epoch_num)))
        return result

    algo.restore = wrapped
