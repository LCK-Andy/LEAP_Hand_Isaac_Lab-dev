"""Simulator-free CLI, resume, and effective PPO parameter boundary."""

import json
import math
import re
from pathlib import Path

import yaml

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
        if key in {"--checkpoint", "--pbt-resume"}:
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
