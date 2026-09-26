"""Strict Stage A -> Stage B policy initialization, never a training resume.

The shared actor/critic MLP sees the unchanged 124-channel prefix followed by
24 reference channels. Zero new input weights preserve both heads. Input RMS
means/variances gain 0/1 entries; the shared scalar count and all value RMS
state stay unchanged. Thus the new channels initially inherit the donor RMS
count (they do not have an independent young normalizer).
"""

import hashlib
import math
import torch
import yaml
from pathlib import Path

from rl_games.common.algo_observer import AlgoObserver

from .runtime import validate_warm_start_architecture


def validate_grasp_source(checkpoint):
    """Require colocated Stage A agent/env manifests, as in the epoch-192 download.

    The legacy checkpoint has no embedded architecture or observation schema.
    YAML BaseLoader reads env scalar fields without constructing its Python tags.
    PBT companion metadata is deliberately neither read nor restored.
    """
    root = Path(checkpoint).parent
    try:
        agent = yaml.safe_load((root / "agent.yaml").read_text())
        env = yaml.load((root / "env.yaml").read_text(), Loader=yaml.BaseLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("Warm-start requires colocated Stage A agent.yaml/env.yaml manifests") from error
    if (
        agent.get("pbt", {}).get("objective") != "grasp_hold"
        or agent.get("params", {}).get("config", {}).get("name") != "leap_grasp"
        or env.get("observation_space") != "124"
        or env.get("action_space") != "16"
        or env.get("state_space") != "0"
    ):
        raise ValueError("Warm-start source manifests must identify Stage A 124-observation/16-action grasp_hold")
    validate_warm_start_architecture(agent["params"])


def convert_grasp_motion_weights(checkpoint, target_state):
    """Return model-only weights for the exact 124 -> 148 shared-MLP contract.

    Validate against a freshly built recipient before loading anything. Unknown
    architectures, missing tensors and nonfinite state fail closed. No optimizer,
    scaler, environment, curriculum, RNG, frame or best-history state is returned.
    """
    source = checkpoint.get("model", {})
    if not source or not target_state:
        raise ValueError("Warm-start requires source and target model state")

    def canonical(state):
        result = {}
        for key, value in state.items():
            key = key.removeprefix("_orig_mod.")
            if key in result:
                raise ValueError("Ambiguous compiled model keys")
            result[key] = value
        return result

    source, target = canonical(source), canonical(target_state)
    # Some RL-Games versions also save RMS outside model. Make precedence explicit.
    for external, prefix in (("running_mean_std", "running_mean_std"), ("reward_mean_std", "value_mean_std")):
        for key, value in checkpoint.get(external, {}).items():
            name = f"{prefix}.{key}"
            if name not in source or not torch.equal(value, source[name]):
                raise ValueError("Conflicting external normalization state")
    expected = {
        "a2c_network.sigma": (16,),
        "a2c_network.actor_mlp.0.weight": (256, 124),
        "a2c_network.actor_mlp.0.bias": (256,),
        "a2c_network.actor_mlp.2.weight": (128, 256),
        "a2c_network.actor_mlp.2.bias": (128,),
        "a2c_network.actor_mlp.4.weight": (64, 128),
        "a2c_network.actor_mlp.4.bias": (64,),
        "a2c_network.mu.weight": (16, 64),
        "a2c_network.mu.bias": (16,),
        "a2c_network.value.weight": (1, 64),
        "a2c_network.value.bias": (1,),
        "running_mean_std.running_mean": (124,),
        "running_mean_std.running_var": (124,),
        "running_mean_std.count": (),
        "value_mean_std.running_mean": (1,),
        "value_mean_std.running_var": (1,),
        "value_mean_std.count": (),
    }
    if source.keys() != expected.keys() or target.keys() != expected.keys():
        raise ValueError("Warm-start requires the shared 256/128/64 actor-critic and both normalizers")
    padded = {
        "a2c_network.actor_mlp.0.weight": 0,
        "running_mean_std.running_mean": 0,
        "running_mean_std.running_var": 1,
    }
    result = {}
    for key, shape in expected.items():
        value, destination = source[key], target[key]
        target_shape = (*shape[:-1], 148) if key in padded else shape
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or tuple(destination.shape) != target_shape
            or value.dtype != destination.dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Incompatible/nonfinite warm-start tensor: {key}")
        if key.endswith("running_var") and bool((value < 0).any()):
            raise ValueError(f"Negative normalization variance: {key}")
        if key.endswith(".count") and not bool(value > 0):
            raise ValueError(f"Invalid normalization count: {key}")
        result[key] = torch.nn.functional.pad(value, (0, 24), value=padded[key]) if key in padded else value.clone()
    return {"model": {key: result[key.removeprefix("_orig_mod.")] for key in target_state}}


class WarmStartObserver(AlgoObserver):
    """Load AFTER exploration initializes fresh sigma, BEFORE any Runner restore/compile.

    Refuse out-of-bound donor exploration rather than silently changing its
    distribution. This observer is installed only on a fresh Stage B launch;
    subsequent full Stage B resumes use the normal checkpoint path instead.
    """

    def __init__(self, checkpoint, *, min_std, max_std):
        self.path = Path(checkpoint).expanduser().resolve(strict=True)
        validate_grasp_source(self.path)
        if not 0 < min_std <= max_std or not math.isfinite(max_std):
            raise ValueError("Invalid warm-start exploration bounds")
        self.low, self.high = math.log(min_std), math.log(max_std)

    def after_init(self, algo):
        if algo.frame != 0 or algo.epoch_num != 0 or algo.optimizer.state:
            raise ValueError("Policy-only warm-start requires a fresh algorithm and optimizer")
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        weights = convert_grasp_motion_weights(checkpoint, algo.model.state_dict())
        sigma = next(value for key, value in weights["model"].items() if key.endswith("a2c_network.sigma"))
        if not bool(((sigma >= self.low) & (sigma <= self.high)).all()):
            raise ValueError("Donor sigma exceeds warm-start bounds; refusing non-exact policy initialization")
        algo.set_weights(weights)
        with self.path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        print(
            f"[warm-start] policy_only=true source={self.path} sha256={digest} "
            f"source_epoch={checkpoint.get('epoch')} source_frame={checkpoint.get('frame')} "
            "obs=124->148 optimizer=fresh frame=0 epoch=0 best=fresh env_state=not_loaded",
            flush=True,
        )
