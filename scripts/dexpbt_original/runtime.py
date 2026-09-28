# Copyright (c) 2026, LCK-Andy.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure launch/restore boundary for the isolated original DexPBT task.

Reference: IsaacGymEnvs 665c32170d84b4be66722eea405a1e08b6e7f761.
The released_source profile follows its launcher/observer, NOT Algorithm 1
of the paper. The paper profile changes only the PBT recipe; it makes no
claim that the released 117-channel task equals the paper's 110-channel task.
"""

import copy
import hashlib
import json
import math
import os
from pathlib import Path

TASK = "Isaac-DexPBT-Allegro-Kuka-Reorientation-Direct-v0"
UPSTREAM = "665c32170d84b4be66722eea405a1e08b6e7f761"
SCHEMA = 1
ENV_PREFIX = "task.env."
PPO_PREFIX = "train.params.config."
ENV_DEFAULTS = {
    "liftingRewScale": 20.0,
    "liftingBonus": 300.0,
    "liftingBonusThreshold": 0.15,
    "keypointRewScale": 200.0,
    "distanceDeltaRewScale": 50.0,
    "reachGoalBonus": 1000.0,
    "kukaActionsPenaltyScale": 0.003,
    "allegroActionsPenaltyScale": 0.0003,
    "fallDistance": 0.24,
}
PPO_MUTATIONS = {
    "grad_norm": "float",
    "critic_coef": "float",
    "bounds_loss_coef": "float",
    "kl_threshold": "float",
    "e_clip": "clip",
    "mini_epochs": "integer",
    "gamma": "discount",
}
MUTATIONS = {**{ENV_PREFIX + k: "float" for k in ENV_DEFAULTS}, **{PPO_PREFIX + k: v for k, v in PPO_MUTATIONS.items()}}


def mutable_values(agent, env_values):
    """Extract exactly the nine source-active environment and seven PPO keys."""
    if set(env_values) != set(ENV_DEFAULTS):
        raise ValueError("original_mutable_params must contain exactly the nine source-active environment keys")
    values = {ENV_PREFIX + k: env_values[k] for k in ENV_DEFAULTS}
    values.update({PPO_PREFIX + k: agent["params"]["config"][k] for k in PPO_MUTATIONS})
    validate_mutable(values)
    return values


def validate_mutable(values):
    if set(values) != set(MUTATIONS):
        raise ValueError("Missing or unsupported original DexPBT mutation keys")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values.values()):
        raise ValueError("Original DexPBT mutation values must be finite numbers")
    if any(v <= 0 for v in values.values()):
        raise ValueError("Original DexPBT active mutation values must be positive")
    if not 0 < values[PPO_PREFIX + "gamma"] < 1:
        raise ValueError("Mutated gamma left (0, 1); refusing invalid training rather than clipping silently")
    if type(values[PPO_PREFIX + "mini_epochs"]) is not int or not 1 <= values[PPO_PREFIX + "mini_epochs"] <= 8:
        raise ValueError("mini_epochs must be an integer from 1 to 8")
    if not 0.01 <= values[PPO_PREFIX + "e_clip"] <= 0.3:
        raise ValueError("e_clip must be between 0.01 and 0.3")


def apply_mutable(agent, env_cfg, values):
    validate_mutable(values)
    if not hasattr(env_cfg, "original_mutable_params"):
        raise TypeError("Environment config lacks original_mutable_params contract")
    env_cfg.original_mutable_params = {key: values[ENV_PREFIX + key] for key in ENV_DEFAULTS}
    for name in PPO_MUTATIONS:
        agent["params"]["config"][name] = values[PPO_PREFIX + name]


def validate_environment(env, values):
    """Do not let donor state silently overwrite the selected reward parameters."""
    for method in ("get_original_mutable_params", "get_pbt_env_state", "set_pbt_env_state"):
        if not callable(getattr(env, method, None)):
            raise TypeError(f"Original task is missing required {method}()")
    expected = {key: values[ENV_PREFIX + key] for key in ENV_DEFAULTS}
    if env.get_original_mutable_params() != expected:
        raise ValueError("Live original task reward/mutation readback differs from selected parameters")
    validate_tolerance_state(env.get_pbt_env_state())


def validate_tolerance_state(state):
    if not isinstance(state, dict) or set(state) != {"success_tolerance"}:
        raise ValueError("Original PBT state must contain exactly success_tolerance, not physical or rollout state")
    value = state["success_tolerance"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.01 <= value <= 0.075
    ):
        raise ValueError("success_tolerance must be finite and within the original [0.01, 0.075] range")


def restart_command(argv, checkpoint):
    """Keep member/device/profile/budget identity; checkpoint owns mutated values."""
    result, skip = [os.path.abspath(argv[0])], False
    for arg in argv[1:]:
        if skip:
            skip = False
            continue
        if arg.split("=", 1)[0] in {"--checkpoint", "--minibatch-size", "--mini-epochs"}:
            skip = "=" not in arg
            continue
        result.append(arg)
    return [*result, f"--checkpoint={Path(checkpoint).resolve()}"]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Isaac config.to_dict() serializes callable/class entries as strings. Fail
    # on any other object instead of accidentally fingerprinting memory addresses.
    raise TypeError(f"Non-serializable configuration value {type(value).__name__}")


def compatibility_stamp(agent, env_manifest, profile, num_envs):
    """Compare physical/task conditions and PPO layout, not reward hyperparameters."""
    env = copy.deepcopy(env_manifest)
    env.pop("original_mutable_params", None)
    env.pop("seed", None)
    if isinstance(env.get("sim"), dict):
        env["sim"].pop("device", None)
    # Rendering differences do not change the state-only task.
    env.pop("viewer", None)
    asset_hashes = {}
    if "robot_usd_path" in env:
        robot = Path(env["robot_usd_path"])
        for asset in (robot, robot.parent / "manifest.json"):
            with asset.open("rb") as stream:
                asset_hashes[asset.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    fixed_ppo = copy.deepcopy(agent["params"]["config"])
    operational = {
        "name",
        "env_name",
        "train_dir",
        "full_experiment_name",
        "device",
        "device_name",
        "num_actors",
        "max_epochs",
        "max_frames",
        "save_best_after",
        "save_frequency",
        "print_stats",
        "player",
    }
    for key in operational | set(PPO_MUTATIONS):
        fixed_ppo.pop(key, None)
    payload = {
        "fixed_ppo": fixed_ppo,
        "model": agent["params"]["model"],
        "algo": agent["params"]["algo"],
        "env_wrapper": agent["params"].get("env", {}),
        "asset_hashes": asset_hashes,
        "upstream": UPSTREAM,
        "profile": profile,
        "env": env,
        "num_envs": num_envs,
        "network": agent["params"]["network"],
        "horizon_length": agent["params"]["config"]["horizon_length"],
        "seq_length": agent["params"]["config"]["seq_length"],
    }
    return hashlib.sha256(json.dumps(_jsonable(payload), sort_keys=True).encode()).hexdigest()


def validate_checkpoint(state, member=None, profile=None):
    """This entrypoint only accepts trusted, self-describing original-task files."""
    required = {"model", "optimizer", "env_state", "frame", "epoch", "dexpbt_original"}
    if not required <= state.keys():
        raise ValueError("Checkpoint is not a complete original DexPBT training checkpoint")
    meta = state["dexpbt_original"]
    if meta.get("schema") != SCHEMA or meta.get("upstream") != UPSTREAM or meta.get("task") != TASK:
        raise ValueError("Checkpoint task/source/schema mismatch")
    if member is not None and meta["member"] != member:
        raise ValueError("Checkpoint belongs to a different member; use a PBT donor transaction")
    if profile is not None and meta["profile"] != profile:
        raise ValueError("Checkpoint PBT profile mismatch")
    validate_mutable(meta["mutable"])
    validate_tolerance_state(state["env_state"])
    return meta


def restore_agent(algo, state, env, mutable):
    """Restore model, optimizer, normalizers/scaler and env; retain recipient age.

    The checkpoint deliberately does NOT claim to restore a live simulator,
    rollout tensors, or recurrent hidden states. Native training starts fresh
    episodes and RNN state, as in the upstream restart flow.
    """
    validate_checkpoint(state)
    algo.set_full_state_weights(state)
    rates = {float(group["lr"]) for group in algo.optimizer.param_groups}
    if len(rates) != 1 or not all(math.isfinite(rate) and rate > 0 for rate in rates):
        raise ValueError("Checkpoint optimizer must have one finite positive adaptive learning rate")
    algo.last_lr = algo.learning_rate = rates.pop()
    algo.config["learning_rate"] = algo.last_lr
    for name in PPO_MUTATIONS:
        expected = mutable[PPO_PREFIX + name]
        attr = "mini_epochs_num" if name == "mini_epochs" else name
        if getattr(algo, attr) != expected:
            raise ValueError(f"Restored PPO constructor did not apply mutated {name}")
        if name == "kl_threshold" and algo.scheduler.kl_threshold != expected:
            raise ValueError("Adaptive scheduler did not apply mutated kl_threshold")
    validate_environment(env, mutable)
    if env.get_pbt_env_state() != state["env_state"]:
        raise ValueError("Restored tolerance readback differs from checkpoint")


def budget_reached(algo):
    return any(
        limit >= 0 and count >= limit
        for count, limit in ((algo.frame, algo.max_frames), (algo.epoch_num, algo.max_epochs))
    )
