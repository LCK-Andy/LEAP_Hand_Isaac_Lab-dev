"""Exact same-task Dexsuite model/normalizer initialization, never a PPO resume.

Only trusted local checkpoints are accepted. Companion PBT state, optimizer,
scaler, environment/curriculum, counters, reward windows and RNG are not loaded.
The separate Stage A -> B warm-start adapter intentionally remains unchanged.
"""

import hashlib
import math
import torch
import yaml
from pathlib import Path

from rl_games.common.algo_observer import AlgoObserver

TASK = "Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0"
GROUP_DIMS = {"policy": (170,), "proprio": (615,), "perception": (960,)}
ENV_CLASS = "DexsuiteReorientPbtEnvCfg"
_ADR = "LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.dexsuite_reorient_pbt_env_cfg:CheckpointableDifficultyScheduler"


def validate_architecture(agent, *, recipient=False):
    """Pin forward/normalization/layout semantics, not mutable PPO hyperparameters."""
    try:
        params = agent["params"]
        network, config, env = params["network"], params["config"], params["env"]
        continuous, mlp = network["space"]["continuous"], network["mlp"]
        groups = env.get("obs_groups")
        if isinstance(groups, list):
            groups = {"obs": groups, "states": groups}
        valid = (
            agent["pbt"]["objective"] == "dexpbt"
            and config["name"] == "dexpbt_reorient"
            and params["algo"]["name"] == "a2c_continuous"
            and params["model"]["name"] == "continuous_a2c_logstd"
            and network["name"] == "actor_critic"
            and network.get("separate") is True
            and network.get("value_activation") in (None, "None")
            and network.get("normalization") in (None, "None")
            and not network.get("central_value", False)
            and not network.get("joint_obs_actions")
            and "rnn" not in network
            and "cnn" not in network
            and mlp["units"] == [512, 256, 128]
            and mlp["activation"] == "elu"
            and not mlp.get("d2rl", False)
            and mlp.get("normalization") in (None, "None")
            and continuous["mu_activation"] in (None, "None")
            and continuous["sigma_activation"] in (None, "None")
            and continuous["fixed_sigma"] is True
            and config["normalize_input"] is True
            and config["normalize_value"] is True
            and not config.get("normalize_rms_advantage", False)
            and not config.get("central_value_config")
            and groups == {"obs": list(GROUP_DIMS), "states": list(GROUP_DIMS)}
            and env.get("concate_obs_groups", True) is True
            and env.get("clip_observations") == 100.0
            and env.get("clip_actions") == 1.0
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("--policy-prior requires complete Dexsuite agent manifests") from error
    if not valid:
        raise ValueError("--policy-prior requires the exact Dexsuite separate ELU 512/256/128 architecture/layout")
    if recipient and (params.get("load_checkpoint", False) or params.get("load_path")):
        raise ValueError("--policy-prior requires a fresh recipient without load_checkpoint/load_path")


def _environment_contract(env):
    """Discard only launch bookkeeping; episode length and physical/MDP knobs remain strict.

    BaseLoader never executes Python tags. Round-tripping the recipient using the
    same YAML dumper as dump_yaml makes tuples/slices/enums comparable to manifests.
    Numeric scalars are normalized so 3 == 3.0 and -0.0 == 0.0. Scene paths use
    Isaac Lab's exact macro expansion: pre-gym configs retain {ENV_REGEX_NS},
    whereas saved post-gym configs contain clone_template.format("[^/]+").
    Existing regex strings are never rewritten or broadened.
    """
    template = env.get("scene", {}).get("clone_cfg", {}).get("clone_template", "/World/envs/env_{}")
    scene_path_fields = {
        "prim_path",
        "filter_prim_paths_expr",
        "sensor_shape_prim_expr",
        "filter_shape_prim_expr",
        "contact_object_prim_path_expr",
    }
    ignored = {
        ("seed",),
        ("log_dir",),
        ("ui_window_class_type",),
        ("scene", "num_envs"),
        ("scene", "clone_cfg", "device"),
        ("sim", "device"),
        ("sim", "log_dir"),
        ("sim", "logging_level"),
        ("sim", "save_logs_to_file"),
        ("sim", "visualizer_cfgs"),
        ("sim", "default_visualizer_cfg"),
    }

    def normalize(value, path=()):
        if isinstance(value, dict):
            return {
                key: normalize(item, (*path, key))
                for key, item in value.items()
                if (*path, key) not in ignored
                and not (path and path[0] == "scene" and key == "cloning_contexts")
                and not (path and path[-1] == "spawn" and key in {"spawn_path", "spawn_paths"})
            }
        if isinstance(value, list):
            return [normalize(item, path) for item in value]
        if isinstance(value, str):
            if path and path[0] == "scene" and path[-1] in scene_path_fields and "{ENV_REGEX_NS}" in value:
                value = value.replace("{ENV_REGEX_NS}", template.format("[^/]+"))
            try:
                number = float(value)
            except ValueError:
                return value
            if math.isfinite(number):
                return number
        return value

    return normalize(env)


def _first_difference(source, target, path="env"):
    if isinstance(source, dict) and isinstance(target, dict):
        if source.keys() != target.keys():
            return f"{path} keys: {sorted(source.keys() ^ target.keys())}"
        # Observation term and action term order determines the network's coordinates.
        if path.startswith(("env.observations", "env.actions")) and list(source) != list(target):
            return f"{path} term order"
        for key in source:
            difference = _first_difference(source[key], target[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(source, list) and isinstance(target, list):
        if len(source) != len(target):
            return f"{path} length"
        for index, (old, new) in enumerate(zip(source, target)):
            difference = _first_difference(old, new, f"{path}[{index}]")
            if difference:
                return difference
        return None
    return None if source == target else path


def validate_source(checkpoint, agent, env):
    """Accept colocated manifests or the ordinary run/{nn,params} layout.

    Multiple complete manifest pairs are ambiguous and refused. Old env.yaml has
    no root configclass/task ID, so the PBT curriculum callable is the identity
    fingerprint, verified against the recipient's native full environment config.
    """
    validate_architecture(agent, recipient=True)
    root = Path(checkpoint).parent
    candidates = [root, root / "params", root.parent / "params"]
    found = list(
        dict.fromkeys(p.resolve() for p in candidates if (p / "agent.yaml").is_file() and (p / "env.yaml").is_file())
    )
    if len(found) != 1:
        raise ValueError("--policy-prior requires one unambiguous agent.yaml/env.yaml manifest pair")
    try:
        source_agent = yaml.safe_load((found[0] / "agent.yaml").read_text())
        source_env = yaml.load((found[0] / "env.yaml").read_text(), Loader=yaml.BaseLoader)
        recipient_env = yaml.load(yaml.dump(env, sort_keys=False), Loader=yaml.BaseLoader)
        validate_architecture(source_agent)
        if source_env["curriculum"]["adr"]["func"] != _ADR or recipient_env["curriculum"]["adr"]["func"] != _ADR:
            raise ValueError("--policy-prior source must identify the Dexsuite PBT training task")
    except (OSError, yaml.YAMLError, KeyError, TypeError) as error:
        raise ValueError("--policy-prior requires complete Dexsuite agent/env manifests") from error
    difference = _first_difference(_environment_contract(source_env), _environment_contract(recipient_env))
    if difference:
        raise ValueError(f"--policy-prior incompatible source/recipient MDP: {difference}")
    return found[0]


def validate_live_layout(env):
    """Check actual built dimensions, before any weights are loaded or training begins."""
    dims = env.observation_manager.group_obs_dim
    if env.action_manager.total_action_dim != 23 or any(
        tuple(dims.get(key, ())) != value for key, value in GROUP_DIMS.items()
    ):
        raise ValueError("--policy-prior requires native 170/615/960 observation groups and 23 actions")


def model_weights(checkpoint, target_state):
    """Strict all-key/all-shape model restore, including critic and both RMS states."""

    def canonical(state):
        result = {}
        for key, value in state.items():
            name = key.removeprefix("_orig_mod.")
            if name in result:
                raise ValueError("Ambiguous compiled model keys")
            result[name] = value
        return result

    source, target = canonical(checkpoint.get("model", {})), canonical(target_state)
    expected = {
        "a2c_network.sigma": (23,),
        "a2c_network.mu.weight": (23, 128),
        "a2c_network.mu.bias": (23,),
        "a2c_network.value.weight": (1, 128),
        "a2c_network.value.bias": (1,),
        "running_mean_std.running_mean": (1745,),
        "running_mean_std.running_var": (1745,),
        "running_mean_std.count": (),
        "value_mean_std.running_mean": (1,),
        "value_mean_std.running_var": (1,),
        "value_mean_std.count": (),
    }
    for branch in ("actor", "critic"):
        for index, (inputs, outputs) in enumerate(((1745, 512), (512, 256), (256, 128))):
            prefix = f"a2c_network.{branch}_mlp.{2 * index}"
            expected[f"{prefix}.weight"] = (outputs, inputs)
            expected[f"{prefix}.bias"] = (outputs,)
    if source.keys() != expected.keys() or target.keys() != expected.keys():
        raise ValueError("--policy-prior requires the full 1745/23 model, critic and both normalizers")
    for key, shape in expected.items():
        value, destination = source[key], target[key]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or tuple(destination.shape) != shape
            or value.dtype != destination.dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Incompatible/nonfinite policy-prior tensor: {key}")
        if key.endswith("running_var") and bool((value < 0).any()):
            raise ValueError(f"Negative normalization variance: {key}")
        if key.endswith(".count") and not bool(value > 0):
            raise ValueError(f"Invalid normalization count: {key}")
    for external, prefix in (("running_mean_std", "running_mean_std"), ("reward_mean_std", "value_mean_std")):
        for key, value in checkpoint.get(external, {}).items():
            name = f"{prefix}.{key}"
            if name not in source or not isinstance(value, torch.Tensor) or not torch.equal(value, source[name]):
                raise ValueError("Conflicting external normalization state")
    return {key: source[key.removeprefix("_orig_mod.")].clone() for key in target_state}


class PolicyPriorObserver(AlgoObserver):
    """Install after ExplorationObserver, before best-history and PBT observers.

    Validate against that observer's actual per-action bounds; in-range raw logstd
    is copied bit-for-bit, not reinitialized. Subsequent PPO updates remain bounded
    by the shared observer. This adapter never calls algo.restore/set_full_state.
    """

    def __init__(self, checkpoint, agent, env, exploration):
        # Keep the lexical path for colocated manifests even if the checkpoint is a symlink.
        self.path = Path(checkpoint).expanduser().absolute()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        validate_source(self.path, agent, env)
        self.exploration = exploration

    def after_init(self, algo):
        if (
            algo.frame != 0
            or algo.epoch_num != 0
            or algo.optimizer.state
            or algo.last_mean_rewards != -1e9
            or any(
                getattr(getattr(algo, name, None), "current_size", 0)
                for name in ("game_rewards", "game_shaped_rewards", "game_lengths")
            )
        ):
            raise ValueError("--policy-prior requires fresh optimizer, counters, best and episode windows")
        if self.exploration.algo is not algo:
            raise ValueError("--policy-prior must load after the shared exploration observer")
        # Match the trusted local full-checkpoint reader at the existing runtime boundary.
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=False)
        weights = model_weights(checkpoint, algo.model.state_dict())
        sigma = next(value for key, value in weights.items() if key.endswith("a2c_network.sigma"))
        low, high = self.exploration._low.cpu(), self.exploration._high.cpu()
        if not bool(((sigma >= low) & (sigma <= high)).all()):
            raise ValueError("Donor sigma exceeds policy-prior bounds; refusing non-exact initialization")
        algo.model.load_state_dict(weights, strict=True)
        with self.path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        print(
            f"[policy-prior] source={self.path} sha256={digest} source_epoch={checkpoint.get('epoch')} "
            f"source_frame={checkpoint.get('frame')} obs=1745 actions=23 model_normalizers_only=true "
            "optimizer=fresh frame=0 epoch=0 best=fresh env_state=not_loaded rng=recipient",
            flush=True,
        )
