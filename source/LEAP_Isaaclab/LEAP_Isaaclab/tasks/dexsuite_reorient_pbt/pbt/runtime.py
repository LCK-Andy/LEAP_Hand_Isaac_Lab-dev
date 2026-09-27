"""Lightweight public runtime exports and application-only launch validators."""

# The public integration package eagerly imports RL-Games/torch. Load its
# standalone runtime from the installed distribution before AppLauncher instead.
# No sys.modules aliases or sibling checkout paths are involved.
import importlib.util
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

try:
    _runtime_path = distribution("async-pbt").locate_file("async_pbt/integrations/rl_games/runtime.py")
except PackageNotFoundError as error:
    raise ImportError(
        "Missing async-pbt. Install the pinned framework in this Python environment with: "
        "python -m pip install --no-deps "
        "'async-pbt @ git+https://github.com/LCK-Andy/async-pbt.git@v0.1.0'"
    ) from error
_spec = importlib.util.spec_from_file_location("_leap_public_pbt_runtime", _runtime_path)
_runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runtime)

PREFIX = _runtime.PREFIX
SUPPORTED = _runtime.SUPPORTED
parse_value = _runtime.parse_value
canonical = _runtime.canonical
split_overrides = _runtime.split_overrides
apply_overrides = _runtime.apply_overrides
resolved_seed_argv = _runtime.resolved_seed_argv
restart_command = _runtime.restart_command
resume_metadata = _runtime.resume_metadata
effective_params = _runtime.effective_params
apply_effective_params = _runtime.apply_effective_params
budget_reached = _runtime.budget_reached
install_train_guard = _runtime.install_train_guard
install_restore = _runtime.install_restore

__all__ = [
    "PREFIX",
    "SUPPORTED",
    "parse_value",
    "canonical",
    "split_overrides",
    "apply_overrides",
    "resolved_seed_argv",
    "restart_command",
    "resume_metadata",
    "effective_params",
    "apply_effective_params",
    "budget_reached",
    "install_train_guard",
    "install_restore",
    "validate_warm_start_options",
    "validate_warm_start_architecture",
    "validate_warm_start_config",
    "validate_warm_start_destination",
    "validate_policy_prior_options",
    "validate_policy_prior_destination",
]


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


def validate_policy_prior_options(args, params_overrides=None):
    """Keep same-task policy priors unambiguously separate from every resume path."""
    if not getattr(args, "policy_prior", None):
        return
    if args.task != "Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0":
        raise ValueError("--policy-prior requires the Dexsuite Reorient PBT training task")
    if any(
        getattr(args, name, None) is not None
        for name in (
            "checkpoint",
            "pbt_resume",
            "warm_start",
            "legacy_best_checkpoint",
            "sigma",
        )
    ):
        raise ValueError("--policy-prior cannot be combined with checkpoint/resume/warm-start/legacy-best/sigma")
    if any(canonical(key) in {"params.load_checkpoint", "params.load_path"} for key in params_overrides or {}):
        raise ValueError("--policy-prior cannot be combined with params.load_checkpoint/load_path overrides")


def validate_policy_prior_destination(member_dir, experiment_dir):
    """A new timestamp alone must not inherit member scores, bests, lineage or RNG."""
    for path in (Path(member_dir), Path(experiment_dir)):
        if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
            raise ValueError(f"--policy-prior requires fresh member and experiment directories: {path}")
