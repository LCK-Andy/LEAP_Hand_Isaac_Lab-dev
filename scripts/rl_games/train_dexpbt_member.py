# Copyright (c) 2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Train one isolated DexPBT member; preserve launch identity across exec/resume."""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

PROCESS_STARTED_AT = time.time()

# Load only the pure helper pre-app: importing the task package here imports pxr
# and breaks pip Isaac Sim startup. Preserve argv BEFORE argparse/Hydra consume it.
ORIGINAL_ARGV = [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
_RUNTIME = (
    Path(__file__).resolve().parents[2] / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/"
    "dexsuite_reorient_pbt/pbt/runtime.py"
)
_spec = importlib.util.spec_from_file_location("dexpbt_runtime", _RUNTIME)
runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runtime)
pbt_cli_overrides, params_cli_overrides, remaining = runtime.split_overrides(sys.argv[1:])

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--warm-start", type=str, default=None, help="Stage A -> B policy-only initialization; NOT resume")
parser.add_argument(
    "--legacy-best-checkpoint",
    type=str,
    default=None,
    help="Explicit recipient conventional best from an old log directory (not donor/latest); seeds missing history only",
)
parser.add_argument(
    "--pbt-resume", type=str, default=None, help="Companion PBT metadata (auto-detected beside checkpoint)"
)
parser.add_argument("--sigma", type=float, default=None, help="RL-Games raw log-standard-deviation override (bounded)")
parser.add_argument("--max_iterations", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
# EA still accepts the launcher kwarg but no longer registers this legacy CLI flag.
if "--headless" not in parser._option_string_actions:
    parser.add_argument("--headless", action="store_true")
args_cli, hydra_args = parser.parse_known_args(remaining)
try:
    runtime.validate_warm_start_options(args_cli)
except ValueError as error:
    parser.error(str(error))
if args_cli.distributed or int(os.environ.get("WORLD_SIZE", 1)) != 1:
    parser.error("DexPBT supports one process per member; distributed PPO restarts are unsupported")
if args_cli.pbt_resume and not args_cli.checkpoint:
    parser.error("--pbt-resume requires --checkpoint")
args_cli.original_argv = ORIGINAL_ARGV
args_cli.process_started_at = PROCESS_STARTED_AT
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib
import math
import random
import yaml
from datetime import datetime

from omegaconf import OmegaConf
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import LEAP_Isaaclab.tasks  # noqa: F401
from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt import MultiObserver, PbtAlgoObserver
from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt.best_checkpoint import BestCheckpointObserver
from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt.exploration import ExplorationObserver
from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt.pbt import initialize_population
from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt.pbt_cfg import PbtCfg
from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt.warm_start import WarmStartObserver


class StatefulRlgEnv(RlGamesGpuEnv):
    """The base IVecEnv checkpoint hooks are no-ops; forward explicitly."""

    def get_env_state(self):
        return self.env.unwrapped.get_env_state()

    def set_env_state(self, state):
        if state is not None:
            self.env.unwrapped.set_env_state(state)


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # One plain config, finalized before env creation, dump, observer, and Runner.load.
    agent_cfg = OmegaConf.to_container(OmegaConf.create(agent_cfg), resolve=True)
    entry = gym.spec(args_cli.task).kwargs["rl_games_cfg_entry_point"]
    module_name, filename = entry.split(":")
    yaml_path = Path(importlib.import_module(module_name).__file__).parent / filename
    with yaml_path.open() as stream:
        agent_cfg["pbt"] = yaml.safe_load(stream).get("pbt", {})
    runtime.apply_overrides(agent_cfg, {f"pbt.{key}": value for key, value in pbt_cli_overrides.items()})
    cfg = PbtCfg(**agent_cfg["pbt"])
    if cfg.fixed_evaluation_adr_fraction is not None:
        raise ValueError("Heldout fixed evaluation must run separately, not inside the training member")

    warm_start_path = str(Path(args_cli.warm_start).expanduser().resolve(strict=True)) if args_cli.warm_start else None
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    metadata = runtime.resume_metadata(resume_path, args_cli.pbt_resume) if resume_path else None
    args_cli.pbt_resume_metadata = metadata
    if metadata:
        if metadata.get("policy_idx", cfg.policy_idx) != cfg.policy_idx:
            raise ValueError("Resume metadata belongs to a different member; use an explicit donor restart transaction")
        runtime.apply_overrides(agent_cfg, metadata["params"])
    elif resume_path:
        print("[pbt] No companion metadata; explicit/base hyperparameters will override restored optimizer settings")
    # Explicit command-line choices win over a companion, including generated mutations.
    runtime.apply_overrides(agent_cfg, params_cli_overrides)
    config = agent_cfg["params"]["config"]
    if warm_start_path:
        runtime.validate_warm_start_config(agent_cfg["params"])
        if cfg.objective != "grasp_motion" or env_cfg.observation_space != 148 or env_cfg.action_space != 16:
            raise ValueError("--warm-start requires the 148-observation/16-action grasp_motion contract")
        if config["name"] != "leap_grasp_motion":
            raise ValueError("--warm-start requires the separate leap_grasp_motion log namespace")
    if config.get("multi_gpu", False):
        raise ValueError("multi_gpu is unsupported for DexPBT")
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 2**31 - 1)
    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = args_cli.seed
    # Resolve -1 once and persist resolved seed in restart identity.
    if agent_cfg["params"]["seed"] == -1:
        agent_cfg["params"]["seed"] = random.randint(0, 2**31 - 1)
    args_cli.original_argv = runtime.resolved_seed_argv(ORIGINAL_ARGV, agent_cfg["params"]["seed"])
    if args_cli.max_iterations is not None:
        config["max_epochs"] = args_cli.max_iterations
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        config["device"] = config["device_name"] = args_cli.device
    env_cfg.seed = agent_cfg["params"]["seed"]
    if resume_path:
        agent_cfg["params"].update(load_checkpoint=True, load_path=resume_path)
    initialize_population(agent_cfg, resumed=bool(resume_path))

    log_root = Path("logs") / "rl_games" / config["name"]
    if cfg.enabled and cfg.directory:
        log_root = Path(cfg.directory) / log_root
    log_root = log_root.resolve()
    log_dir = config.get("full_experiment_name") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config["train_dir"] = str(log_root)
    config["full_experiment_name"] = log_dir
    history_dir = log_root / cfg.workspace / f"{cfg.policy_idx:03d}" / "best"
    if warm_start_path:
        runtime.validate_warm_start_destination(history_dir, log_root / log_dir)
    print(f"[INFO] Logging member {cfg.policy_idx}/{cfg.num_policies} to {log_root / log_dir}")

    rl_device = config["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concatenate = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if cfg.objective == "allegro_goals":
        if args_cli.task != "Isaac-Repose-Cube-Allegro-Direct-v0":
            raise ValueError("allegro_goals requires the unmodified official Allegro Direct task")
        from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt.allegro import attach_measurement

        attach_measurement(env.unwrapped)
    elif cfg.objective == "contact_goals":
        if args_cli.task != "Isaac-Contact-Cube-LEAP-Direct-v0":
            raise ValueError("contact_goals requires the native LEAP contact Direct task")
        from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt.contact import attach_measurement

        attach_measurement(env.unwrapped)
    exploration = None
    try:
        if cfg.objective == "grasp_hold":
            from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt.grasp import validate_grasp_environment

            validate_grasp_environment(args_cli.task, env.unwrapped)
        elif cfg.objective == "grasp_motion":
            from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt.grasp_motion import validate_motion_environment

            validate_motion_environment(args_cli.task, env.unwrapped)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        try:
            env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concatenate)
        except TypeError:
            groups = {"obs": obs_groups, "states": obs_groups} if isinstance(obs_groups, list) else obs_groups
            env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, groups)
        vecenv.register(
            "IsaacRlgWrapper", lambda config_name, num_actors, **kw: StatefulRlgEnv(config_name, num_actors, **kw)
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
        config["num_actors"] = env.unwrapped.num_envs
        exploration_options = dict(config.get("exploration", {}))
        exploration_options.setdefault("action_clip", clip_actions)
        exploration = ExplorationObserver(**exploration_options)
        # Commit raw-reward bests before PBT's callback can exec. Instance restore
        # wrappers compose as PBT -> best-history -> exploration -> base restore.
        observers = [IsaacAlgoObserver(), exploration]
        if warm_start_path:
            observers.append(
                WarmStartObserver(warm_start_path, min_std=exploration.min_std, max_std=exploration.max_std)
            )
        observers.append(BestCheckpointObserver(metadata, history_dir, args_cli.legacy_best_checkpoint))
        # Restore semantics remain active for disabled-PBT diagnostic ablations.
        observers.append(PbtAlgoObserver(agent_cfg, args_cli))
        dump_yaml(str(log_root / log_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(log_root / log_dir / "params" / "agent.yaml"), agent_cfg)
        runner = Runner(MultiObserver(observers))
        runner.load(agent_cfg)
        runner.reset()
        options = {"train": True, "play": False, "sigma": args_cli.sigma}
        if resume_path:
            options["checkpoint"] = resume_path
        runner.run(options)
    finally:
        if exploration is not None:
            exploration.close()
        env.close()


if __name__ == "__main__":
    import traceback

    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        exit_code = 130
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        # Some Kit versions raise SystemExit(0) during close; don't hide a training
        # failure behind a successful shutdown status or suppress its traceback.
        try:
            simulation_app.close(exit_code=exit_code)
        except SystemExit as exc:
            exit_code = exit_code or exc.code or 0
    sys.exit(exit_code)
