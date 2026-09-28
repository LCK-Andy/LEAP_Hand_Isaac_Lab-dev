# Copyright (c) 2026, LCK-Andy.
# SPDX-License-Identifier: BSD-3-Clause
"""Train a separate original-DexPBT recurrent PPO learner, never Dexsuite.

Example (one of eight separately launched members sharing an absolute workspace):
  python scripts/dexpbt_original/train.py --headless --workspace /tmp/original-pbt --member 0

``released_source`` reproduces July2023's bottom-only PBT recipe. ``paper``
selects Algorithm1's top/middle/bottom recipe, .2 mutation and50M adaptation;
it does NOT turn the released117-channel task into the paper110-channel task.
Checkpoints under WORKSPACE/000/{iteration,final,best,restart}.pth are the only
supported full resumes. Conventional RL-Games raw checkpoints lack PBT state.
"""

import argparse
import copy
import importlib.metadata
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Direct-file execution requires the repository path bootstrap before local imports.
from scripts.dexpbt_original.pbt import MultiObserver, OriginalPbtObserver, PbtConfig, atomic_json, mutate  # noqa: E402
from scripts.dexpbt_original.runtime import (  # noqa: E402
    ENV_DEFAULTS,
    TASK,
    UPSTREAM,
    apply_mutable,
    budget_reached,
    compatibility_stamp,
    mutable_values,
    restore_agent,
    validate_checkpoint,
    validate_environment,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default=TASK, choices=[TASK])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--allow-physics-mismatch",
        action="store_true",
        help="Explicitly allow experimental PhysX port: signed inertia differs from original Gym",
    )
    parser.add_argument("--member", type=int, default=0)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument(
        "--profile",
        choices=["released_source", "paper"],
        default="released_source",
        help="paper changes PBT recipe only, not task/observation semantics",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--checkpoint", type=Path, help="Trusted original-task full checkpoint; no raw/prior restore")
    parser.add_argument("--no-pbt", action="store_true", help="Standalone original recurrent PPO, no initial mutations")
    parser.add_argument("--interval", type=int, default=20_000_000)
    parser.add_argument("--initial-delay", type=int, default=200_000_000)
    parser.add_argument("--adaptation", type=int)
    parser.add_argument("--minimum-runtime-seconds", type=float, default=30.0)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--minibatch-size", type=int, help="Execution checks only; source default32768")
    parser.add_argument("--mini-epochs", type=int, help="Initial PPO setting for an execution check, before mutations")
    return parser


def main():
    # Nothing from LEAP/pxr/Isaac tasks may be imported until AppLauncher exists.
    parser = build_parser()
    preflight, _ = parser.parse_known_args()
    if not preflight.allow_physics_mismatch:
        parser.error(
            "Modern PhysX cannot preserve source signed inertia; use --allow-physics-mismatch only for the experimental port, not faithful physical reproduction"
        )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    if "--headless" not in parser._option_string_actions:
        parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        parser.error("One process per member; distributed PPO is not supported")
    if args.seed < 0:
        parser.error("Use a resolved nonnegative seed so process restarts retain identity")
    if importlib.metadata.version("rl-games") != "1.6.5":
        parser.error("This isolated adapter is audited for rl-games1.6.5; do not silently change trainer version")
    cfg = PbtConfig(
        profile=args.profile,
        enabled=not args.no_pbt,
        member=args.member,
        population=args.population,
        interval=args.interval,
        initial_delay=args.initial_delay,
        adaptation=args.adaptation,
        minimum_runtime_seconds=args.minimum_runtime_seconds,
    )
    args.workspace = args.workspace.expanduser().resolve()
    member_dir = args.workspace / f"{args.member:03d}"
    if not args.checkpoint and member_dir.exists() and any(member_dir.iterdir()):
        parser.error("Refusing fresh initialization in a populated member directory; use --checkpoint")
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    launcher = AppLauncher(args)
    app = launcher.app
    exit_code = 0
    try:
        train(args, cfg)
    except KeyboardInterrupt:
        exit_code = 130
    except BaseException:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            app.close(exit_code=exit_code)
        except SystemExit as exc:
            exit_code = exit_code or exc.code or 0
    return exit_code


def train(args, cfg):
    import gymnasium as gym
    import torch

    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    import LEAP_Isaaclab.tasks  # noqa: F401

    agent = copy.deepcopy(load_cfg_from_registry(TASK, "rl_games_cfg_entry_point"))
    saved = None
    metadata = None
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        metadata = validate_checkpoint(saved, args.member, args.profile)
        # Restore effective architecture/config, not today's potentially changed defaults.
        if not isinstance(metadata.get("agent_config"), dict):
            raise ValueError("Checkpoint is missing the effective original PPO configuration")
        agent = copy.deepcopy(metadata["agent_config"])
        for key, value in metadata["pbt_config"].items():
            if getattr(cfg, key) != value:
                raise ValueError(f"Full resume PBT configuration mismatch for {key}")
    if metadata and agent["params"]["seed"] != args.seed:
        raise ValueError("Full resume seed must match checkpoint recipient seed")
    num_envs = args.num_envs if args.num_envs is not None else agent["params"]["config"]["num_actors"]
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=num_envs)
    env_cfg.seed = args.seed
    agent["params"]["seed"] = args.seed
    config = agent["params"]["config"]
    config.update(
        device=args.device,
        device_name=args.device,
        num_actors=env_cfg.scene.num_envs,
        train_dir=str(args.workspace / "rl_games"),
        full_experiment_name=f"member_{args.member:03d}",
    )
    if config.get("torch_compile") is not False or config.get("multi_gpu"):
        raise ValueError("Original adapter requires eager, single-process recurrent PPO")
    for flag, name in (
        (args.max_epochs, "max_epochs"),
        (args.max_frames, "max_frames"),
        (args.minibatch_size, "minibatch_size"),
        (args.mini_epochs, "mini_epochs"),
    ):
        if flag is not None:
            if name in {"minibatch_size", "mini_epochs"} and saved is not None:
                raise ValueError(f"Full resume cannot override {name}; selected checkpoint configuration wins")
            config[name] = flag
    horizon = config["horizon_length"]
    batch = env_cfg.scene.num_envs * horizon
    if config["seq_length"] != horizon or batch % config["minibatch_size"] or config["minibatch_size"] % horizon:
        raise ValueError("Recurrent PPO requires seq_length=horizon16 and minibatch dividing num_envs*horizon")
    values = mutable_values(agent, {**ENV_DEFAULTS, **env_cfg.original_mutable_params})
    initial_rng = random.Random(args.seed + args.member)
    if metadata:
        values = metadata["mutable"]
        metadata = {**metadata, "resume_frame": int(saved["frame"])}
    elif cfg.enabled:
        # Every member including0 gets the source Bernoulli initial mutation.
        values = mutate(values, cfg.mutation_rate, initial_rng)
    apply_mutable(agent, env_cfg, values)
    stamp = compatibility_stamp(agent, env_cfg.to_dict(), cfg.profile, env_cfg.scene.num_envs)
    if metadata and metadata["stamp"] != stamp:
        raise ValueError("Resume environment/layout differs from saved experiment stamp")
    env = gym.make(TASK, cfg=env_cfg)
    try:
        base = env.unwrapped
        validate_environment(base, values)
        wrapped = RlGamesVecEnvWrapper(
            env, args.device, 10.0, 1.0, obs_groups={"obs": ["policy"], "states": ["policy"]}
        )

        class StatefulEnv(RlGamesGpuEnv):
            def get_env_state(self):
                return self.env.unwrapped.get_pbt_env_state()

            def set_env_state(self, state):
                if state is None:
                    raise ValueError("Missing original DexPBT tolerance/curriculum state")
                self.env.unwrapped.set_pbt_env_state(state)

        vecenv.register(
            "OriginalDexPBT", lambda config_name, num_actors, **kw: StatefulEnv(config_name, num_actors, **kw)
        )
        env_configurations.register(
            "dexpbt_original", {"vecenv_type": "OriginalDexPBT", "env_creator": lambda **kw: wrapped}
        )
        observer = OriginalPbtObserver(
            cfg,
            args.workspace,
            values,
            stamp,
            base.num_envs,
            args.seed,
            [str(Path(__file__).resolve()), *sys.argv[1:]],
            metadata,
            agent_config=agent,
        )
        if metadata is None:
            observer.rng.setstate(initial_rng.getstate())
        manifest = {
            "task": TASK,
            "upstream": UPSTREAM,
            "profile": cfg.profile,
            "paper_recipe_only": cfg.profile == "paper",
            "physics_parity": "FAILED: modern PhysX rejects original signed inertia components",
            "allow_physics_mismatch": args.allow_physics_mismatch,
            "stamp": stamp,
            "mutable": values,
            "inert_source_keys": ["task.env.fallDistance"],
            "restore_scope": "model/optimizer/normalizers/scaler/tolerance; fresh curriculum timers, episodes and RNN states",
            "runtime_adaptations": ["post_update_publication", "atomic_transactions", "recipient_frame_and_epoch"],
        }
        atomic_json(observer.directory / "manifest.json", manifest)
        dump_yaml(str(observer.directory / "agent.yaml"), agent)
        dump_yaml(str(observer.directory / "env.yaml"), env_cfg)
        print("[dexpbt-original] " + json.dumps(manifest, sort_keys=True), flush=True)
        runner = Runner(MultiObserver([IsaacAlgoObserver(), observer]))
        runner.load(agent)
        # Runner's public factories preserve native PPO; explicit restoration avoids
        # global patches and its default compilation path. The profile is eager.
        algo = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
        if saved is not None:
            restore_agent(algo, saved, base, values)
            del saved
        if not budget_reached(algo):
            algo.train()
        observer.iteration = int(algo.frame) // cfg.interval
        observer.publish(terminal=True)
        algo.writer.flush()
        algo.writer.close()
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
