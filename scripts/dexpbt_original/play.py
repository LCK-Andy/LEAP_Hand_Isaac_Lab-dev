# Copyright (c) 2026, LCK-Andy.
# SPDX-License-Identifier: BSD-3-Clause
"""Play/evaluate an isolated original-DexPBT checkpoint with native LSTM state.

Policy-only evaluation at fixed1cm tolerance by default; donor curriculum and
training optimizer are not restored. Reports Eq5 terminal objective, not an
unjustified claim of paper-level acquisition. Source playback is stochastic;
--deterministic-policy is an explicit evaluation override.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Direct-file execution requires the repository path bootstrap before local imports.
from scripts.dexpbt_original.runtime import TASK, apply_mutable, validate_checkpoint, validate_environment  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--allow-physics-mismatch",
        action="store_true",
        help="Allow experimental modern PhysX port, which cannot reproduce original signed inertia",
    )
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--deterministic-policy", action="store_true")
    parser.add_argument("--output", type=Path, help="Write the evaluation JSON report")
    return parser


def main():
    parser = build_parser()
    preflight, _ = parser.parse_known_args()
    if not preflight.allow_physics_mismatch:
        parser.error(
            "Modern PhysX cannot preserve source signed inertia; pass --allow-physics-mismatch only for experimental port evaluation"
        )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    if "--headless" not in parser._option_string_actions:
        parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.num_envs <= 0 or not 0.01 <= args.tolerance <= 0.075:
        parser.error("Require positive steps/envs and tolerance between0.01 and0.075")
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    launcher = AppLauncher(args)
    code = 0
    try:
        evaluate(args, launcher.app)
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        try:
            launcher.app.close(exit_code=code)
        except SystemExit as exc:
            code = code or exc.code or 0
    return code


def evaluate(args, app):
    import gymnasium as gym
    import torch

    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import LEAP_Isaaclab.tasks  # noqa: F401

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = validate_checkpoint(checkpoint)
    agent = copy.deepcopy(metadata["agent_config"])
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    # Keep both endpoints equal to disable training curriculum during evaluation.
    env_cfg.initial_success_tolerance = args.tolerance
    env_cfg.target_success_tolerance = args.tolerance
    apply_mutable(agent, env_cfg, metadata["mutable"])
    agent["params"]["seed"] = args.seed
    agent["params"]["config"].update(device=args.device, device_name=args.device, num_actors=args.num_envs)
    env = gym.make(TASK, cfg=env_cfg)
    try:
        validate_environment(env.unwrapped, metadata["mutable"])
        if env.unwrapped.success_tolerance != args.tolerance:
            raise ValueError("Evaluation tolerance was not applied to the live task")
        wrapped = RlGamesVecEnvWrapper(
            env, args.device, 10.0, 1.0, obs_groups={"obs": ["policy"], "states": ["policy"]}
        )
        vecenv.register(
            "OriginalDexPBT", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw)
        )
        env_configurations.register(
            "dexpbt_original", {"vecenv_type": "OriginalDexPBT", "env_creator": lambda **kw: wrapped}
        )
        runner = Runner()
        runner.load(agent)
        player = runner.create_player()
        player.model.load_state_dict(checkpoint["model"])
        if player.normalize_input and "running_mean_std" in checkpoint:
            player.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        player.model.eval()
        player.reset()
        obs = wrapped.reset()
        player.get_batch_size(obs["obs"] if isinstance(obs, dict) else obs, 1)
        if not player.is_rnn:
            raise ValueError("Original DexPBT evaluation requires a recurrent policy")
        player.init_rnn()
        episodes, total_objective = 0, 0.0
        for step in range(args.steps):
            if not app.is_running():
                break
            with torch.inference_mode():
                action = player.get_action(player.obs_to_torch(obs), is_deterministic=args.deterministic_policy)
                obs, _, dones, infos = wrapped.step(action)
                terminal = dones.bool().reshape(-1)
                objective = infos["true_objective"][terminal]
                if not torch.isfinite(objective).all():
                    raise ValueError("Nonfinite evaluation objective")
                episodes += int(terminal.sum())
                total_objective += float(objective.sum())
                for hidden in player.states:
                    hidden[:, terminal, :] = 0.0
        if env.unwrapped.success_tolerance != args.tolerance:
            raise ValueError("Evaluation curriculum changed the requested fixed tolerance")
        with args.checkpoint.open("rb") as stream:
            checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        mean_objective = total_objective / episodes if episodes else None
        report = {
            "task": TASK,
            "physics_parity": "FAILED: modern PhysX rejects original signed inertia components",
            "allow_physics_mismatch": args.allow_physics_mismatch,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "upstream": metadata["upstream"],
            "training_profile": metadata["profile"],
            "training_config_stamp": metadata["stamp"],
            "seed": args.seed,
            "num_envs": args.num_envs,
            "evaluation": "policy_only_fixed_tolerance",
            "tolerance": args.tolerance,
            "success_threshold": "max keypoint error <= 1.5 * tolerance (source predicate)",
            "deterministic": args.deterministic_policy,
            "completed_episodes": episodes,
            "mean_terminal_objective": mean_objective,
            "mean_completed_successes": mean_objective - 1.0 if mean_objective is not None else None,
            "steps": step + 1,
        }
        print(json.dumps(report, sort_keys=True), flush=True)
        if args.output:
            from scripts.dexpbt_original.pbt import atomic_json

            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output, report)
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
