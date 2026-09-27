#!/usr/bin/env python3
"""Native EA Dexsuite with the real FlashSAC learner, never its obsolete environment wrapper."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.flashsac.runtime import build_parser, validate_args
from scripts.tdmpc2.standalone import WorkspaceOwner


def train(args, simulation_app):
    # No task, pxr, or FlashSAC imports before AppLauncher.
    import dataclasses
    import gymnasium as gym
    import json
    import math
    import numpy as np
    import os
    import random
    import signal
    import time
    import torch
    from torch.utils.tensorboard import SummaryWriter

    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.utils import parse_env_cfg

    import LEAP_Isaaclab.tasks  # noqa: F401
    from scripts.flashsac.runtime import (
        RESUME_SEMANTICS,
        VectorProgress,
        finite_metrics,
        json_write,
        learner_state,
        learning_diagnostics,
        learning_snapshot,
        load_checkpoint,
        make_agent,
        make_agent_config,
        native_transition,
        restore_learner,
        save_checkpoint,
        source_manifest,
    )
    from scripts.tdmpc2.env import DexsuiteVectorEnv
    from scripts.tdmpc2.standalone import EpisodeMetrics

    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        print(f"[flashsac] Signal {signum}; checkpoint after the current vector/update boundary", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    agent_cfg = make_agent_config(args)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed, cfg.compute_final_obs = args.seed, False
    if not (
        cfg.episode_length_s == 6
        and cfg.decimation == 2
        and math.isclose(cfg.sim.dt, 1 / 120)
        and cfg.scene.robot.spawn.articulation_props[0].solver_position_iteration_count == 32
    ):
        raise ValueError("Native 360-step/solver32 physics contract changed")
    if not math.isclose(cfg.actions.action.scale, 0.1):
        raise ValueError("Native relative joint action scale must remain 0.1")
    native_env = gym.make(args.task, cfg=cfg)
    writer = None
    try:
        env = DexsuiteVectorEnv(native_env)
        obs = env.reset(seed=args.seed)
        if (env.obs_dim, env.action_dim) != (1745, 23):
            raise ValueError("Native observation/action dimensions changed")
        agent = make_agent(args, agent_cfg)
        contract = json.loads(
            json.dumps(
                {
                    "algorithm": "FlashSAC",
                    "pbt": False,
                    "task": args.task,
                    "num_envs": args.num_envs,
                    "observation_groups": ["policy", "proprio", "perception"],
                    "obs_dim": 1745,
                    "action_dim": 23,
                    "action_bounds": [-1, 1],
                    "native_action_scale": 0.1,
                    "n_step_discount_version": 1,
                    "physics_dt": cfg.sim.dt,
                    "decimation": cfg.decimation,
                    "solver_position_iterations": 32,
                    "max_steps": args.max_steps,
                    "updates_per_vector_step": args.updates_per_vector_step,
                    "agent": dataclasses.asdict(agent.cfg),
                    "source_sha256": source_manifest(),
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        # Resume may write only to this contract's workspace, never overwrite foreign run metadata.
        contract_path = args.workspace / "contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("Workspace belongs to a different FlashSAC contract")
        if not contract_path.exists() and any(path.name != "writer.lock" for path in args.workspace.iterdir()):
            raise ValueError("Populated workspace has no matching FlashSAC contract")
        progress = VectorProgress(args.num_envs, args.updates_per_vector_step)
        episodes = EpisodeMetrics(args.num_envs, env.device)
        if args.checkpoint:
            payload = load_checkpoint(args.checkpoint, contract)
            progress.load_state_dict(payload["trainer"])
            episodes.load_state_dict(payload["episodes"])
            env.set_state(payload["environment"])
            obs = env.reset()
            restore_learner(agent, payload["agent"])
            if agent._update_step != progress.updates:
                raise ValueError("Learner/trainer update counters disagree")
            del payload
            print(f"[flashsac] Resume frame={progress.frame}; {RESUME_SEMANTICS}", flush=True)
        writer = SummaryWriter(str(args.workspace / "summaries"))
        dump_yaml(str(args.workspace / "env.yaml"), cfg)
        json_write(args.workspace / "agent.json", dataclasses.asdict(agent.cfg))
        json_write(contract_path, contract)
        json_write(
            args.workspace / "launch.json",
            {
                "argv": sys.argv,
                "pid": os.getpid(),
                "started_at": time.time(),
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            },
        )
        print(
            f"[flashsac] Single learner, NO PBT; {args.num_envs} envs, 1745 obs / 23 actions; "
            f"budget={args.max_steps} transitions, {args.updates_per_vector_step} updates/vector step; "
            f"replay={args.replay_capacity}@{args.buffer_device}, nstep=3, gamma=.99; "
            f"compile={agent.cfg.use_compile}/{agent.cfg.compile_mode}, AMP={agent.cfg.use_amp}",
            flush=True,
        )
        initial = learning_snapshot(agent)
        initial_updates, initial_vectors = progress.updates, progress.vector_steps
        started = tick = time.perf_counter()
        last_log_frame = last_checkpoint_frame = progress.frame
        latest_metrics, diagnostics = {}, {}

        def status():
            nonlocal last_log_frame, tick
            now = time.perf_counter()
            values = finite_metrics(
                {
                    **episodes.values(),
                    **latest_metrics,
                    **diagnostics,
                    "training/updates": progress.updates,
                    "replay/transitions": len(agent._replay_buffer),
                    "performance/transitions_per_second": (progress.frame - last_log_frame) / max(now - tick, 1e-9),
                }
            )
            if env.device.type == "cuda":
                values.update(
                    {
                        "memory/allocated_gib": torch.cuda.memory_allocated(env.device) / 2**30,
                        "memory/peak_allocated_gib": torch.cuda.max_memory_allocated(env.device) / 2**30,
                        "memory/reserved_gib": torch.cuda.memory_reserved(env.device) / 2**30,
                    }
                )
            for key, value in values.items():
                writer.add_scalar(key, value, progress.frame)
            result = {
                **progress.state_dict(),
                "metrics": values,
                "updated_at": time.time(),
                "resume_semantics": RESUME_SEMANTICS,
            }
            json_write(args.workspace / "status.json", result)
            writer.flush()
            print(
                f"[flashsac] frame={progress.frame} updates={progress.updates} "
                f"episodes={episodes.totals['completed']} successes={episodes.totals['success_count']} "
                f"metrics={values}",
                flush=True,
            )
            last_log_frame, tick = progress.frame, now
            return result

        def checkpoint():
            nonlocal last_checkpoint_frame
            path = save_checkpoint(
                args.workspace / "checkpoints",
                contract,
                {
                    "agent": learner_state(agent),
                    "trainer": progress.state_dict(),
                    "episodes": episodes.state_dict(),
                    "environment": env.get_state(),
                    "diagnostics": diagnostics,
                },
            )
            last_checkpoint_frame = progress.frame
            return path

        while progress.frame < args.max_steps and not stop and simulation_app.is_running():
            if args.stop_after_vector_steps and progress.vector_steps - initial_vectors >= args.stop_after_vector_steps:
                break
            action = (
                agent.sample_actions(progress.vector_steps, {"next_observation": obs}, training=True)
                if agent.can_start_training()
                else torch.rand((args.num_envs, 23), device=env.device) * 2 - 1
            )
            result = env.step(action)
            agent.process_transition(native_transition(obs, action, result))
            obs = result.obs
            episodes.observe(result.reward, result.terminated | result.truncated, result.infos)
            for _ in range(progress.collect(agent.can_start_training())):
                latest_metrics.update(finite_metrics(agent.update()))
                progress.updated()
            # Always finish the bounded vector update block before saving a requested stop.
            if progress.updates > initial_updates and (
                not diagnostics or progress.frame - last_log_frame >= args.log_every
            ):
                diagnostics = learning_diagnostics(agent, initial)
            if progress.frame - last_log_frame >= args.log_every or (
                progress.updates > initial_updates
                and progress.updates <= initial_updates + args.updates_per_vector_step
            ):
                status()
            if progress.frame - last_checkpoint_frame >= args.checkpoint_interval:
                checkpoint()
        path = checkpoint()
        result = status()
        json_write(
            args.workspace / "result.json",
            {
                **result,
                "checkpoint": str(path),
                "elapsed_seconds": time.perf_counter() - started,
                "stopped_by_signal": stop,
                "vector_steps_this_process": progress.vector_steps - initial_vectors,
            },
        )
        print(f"FLASHSAC_NATIVE_DONE frame={progress.frame} updates={progress.updates} checkpoint={path}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        native_env.close()


def main(argv=None):
    import traceback

    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    if "--headless" not in parser._option_string_actions:
        parser.add_argument("--headless", action="store_true")
    args = validate_args(parser, parser.parse_args(argv))
    with WorkspaceOwner(args.workspace, resume=args.checkpoint is not None):
        app = AppLauncher(args).app
        exit_code = 0
        try:
            train(args, app)
        except Exception:
            traceback.print_exc()
            exit_code = 1
        finally:
            app.close(exit_code=exit_code)
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
