#!/usr/bin/env python3
"""One TD-MPC2 learner on native Dexsuite or Stage A/B, with full-state local checkpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.tdmpc2.standalone import GRASP_MOTION_TASK, GRASP_TASK, WorkspaceOwner, build_parser, validate_args


def train(args, simulation_app):
    # Task/pxr imports must happen AFTER AppLauncher initialization.
    import dataclasses
    import gymnasium as gym
    import hashlib
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
    from scripts.tdmpc2.agent import Agent
    from scripts.tdmpc2.config import TDMPC2Config
    from scripts.tdmpc2.env import DexsuiteVectorEnv
    from scripts.tdmpc2.replay import SequenceReplay
    from scripts.tdmpc2.standalone import (
        EpisodeMetrics,
        TrainingProgress,
        compact_replay,
        finite_metrics,
        json_write,
        load_checkpoint,
        restore_replay,
        save_checkpoint,
    )

    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        print(f"[tdmpc2] Signal {signum}; saving at next safe update boundary", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    stage_a, stage_b = args.task == GRASP_TASK, args.task == GRASP_MOTION_TASK
    if stage_a:
        from scripts.tdmpc2.grasp import GraspEpisodeMetrics, GraspVectorEnv, configure_grasp

        configure_grasp(cfg, args.seed)
        adapter, metrics_type = GraspVectorEnv, GraspEpisodeMetrics
    elif stage_b:
        from scripts.tdmpc2.grasp_motion import (
            GraspMotionEpisodeMetrics,
            GraspMotionVectorEnv,
            configure_grasp_motion,
        )

        configure_grasp_motion(cfg, args.seed)
        adapter, metrics_type = GraspMotionVectorEnv, GraspMotionEpisodeMetrics
    else:
        cfg.seed, cfg.compute_final_obs = args.seed, False
        if not (
            cfg.episode_length_s == 6
            and cfg.decimation == 2
            and math.isclose(cfg.sim.dt, 1 / 120)
            and cfg.scene.robot.spawn.articulation_props[0].solver_position_iteration_count == 32
        ):
            raise ValueError("Native 360-step/solver32 physics contract changed")
        adapter, metrics_type = DexsuiteVectorEnv, EpisodeMetrics
    native_env = gym.make(args.task, cfg=cfg)
    try:
        env = adapter(native_env)
    except Exception:
        native_env.close()
        raise
    writer = SummaryWriter(str(args.workspace / "summaries"))
    try:
        obs = env.reset(seed=args.seed)
        agent_cfg = TDMPC2Config(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            num_envs=args.num_envs,
            episode_length=int(env.env.max_episode_length),
            batch_size=args.batch_size,
            plan_chunk_size=args.plan_chunk_size,
            num_samples=args.num_samples,
            num_elites=args.num_elites,
            num_pi_trajs=args.num_pi_trajs,
            iterations=args.planner_iterations,
            compile=args.compile,
        )
        root = Path(__file__).parent
        sources = (
            "agent.py",
            "vendor/common/layers.py",
            "vendor/common/world_model.py",
            "vendor/tdmpc2.py",
            "config.py",
            "env.py",
            "replay.py",
            "schedule.py",
        )
        task_contract = {
            "observation_groups": ["policy", "proprio", "perception"],
            "solver_position_iterations": 32,
        }
        if stage_a or stage_b:
            sources += ("grasp.py", "standalone.py", "train.py")
            if stage_b:
                sources += ("grasp_motion.py",)
            task_contract = env.contract_metadata()
        contract = json.loads(
            json.dumps(
                {
                    "version": 2,
                    "algorithm": "tdmpc2_standalone",
                    "task": args.task,
                    "seed": args.seed,
                    "upstream_commit": "e9f59321933cbc8e11a002b842adc7d4ffae8ff1",
                    "implementation_sha256": {
                        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
                    },
                    "num_envs": args.num_envs,
                    "replay_capacity": args.replay_capacity,
                    "seed_steps": args.seed_steps,
                    "updates_per_transition": args.updates_per_transition,
                    **task_contract,
                    "physics_dt": cfg.sim.dt,
                    "decimation": cfg.decimation,
                    "model": {
                        k: v
                        for k, v in dataclasses.asdict(agent_cfg).items()
                        if k not in {"device", "compile", "plan_chunk_size"}
                    },
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        agent = Agent(agent_cfg, device=env.device)
        replay = SequenceReplay(
            args.replay_capacity,
            args.num_envs,
            env.obs_dim,
            env.action_dim,
            agent_cfg.horizon,
            args.batch_size,
            sample_device=env.device,
            seed=args.seed,
        )
        progress = TrainingProgress(args.seed_steps, args.updates_per_transition)
        episodes = metrics_type(args.num_envs, env.device)
        if args.checkpoint:
            payload = load_checkpoint(args.checkpoint, contract)
            restore_replay(replay, payload["replay"])
            replay.start_new_stream()
            progress.load_state_dict(payload["trainer"])
            episodes.load_state_dict(payload["episodes"])
            env.set_state(payload["environment"])
            obs = env.reset()
            # Restore learner RNG last: simulator reset consumes random numbers but is not physics restore.
            agent.load_state_dict(payload["agent"], reset_planner=True)
            if agent.update_count != progress.updates:
                raise ValueError("Learner/trainer update counts differ")
            del payload
            print(f"[tdmpc2] Full resume frame={progress.frame} updates={progress.updates}", flush=True)
        reset_mask = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
        dump_yaml(str(args.workspace / "env.yaml"), cfg)
        json_write(args.workspace / "agent.json", dataclasses.asdict(agent_cfg))
        json_write(args.workspace / "contract.json", contract)
        json_write(
            args.workspace / "launch.json",
            {
                "argv": sys.argv,
                "pid": os.getpid(),
                "started_at": time.time(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
        )
        replay_gib = sum(t.numel() * t.element_size() for t in replay.state_dict()["buffers"].values()) / 2**30
        print(
            f"[tdmpc2] standalone task={args.task} seed={args.seed} envs={args.num_envs} "
            f"obs={env.obs_dim} actions={env.action_dim} capacity={replay.capacity} "
            f"UTD={args.updates_per_transition} compile={args.compile} "
            f"planning={args.num_samples}/{args.num_elites}/{args.num_pi_trajs}/{agent.core.cfg.iterations}; "
            f"full replay checkpoints reach ~{replay_gib:.2f} GiB plus model/optimizer; physics resets on resume",
            flush=True,
        )
        started = tick = time.perf_counter()
        last_log_frame = last_checkpoint_frame = progress.frame
        initial_updates, vector_steps = progress.updates, 0
        latest_metrics = {}
        last_fps = None
        timing = {"plan": 0.0, "environment": 0.0, "update": 0.0, "checkpoint": 0.0}

        def probe_done():
            return bool(args.stop_after_updates and progress.updates - initial_updates >= args.stop_after_updates)

        def log_status(*, rollout=False):
            nonlocal last_log_frame, tick, last_fps
            now = time.perf_counter()
            diagnostics = finite_metrics(agent.critic_diagnostics())
            values = {
                **episodes.values(),
                **diagnostics,
                **{f"train/{k}": v for k, v in latest_metrics.items()},
                "training/updates": progress.updates,
                "training/pending_updates": progress.pending_updates,
                "replay/transitions": len(replay),
            }
            if rollout and progress.frame > last_log_frame:
                last_fps = (progress.frame - last_log_frame) / max(now - tick, 1e-9)
                values["performance/transitions_per_second"] = last_fps
                last_log_frame, tick = progress.frame, now
            for key, value in finite_metrics(values).items():
                writer.add_scalar(key, value, progress.frame)
            status = {
                **progress.state_dict(),
                **episodes.values(),
                "critic": diagnostics,
                "learner_metrics": latest_metrics,
                "replay_size": len(replay),
                "timing_seconds": dict(timing),
                "updated_at": time.time(),
                "last_rollout_transitions_per_second": last_fps,
            }
            json_write(args.workspace / "status.json", status)
            writer.flush()
            print(
                f"[tdmpc2] frame={progress.frame} updates={progress.updates} pending={progress.pending_updates} "
                f"episodes={episodes.totals['completed']} successes={episodes.totals['success_count']} "
                f"value_loss={latest_metrics.get('value_loss')} critic={diagnostics}",
                flush=True,
            )
            return status

        def checkpoint():
            nonlocal last_checkpoint_frame
            t0 = time.perf_counter()
            path = save_checkpoint(
                args.workspace / "checkpoints",
                contract,
                {
                    "agent": agent.state_dict(),
                    "replay": compact_replay(replay),
                    "trainer": progress.state_dict(),
                    "environment": env.get_state(),
                    "episodes": episodes.state_dict(),
                    "critic": finite_metrics(agent.critic_diagnostics()),
                },
            )
            timing["checkpoint"] += time.perf_counter() - t0
            last_checkpoint_frame = progress.frame
            return path

        while progress.has_work(args.max_steps) and not stop and not probe_done() and simulation_app.is_running():
            # Drain old debt BEFORE collecting, even when a resumed run is already at its frame budget.
            while progress.pending_updates and not stop and not probe_done() and simulation_app.is_running():
                t0 = time.perf_counter()
                latest_metrics = finite_metrics(agent.update(replay))
                progress.updated()
                timing["update"] += time.perf_counter() - t0
                if progress.updates == initial_updates + 1 or progress.updates % 100 == 0:
                    log_status()
            if stop or probe_done() or progress.frame >= args.max_steps or not simulation_app.is_running():
                break
            if args.stop_after_vector_steps and vector_steps >= args.stop_after_vector_steps:
                break
            t0 = time.perf_counter()
            action = (
                torch.rand((args.num_envs, env.action_dim), device=env.device) * 2 - 1
                if progress.frame < args.seed_steps
                else agent.act(obs, reset_mask=reset_mask)
            )
            timing["plan"] += time.perf_counter() - t0
            condition = (
                env.condition()
                if stage_a or stage_b
                else json.dumps(env.env.get_evaluation_condition(), sort_keys=True, allow_nan=False)
            )
            t0 = time.perf_counter()
            result = env.step(action)
            timing["environment"] += time.perf_counter() - t0
            replay.add(
                obs, action, result.reward, result.next_obs, result.terminated, result.truncated, condition=condition
            )
            obs, reset_mask = result.obs, result.terminated | result.truncated
            episodes.observe(result.reward, reset_mask, result.infos)
            progress.collect(args.num_envs, ready=replay.can_sample())
            vector_steps += 1
            if progress.frame - last_log_frame >= args.log_every:
                log_status(rollout=True)
            if progress.frame - last_checkpoint_frame >= args.checkpoint_interval:
                checkpoint()
        path = checkpoint()
        status = log_status(rollout=True)
        json_write(
            args.workspace / "result.json",
            {
                **status,
                "checkpoint": str(path),
                "vector_steps_this_process": vector_steps,
                "elapsed_seconds": time.perf_counter() - started,
                "stopped_by_signal": stop,
            },
        )
        print(f"TDMPC2_STANDALONE_DONE frame={progress.frame} updates={progress.updates} checkpoint={path}", flush=True)
    finally:
        writer.close()
        env.close()


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
