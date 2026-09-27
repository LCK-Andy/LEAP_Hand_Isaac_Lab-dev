#!/usr/bin/env python3
"""Train a fresh TD-MPC2 member on native Dexsuite physics with replay-aware PBT."""

import argparse
import fcntl
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

# No LEAP/pxr/task imports before AppLauncher.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--member", type=int, default=0)
parser.add_argument("--population", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--workspace", type=Path, required=True)
parser.add_argument("--max-steps", type=int, default=10_000_000)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--replay-capacity", type=int, default=131_072)
parser.add_argument("--seed-steps", type=int, default=1800)
parser.add_argument("--updates-per-transition", type=float, default=1.0)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--plan-chunk-size", type=int, default=32)
parser.add_argument("--num-samples", type=int, default=512)
parser.add_argument("--num-elites", type=int, default=64)
parser.add_argument("--num-pi-trajs", type=int, default=24)
parser.add_argument("--planner-iterations", type=int, default=6, help="Base iterations; TD-MPC2 adds 2 for 23 actions")
parser.add_argument("--interval-steps", type=int, default=1_000_000)
parser.add_argument("--warmup-steps", type=int, default=5_000_000)
parser.add_argument("--adaptation-steps", type=int, default=3_000_000)
parser.add_argument("--min-completed-episodes", type=int, default=128)
parser.add_argument("--max-score-age-seconds", type=float, default=1800.0)
parser.add_argument("--disable-selection", action="store_true")
parser.add_argument("--disable-initial-diversity", action="store_true")
parser.add_argument("--log-every", type=int, default=2560)
parser.add_argument("--stop-after-vector-steps", type=int, default=0, help="Bounded execution probe, not a task change")
parser.add_argument(
    "--collect-only", action="store_true", help="Native episode/terminal replay probe without model updates"
)
parser.add_argument("--compile", action="store_true", help="Opt-in torch.compile after eager validation")
AppLauncher.add_app_launcher_args(parser)
if "--headless" not in parser._option_string_actions:
    parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
if args.task != "Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0":
    parser.error("This integration is validated only for the native Dexsuite PBT task")
if args.num_envs <= 0 or not 0 <= args.member < args.population or args.max_steps <= 0:
    parser.error("Invalid environment count, member identity, or transition budget")
if args.replay_capacity % args.num_envs:
    parser.error("Replay capacity must be divisible by num_envs")
args.workspace = args.workspace.expanduser().resolve()
member_dir = args.workspace / "members" / f"{args.member:03d}"
archive_dir = args.workspace / "tdmpc2" / f"{args.member:03d}"
if not args.checkpoint and any(path.exists() and any(path.iterdir()) for path in (member_dir, archive_dir)):
    parser.error("Refusing a fresh start in a populated member/archive directory; use --checkpoint")
member_dir.mkdir(parents=True, exist_ok=True)
member_lock = (member_dir / "writer.lock").open("a+")
try:
    fcntl.flock(member_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    parser.error("Another process already owns this population member")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import dataclasses
import gymnasium as gym
import math
import numpy as np
import random
import torch
from torch.utils.tensorboard import SummaryWriter

from async_pbt.pbt_cfg import PbtCfg

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401
from scripts.tdmpc2.agent import Agent
from scripts.tdmpc2.config import TDMPC2Config
from scripts.tdmpc2.env import DexsuiteVectorEnv
from scripts.tdmpc2.pbt import PopulationMember, load_checkpoint
from scripts.tdmpc2.replay import SequenceReplay
from scripts.tdmpc2.runtime import canonical_contract, recipient_trainer_state
from scripts.tdmpc2.schedule import UpdateSchedule

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True
    print(f"[tdmpc2] Signal {signum}: save at the next safe update boundary", flush=True)


for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, request_stop)


def json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.compute_final_obs = False
    assert cfg.episode_length_s == 6.0 and cfg.decimation == 2
    assert math.isclose(cfg.sim.dt, 1 / 120)
    assert cfg.scene.robot.spawn.articulation_props[0].solver_position_iteration_count == 32
    raw_env = gym.make(args.task, cfg=cfg)
    env = DexsuiteVectorEnv(raw_env)
    writer = SummaryWriter(str(member_dir / "summaries"))
    try:
        obs = env.reset(seed=args.seed)
        agent_cfg = TDMPC2Config(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            num_envs=args.num_envs,
            episode_length=int(raw_env.unwrapped.max_episode_length),
            batch_size=args.batch_size,
            plan_chunk_size=args.plan_chunk_size,
            num_samples=args.num_samples,
            num_elites=args.num_elites,
            num_pi_trajs=args.num_pi_trajs,
            iterations=args.planner_iterations,
            compile=args.compile,
        )
        contract = {
            "version": 1,
            "algorithm": "tdmpc2",
            "upstream_commit": "e9f59321933cbc8e11a002b842adc7d4ffae8ff1",
            "task": args.task,
            "obs_dim": env.obs_dim,
            "action_dim": env.action_dim,
            "observation_groups": ["policy", "proprio", "perception"],
            "num_envs": args.num_envs,
            "population": args.population,
            "pbt_schedule": {
                "interval_steps": args.interval_steps,
                "warmup_steps": args.warmup_steps,
                "adaptation_steps": args.adaptation_steps,
                "min_completed_episodes": args.min_completed_episodes,
                "max_score_age_seconds": args.max_score_age_seconds,
            },
            "replay_capacity": args.replay_capacity,
            "updates_per_transition": args.updates_per_transition,
            "seed_steps": args.seed_steps,
            "collect_only": args.collect_only,
            "physics_dt": cfg.sim.dt,
            "decimation": cfg.decimation,
            "solver_position_iterations": 32,
            "model": {
                k: v
                for k, v in dataclasses.asdict(agent_cfg).items()
                if k not in {"lr", "entropy_coef", "grad_clip_norm", "device", "compile", "plan_chunk_size"}
            },
        }
        contract = canonical_contract(contract)
        pbt_cfg = PbtCfg(
            enabled=True,
            objective="dexpbt",
            policy_idx=args.member,
            num_policies=args.population,
            interval_steps=args.interval_steps,
            warmup_steps=args.warmup_steps,
            adaptation_steps=args.adaptation_steps,
            min_completed_episodes=args.min_completed_episodes,
            max_score_age_seconds=args.max_score_age_seconds,
            selection_enabled=not args.disable_selection and not args.collect_only,
            initial_diversity=not args.disable_initial_diversity,
            mutation_rate=0.25,
            change_range=(1.1, 1.5),
            mutation={"lr": "mutate_float", "entropy_coef": "mutate_float", "grad_clip_norm": "mutate_float"},
            mutation_bounds={"lr": (1e-5, 1e-3), "entropy_coef": (1e-5, 1e-3), "grad_clip_norm": (1.0, 100.0)},
        )
        hparams = {name: getattr(agent_cfg, name) for name in pbt_cfg.mutation}
        if not args.checkpoint and pbt_cfg.initial_diversity and args.member:
            rng = random.Random(f"tdmpc2_initial:{args.seed}:{args.member}")
            for key, value in hparams.items():
                low, high = pbt_cfg.mutation_bounds[key]
                hparams[key] = min(high, max(low, value * math.exp(rng.uniform(-math.log(1.25), math.log(1.25)))))
        agent = Agent(agent_cfg, device=env.device)
        agent.set_hparams(hparams)
        replay = SequenceReplay(
            capacity=args.replay_capacity,
            num_envs=args.num_envs,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            horizon=agent_cfg.horizon,
            batch_size=agent_cfg.batch_size,
            device="cpu",
            sample_device=env.device,
            seed=args.seed,
        )
        schedule = UpdateSchedule(args.seed_steps, args.updates_per_transition)

        def pbt_log(event):
            with (member_dir / "pbt_events.jsonl").open("a") as stream:
                stream.write(json.dumps(event, default=str) + "\n")
            for key in ("true_objective", "objective_valid", "incompatible_records"):
                if isinstance(event.get(key), (int, float, bool)):
                    writer.add_scalar(f"pbt/{key}", event[key], event.get("frame", 0))
            print(
                f"[tdmpc2-pbt] {event['event']} frame={event.get('frame')} " f"reason={event.get('reason', '')}",
                flush=True,
            )

        population = PopulationMember(args.workspace, pbt_cfg, args.seed, hparams, contract, log_callback=pbt_log)
        frame = updates = vector_steps = pending_updates = 0
        reset_mask = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
        if args.checkpoint:
            payload = load_checkpoint(args.checkpoint)
            if payload["pbt"].get("contract") != contract:
                raise ValueError("Checkpoint task/algorithm/training contract differs")
            state = payload["training_state"]
            agent.load_state_dict(state["agent"], reset_planner=True)
            replay.load_state_dict(state["replay"])
            replay.start_new_stream()
            accounting = recipient_trainer_state(payload)
            schedule.load_state_dict(accounting["schedule"])
            frame = int(accounting["frame"])
            updates = int(accounting["updates"])
            pending_updates = int(accounting.get("pending_updates", 0))
            population.restore_metadata(payload["pbt"])
            agent.set_hparams(population.hparams)
            env.set_state(payload["env_state"])
            obs = env.reset()
            print(f"[tdmpc2] Full resume frame={frame} updates={updates} replay={len(replay)}", flush=True)
            del payload, state, accounting
        dump_yaml(str(member_dir / "env.yaml"), cfg)
        json_write(member_dir / "agent.json", dataclasses.asdict(agent_cfg))
        json_write(member_dir / "contract.json", contract)
        json_write(
            member_dir / "launch.json",
            {
                "argv": sys.argv,
                "pid": os.getpid(),
                "started_at": time.time(),
                "seed": args.seed,
                "hparams": population.hparams,
                "num_envs": args.num_envs,
            },
        )
        print(
            f"[tdmpc2] member={args.member}/{args.population} obs={env.obs_dim} actions={env.action_dim} "
            f"envs={args.num_envs} UTD={args.updates_per_transition} capacity={args.replay_capacity} "
            f"samples={agent_cfg.num_samples} iterations={args.planner_iterations}+2",
            flush=True,
        )
        started = tick = time.perf_counter()
        last_log_frame = frame
        latest_metrics = {}
        episode_return = torch.zeros(args.num_envs, device=env.device)
        episode_length = torch.zeros_like(episode_return)
        completed = 0
        timing = {"plan": 0.0, "environment": 0.0, "update": 0.0}

        def training_state():
            return {
                "agent": agent.state_dict(),
                "replay": replay.state_dict(),
                "trainer": {
                    "frame": frame,
                    "updates": updates,
                    "pending_updates": pending_updates,
                    "schedule": schedule.state_dict(),
                },
            }

        def run_pending_updates():
            nonlocal updates, pending_updates, latest_metrics
            while pending_updates > 0 and not STOP:
                t0 = time.perf_counter()
                latest_metrics = agent.update(replay)
                timing["update"] += time.perf_counter() - t0
                updates += 1
                pending_updates -= 1
                if updates % 100 == 0:
                    print(f"[tdmpc2] update={updates} frame={frame} pending={pending_updates}", flush=True)

        while (frame < args.max_steps or pending_updates) and not STOP and simulation_app.is_running():
            if pending_updates:
                run_pending_updates()
                if STOP:
                    break
            if frame >= args.max_steps:
                break
            t0 = time.perf_counter()
            if frame < args.seed_steps or args.collect_only:
                action = torch.rand((args.num_envs, env.action_dim), device=env.device) * 2 - 1
            else:
                action = agent.act(obs, reset_mask=reset_mask)
            timing["plan"] += time.perf_counter() - t0
            condition = raw_env.unwrapped.get_evaluation_condition()
            t0 = time.perf_counter()
            result = env.step(action)
            timing["environment"] += time.perf_counter() - t0
            replay.add(
                obs,
                action,
                result.reward,
                result.next_obs,
                result.terminated,
                result.truncated,
                condition=json.dumps(condition, sort_keys=True, allow_nan=False),
            )
            obs = result.obs
            reset_mask = result.terminated | result.truncated
            frame += args.num_envs
            vector_steps += 1
            population.observe(result.infos, frame)
            episode_return += result.reward.reshape(-1)
            episode_length += 1
            if reset_mask.any():
                writer.add_scalar("episode/return", episode_return[reset_mask].mean().item(), frame)
                writer.add_scalar("episode/length", episode_length[reset_mask].mean().item(), frame)
                completed += int(reset_mask.sum())
                episode_return[reset_mask] = 0
                episode_length[reset_mask] = 0
            if not args.collect_only:
                pending_updates += schedule.collect(args.num_envs, ready=replay.can_sample())
                run_pending_updates()
            if frame - last_log_frame >= args.log_every:
                now = time.perf_counter()
                fps = (frame - last_log_frame) / max(now - tick, 1e-9)
                writer.add_scalar("performance/transitions_per_second", fps, frame)
                writer.add_scalar("training/updates", updates, frame)
                writer.add_scalar("replay/transitions", len(replay), frame)
                writer.add_scalar("episode/completed", completed, frame)
                for name, value in latest_metrics.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        value = value.item()
                    if isinstance(value, (float, int)) and math.isfinite(value):
                        writer.add_scalar(f"train/{name}", value, frame)
                for name, value in timing.items():
                    writer.add_scalar(f"performance/{name}_seconds", value, frame)
                for name, value in population.hparams.items():
                    writer.add_scalar(f"pbt/{name}", value, frame)
                print(
                    f"[tdmpc2] frame={frame} updates={updates} replay={len(replay)} "
                    f"episodes={completed} fps={fps:.2f}",
                    flush=True,
                )
                json_write(
                    member_dir / "status.json",
                    {
                        "frame": frame,
                        "updates": updates,
                        "completed_episodes": completed,
                        "replay_size": len(replay),
                        "fps": fps,
                        "updated_at": time.time(),
                        "timing_seconds": timing,
                        "pending_updates": pending_updates,
                    },
                )
                writer.flush()
                last_log_frame, tick = frame, now
            if population.should_publish(frame) and not STOP:
                record = population.publish(training_state(), frame, updates, env.get_state())
                decision = population.maybe_exploit(record)
                if decision is not None:
                    recipient_schedule = schedule.state_dict()
                    agent.load_state_dict(decision.training_state["agent"], reset_planner=True)
                    replay.load_state_dict(decision.training_state["replay"])
                    replay.start_new_stream()
                    schedule.load_state_dict(recipient_schedule)
                    agent.set_hparams(decision.hparams)
                    env.set_state(decision.env_state)
                    obs = env.reset()
                    reset_mask.fill_(True)
                    episode_return.zero_()
                    episode_length.zero_()
                    print(
                        f"[tdmpc2] EXPLOIT member={args.member} frame={frame} "
                        f"donor={decision.metadata.get('donor_policy_idx')} replay={len(replay)}",
                        flush=True,
                    )
                    writer.add_scalar("pbt/exploit", 1, frame)
                    del decision
            if args.stop_after_vector_steps and vector_steps >= args.stop_after_vector_steps:
                break
        record = population.publish(training_state(), frame, updates, env.get_state(), final=True)
        json_write(
            member_dir / "result.json",
            {
                "frame": frame,
                "updates": updates,
                "vector_steps_this_process": vector_steps,
                "completed_episodes_this_process": completed,
                "checkpoint": record["checkpoint"],
                "elapsed_seconds": time.perf_counter() - started,
                "stopped_by_signal": STOP,
                "pending_updates": pending_updates,
                "replay_size": len(replay),
            },
        )
        print(f"TDMPC2_MEMBER_DONE frame={frame} updates={updates} checkpoint={record['checkpoint']}", flush=True)
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close(exit_code=exit_code)
    sys.exit(exit_code)
