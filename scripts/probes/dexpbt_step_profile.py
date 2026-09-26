# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Isolated checkpoint-driven DexPBT rollout timing; never launches a population.

Run with the Isaac Lab Python, --checkpoint, and --output (JSON). Defaults match
one production member: 512 envs, native physics/MDP, stochastic frozen PPO policy.
No optimizer updates are performed. Component timers are separate from baseline
rounds, nested/inclusive, and report exclusive host time too. --sync-components
serializes CUDA at boundaries and is diagnostic, NOT a throughput benchmark.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--warmup", type=int, default=128)
parser.add_argument("--steps", type=int, default=360)
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--profile-steps", type=int, default=128)
parser.add_argument("--sync-components", action="store_true")
parser.add_argument("--reset-detail", action="store_true", help="Time individual reset managers and asset writers")
parser.add_argument("--native-reset-dtype", choices=("int32", "int64"), help="Isolated reset-index A/B experiment")
parser.add_argument("--reset-acceptance", action="store_true", help="Compare public seeded reset results and cost")
parser.add_argument("--cuda-capture", action="store_true", help="Bracket profile steps with CUDA profiler API/NVTX")
parser.add_argument("--torch-trace", type=Path, help="Separate CPU/CUDA trace using installed PyTorch CUPTI")
parser.add_argument("--seed", type=int, default=45)
parser.add_argument(
    "--trajectory-check", type=int, default=0, help="Untimed seeded deterministic-policy fingerprint steps"
)
parser.add_argument(
    "--solver-position-iterations",
    type=int,
    choices=(16, 32),
    default=32,
    help="Isolated numerical-fidelity experiment; never changes the task default",
)
parser.add_argument(
    "--diagnostic-steps",
    type=int,
    default=0,
    help="Untimed seeded checkpoint contact/reset rollouts at fixed ADR 0 and 1",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import contextlib
import cProfile
import functools
import gymnasium as gym
import hashlib
import io
import json
import pstats
import statistics
import time
import torch
import traceback
import yaml

from isaaclab_physx.physics.physx_manager import PhysxManager
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401

TASK = "Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0"


class ComponentTimers:
    """Reversible process-local instrumentation; no installed-package edits."""

    def __init__(self, sync=False, nvtx=False):
        self.sync = sync
        self.nvtx = nvtx
        self.stack = []
        self.stats = {}
        self.originals = []

    def wrap(self, obj, name, label):
        if not hasattr(obj, "__dict__"):
            obj = type(obj)
        original = getattr(obj, name)
        own = name in vars(obj)
        descriptor = vars(obj).get(name)
        self.originals.append((obj, name, own, descriptor))

        @functools.wraps(original)
        def measured(*a, **kw):
            if self.sync:
                torch.cuda.synchronize()
            start = time.perf_counter()
            frame = [0.0]
            self.stack.append(frame)
            if self.nvtx:
                torch.cuda.nvtx.range_push(label)
            try:
                return original(*a, **kw)
            finally:
                if self.sync:
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if self.nvtx:
                    torch.cuda.nvtx.range_pop()
                self.stack.pop()
                if self.stack:
                    self.stack[-1][0] += elapsed
                row = self.stats.setdefault(label, {"calls": 0, "inclusive_s": 0.0, "exclusive_s": 0.0})
                row["calls"] += 1
                row["inclusive_s"] += elapsed
                row["exclusive_s"] += elapsed - frame[0]

        setattr(obj, name, measured)

    def restore(self):
        while self.originals:
            obj, name, own, descriptor = self.originals.pop()
            if own:
                setattr(obj, name, descriptor)
            else:
                delattr(obj, name)


def install_reset_timers(raw, timers):
    for name in (
        "scene",
        "observation_manager",
        "action_manager",
        "reward_manager",
        "curriculum_manager",
        "command_manager",
        "event_manager",
        "termination_manager",
    ):
        timers.wrap(getattr(raw, name), "reset", f"reset.{name}")
    for name in raw.event_manager.active_terms.get("reset", []):
        term = raw.event_manager.get_term_cfg(name)
        original_func = term.func
        timers.wrap(term, "func", f"event.{name}")
        if hasattr(original_func, "reset"):
            term.func.reset = original_func.reset
    for name in ("robot", "object"):
        asset = raw.scene[name]
        timers.wrap(asset, "reset", f"reset.asset.{name}")
        for method in (
            "write_root_pose_to_sim_index",
            "write_root_velocity_to_sim_index",
            "write_joint_position_to_sim_index",
            "write_joint_velocity_to_sim_index",
        ):
            if hasattr(asset, method):
                timers.wrap(asset, method, f"reset.{name}.{method}")
        for method in (
            "set_root_velocities",
            "set_velocities",
            "get_link_accelerations",
            "get_accelerations",
            "get_root_velocities",
            "get_velocities",
        ):
            if hasattr(asset.root_view, method):
                timers.wrap(asset.root_view, method, f"view.{name}.{method}")


def reset_acceptance_counts(num_envs):
    """Representative nonempty reset batches, including the whole available scene."""
    return sorted({count for count in (1, 4, 32, num_envs) if count <= num_envs})


def check_public_resets(raw, set_reset_dtype):
    acceptance = {"reset_rounds": [], "parity": []}
    # Same live scene, same public reset seed and selected indices. No
    # physics steps between A/B: separate reset correctness from solver noise.
    state = raw.get_env_state()
    for count in reset_acceptance_counts(raw.num_envs):
        ids = torch.arange(count - 1, -1, -1, device=raw.device, dtype=torch.int32)
        snapshots = []
        for dtype in (torch.int32, torch.int64, torch.int64, torch.int32):
            set_reset_dtype(dtype)
            raw.set_env_state(state)
            raw.reset(seed=args.seed)
            before = raw.scene["object"].data.root_pos_w.torch.clone()
            for _ in range(10):
                raw.reset(seed=args.seed, env_ids=ids)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(40):
                reset_obs, info = raw.reset(seed=args.seed, env_ids=ids)
            torch.cuda.synchronize()
            acceptance["reset_rounds"].append(
                {"count": count, "dtype": str(dtype), "ms_per_reset": (time.perf_counter() - started) * 1000 / 40}
            )
            robot, obj = raw.scene["robot"].data, raw.scene["object"].data
            snapshot = {
                "joint_pos": robot.joint_pos.torch.clone(),
                "joint_vel": robot.joint_vel.torch.clone(),
                "object_pose": obj.root_pose_w.torch.clone(),
                "object_vel": obj.root_vel_w.torch.clone(),
                "cpu_rng": torch.get_rng_state().clone(),
                "cuda_rng": torch.cuda.get_rng_state().clone(),
                **{f"obs_{name}": value[ids.long()].clone() for name, value in reset_obs.items()},
            }
            assert torch.equal(obj.root_pos_w.torch[count:], before[count:])
            assert info["dexpbt"]["completed_episodes"] == 0
            assert raw.get_env_state() == state
            snapshots.append(snapshot)
        for snapshot in snapshots[1:]:
            for name, value in snapshot.items():
                assert torch.equal(value, snapshots[0][name]), (count, name)
        acceptance["parity"].append({"count": count, "exact": True, "fields": list(snapshots[0])})
    print("RESET_ACCEPTANCE", json.dumps(acceptance), flush=True)
    return acceptance


def fingerprint_trajectory(raw, wrapped, player):
    trajectory = []
    raw.seed(args.seed)
    obs = wrapped.reset()

    def fingerprint(value):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    for _ in range(args.trajectory_check):
        actions = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
        obs, reward, dones, info = wrapped.step(actions)
        robot, obj = raw.scene["robot"].data, raw.scene["object"].data
        trajectory.append(
            {
                "actions": fingerprint(actions),
                "obs": fingerprint(obs["obs"]),
                "reward": fingerprint(reward),
                "done": fingerprint(dones),
                "q": fingerprint(robot.joint_pos.torch),
                "qd": fingerprint(robot.joint_vel.torch),
                "object_pose": fingerprint(obj.root_pose_w.torch),
                "object_vel": fingerprint(obj.root_vel_w.torch),
                "cpu_rng": fingerprint(torch.get_rng_state()),
                "cuda_rng": fingerprint(torch.cuda.get_rng_state()),
                "report": info["dexpbt"],
            }
        )
    raw.seed(args.seed)
    obs = wrapped.reset()
    return trajectory, obs


# Read tags as data only; saved scene slices/tuples need no Python execution.
class ConfigLoader(yaml.SafeLoader):
    pass


def tagged(loader, tag, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", tagged)


def verify_settings(cfg, saved_cfg):
    # Only physical/MDP settings expected to match the checked-in task are used.
    # Refuse drift rather than silently turning an experiment into another task.
    actual = cfg.to_dict()
    checks = [
        ("sim", "dt"),
        ("decimation",),
        ("episode_length_s",),
        ("scene", "robot", "spawn", "articulation_props", 0, "solver_position_iteration_count"),
        ("scene", "robot", "spawn", "articulation_props", 0, "solver_velocity_iteration_count"),
        ("scene", "robot", "spawn", "articulation_props", 0, "enabled_self_collisions"),
        ("scene", "object", "spawn", "rigid_props", "solver_position_iteration_count"),
        ("scene", "object", "spawn", "rigid_props", "solver_velocity_iteration_count"),
    ]
    verified = {}
    for keys in checks:
        left, right = actual, saved_cfg
        for key in keys:
            left, right = left[key], right[key]
        assert left == right, (keys, left, right)
        verified[".".join(map(str, keys))] = left
    for name in ("good_finger_contact", "contact_count"):
        term = getattr(cfg.rewards, name)
        assert actual["rewards"][name]["func"] == saved_cfg["rewards"][name]["func"], name
        assert term.weight == saved_cfg["rewards"][name]["weight"], name
    return verified


def main():
    if args.num_envs < 1 or min(args.warmup, args.steps, args.rounds, args.profile_steps) < 1:
        raise ValueError("Counts must be positive")
    torch.set_num_threads(1)
    base = args.checkpoint.resolve().parents[1]
    agent_cfg = yaml.safe_load((base / "params/agent.yaml").read_text())

    saved_cfg = yaml.load((base / "params/env.yaml").read_text(), Loader=ConfigLoader)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    verified = verify_settings(cfg, saved_cfg)
    cfg.scene.robot.spawn.articulation_props[0].solver_position_iteration_count = args.solver_position_iterations
    agent_cfg["params"]["config"].update(device=args.device, device_name=args.device, num_actors=args.num_envs)
    agent_cfg["params"]["load_checkpoint"] = False
    env = gym.make(TASK, cfg=cfg)
    raw = env.unwrapped
    from isaaclab.envs import ManagerBasedRLEnv

    native_reset = ManagerBasedRLEnv._reset_idx

    def set_reset_dtype(dtype):
        def reset_with_dtype(instance, env_ids):
            return native_reset(instance, env_ids.to(dtype=dtype))

        ManagerBasedRLEnv._reset_idx = reset_with_dtype if dtype is not None else native_reset

    if args.native_reset_dtype:
        set_reset_dtype(getattr(torch, args.native_reset_dtype))
    groups = agent_cfg["params"]["env"]["obs_groups"]
    wrapped = RlGamesVecEnvWrapper(env, args.device, 100.0, 1.0, {"obs": groups, "states": groups})
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in saved["model"].items()})
    if player.normalize_input and "running_mean_std" in saved:
        player.model.running_mean_std.load_state_dict(saved["running_mean_std"])
    player.reset()
    if args.num_envs == len(saved["env_state"]["dexpbt"]["adr_difficulties"]):
        raw.set_env_state(saved["env_state"])
    else:
        state = saved["env_state"]["dexpbt"]
        assert len(set(state["adr_difficulties"])) == 1, "Small-env reproduction requires uniform ADR"
        reduced = {**state, "adr_difficulties": [state["adr_difficulties"][0]] * args.num_envs}
        raw.set_env_state({"dexpbt": reduced})
    obs = wrapped.reset()
    player.get_batch_size(obs["obs"], 1)
    assert obs["obs"].shape == (args.num_envs, 1745), obs["obs"].shape
    assert not player.is_rnn
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_frame": saved.get("frame"),
        "checkpoint_epoch": saved.get("epoch"),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "policy": "stochastic frozen PPO",
        "verified_settings": verified,
        "initial_condition": raw.get_evaluation_condition(),
        "solver_position_iterations": args.solver_position_iterations,
        "torch_threads": torch.get_num_threads(),
        "warmup_steps": args.warmup,
        "round_steps": args.steps,
        "profile_steps": args.profile_steps,
        "sync_components": args.sync_components,
        "native_reset_dtype": args.native_reset_dtype,
        "rounds": [],
    }
    del saved
    completed = 0

    def step():
        nonlocal obs, completed
        actions = player.get_action(player.obs_to_torch(obs), is_deterministic=False)
        obs, reward, dones, info = wrapped.step(actions)
        completed += info["dexpbt"]["completed_episodes"]
        return reward

    timers = ComponentTimers(args.sync_components, args.cuda_capture)
    try:
        with torch.inference_mode():
            if args.trajectory_check:
                result["trajectory"], obs = fingerprint_trajectory(raw, wrapped, player)
            for _ in range(args.warmup):
                step()
            for repeat in range(args.rounds):
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(args.steps):
                    reward = step()
                torch.cuda.synchronize()
                duration = time.perf_counter() - start
                row = {
                    "round": repeat,
                    "seconds": duration,
                    "ms_per_step": duration * 1000 / args.steps,
                    "env_fps": args.num_envs * args.steps / duration,
                }
                result["rounds"].append(row)
                print("BASELINE", json.dumps(row), flush=True)
                assert torch.isfinite(obs["obs"]).all() and torch.isfinite(reward).all()
            for obj, method, label in [
                (player, "get_action", "policy"),
                (raw, "step", "env.step"),
                (raw.sim, "step", "sim.step"),
                (PhysxManager, "step", "physics.simulate_fetch"),
                (raw.action_manager, "process_action", "action.process"),
                (raw.action_manager, "apply_action", "action.apply"),
                (raw.scene, "write_data_to_sim", "scene.write"),
                (raw.scene, "update", "scene.update"),
                (raw.termination_manager, "compute", "termination"),
                (raw.reward_manager, "compute", "reward+measure"),
                (raw, "_measure_episode_step", "pbt.measure"),
                (raw, "get_evaluation_condition", "pbt.condition"),
                (raw, "_adr_values", "pbt.adr_values"),
                (raw._dexpbt_metrics, "observe", "pbt.observe"),
                (raw._dexpbt_metrics, "complete", "pbt.complete"),
                (raw, "_reset_idx", "reset"),
                (raw.curriculum_manager, "compute", "curriculum"),
                (raw.command_manager, "compute", "command"),
                (raw.event_manager, "apply", "events"),
                (raw.observation_manager, "compute", "observations"),
            ]:
                timers.wrap(obj, method, label)
            for name, sensor in raw.scene.sensors.items():
                timers.wrap(sensor, "_update_outdated_buffers", f"sensor.{name}")
            if args.reset_detail:
                install_reset_timers(raw, timers)
            reset_sizes = []
            profile_step_rows = []
            timed_reset = raw._reset_idx

            def count_reset(env_ids):
                reset_sizes.append(len(env_ids))
                return timed_reset(env_ids)

            raw._reset_idx = count_reset
            profiler = cProfile.Profile()
            torch_profiler = (
                torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                )
                if args.torch_trace
                else contextlib.nullcontext()
            )
            torch.cuda.synchronize()
            if args.cuda_capture:
                torch.cuda.cudart().cudaProfilerStart()
            start = time.perf_counter()
            with torch_profiler, profiler if not (args.cuda_capture or args.torch_trace) else contextlib.nullcontext():
                for _ in range(args.profile_steps):
                    before = len(reset_sizes)
                    step_start = time.perf_counter()
                    step()
                    profile_step_rows.append(
                        {"seconds": time.perf_counter() - step_start, "reset_envs": sum(reset_sizes[before:])}
                    )
            torch.cuda.synchronize()
            result["profile_wall_s"] = time.perf_counter() - start
            if args.cuda_capture:
                torch.cuda.cudart().cudaProfilerStop()
            timers.restore()
            if args.torch_trace:
                torch_profiler.export_chrome_trace(str(args.torch_trace))
                result["torch_trace"] = str(args.torch_trace.resolve())
            result["components"] = timers.stats
            result["reset_batches"] = reset_sizes
            result["profile_step_rows"] = profile_step_rows
            result["completed_episodes"] = completed
            result["final_condition"] = raw.get_evaluation_condition()
            result["median_ms_per_step"] = statistics.median(row["ms_per_step"] for row in result["rounds"])
            stream = io.StringIO()
            if not (args.cuda_capture or args.torch_trace):
                pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(
                    120 if args.reset_detail else 60
                )
                if args.reset_detail:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    profiler.dump_stats(str(args.output.with_suffix(".cprofile")))
            result["cprofile"] = stream.getvalue()
            if args.diagnostic_steps:
                diagnostics = []
                for fraction in (0.0, 1.0):
                    raw.configure_fixed_evaluation(fraction)
                    raw.seed(args.seed + 100)
                    obs = wrapped.reset()
                    records = []
                    reports = []
                    native_measure = raw._measure_episode_step

                    def measure_physics():
                        native_measure()
                        robot = raw.scene["robot"].data
                        obj = raw.scene["object"].data
                        forces = torch.stack(
                            [
                                raw._tensor(raw.scene.sensors[name].data.force_matrix_w).reshape(args.num_envs, 3)
                                for name in [
                                    raw._dexpbt_measurement["thumb_name"],
                                    *raw._dexpbt_measurement["finger_names"],
                                ]
                            ],
                            dim=1,
                        ).norm(dim=-1)
                        q = raw._tensor(robot.joint_pos)
                        qd = raw._tensor(robot.joint_vel)
                        target = raw._tensor(robot.joint_pos_target)
                        object_vel = raw._tensor(obj.root_vel_w)
                        values = torch.stack(
                            [
                                forces.max(),
                                (forces > 0.01).float().mean(),
                                ((forces[:, 0] > 0.01) & (forces[:, 1:] > 0.01).any(dim=1)).float().mean(),
                                qd.abs().max(),
                                object_vel[:, :3].norm(dim=-1).max(),
                                object_vel[:, 3:].norm(dim=-1).max(),
                                (target - q).abs().mean(),
                                raw._tensor(obj.root_pos_w)[:, 2].mean(),
                            ]
                        )
                        assert torch.isfinite(values).all(), values
                        records.append(values.cpu().tolist())

                    raw._measure_episode_step = measure_physics
                    try:
                        for _ in range(args.diagnostic_steps):
                            step()
                            reports.append(dict(raw.extras["dexpbt"]))
                    finally:
                        raw._measure_episode_step = native_measure
                    rows = torch.tensor(records)
                    diagnostics.append(
                        {
                            "adr_fraction": fraction,
                            "steps": args.diagnostic_steps,
                            "condition": raw.get_evaluation_condition(),
                            "force_max_n": float(rows[:, 0].max()),
                            "contact_fraction": float(rows[:, 1].mean()),
                            "good_contact_fraction": float(rows[:, 2].mean()),
                            "joint_speed_max_rad_s": float(rows[:, 3].max()),
                            "object_linear_speed_max_m_s": float(rows[:, 4].max()),
                            "object_angular_speed_max_rad_s": float(rows[:, 5].max()),
                            "joint_target_error_mean_rad": float(rows[:, 6].mean()),
                            "completed_episodes": sum(row["completed_episodes"] for row in reports),
                            "success_count": sum(row["success_count"] for row in reports),
                            "mixed_condition_episodes": sum(row["mixed_condition_episodes"] for row in reports),
                            "per_step_physics": records,
                        }
                    )
                    print(
                        "DIAGNOSTIC",
                        json.dumps({k: v for k, v in diagnostics[-1].items() if k != "per_step_physics"}),
                        flush=True,
                    )
                result["diagnostics"] = diagnostics
            if args.reset_acceptance:
                result["reset_acceptance"] = check_public_resets(raw, set_reset_dtype)
                set_reset_dtype(getattr(torch, args.native_reset_dtype) if args.native_reset_dtype else None)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
            print("PROFILE_RESULT", str(args.output.resolve()), flush=True)
            print(json.dumps(result["components"], indent=2), flush=True)
            print(result["cprofile"], flush=True)
    finally:
        timers.restore()
        ManagerBasedRLEnv._reset_idx = native_reset
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        app.close(exit_code=exit_code)
