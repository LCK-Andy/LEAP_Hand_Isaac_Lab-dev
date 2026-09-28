"""Expose one paused-between-commands humanoid to an external, state-feedback controller."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

from astra_humanoid_protocol import atomic_write_json, validate_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="New directory; existing paths are refused.")
    parser.add_argument("--task", default="Isaac-Humanoid-Direct-v0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--neutral-reset", action="store_true", help="Disable only joint reset noise for the demo.")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum control steps per request.")
    parser.add_argument(
        "--total-steps", type=int, default=600, help="Lifetime step budget, including resets' episodes."
    )
    parser.add_argument("--idle-timeout", type=float, default=900.0)
    parser.add_argument("--controller-label", default="external JSON client; model identity is not authenticated")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 60 or args.total_steps < 1 or args.idle_timeout <= 0:
        parser.error("Require max-steps in [1,60], positive total-steps and idle-timeout.")
    return args


def run(args: argparse.Namespace, simulation_app) -> None:
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_visualizers.kit import KitVisualizerCfg

    task = args.task
    if task not in gym.registry and task == "Isaac-Humanoid-Direct-v0":
        task = "Isaac-Humanoid-Direct"
        print(f"Requested {args.task}; installed successor is {task} (not historical MDP parity).", flush=True)
    if task not in {"Isaac-Humanoid-Direct-v0", "Isaac-Humanoid-Direct"}:
        raise ValueError("This demo is scoped to the direct humanoid task.")
    cfg = parse_env_cfg(task, device=args.device, num_envs=1, overrides=["physics=isaacsim_physx"])
    cfg.seed = args.seed
    cfg.compute_final_obs = True
    if args.neutral_reset:
        cfg.initial_joint_pos_range = (0.0, 0.0)
        cfg.initial_joint_vel_range = (0.0, 0.0)
    cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(
            headless=not args.gui,
            window_width=960,
            window_height=720,
            eye=(3.2, -3.2, 2.0),
            lookat=(0.0, 0.0, 1.0),
        )
    ]
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    viz = next(v for v in u.sim.visualizers if v.cfg.visualizer_type == "kit")
    names = list(u.robot.joint_names)
    if len(names) != 21 or cfg.action_space != 21:
        raise RuntimeError(f"Expected 21 humanoid actions, got {names} / {cfg.action_space}.")
    frame_index = 0
    total_steps = 0
    episode_steps = 0
    episode = 0
    needs_reset = False
    terminated = truncated = False
    final_observation = None
    latest_image = None
    with torch.inference_mode():
        observation, _ = env.reset(seed=args.seed)
    for _ in range(20):
        u.sim.render()
        viz.render_rgb_array()

    def values(value):
        tensor = value.torch if hasattr(value, "torch") else value
        return tensor.detach().cpu().tolist()

    def capture() -> str:
        nonlocal frame_index, latest_image
        for _ in range(30):
            u.sim.render()
            image = viz.render_rgb_array()
            if (
                image is not None
                and image.ndim == 3
                and image.shape[-1] >= 3
                and image.size > 0
                and np.isfinite(image).all()
                and np.any(image[..., :3])
            ):
                break
        else:
            raise RuntimeError("Kit did not produce a nonblank, finite RGB frame.")
        path = args.run_dir / "frames" / f"{frame_index:06d}.png"
        Image.fromarray(image[..., :3].astype(np.uint8)).save(path)
        frame_index += 1
        latest_image = str(path)
        return latest_image

    manifest = {
        "requested_task": args.task,
        "resolved_task": task,
        "task_renamed": task != args.task,
        "isaaclab_version": importlib.metadata.version("isaaclab"),
        "backend": u.scene.physics_backend,
        "controller_label": args.controller_label,
        "controller_identity_verified": False,
        "seed": args.seed,
        "neutral_reset": args.neutral_reset,
        "step_dt": u.step_dt,
        "action_interface": "normalized feedforward joint efforts; omitted joints receive zero",
        "effort_scale_nm": dict(zip(names, values(u.joint_gears))),
        "stock_implicit_stiffness": cfg.robot.actuators["body"].stiffness,
        "stock_implicit_damping": cfg.robot.actuators["body"].damping,
        "control_note": "Stock passive joint drives retained; no learned policy, IK, or extra balance controller.",
        "timing_note": "Physics advances only in env.step; model/tool wall latency is not simulated.",
        "max_steps": args.max_steps,
        "total_step_budget": args.total_steps,
    }
    atomic_write_json(args.run_dir / "manifest.json", manifest)

    def state(status: str = "ready") -> dict:
        data = u.robot.data
        result = {
            "status": status,
            "requested_task": args.task,
            "resolved_task": task,
            "joint_names": names,
            "max_steps": args.max_steps,
            "step_dt": u.step_dt,
            "total_steps": total_steps,
            "remaining_steps": args.total_steps - total_steps,
            "episode": episode,
            "episode_steps": episode_steps,
            "episode_sim_time": episode_steps * u.step_dt,
            "needs_reset": needs_reset,
            "terminated": terminated,
            "truncated": truncated,
            "state_is_after_auto_reset": needs_reset,
            "state_episode": episode + int(needs_reset),
            "state_episode_steps": 0 if needs_reset else episode_steps,
            "terminal_policy_observation": final_observation,
            "terminal_image_available": False,
            "root_position_m": values(data.root_pos_w)[0],
            "root_quaternion_xyzw": values(data.root_quat_w)[0],
            "root_linear_velocity_world": values(data.root_lin_vel_w)[0],
            "root_angular_velocity_body": values(data.root_ang_vel_b)[0],
            "projected_gravity_body": values(data.projected_gravity_b)[0],
            "joint_position_rad": dict(zip(names, values(data.joint_pos)[0])),
            "joint_velocity_rad_s": dict(zip(names, values(data.joint_vel)[0])),
            "joint_limits_rad": dict(zip(names, values(data.soft_joint_pos_limits)[0])),
            "body_positions_m": dict(zip(u.robot.body_names, values(data.body_pos_w)[0])),
            "policy_observation": values(observation["policy"])[0],
            "image": latest_image,
        }
        atomic_write_json(args.run_dir / "state.json", result)
        return result

    capture()
    state()
    print(f"READY {args.run_dir / 'state.json'}", flush=True)
    last_command = time.monotonic()
    trace_file = (args.run_dir / "transitions.jsonl").open("a", encoding="utf-8")
    commands_file = (args.run_dir / "commands.jsonl").open("a", encoding="utf-8")
    stop_reason = "application_closed"
    try:
        while simulation_app.is_running():
            pending = sorted((args.run_dir / "commands").glob("*.json"))
            pending = [p for p in pending if not (args.run_dir / "responses" / p.name).exists()]
            if not pending:
                if time.monotonic() - last_command > args.idle_timeout:
                    stop_reason = "idle_timeout"
                    break
                if args.gui:
                    u.sim.render()
                time.sleep(0.05)
                continue
            request = pending[0]
            response_path = args.run_dir / "responses" / request.name
            last_command = time.monotonic()
            try:
                command = validate_command(json.loads(request.read_text()), names, args.max_steps)
                if command["op"] == "action":
                    if needs_reset:
                        raise ValueError("Episode ended; send reset before another action.")
                    if total_steps + command["steps"] > args.total_steps:
                        raise ValueError("Request exceeds remaining lifetime step budget.")
            except (ValueError, TypeError) as exc:
                atomic_write_json(response_path, {"ok": False, "error": str(exc)})
                continue
            commands_file.write(json.dumps({"id": request.stem, "command": command, "before": state()}) + "\n")
            commands_file.flush()
            if command["op"] == "close":
                stop_reason = "client_close"
                atomic_write_json(response_path, {"ok": True, "state": state("closed")})
                break
            if command["op"] == "reset":
                # A terminal step already reset the native env; acknowledge it without sampling another pose.
                if not needs_reset:
                    with torch.inference_mode():
                        observation, _ = env.reset()
                episode += 1
                episode_steps = 0
                needs_reset = False
                terminated = truncated = False
                final_observation = None
                capture()
                atomic_write_json(response_path, {"ok": True, "state": state()})
                continue
            action = torch.tensor([[command["torques"].get(n, 0.0) for n in names]], device=u.device)
            reward_sum = 0.0
            executed = 0
            terminated = truncated = False
            for _ in range(command["steps"]):
                # Native extras persist; a GUI-requested reset may terminate without producing final_obs.
                u.extras.pop("final_obs", None)
                with torch.inference_mode():
                    observation, reward, term, trunc, extras = env.step(action)
                if not torch.isfinite(observation["policy"]).all() or not torch.isfinite(reward).all():
                    raise RuntimeError("Nonfinite environment observation/reward; aborting demo.")
                total_steps += 1
                episode_steps += 1
                executed += 1
                reward_sum += float(reward.item())
                terminated, truncated = bool(term.item()), bool(trunc.item())
                needs_reset = terminated or truncated
                if needs_reset:
                    final_obs = extras.get("final_obs")
                    final_observation = values(final_obs["policy"])[0] if final_obs is not None else None
                capture()
                snapshot = state()
                trace_file.write(
                    json.dumps(
                        {
                            "request_id": request.stem,
                            "action": values(action)[0],
                            "reward": float(reward.item()),
                            "state": snapshot,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                trace_file.flush()
                if needs_reset:
                    break
            atomic_write_json(
                response_path,
                {"ok": True, "executed_steps": executed, "reward_sum": reward_sum, "state": snapshot},
            )
    except Exception:
        stop_reason = "execution_error"
        if "response_path" in locals():
            atomic_write_json(response_path, {"ok": False, "error": traceback.format_exc()})
        raise
    finally:
        state("closed")
        atomic_write_json(
            args.run_dir / "summary.json",
            {"stop_reason": stop_reason, "total_steps": total_steps, "episodes": episode + 1, "frames": frame_index},
        )
        trace_file.close()
        commands_file.close()
        env.close()


def main() -> int:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    for child in ("commands", "responses", "frames"):
        (args.run_dir / child).mkdir()
    from isaaclab.app import AppLauncher

    app = AppLauncher(
        headless=not args.gui,
        enable_cameras=True,
        device=args.device,
        visualizer_intent={"has_any_visualizers": True, "has_kit_visualizer": True},
    ).app
    exit_code = 0
    try:
        run(args, app)
    except Exception:
        exit_code = 1
        traceback.print_exc()
        atomic_write_json(args.run_dir / "error.json", {"error": traceback.format_exc()})
    finally:
        app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
