"""Exercise original-task control, physical assets, deferred resets and checkpoint state."""

# flake8: noqa: E402

import argparse
import json
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=620)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
if "--headless" not in parser._option_string_actions:
    parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401
from LEAP_Isaaclab.tasks.dexpbt_original import mdp
from LEAP_Isaaclab.tasks.dexpbt_original.env_cfg import JOINT_NAMES

TASK = "Isaac-DexPBT-Allegro-Kuka-Reorientation-Direct-v0"


def main():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = 42
    cfg.original_mutable_params = {"distanceDeltaRewScale": 55.0}
    env = gym.make(TASK, cfg=cfg)
    try:
        raw = env.unwrapped
        observation, _ = env.reset(seed=42)
        assert observation["policy"].shape == (args.num_envs, 117)
        assert not observation["policy"].any(), "The released VecTask starts with zero observations"
        assert raw.reset_buf.all() and raw.reset_goal_buf.all()
        assert raw.robot.is_fixed_base
        assert tuple(raw.robot.joint_names) == JOINT_NAMES
        assert abs(raw.step_dt - 0.01667) < 1e-9
        assert raw.get_original_mutable_params()["distanceDeltaRewScale"] == 55.0
        initial_root = raw.robot.data.root_link_pose_w.torch.clone()
        measured_mass = raw.object.data.body_mass.torch[:, 0]
        expected_mass = 400.0 * (0.05 * raw.object_scales).prod(dim=-1)
        torch.testing.assert_close(measured_mass, expected_mass, atol=1e-5, rtol=1e-4)
        action = torch.full((args.num_envs, 23), 0.05, device=raw.device)
        env.step(action)
        previous_targets = raw.joint_targets.clone()
        expected_targets = mdp.joint_targets(action, previous_targets, raw.joint_lower, raw.joint_upper)
        env.step(action)
        torch.testing.assert_close(raw.joint_targets, expected_targets)
        initial_tolerance = raw.success_tolerance
        checkpoint = raw.get_env_state()
        raw.success_tolerance = 0.04
        raw.set_env_state(checkpoint)
        assert raw.success_tolerance == initial_tolerance
        assert raw.get_original_mutable_params()["distanceDeltaRewScale"] == 55.0
        torch.testing.assert_close(raw.state.successes.cpu(), checkpoint["mdp_state"]["successes"])
        completed = 0
        timeout_count = 0
        successes = 0
        maximum_joint_velocity = 0.0
        maximum_object_speed = 0.0
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(args.steps):
            action = 0.2 * (2.0 * torch.rand_like(action) - 1.0)
            observation, reward, terminated, truncated, info = env.step(action)
            assert observation["policy"].shape == (args.num_envs, 117)
            assert torch.isfinite(observation["policy"]).all(), f"nonfinite observation at {step}"
            assert reward.shape == (args.num_envs,) and torch.isfinite(reward).all()
            torch.testing.assert_close(observation["policy"][:, 116], (reward * 0.01).clamp(-10, 10))
            assert info["true_objective"].shape == (args.num_envs,)
            assert torch.isfinite(info["true_objective"]).all()
            done = terminated | truncated
            completed += int(done.sum())
            timeout_count += int(truncated.sum())
            successes += int(raw.reset_goal_buf.sum())
            if done.any():
                assert raw.reset_buf[done].all(), "Terminals must remain pending until next pre-step"
                assert (raw.state.progress_buf[truncated] >= 599).all()
            maximum_joint_velocity = max(maximum_joint_velocity, float(raw.robot.data.joint_vel.torch.abs().max()))
            maximum_object_speed = max(maximum_object_speed, float(raw.object.data.root_com_vel_w.torch.abs().max()))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        torch.testing.assert_close(raw.robot.data.root_link_pose_w.torch, initial_root, atol=1e-5, rtol=1e-5)
        if args.steps >= 620:
            assert completed > 0, "Native failure/timeout transitions were never exercised"
        manifest = raw.asset_manifest
        body_properties = {
            "names": list(raw.robot.body_names),
            "mass": raw.robot.data.body_mass.torch[0].cpu().tolist(),
            "com_pose_b": raw.robot.data.body_com_pose_b.torch[0].cpu().tolist(),
            "inertia": raw.robot.data.body_inertia.torch[0].cpu().tolist(),
            "scope": "Live EA tensor properties after stepping; compare inertia frames before interpreting differences.",
        }
        report = {
            "task": TASK,
            "profile": mdp.PROFILE,
            "source_commit": mdp.SOURCE_COMMIT,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "elapsed_s": elapsed,
            "transitions_per_second": args.steps * args.num_envs / elapsed,
            "observation_dim": 117,
            "action_dim": 23,
            "control_dt": raw.step_dt,
            "completed_episodes": completed,
            "timeouts": timeout_count,
            "goal_success_events": successes,
            "object_variant_ids": raw.object_variant_ids.cpu().tolist(),
            "object_scales": raw.object_scales.cpu().tolist(),
            "measured_object_masses": measured_mass.cpu().tolist(),
            "maximum_joint_speed": maximum_joint_velocity,
            "maximum_object_velocity_component": maximum_object_speed,
            "asset_source_sha256": manifest["source_urdf_sha256"],
            "converted_asset_sha256": raw.asset_sha256,
            "robot_live_properties": body_properties,
            "checks": {
                "fixed_base": True,
                "source_joint_order": True,
                "shape_mass_assignment": True,
                "hybrid_action_targets": True,
                "current_reward_observation": True,
                "deferred_terminal_reset": True,
                "task_state_restore": True,
                "mutation_readback": True,
            },
            "scope": "Execution and contract smoke only; not learned-policy or cross-simulator dynamics equivalence.",
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("DEXPBT_ORIGINAL_SMOKE", json.dumps(report, allow_nan=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        app.close(exit_code=code)
    raise SystemExit(code)
