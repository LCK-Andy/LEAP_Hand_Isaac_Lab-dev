"""Exercise Cartesian action bounds, palm motion, reset isolation and native episodes."""

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--axis-sweep", action="store_true", help="Also test all six signed Cartesian axes")
AppLauncher.add_app_launcher_args(parser)
if "--headless" not in parser._option_string_actions:
    parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import compute_pose_error
from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401
from LEAP_Isaaclab.tasks.dexsuite_reorient_cartesian.control import OBS_DIM

TASK = "Isaac-Dexsuite-Kuka-Allegro-Reorient-Cartesian-v0"


def main():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    assert cfg.sim.dt == 1 / 120 and cfg.decimation == 2 and cfg.episode_length_s == 6.0
    assert cfg.scene.robot.spawn.articulation_props[0].solver_position_iteration_count == 32
    env = gym.make(TASK, cfg=cfg)
    result = {
        "task": TASK,
        "num_envs": args.num_envs,
        "physics_dt": cfg.sim.dt,
        "decimation": cfg.decimation,
    }
    try:
        raw = env.unwrapped
        obs, _ = env.reset(seed=42)
        dims = {key: value.shape[-1] for key, value in obs.items()}
        assert dims == {
            "policy": 165,
            "proprio": 615,
            "perception": 960,
            "controller": 22,
        }, dims
        assert sum(dims.values()) == OBS_DIM
        assert raw.action_manager.total_action_dim == 22
        arm = raw.action_manager.get_term("arm")
        result["observation_dims"] = dims
        result["arm_joints"] = arm._joint_names
        result["control_contract"] = raw.control_contract()
        assert raw.get_evaluation_condition()["cartesian_control"]
        initial_state = raw.get_env_state()
        bad = dict(initial_state)
        bad.pop("cartesian_control")
        try:
            raw.set_env_state(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Joint-control checkpoint was accepted")
        raw.set_env_state(initial_state)
        assert raw.get_env_state() == initial_state

        zero = torch.zeros(args.num_envs, 22, device=raw.device)
        # Use the native default robot posture, away from random wrist-limit resets,
        # for the mechanical axis probe only. The episode test below uses native resets.
        raw.configure_fixed_evaluation(0.0)
        env.reset(seed=43)
        robot = raw.scene["robot"]
        ids = torch.arange(args.num_envs, device=raw.device)
        q = robot.data.default_joint_pos.torch.clone()
        qd = torch.zeros_like(q)
        robot.write_joint_state_to_sim_index(position=q, velocity=qd, env_ids=ids)
        robot.set_joint_position_target_index(target=q)
        raw.scene.write_data_to_sim()
        raw.sim.forward()
        raw.scene.update(raw.physics_dt)
        arm.reset()
        for _ in range(20):
            env.step(zero)
        probe_start = arm._compute_frame_pose()[0].clone()
        action = zero.clone()
        action[:, 0] = 0.5
        max_target_slew = 0.0
        max_joint_error = 0.0
        for _ in range(60):
            previous = arm.joint_target.clone()
            obs, reward, terminated, truncated, _ = env.step(action)
            assert not (terminated | truncated).any(), "Axis fixture ended unexpectedly"
            assert torch.isfinite(reward).all() and all(torch.isfinite(value).all() for value in obs.values())
            slew = (arm.joint_target - previous).abs().max().item()
            max_target_slew = max(max_target_slew, slew)
            assert slew <= max(cfg.actions.arm.joint_target_speed) * raw.step_dt + 1e-5, slew
            q = robot.data.joint_pos.torch[:, arm._joint_ids]
            max_joint_error = max(max_joint_error, (arm.joint_target - q).abs().max().item())
        displacement = arm._compute_frame_pose()[0] - probe_start
        result["x_probe_displacement_m"] = displacement.cpu().tolist()
        result["max_command_slew_rad_per_policy_step"] = max_target_slew
        result["max_command_tracking_error_rad"] = max_joint_error
        assert displacement[:, 0].median() > 0.001, displacement

        # Action reset must clear only the selected environment, and observations
        # must expose the reset robot pose rather than an old controller target.
        previous_target = arm.joint_target.clone()
        reset_ids = torch.tensor([0], device=raw.device)
        raw._reset_idx(reset_ids)
        assert not arm._initialized[0] and arm._initialized[1:].all()
        assert torch.equal(arm.joint_target[1:], previous_target[1:])
        controller_obs = arm.observation()
        assert torch.isfinite(controller_obs).all()
        assert torch.equal(controller_obs[0, 14:20], torch.zeros(6, device=raw.device))
        for fraction in (0.0, 1.0):
            raw.configure_fixed_evaluation(fraction)
            env.reset(seed=44)
            completed = 0
            success = 0
            for _ in range(raw.max_episode_length + 1):
                obs, reward, terminated, truncated, info = env.step(zero)
                assert reward.shape == (args.num_envs,) and torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                completed += info["dexpbt"]["completed_episodes"]
                success += info["dexpbt"]["success_count"]
                assert info["dexpbt"]["mixed_condition_episodes"] == 0
            assert completed >= args.num_envs, completed
            result[f"adr_{fraction}_completed_episodes"] = completed
            result[f"adr_{fraction}_successes"] = success
        if args.axis_sweep:
            raw.configure_fixed_evaluation(0.0)
            sweep = []
            for axis in range(6):
                for sign in (-1.0, 1.0):
                    env.reset(seed=43)
                    posture = robot.data.default_joint_pos.torch.clone()
                    robot.write_joint_state_to_sim_index(
                        position=posture, velocity=torch.zeros_like(posture), env_ids=ids
                    )
                    robot.set_joint_position_target_index(target=posture)
                    raw.scene.write_data_to_sim()
                    raw.sim.forward()
                    raw.scene.update(raw.physics_dt)
                    arm.reset()
                    for _ in range(20):
                        env.step(zero)
                    start_pos, start_quat = (value.clone() for value in arm._compute_frame_pose())
                    command = zero.clone()
                    command[:, axis] = sign * 0.5
                    for _ in range(60):
                        axis_obs, axis_reward, terminated, truncated, _ = env.step(command)
                        assert not (terminated | truncated).any()
                        assert torch.isfinite(axis_reward).all()
                        assert all(torch.isfinite(value).all() for value in axis_obs.values())
                    end_pos, end_quat = arm._compute_frame_pose()
                    translation, rotation = compute_pose_error(
                        start_pos, start_quat, end_pos, end_quat, rot_error_type="axis_angle"
                    )
                    delta = torch.cat((translation, rotation), dim=-1)
                    signed = sign * delta[:, axis]
                    row = {
                        "axis": axis,
                        "sign": sign,
                        "delta": delta.cpu().tolist(),
                        "signed_median": signed.median().item(),
                    }
                    sweep.append(row)
                    threshold = 0.001 if axis < 3 else 0.005
                    assert signed.median() > threshold, row
            result["axis_sweep"] = sweep
        result["status"] = "pass"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("CARTESIAN_SMOKE_PASS", json.dumps(result), flush=True)
    finally:
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
