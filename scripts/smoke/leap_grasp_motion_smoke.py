"""Exercise the Stage B MDP with real physics; zero-action holding is not motion acquisition."""

# flake8: noqa: E402

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=330)
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
from LEAP_Isaaclab.tasks.leap_hand_grasp.leap_grasp_env import LeapGraspEnv


def main():
    task = "Isaac-Grasp-Motion-Cube-LEAP-Direct-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 42
    cfg.motion.randomize_phase = False
    env = gym.make(task, cfg=cfg)
    uw = env.unwrapped
    obs, _ = env.reset(seed=42)
    assert obs["policy"].shape == (args.num_envs, 148)
    torch.testing.assert_close(obs["policy"][:, :124], LeapGraspEnv._get_observations(uw)["policy"])
    torch.testing.assert_close(uw.reference.quaternion, uw._fixture_pose[3:].expand(uw.num_envs, -1))
    zero = torch.zeros(args.num_envs, 16, device=uw.device)
    phases = uw._motion_phase.clone()
    target = uw.cur_targets.clone()
    holding, tracking, rotations, reference_rotations, completed = [], [], [], [], []
    from LEAP_Isaaclab.tasks.leap_hand_grasp_motion.motion_logic import rotation_error_vector

    original_pose_writer = uw._write_obj_root_pose
    original_velocity_writer = uw._write_obj_root_vel

    def reset_only_pose(**kwargs):
        assert (uw.episode_length_buf[kwargs["env_ids"]] == 0).all(), "Object pose written outside reset"
        return original_pose_writer(**kwargs)

    def reset_only_velocity(**kwargs):
        assert (uw.episode_length_buf[kwargs["env_ids"]] == 0).all(), "Object velocity written outside reset"
        return original_velocity_writer(**kwargs)

    uw._write_obj_root_pose = reset_only_pose
    uw._write_obj_root_vel = reset_only_velocity
    for _ in range(args.steps):
        obs, reward, terminated, truncated, extras = env.step(zero)
        assert not terminated.any(), "Holding fixture should not fail a hard safety limit"
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        torch.testing.assert_close(uw.cur_targets, target)
        torch.testing.assert_close(uw._motion_phase, phases)
        holding.append(uw._score.valid.clone())
        tracking.append(uw.tracking.tracking_valid.clone())
        rotations.append(rotation_error_vector(uw._fixture_pose[3:].expand(uw.num_envs, -1), uw.object_rot))
        reference_rotations.append(
            rotation_error_vector(uw._fixture_pose[3:].expand(uw.num_envs, -1), uw.reference.quaternion)
        )
        state = uw.motion_state.angular_velocity.clone()
        ref_velocity = uw._filtered_ref_angular_velocity.clone()
        uw._get_observations()
        uw._get_observations()
        torch.testing.assert_close(uw.motion_state.angular_velocity, state)
        torch.testing.assert_close(uw._filtered_ref_angular_velocity, ref_velocity)
        if extras.get("grasp_motion"):
            completed.append(
                {key: value.tolist() for key, value in extras["grasp_motion"].items() if torch.is_tensor(value)}
            )
        if truncated.any():
            assert not uw._tracking_steps.any()
            assert not uw.motion_state.angular_velocity.any()
            assert torch.isfinite(extras["final_obs"]["policy"]).all()
    report = {
        "task": task,
        "num_envs": uw.num_envs,
        "steps": args.steps,
        "observation_dim": 148,
        "holding_valid_fraction": torch.stack(holding).float().mean().item(),
        "stationary_tracking_fraction": torch.stack(tracking).float().mean().item(),
        "reference_rotation_max_rad": torch.stack(reference_rotations).norm(dim=-1).amax().item(),
        "actual_rotation_max_rad": torch.stack(rotations).norm(dim=-1).amax().item(),
        "completed": completed,
        "condition_key": uw.condition_key,
        "meaning": "MDP execution/reset check; not learned tracking or inertial identification",
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["holding_valid_fraction"] > 0.95
    assert report["reference_rotation_max_rad"] > 0.001
    assert report["stationary_tracking_fraction"] < 0.2, "Stationary holding must not solve motion tracking"
    if args.steps >= uw.max_episode_length:
        assert completed and all(completed[0]["completed_retained"])
    env.close()
    print("STAGE B EXECUTION SMOKE PASSED")


exit_code = 0
try:
    main()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    app.close(exit_code=exit_code)
