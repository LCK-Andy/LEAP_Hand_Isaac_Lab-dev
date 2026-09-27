"""Check TD-MPC2 Stage A/B terminal replay and native grasp metrics in real PhysX."""

# flake8: noqa: E402

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--stage", choices=("a", "b"), default="b")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
if "--headless" not in parser._option_string_actions:
    parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
app = AppLauncher(args).app


def main():
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    import LEAP_Isaaclab.tasks  # noqa: F401
    from scripts.tdmpc2.grasp import GraspEpisodeMetrics, GraspVectorEnv, configure_grasp
    from scripts.tdmpc2.grasp_motion import GraspMotionEpisodeMetrics, GraspMotionVectorEnv, configure_grasp_motion
    from scripts.tdmpc2.replay import SequenceReplay

    stage_b = args.stage == "b"
    if stage_b:
        task = "Isaac-Grasp-Motion-Cube-LEAP-Direct-v0"
        adapter, metrics_type, configure = GraspMotionVectorEnv, GraspMotionEpisodeMetrics, configure_grasp_motion
    else:
        task = "Isaac-Grasp-Cube-LEAP-Direct-v0"
        adapter, metrics_type, configure = GraspVectorEnv, GraspEpisodeMetrics, configure_grasp
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    configure(cfg, 1)
    native_env = gym.make(task, cfg=cfg)
    try:
        env = adapter(native_env)
        obs = env.reset(seed=1)
        metrics = metrics_type(env.num_envs, env.device)
        replay = SequenceReplay(330 * env.num_envs, env.num_envs, env.obs_dim, 16, 3, 256, sample_device=env.device)
        zero = torch.zeros(env.num_envs, 16, device=env.device)
        native = env.env
        targets = native.cur_targets.clone()
        pose_writer, velocity_writer = native._write_obj_root_pose, native._write_obj_root_vel

        def reset_pose(**kwargs):
            assert (native.episode_length_buf[kwargs["env_ids"]] == 0).all()
            return pose_writer(**kwargs)

        def reset_velocity(**kwargs):
            assert (native.episode_length_buf[kwargs["env_ids"]] == 0).all()
            return velocity_writer(**kwargs)

        native._write_obj_root_pose, native._write_obj_root_vel = reset_pose, reset_velocity
        timeout_count = 0
        valid, tracking = [], []
        for step in range(330):
            result = env.step(zero)
            assert torch.isfinite(result.obs).all() and torch.isfinite(result.next_obs).all()
            assert torch.isfinite(result.reward).all()
            torch.testing.assert_close(native.cur_targets, targets)
            done = result.terminated | result.truncated
            if done.any():
                assert step == 299, "Stationary fixture failed before the finite endpoint"
                assert result.truncated.all() and result.terminated.all()
                assert not torch.equal(result.next_obs, result.obs), "Replay received reset states"
                timeout_count += int(result.truncated.sum())
            else:
                torch.testing.assert_close(result.next_obs, result.obs)
            replay.add(obs, zero, result.reward, result.next_obs, result.terminated, result.truncated, env.condition())
            metrics.observe(result.reward, done, result.infos)
            obs = result.obs
            valid.append(native._score.valid.float().mean().item())
            hold = native.grasp_state.hold_seconds.clone()
            if stage_b:
                tracking.append(native.tracking.tracking_valid.float().mean().item())
                velocity = native.motion_state.angular_velocity.clone()
                native._get_observations()
                torch.testing.assert_close(native.motion_state.angular_velocity, velocity)
            else:
                native._get_observations()
            torch.testing.assert_close(native.grasp_state.hold_seconds, hold)
        assert timeout_count == env.num_envs
        assert replay.terminated[299].eq(1).all() and replay.truncated[299].all()
        assert sum(valid) / len(valid) > 0.95
        if stage_b:
            assert sum(tracking) / len(tracking) < 0.2
        else:
            assert metrics.values()["episode/success_count"] == env.num_envs
        assert metrics.values()["episode/completed"] == env.num_envs
        report = {
            "task": task,
            "num_envs": env.num_envs,
            "steps": 330,
            "finite_timeouts": timeout_count,
            "holding_valid_fraction": sum(valid) / len(valid),
            "stationary_tracking_fraction": sum(tracking) / len(tracking) if stage_b else None,
            "metrics": metrics.values(),
            "contract": env.contract_metadata(),
            "replay": replay.stats,
            "meaning": "Fixed-fixture execution, finite timeout, reset and replay test; not learned acquisition",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        print(f"TDMPC2_STAGE_{args.stage.upper()}_SMOKE_PASSED", flush=True)
    finally:
        native_env.close()


exit_code = 1
try:
    main()
    exit_code = 0
except KeyboardInterrupt:
    exit_code = 130
except Exception:
    traceback.print_exc()
finally:
    app.close(exit_code=exit_code)
