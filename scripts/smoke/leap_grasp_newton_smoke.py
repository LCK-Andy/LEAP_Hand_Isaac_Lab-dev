"""Validate Stage A full-force grasp on Newton/MJWarp without launching Kit.

Run with env_isaaclab Python. Configuration is resolved with the official
``newton_mjwarp`` preset before entering ``launch_simulation``. This tests the
current backend and the shared fixture separately; live contact reporting does
not imply a PhysX-calibrated fixture satisfies Newton's grasp-force bands.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

TASK = "Isaac-Grasp-Cube-LEAP-Direct-v0"


def run(args, result):
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab.app.sim_launcher import launch_simulation
    from isaaclab.utils import has_kit
    from isaaclab_tasks.utils.hydra import resolve_task_config

    import LEAP_Isaaclab.tasks  # noqa: F401
    from LEAP_Isaaclab.tasks.leap_hand_grasp.contact_forces import read_contact_sample

    sys.argv = [sys.argv[0], "presets=newton_mjwarp"]
    cfg, _ = resolve_task_config(TASK, "rl_games_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 42
    # This script validates an experimental backend; normal training stays gated.
    cfg.allow_unvalidated_backend = True
    assert type(cfg.sim.physics).__name__ == "NewtonCfg", "Newton preset did not resolve"
    assert cfg.sim.physics.solver_cfg.use_mujoco_contacts, "MJWarp internal contacts are required for this probe"
    assert not cfg.sim.physics.solver_cfg.use_mujoco_cpu, "MuJoCo CPU update_contacts is not implemented"
    assert cfg.sim.physics.collision_cfg is None
    assert not cfg.sim.use_newton_actuators, "Native actuator path fuses decimation and loses physics-tick history"
    assert cfg.decimation == 4 and abs(cfg.sim.dt - 1 / 120) < 1e-9
    result.update(physics_dt=cfg.sim.dt, decimation=cfg.decimation)
    launcher = {
        "headless": True,
        "device": args.device,
        "enable_cameras": False,
        "visualizer": None,
        "visualizer_explicit": True,
    }
    with launch_simulation(cfg, launcher):
        assert not has_kit(), "Unexpected Kit launch: this must be a kitless test"
        result["kit_launched"] = False
        env = gym.make(TASK, cfg=cfg)
        try:
            uw = env.unwrapped
            assert uw._physics_backend == "newton" and not uw._physics_handles_decimation
            from isaaclab_newton.physics import NewtonManager

            from LEAP_Isaaclab.tasks.leap_hand_grasp.newton_geometry import validate_newton_collision_coverage

            result["collision_coverage"] = validate_newton_collision_coverage(
                NewtonManager.get_model(), [*uw.fingertip_contact_sensors.values(), uw.support_sensor]
            )
            result["geometry_preparation"] = getattr(uw, "_newton_geometry_report", None)
            obs, _ = env.reset(seed=42)
            assert obs["policy"].shape == (args.num_envs, cfg.observation_space)
            assert torch.isfinite(obs["policy"]).all()
            assert torch.all(uw.fingertip_contact_forces == 0), "Reset retained stale tactile force"
            target = uw.cur_targets.clone()
            root = uw.hand.data.root_pos_w.torch.clone()
            zero = torch.zeros(args.num_envs, 16, device=uw.device)
            normals, frictions, valid, support, positions = [], [], [], [], []
            max_root_drift = 0.0
            terminated_count = 0
            persistent_hold_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=uw.device)
            completed = []
            for step in range(args.steps):
                obs, reward, terminated, _, extras = env.step(zero)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                torch.testing.assert_close(uw.cur_targets, target)
                torch.testing.assert_close(uw.fingertip_contact_forces, uw._normal + uw._friction)
                terminated_count += int(terminated.sum())
                point_valid = []
                for name in cfg.fingertip_body_names:
                    sensor = uw.fingertip_contact_sensors[name]
                    sample = read_contact_sample(sensor, args.num_envs, uw.physics_dt)
                    assert sample.history_normal.shape == (args.num_envs, 4, 3)
                    assert sample.history_friction.shape == (args.num_envs, 4, 3)
                    # Independent backend total: proves the decomposition does not discard friction.
                    raw_total = torch.from_dlpack(sensor.contact_view.force_matrix).reshape(args.num_envs, 3)
                    live = ~uw._fresh_reset
                    torch.testing.assert_close(sample.total[live], raw_total[live], rtol=1e-4, atol=1e-4)
                    assert torch.all(sample.total[~live] == 0), "Reset reintroduced stale native contact forces"
                    point_valid.append(sample.position_valid)
                positions.append(torch.stack(point_valid, dim=1))
                assert uw._support_force.shape == (args.num_envs, 4)
                normals.append(torch.linalg.vector_norm(uw._normal, dim=-1).clone())
                frictions.append(torch.linalg.vector_norm(uw._friction, dim=-1).clone())
                valid.append(uw._score.valid.clone())
                support.append(uw._support_force.amax(dim=1).clone())
                persistent_hold_seen |= uw.grasp_state.hold_seconds > cfg.hold_duration + uw.step_dt
                max_root_drift = max(max_root_drift, float((uw.hand.data.root_pos_w.torch - root).norm(dim=-1).amax()))
                metrics = extras.get("grasp_hold", {})
                if metrics:
                    completed.append({key: value.tolist() for key, value in metrics.items() if torch.is_tensor(value)})
                result["steps_completed"] = step + 1
            normal = torch.stack(normals)
            friction = torch.stack(frictions)
            result.update(
                normal_force_peak_per_tip=normal.amax(dim=(0, 1)).tolist(),
                friction_force_peak_per_tip=friction.amax(dim=(0, 1)).tolist(),
                loaded_position_fraction=torch.stack(positions).float().mean(dim=(0, 1)).tolist(),
                valid_fraction=torch.stack(valid).float().mean(dim=0).tolist(),
                support_force_max=float(torch.stack(support).amax()),
                root_drift_max=max_root_drift,
                terminated=terminated_count,
                persistent_hold_seen=persistent_hold_seen.tolist(),
                completed=completed,
                condition_key=uw.condition_key,
                full_force_reporting_live=bool((normal.amax(dim=(0, 1)) > 0.1).all() and friction.amax() > 0.01),
            )
            assert result["full_force_reporting_live"], "Not all tips report normal force or friction is not live"
            assert max_root_drift < 1e-3, "Hand root is not fixed"
            assert terminated_count == 0, "Newton fixture dropped or exceeded force/safety limits"
            assert torch.stack(valid).float().mean() > 0.95, "Newton fixture four-pulp valid occupancy below 95%"
            if args.steps * uw.step_dt > cfg.hold_duration + 0.1:
                assert persistent_hold_seen.all(), "Newton fixture did not maintain the requested persistent hold"
            if args.steps >= uw.max_episode_length:
                assert completed and all(completed[0]["completed_retained"]), "Retention metrics missing or failed"
            result["passed"] = True
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=330)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num_envs and --steps must be positive")
    result = {"task": TASK, "backend": "newton_mjwarp", "num_envs": args.num_envs, "steps": args.steps, "passed": False}
    try:
        run(args, result)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    print(text, flush=True)
    if args.output:
        args.output.write_text(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
