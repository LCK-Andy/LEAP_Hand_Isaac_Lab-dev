"""Physically step the Stage A grasp and verify full-force, retention, and reset contracts."""

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


def main():
    cfg = parse_env_cfg("Isaac-Grasp-Cube-LEAP-Direct-v0", device=args.device, num_envs=args.num_envs)
    cfg.seed = 42
    env = gym.make("Isaac-Grasp-Cube-LEAP-Direct-v0", cfg=cfg)
    uw = env.unwrapped
    from pxr import Usd, UsdGeom

    from isaaclab.sim.utils.stage import get_current_stage

    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], True)
    extent = bound.ComputeWorldBound(
        get_current_stage().GetPrimAtPath("/World/envs/env_0/object")
    ).ComputeAlignedRange()
    measured_extent = [float(extent.GetSize()[axis]) for axis in range(3)]
    print(f"Cube world-aligned spawn extent: {measured_extent}", flush=True)
    obs, _ = env.reset(seed=42)
    assert obs["policy"].shape == (args.num_envs, cfg.observation_space)
    assert torch.isfinite(obs["policy"]).all()
    assert torch.all(uw.fingertip_contact_forces == 0), "Reset must clear stale tactile observations"
    target = uw.cur_targets.clone()
    zero = torch.zeros(args.num_envs, 16, device=uw.device)
    valid, normal, friction, support, slips = [], [], [], [], []
    completed = []
    terminated_count = 0
    for step in range(args.steps):
        obs, reward, terminated, truncated, extras = env.step(zero)
        assert reward.shape == (args.num_envs,) and torch.isfinite(reward).all()
        assert torch.isfinite(obs["policy"]).all()
        torch.testing.assert_close(uw.cur_targets, target)
        terminated_count += int(terminated.sum())
        torch.testing.assert_close(uw.fingertip_contact_forces, uw._normal + uw._friction)
        valid.append(uw._score.valid.clone())
        normal.append(torch.linalg.vector_norm(uw._normal, dim=-1))
        friction.append(torch.linalg.vector_norm(uw._friction, dim=-1))
        support.append(uw._support_force.amax(dim=1))
        slips.append(uw._slip_speed.clone())
        data = extras.get("grasp_hold", {})
        if data:
            completed.append({key: value.tolist() for key, value in data.items() if torch.is_tensor(value)})
        if step == 89:
            assert uw.grasp_state.success.all(), "Verified fixture must achieve a persistent two-second hold"
            assert (uw.grasp_state.hold_seconds > 2.0).all(), "Success must not reset the hold clock"
    # Exercise real contact loss and same-step autoreset, not just a static fixture.
    free_pose = uw.object.data.root_link_pose_w.torch.clone()
    free_pose[:, 2] += 0.20
    ids = torch.arange(uw.num_envs, device=uw.device)
    uw.object.write_root_pose_to_sim_index(root_pose=free_pose, env_ids=ids)
    uw.object.write_root_velocity_to_sim_index(root_velocity=torch.zeros(uw.num_envs, 6, device=uw.device), env_ids=ids)
    obs, reward, dropped, _, extras = env.step(zero)
    assert dropped.all() and torch.isfinite(reward).all()
    assert torch.isfinite(obs["policy"]).all()
    assert torch.isfinite(extras["final_obs"]["policy"]).all(), "Contact loss must not poison terminal observations"
    assert not uw._score.unsafe.any(), "No contact is not a force overload"
    assert not uw.grasp_state.success.any() and not uw.grasp_state.steps.any()
    assert not uw.fingertip_contact_forces.any(), "Autoreset returned stale forces"
    reset_target = uw.cur_targets.clone()
    env.step(torch.full_like(zero, 0.5))
    expected_target = torch.clamp(
        reset_target + 0.5 * cfg.max_joint_delta, min=uw.hand_dof_lower_limits, max=uw.hand_dof_upper_limits
    )
    torch.testing.assert_close(uw.cur_targets, expected_target)
    results = {
        "backend": uw._physics_backend,
        "steps": args.steps,
        "num_envs": args.num_envs,
        "observation_dim": cfg.observation_space,
        "cube_extent_m": measured_extent,
        "contact_loss_reset_passed": True,
        "one_increment_per_control_step_passed": True,
        "terminated": terminated_count,
        "valid_fraction": torch.stack(valid).float().mean(dim=0).tolist(),
        "normal_force_mean": torch.stack(normal).mean(dim=(0, 1)).tolist(),
        "friction_force_mean": torch.stack(friction).mean(dim=(0, 1)).tolist(),
        "support_force_max": float(torch.stack(support).amax()),
        "slip_speed_max": float(torch.stack(slips).amax()),
        "completed": completed,
        "condition_key": uw.condition_key,
    }
    print(json.dumps(results, indent=2))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    assert terminated_count == 0, "Zero-action fixture dropped or exceeded the safety ceiling"
    assert torch.stack(valid).float().mean() > 0.95, "Four-pulp valid occupancy is too low"
    assert torch.stack(friction).amax() > 0.01, "Tangential forces must be live, not zeros"
    if args.steps >= uw.max_episode_length:
        assert completed and all(completed[0]["completed_retained"]), "Finite-horizon retention metrics missing"
    env.close()
    print("STAGE A GRASP SMOKE PASSED")


exit_code = 0
try:
    main()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    app.close(exit_code=exit_code)
