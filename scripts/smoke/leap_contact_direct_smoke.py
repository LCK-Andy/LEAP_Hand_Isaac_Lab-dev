"""
Smoke test for the Isaac-Contact-Cube-LEAP-Direct-v0 task (touch 4 sampled
cube-surface contact points with the fingertip pulps).

Checks, headless, with 4 envs:
  1. The task is registered and the LEAP hand articulation is in the scene
     (16 joints a_0..a_15) with the 4 LEAP fingertip bodies and pulp sensors.
  2. Reset works and the policy observation is the 184-d vector (136 repose
     + 4x4 tip->point relative vectors); the sampled points sit on the cube
     surface (one coordinate at +-half extent) and respect the minimum
     pairwise separation.
  3. The spawned cube's bounding box is measured and compared against
     cfg.cube_half_extent (pin the cfg from this measurement).
  4. Stepping with zero actions keeps rewards finite and the intermediate
     buffers (matched_dist / touched / hold_timer) sane.
  5. Success path: after curling the fingers onto the cube (pulp forces live),
     the points are colocated with the pulp positions in a frozen geometry
     snapshot. The real 0.1 s timer must wait three reward steps, then score
     and resample the points without bypassing the measured-force gate.

Run:
  ~/env_isaaclab/bin/python scripts/smoke/leap_contact_direct_smoke.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Contact-Cube-LEAP-Direct smoke test.")
parser.add_argument("--task", default="Isaac-Contact-Cube-LEAP-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from pxr import Usd, UsdGeom

from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401  (registers Isaac-Contact-Cube-LEAP-Direct-v0)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 42
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env_cfg = env.unwrapped.cfg
    from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt.contact import attach_measurement

    attach_measurement(env.unwrapped)

    # [1] scene check: LEAP articulation with 16 joints, 4 pulps with sensors
    hand = env.unwrapped.hand
    joint_names = list(hand.joint_names)
    assert len(joint_names) == 16, f"expected 16 LEAP joints, got {joint_names}"
    for fingertip in env_cfg.fingertip_body_names:
        assert fingertip in hand.body_names, f"fingertip body '{fingertip}' not in hand bodies"
    assert env.unwrapped.fingertip_contact_sensors, "pulp contact sensors missing"
    print(f"[1] robot OK: 16 LEAP joints, pulps {env_cfg.fingertip_body_names} with sensors")

    # [2] reset + observation + point-set validity
    obs, _ = env.reset()
    policy_obs = obs["policy"]
    assert policy_obs.shape == (args_cli.num_envs, 184), f"unexpected obs shape {tuple(policy_obs.shape)}"
    assert torch.isfinite(policy_obs).all(), "non-finite values in policy obs"
    half, min_sep = env_cfg.cube_half_extent, env_cfg.min_point_separation
    points = env.unwrapped.contact_points_local  # (N, 4, 3) cube frame
    face_coord = points.abs().amax(dim=-1)
    assert torch.allclose(face_coord, torch.full_like(face_coord, half), atol=1e-5), (
        f"points not on the cube surface: max |coord| {face_coord.min():.4f}..{face_coord.max():.4f} vs h={half}"
    )
    assert (points.abs() <= half + 1e-5).all(), "point coordinate outside the cube"
    pairwise = torch.cdist(points, points)
    pairwise[:, torch.arange(4), torch.arange(4)] = float("inf")
    assert pairwise.amin(dim=(1, 2)).min() >= min_sep - 1e-5, "minimum point separation violated"
    print(f"[2] reset OK, obs {tuple(policy_obs.shape)}; points on surface, min pairwise "
          f"{pairwise.amin(dim=(1, 2)).min():.3f} m >= {min_sep} m")

    # [3] measure the spawned cube's bbox (untransformed = geometry frame)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], True)
    obj_prim = get_current_stage().GetPrimAtPath("/World/envs/env_0/object")
    extent = bbox_cache.ComputeUntransformedBound(obj_prim).ComputeAlignedRange()
    measured = [(extent.GetMax()[i] - extent.GetMin()[i]) / 2.0 for i in range(3)]
    print(f"[3] cube half extents measured: {[f'{m:.4f}' for m in measured]} m vs cfg {half} m")
    assert all(abs(m - half) / half < 0.25 for m in measured), (
        f"cfg.cube_half_extent {half} is >25% off the measured bbox {measured} -- pin the cfg to the measurement"
    )

    # [4] zero-action stepping: finite rewards, sane intermediates
    zero_action = torch.zeros(args_cli.num_envs, env.action_space.shape[1], device=env.unwrapped.device)
    for _ in range(60):
        obs, reward, terminated, truncated, extras = env.step(zero_action)
        assert reward.shape == (args_cli.num_envs,), f"reward shape {tuple(reward.shape)}"
        assert torch.isfinite(reward).all(), "non-finite reward"
        assert torch.isfinite(obs["policy"]).all(), "non-finite obs while stepping"
    uw = env.unwrapped
    assert (uw.matched_dist >= 0).all() and torch.isfinite(uw.matched_dist).all(), "bad matched_dist"
    assert (uw.hold_timer >= 0).all(), "negative hold timer"
    print(f"[4] 60 zero-action steps OK; matched_dist max {uw.matched_dist.max():.3f} m, "
          f"touched/step max {uw.touched.sum(-1).max():.0f}, successes {uw.successes.sum():.0f}")

    # [5] success path: curl onto the cube, colocate points with pressing pulps,
    # and exercise the real three-step hold on a controlled geometry snapshot.
    env.reset(seed=42)
    curl = torch.ones(args_cli.num_envs, env.action_space.shape[1], device=uw.device)
    # All-ones thumb targets press the hand, not necessarily the cube. This
    # measured fixture opposes the thumb; stop at an actual four-partner contact.
    thumb_targets = {"a_12": 0.2259197, "a_13": 0.9754373, "a_14": 0.6517336, "a_15": 0.4591985}
    for joint_index, name in enumerate(uw.hand.joint_names):
        if name in thumb_targets:
            curl[:, joint_index] = thumb_targets[name]
    for _ in range(90):
        env.step(curl)
        forces = uw.fingertip_contact_forces
        force_mag = torch.linalg.norm(forces, dim=-1)
        if (force_mag > env_cfg.touch_force).all(-1).any():
            break
    print(f"    after curling: max pulp force per tip {force_mag.max(dim=0).values.tolist()} N "
          f"(threshold {env_cfg.touch_force} N)")

    def pulp_colocated_points() -> torch.Tensor:
        # points AT the pulp centers, in cube-local frame (flat inputs for the compiled quat helper)
        tips_rel = (uw.fingertip_pos - uw.object_pos.unsqueeze(1)).reshape(-1, 3)
        quat = uw.object_rot.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
        return quat_apply_inverse(quat, tips_rel).reshape(uw.num_envs, 4, 3)

    assert forces.shape == (args_cli.num_envs, 4, 3)
    assert (force_mag.max(dim=0).values > env_cfg.touch_force).all(), "one or more pulp sensors are dead"
    pressing = (force_mag > env_cfg.touch_force).all(-1)
    assert pressing.any(), "curl probe never established a four-pulp contact snapshot"
    # Controlled geometry snapshot: exercise the REAL timer and measured forces
    # without integrating physics between reward calls. Production point sampling
    # and hold ranges are untouched. A target below reward epsilon would falsely
    # confirm even with zero timer, so use the real 0.1 s native hold duration.
    uw.contact_points_local[:] = pulp_colocated_points()
    uw.hold_target[:] = 3 * uw.step_dt
    uw.hold_timer.zero_()
    uw._compute_intermediate_values()
    assert (uw.touched[pressing].sum(-1) == 4).all()
    successes_before = uw.successes.sum().item()
    for _ in range(2):
        uw._get_rewards()
        assert uw.successes.sum().item() == successes_before, "native hold confirmed early"
    uw._get_rewards()
    successes_after = uw.successes.sum().item()
    resampled = not torch.allclose(uw.contact_points_local, pulp_colocated_points(), atol=1e-6)
    print(f"[5] colocated-points confirm: successes {successes_before:.0f} -> {successes_after:.0f}, "
          f"points resampled after confirm: {resampled}")
    assert successes_after >= successes_before + 1, (
        "hold-confirm never fired with points colocated on pressing pulps -- check the force gate pairing"
    )
    assert "consecutive_successes" in uw.extras.get("log", {}), "missing consecutive_successes log"

    env.close()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
    simulation_app.close()
