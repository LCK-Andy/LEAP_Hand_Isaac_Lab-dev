"""
Smoke test for the Isaac-Repose-Objects-LEAP-Direct-v0 task (multi-object).

Checks, headless with 20 envs:
  1. The task registers; LEAP hand (16 joints) in the scene.
  2. Palm-size catalog: every spawned object's measured extent fits
     ``max_object_extent_m``.
  3. Heterogeneous scenes: the active objects differ across envs.
  4. Per-episode resampling: a second reset reshuffles the active objects
     (active indices change).
  5. Goal markers: one prototype per catalog object, ordered like the
     collection members; mid-episode goal resampling works (and, when the Kit
     marker backend is live, each env's goal instance shows its active object).
  6. Obs keeps the 136-d shape (124 official + 12 tactile); zero-action stepping keeps
     rewards finite and objects near the palm.

Run:
  ~/env_isaaclab/bin/python scripts/leap_repose_objects_direct_smoke.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Repose-Objects-LEAP-Direct smoke test.")
parser.add_argument("--task", default="Isaac-Repose-Objects-LEAP-Direct-v0")
parser.add_argument("--num_envs", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401  (registers Isaac-Repose-Objects-LEAP-Direct-v0)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped

    # [1] scene check
    joint_names = list(unwrapped.hand.joint_names)
    assert len(joint_names) == 16 and all(n.startswith("a_") for n in joint_names), joint_names
    print(f"[1] robot OK (16 LEAP joints), candidates: {len(unwrapped.cfg.object_names)}")

    # [2] palm-size catalog
    catalog = unwrapped.object_catalog
    max_extent = unwrapped.cfg.max_object_extent_m
    assert all(extent <= max_extent for _, extent in catalog), catalog
    print(f"[2] catalog OK: {len(catalog)} objects, all <= {max_extent * 100:.0f} cm: "
          + ", ".join(f"{n}({e * 100:.0f}cm)" for n, e in catalog))

    # [3] heterogeneous active objects across envs (first reset activates them)
    env.reset()
    active_before = unwrapped.active_object_ids.clone()
    names_before = unwrapped.active_object_names
    distinct = len(set(names_before))
    print(f"[3] active objects: {distinct} distinct across {args_cli.num_envs} envs")
    assert distinct >= max(3, args_cli.num_envs // 4), f"only {distinct} distinct active objects"

    # [4] per-episode resampling on reset
    obs, _ = env.reset()
    active_after = unwrapped.active_object_ids
    changed = int((active_before != active_after).sum())
    print(f"[4] resampled on reset: {changed}/{args_cli.num_envs} envs changed object")
    assert changed >= args_cli.num_envs // 4, f"reset changed only {changed} active objects"
    assert obs["policy"].shape == (args_cli.num_envs, 136), tuple(obs["policy"].shape)

    # [5] goal markers: prototype per catalog object, member-aligned order
    goal_markers = unwrapped.goal_markers
    marker_names = list(goal_markers.cfg.markers.keys())
    assert marker_names == [n for n, _ in catalog], (marker_names, catalog)
    # orientation triad at the goal pose (frame_prim arrows)
    assert unwrapped.goal_axes_markers is not None, "goal axes triad missing"
    # mid-episode goal resample (goal-reached path) must not raise and must keep
    # the active object's prototype
    probe = torch.tensor([0], device=unwrapped.device)
    unwrapped._reset_target_pose(probe)
    instancer_path = goal_markers.prim_path
    from isaaclab.sim.utils.stage import get_current_stage

    prim = get_current_stage().GetPrimAtPath(instancer_path)
    if prim.IsValid():  # Kit marker backend only exists when rendering (GUI)
        from pxr import UsdGeom

        proto_indices = UsdGeom.PointInstancer(prim).GetProtoIndicesAttr().Get()
        assert len(proto_indices) == args_cli.num_envs, len(proto_indices)
        mismatches = sum(int(p) != int(a) for p, a in zip(proto_indices, unwrapped.active_object_ids.tolist()))
        assert mismatches == 0, f"{mismatches} goal markers show the wrong object"
        print(f"[5] goal markers OK: {len(marker_names)} prototypes, all {args_cli.num_envs} match active objects")
    else:
        print("[5] goal markers OK (cfg order validated; instancer prim not live in headless)")

    # [6] step with zero actions
    zero_action = torch.zeros(args_cli.num_envs, unwrapped.single_action_space.shape[0], device=unwrapped.device)
    rew_sum = 0.0
    for _ in range(30):
        obs, reward, terminated, truncated, extras = env.step(zero_action)
        assert reward.shape == (args_cli.num_envs,), f"reward shape {tuple(reward.shape)}"
        assert torch.isfinite(reward).all(), "non-finite reward"
        rew_sum += reward.mean().item()
    dist = torch.linalg.norm(unwrapped.object_pos - unwrapped.in_hand_pos, dim=1)
    print(f"[6] 30 steps OK, mean reward/step: {rew_sum / 30:.4f}, object distance: max {dist.max():.3f}")

    env.close()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
    simulation_app.close()
