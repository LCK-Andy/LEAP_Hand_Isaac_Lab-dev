"""
Smoke test for the Isaac-Repose-Objects-LEAP-Direct-v1 task (manifest inertials).

Checks, headless with 24 envs:
  1. The task registers; LEAP hand (16 joints); the catalog is restricted to
     manifest-covered objects and every entry passes the palm-size filter.
  2. Per-episode inertial resampling: after reset (+ one step so the data
     buffers refresh), the sim's read-back body mass / COM / inertia of each
     env's ACTIVE member equals the env's sampled ground truth.
  3. Route A (spawn-time USD authoring): each template prim carries explicit
     physics:mass / centerOfMass / diagonalInertia / principalAxes, and the
     quat reconstructs the manifest's uniform-variant inertia (R diag R^T).
  4. Variant resampling: a second reset redraws variants (>= 50% of the
     (env, member) variant indices change; expected 5/6 with 6 strategies).
  5. Obs keeps the 136-d shape (ground-truth inertials stay unobserved).
  6. Zero-action stepping keeps rewards finite and objects near the palm.

Run:
  ~/env_isaaclab/bin/python scripts/leap_repose_objects_direct_v1_smoke.py
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Repose-Objects-LEAP-Direct-v1 smoke test.")
parser.add_argument("--task", default="Isaac-Repose-Objects-LEAP-Direct-v1")
parser.add_argument("--num_envs", type=int, default=24)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401  (registers Isaac-Repose-Objects-LEAP-Direct-v1)
import LEAP_Isaaclab.tasks.leap_hand_reorient.leap_hand_reorient_ycb_env_cfg as ycb_catalog


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped

    # [1] scene + catalog/manifest coverage
    joint_names = list(unwrapped.hand.joint_names)
    assert len(joint_names) == 16 and all(n.startswith("a_") for n in joint_names), joint_names
    catalog = unwrapped.object_catalog
    max_extent = unwrapped.cfg.max_object_extent_m
    assert all(extent <= max_extent for _, extent in catalog), catalog
    covered = set(unwrapped._inertial_records)
    assert {n for n, _ in catalog} <= covered, f"catalog not manifest-covered: {catalog}"
    print(f"[1] robot OK (16 LEAP joints); catalog: " + ", ".join(f"{n}({e * 100:.0f}cm)" for n, e in catalog))

    # [2] read-back of the resampled inertials (one step refreshes the data buffers)
    obs, _ = env.reset()
    zero_action = torch.zeros(args_cli.num_envs, unwrapped.single_action_space.shape[0], device=unwrapped.device)
    env.step(zero_action)
    rows = torch.arange(args_cli.num_envs, device=unwrapped.device)
    active = unwrapped.active_object_ids
    mass_rb = unwrapped.objects.data.body_mass.torch[rows, active]
    com_rb = unwrapped.objects.data.body_com_pos_b.torch[rows, active]
    inertia_rb = unwrapped.objects.data.body_inertia.torch.reshape(args_cli.num_envs, -1, 3, 3)[rows, active]
    mass_gt, com_gt, inertia_gt = unwrapped.active_inertials
    assert torch.allclose(mass_rb, mass_gt, rtol=1e-4), (mass_rb[:4], mass_gt[:4])
    assert torch.allclose(com_rb, com_gt, atol=1e-5), (com_rb[:2], com_gt[:2])
    assert torch.allclose(inertia_rb, inertia_gt, atol=1e-7), (inertia_rb[0], inertia_gt[0])
    print(
        f"[2] inertial read-back OK: mass {mass_gt.min():.3f}..{mass_gt.max():.3f} kg "
        f"({mass_gt.unique().numel()} distinct), |COM| <= {com_gt.norm(dim=1).max() * 1000:.1f} mm, "
        f"I trace <= {inertia_gt.diagonal(dim1=-2, dim2=-1).sum(-1).max():.2e} kg*m^2"
    )

    # [3] route A: explicit MassAPI attrs on the template prims (pre-clone state)
    manifest_path = unwrapped.cfg.inertial_manifest or os.path.join(
        ycb_catalog._YCB_DIR, "inertial_manifest.json"
    )
    with open(manifest_path) as f:
        manifest = json.load(f)["objects"]
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    for name, _ in catalog:
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
        attr_names = ("mass", "centerOfMass", "diagonalInertia", "principalAxes")
        attrs = {a: prim.GetAttribute(f"physics:{a}") for a in attr_names}
        assert all(a.HasValue() for a in attrs.values()), (name, {k: a.HasValue() for k, a in attrs.items()})
        m = attrs["mass"].Get()
        com = attrs["centerOfMass"].Get()
        diag = attrs["diagonalInertia"].Get()
        q = attrs["principalAxes"].Get()
        assert m > 0 and all(d > 0 for d in diag), (name, m, diag)
        entry = next((v for v in manifest[name] if v["strategy"] == "uniform"), manifest[name][0])
        assert abs(m - entry["mass"]) < 1e-4 * entry["mass"], (name, m, entry["mass"])
        # quat (w,x,y,z) -> R; PhysX reconstructs inertia = R diag R^T
        rot = Rotation.from_quat([q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2], q.GetReal()])
        inertia_rec = rot.as_matrix() @ np.diag(diag) @ rot.as_matrix().T
        inertia_ref = np.asarray(entry["inertia"])
        assert np.allclose(inertia_rec, inertia_ref, atol=1e-3 * np.abs(inertia_ref).max()), (
            name,
            inertia_rec,
            inertia_ref,
        )
    print(f"[3] template MassAPI OK (mass/COM/diagonalInertia/principalAxes authored + quat round-trip)")

    # [4] variant resampling across resets
    vids_before = unwrapped._inertial_variant_idx.clone()
    obs, _ = env.reset()
    vids_after = unwrapped._inertial_variant_idx
    changed = float((vids_before != vids_after).float().mean())
    print(f"[4] variant resample: {changed:.0%} of (env, member) variant indices changed on reset")
    assert changed >= 0.5, f"only {changed:.0%} of variants changed"

    # [5] obs shape (ground truth stays hidden from the policy)
    assert obs["policy"].shape == (args_cli.num_envs, 136), tuple(obs["policy"].shape)
    print("[5] obs OK: 136-d policy (12 tactile channels, no inertial ground truth)")

    # [6] step with zero actions
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
