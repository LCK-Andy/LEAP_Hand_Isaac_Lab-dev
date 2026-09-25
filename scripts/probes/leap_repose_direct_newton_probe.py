# Copyright (c) 2024-2025, The LEAP Hand Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Kit-less probe for the LEAP repose direct tasks on the Newton backend.

Steps the env with zero or curling actions and reports the two PhysX->Newton
risks directly:

* hand-root stability (did dropping the PhysX ``fix_root_link`` world joint
  leave the hand anchored?)
* pulp contact-force liveness (do the re-plumbed Newton contact sensors report
  non-zero filtered forces?)

plus obs/reward shapes. Kit-less only: resolves the config with
``presets=newton_mjwarp`` and never launches Kit.

Usage:
    python scripts/probes/leap_repose_direct_newton_probe.py [--task ID] [--num_envs 16] [--steps 150] [--curl]
"""

import argparse
import sys

import gymnasium as gym
import torch

import LEAP_Isaaclab.tasks  # noqa: F401  (registers the custom gym ids)

parser = argparse.ArgumentParser(description="Newton-backend probe for the LEAP repose direct tasks.")
parser.add_argument("--task", type=str, default="Isaac-Repose-Cube-LEAP-Direct-v0", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--steps", type=int, default=150, help="Number of env steps.")
parser.add_argument(
    "--curl",
    action="store_true",
    help="Alternate full-curl/open actions instead of zeros to force pulp contacts.",
)
args_cli, hydra_args = parser.parse_known_args()

# hydra-backed resolution must only see the preset argument (mirrors train_kitless.py);
# default to the newton_mjwarp preset (this is a kit-less probe) unless overridden
if not any(a.startswith("presets=") for a in hydra_args):
    hydra_args.append("presets=newton_mjwarp")
sys.argv = [sys.argv[0]] + hydra_args

from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402
from isaaclab.app.sim_launcher import launch_simulation  # noqa: E402

TASK = args_cli.task
env_cfg, _ = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
env_cfg.sim.device = "cuda:0"
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.seed = 42

with launch_simulation(env_cfg, {"headless": True, "device": "cuda:0", "enable_cameras": False}):
    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    device = base.device

    # count solver contact conversions and their contact counts
    update_stats = {"calls": 0, "nacon": []}
    chain_stats = {"step": 0, "full": 0, "physics_only": 0, "update_sensors": 0}

    try:
        from newton._src.solvers.mujoco.solver_mujoco import SolverMuJoCo

        _orig_update_contacts = SolverMuJoCo.update_contacts

        def _counting_update_contacts(solver_self, contacts, state=None):
            update_stats["calls"] += 1
            return _orig_update_contacts(solver_self, contacts, state)

        SolverMuJoCo.update_contacts = _counting_update_contacts

        from isaaclab_newton.physics.newton_manager import NewtonManager

        _orig_full = NewtonManager._simulate_full.__func__
        _orig_po = NewtonManager._simulate_physics_only.__func__
        _orig_us = NewtonManager._update_sensors.__func__
        _orig_step = NewtonManager.step.__func__

        def _full(cls):
            chain_stats["full"] += 1
            return _orig_full(cls)

        def _po(cls):
            chain_stats["physics_only"] += 1
            return _orig_po(cls)

        def _us(cls, contacts):
            chain_stats["update_sensors"] += 1
            return _orig_us(cls, contacts)

        def _step(cls, *args, **kwargs):
            chain_stats["step"] += 1
            return _orig_step(cls, *args, **kwargs)

        NewtonManager._simulate_full = classmethod(_full)
        NewtonManager._simulate_physics_only = classmethod(_po)
        NewtonManager._update_sensors = classmethod(_us)
        NewtonManager.step = classmethod(_step)
    except Exception as err:  # noqa: BLE001
        print(f"[probe] could not instrument chain: {err}")
        _orig_update_contacts = None

    obs, _ = env.reset(seed=42)
    print(f"[probe] obs shape: {tuple(obs['policy'].shape)} (expect (*, 136))")
    print(f"[probe] action space: {base.single_action_space.shape}, backend: {base._physics_backend}")

    root0 = base.hand.data.root_pos_w.clone()
    # cube task: plain RigidObject view; objects task: duck-typed active-member
    # view without root_pos_w -- read through the collection instead
    if hasattr(base.object.data, "root_pos_w"):
        def _obj_pos():
            return base.object.data.root_pos_w
    else:
        def _obj_pos():
            rows = torch.arange(base.num_envs, device=base.device)
            return base.objects.data.body_pos_w.torch[rows, base._active_idx]
    obj0 = _obj_pos().clone()
    forces_seen = []
    for step in range(args_cli.steps):
        if args_cli.curl:
            curl = 1.0 if (step // 25) % 2 == 0 else -0.5
            action = torch.full((base.num_envs, base.cfg.action_space), curl, device=device)
        else:
            action = torch.zeros((base.num_envs, base.cfg.action_space), device=device)
        obs, rew, terminated, truncated, _ = env.step(action)
        forces = base.fingertip_contact_forces
        forces_seen.append(forces.norm(dim=-1))  # (N, 4)
        if step % 25 == 0 or step == args_cli.steps - 1:
            drift = (base.hand.data.root_pos_w - root0).norm(dim=-1)
            obj_move = (_obj_pos() - obj0).norm(dim=-1)
            print(
                f"[probe] step {step:4d}: root drift mean {drift.mean():.6f} m / max {drift.max():.6f} m | "
                f"obj move mean {obj_move.mean():.4f} m | "
                f"force |F| per tip mean {forces_seen[-1].mean(0).tolist()} | "
                f"rew mean {rew.mean():.4f} finite={bool(torch.isfinite(rew).all())}"
            )
            if step == 0 and base.fingertip_contact_sensors:
                name = next(iter(base.fingertip_contact_sensors))
                s = base.fingertip_contact_sensors[name]
                # backend-specific internals: newton names the filter count
                # _num_filter_objects, ovphysx _num_filter_shapes; the raw view
                # only exists on newton
                n_filter = getattr(s, "_num_filter_objects", getattr(s, "_num_filter_shapes", "?"))
                raw_view = getattr(s, "_newton_total_force_view", None)
                print(
                    f"[probe] sensor '{name}': num_sensors={s._num_sensors} "
                    f"num_filter={n_filter} "
                    f"net_forces_w shape={tuple(s.data.net_forces_w.shape)} "
                    f"raw view shape={tuple(raw_view.shape) if raw_view is not None else 'n/a (non-newton)'}"
                )
            if step % 25 == 0:
                try:
                    from isaaclab_newton.physics.newton_manager import NewtonManager

                    c = NewtonManager.get_contacts()
                    if step == 0:
                        print(
                            f"[probe] manager state: report_contacts={NewtonManager._report_contacts} "
                            f"solver={type(NewtonManager._solver).__name__ if NewtonManager._solver else None} "
                            f"needs_pipeline={NewtonManager._needs_collision_pipeline} "
                            f"requested_attrs={NewtonManager._model.get_requested_contact_attributes()}"
                        )
                    if c is None:
                        print("[probe] newton contacts buffer: None")
                    else:
                        ncon_np = c.n_contacts.numpy()
                        force_np = c.force.numpy()
                        nz = int((abs(force_np) > 1e-8).any(axis=-1).sum()) if force_np.size else 0
                        print(
                            f"[probe] newton contacts: n_contacts={ncon_np} "
                            f"force shape={force_np.shape} nonzero_rows={nz}"
                        )
                        sensors = NewtonManager._newton_contact_sensors
                        for key, sensor in sensors.items():
                            tf = sensor.total_force
                            tf_np = tf.numpy() if hasattr(tf, "numpy") else tf
                            nz = int((abs(tf_np) > 1e-8).sum())
                            mx = float(abs(tf_np).max()) if tf_np.size else 0.0
                            print(f"[probe] SensorContact {key[0]!r}: total_force nonzero={nz} max={mx}")
                except Exception as err:  # noqa: BLE001
                    print(f"[probe] contacts introspection failed: {err}")

    forces = torch.stack(forces_seen)  # (steps, N, 4)
    contact_frac = (forces > 0.1).float().mean()
    print(
        f"[probe] update_contacts calls: {update_stats['calls']}, nacon tail: {update_stats['nacon'][-5:]}"
    )
    print(f"[probe] step chain: {chain_stats}")
    print(f"[probe] pulp-force liveness: fraction of (step, env, tip) readings > 0.1 N: {contact_frac:.4f}")
    print(f"[probe] pulp-force |F| max: {forces.max():.3f} N, median: {forces.median():.3f} N")
    if args_cli.curl and contact_frac == 0:
        print("[probe] FAIL: no pulp contacts registered under curling actions")
    elif not args_cli.curl:
        print("[probe] NOTE: zero actions -- rerun with --curl to force contacts for the liveness check")
    print("[probe] done")
