# Stage A: persistent force-regulated LEAP grasp

New tasks:

- `Isaac-Grasp-Cube-LEAP-Direct-v0`
- `Isaac-Grasp-Cube-LEAP-Direct-Play-v0`

The existing `Isaac-Contact-Cube-LEAP-Direct-v0` and GRU task are unchanged. This is a new MDP and observation/action contract; their checkpoints are not compatible.

For the subsequent single-learner, no-PBT model-based training path, see [Stage A TD-MPC2](tdmpc2-stage-a.md). It uses the same task and a fresh TD-MPC2 model, not a PPO checkpoint transfer.

## Scope

This first stage retains a nominal cube in a measured four-pulp pregrasp. It does not pick up an object, select arbitrary contact locations, reorient it, estimate inertial parameters, or optimize excitation. It is a state-based simulation baseline, not a claim of tactile-only hardware deployment. Training starts from the same verified grasp rather than a broad distribution; robustness/randomized acquisition is a subsequent curriculum.

A valid grasp requires all four named pulps in their normal-force bands, a conservative shear/friction margin, low relative tangential contact velocity, no detected non-pulp support, and the object inside its retention region. Success is a continuous two-second valid hold, recorded once per episode; holding continues afterwards. There is no success-triggered reset or goal resampling. Both train and Play use finite ten-second episodes (`is_finite_horizon=True`), with remaining time included in the 124-dimensional actor observation. RL-Games therefore does not apply continuing-task timeout bootstrapping.

## Full three-dimensional contact force

The task exposes `env.fingertip_contact_forces` as **normal + friction force on the pulp, in world axes, in newtons**. The actor receives the full vectors and separate normal vectors, so it can recover shear without losing force magnitude. Force observations divide by 10 N; RL-Games clips only at +/-100 observation units. The controller uses normal components for preload bands, shear for slip margin, and full-vector magnitude for the hard safety limit.

Contact forces and support loads use four physics samples per 30-Hz control interval (120-Hz physics). Contact-location/alignment and tangential-velocity gates are evaluated at the final control sample; this version does not certify the absence of substep geometric/slip transients. Missing normal, friction, partner, or history data raises; there is no normal-only or geometry-only fallback. Contact locations use positive-force-weighted raw data under PhysX rather than the API's speculative/unweighted average. They are effective contact-location measurements, not a complete torsional contact wrench.

Pulp identity is a geometric approximation: force-bearing contact location must lie inside the measured ventral white-pulp region with aligned normal. The asset merges distal collision geometry, so this is not shape-resolved proof of a real tactile patch. The outward pad normal is local -X, not the distal -Y direction. Non-pulp support is measured with separate partner columns and summed per-partner load magnitudes, preventing cancellation between opposing support bodies. The object is terminated well above the ground, so ground support cannot be accepted as an in-hand hold.

## Control, rewards, and thresholds

Actions are 16 name-consistent bounded joint-target increments. A zero action holds the previous target. The default maximum increment is 0.015 rad **per control step**, not per physics substep. Saturation has no accumulated target windup. The PPO profile starts with std 0.15 and bounds learned std to [0.03, 0.3] through the PBT-member entrypoint's exploration observer.

Finger order: `fingertip`, `thumb_fingertip`, `fingertip_2`, `fingertip_3`.

| Quantity | Fingers | Thumb |
|---|---:|---:|
| Normal-force lower bound | 0.5 N | 1 N |
| Normal-force upper bound | 8 N | 16 N |
| Full-force hard ceiling | 16 N | 24 N |

Other initial settings: assumed friction coefficient 1.0; utilization limit 0.9; tangential velocity limit 0.02 m/s; non-pulp support limit 0.1 N; retention radius 0.08 m. These are simulation development settings, **not hardware-certified force limits**.

Reward combines bounded per-finger/weakest-contact quality, valid-hold occupancy, a small smooth surface-approach term, retention, overload/support/slip costs, and target-increment regularization. Drop and hard-overload events terminate with explicit penalties. No reward is proportional to unlimited squeeze. Completed-episode metrics divide valid/overload steps by the full nominal horizon; an early drop does not obtain a high percentage by shortening the denominator.

## Pregrasp provenance

`tasks/leap_hand_grasp/pregrasp.json` stores named joint positions, preload targets, cube pose, pulp regions and the physical calibration provenance. The original search used actual-USD joint-transform IK and bounded target search; a candidate was replayed in 32 environments with one-second settling and three-second measured fixed-target holding. This is a scripted fixture, not a trained policy. The standalone smoke below independently tests the new task through reset, continuous physics stepping, success persistence, and timeout.

The fixture JSON retains its original calibration-time validation flags and notes. Those
historical `false` flags do not describe the later task smoke results documented below;
the raw calibration record has deliberately not been rewritten after validation.

The legacy DexCube density setting did not imply the expected cube mass in the probe. This task explicitly sets mass to the measured fixture value, 0.216 kg, rather than relying on density authoring. The integration smoke measured a 0.072 m cube at the legacy 1.2 asset scale; the new task pins half-extent to 0.036 m. Old contact-task documentation's 0.03 m assumption is not used here, and the old task remains unchanged.

## Commands

Run from the repository root with the validated environment:

```bash
/home/andy/env_isaaclab/bin/python scripts/smoke/leap_grasp_smoke.py --headless
```

Train one bounded-exploration PPO member, without population replacement:

```bash
/home/andy/env_isaaclab/bin/python scripts/rl_games/train_dexpbt_member.py \
  --task Isaac-Grasp-Cube-LEAP-Direct-v0 --headless --num_envs 256 \
  pbt.enabled=false pbt.workspace=grasp_stage_a_run1
```

Use `train_dexpbt_member.py`, not ordinary `train.py`, to install the learned-exploration bounds. Population replacement is opt-in. The `grasp_hold` PBT objective compares safe full-horizon retention, then safe four-pulp-valid occupancy; unsafe episodes receive no selection credit and donors may not increase overload incidence. Scores are compared only when the versioned physical/task condition stamp matches. Do not reuse an existing contact population workspace.

## Validation and limitations

Simulator-free tests cover bounded control, force-quality ordering, unsafe measurements, substep dropouts, full-force overload, persistent success, horizon-normalized episode accounting, backend force composition, and PBT isolation/selection/restart behavior.

The smoke requires a real stepped ten-second hold with live friction signals, finite observations/rewards, no hard termination, >=95% valid occupancy, persistent success and completed-episode metrics. It is not a learning-quality experiment or a robustness guarantee across objects, masses, friction, sensor noise or real hardware.

### Backend results (2026-09-25)

- **PhysX:** passed four-environment, 330-control-step holding (including a ten-second completed episode), with approximately 99.7% valid occupancy, no drop/overload terminations, no measured non-pulp support, and live friction forces. This is a fixed pregrasp execution check, not trained-policy performance.
- **Newton / MuJoCo Warp:** full normal **and friction** reporting works in the current environment. The original asset imported no hand colliders because CollisionAPI was on instanceable `collisions` Xforms rather than actual geometry. A new-task-only session-layer adapter deinstances those groups and applies collision schemas to their 67 boxes and four pulp meshes, retaining transforms and requested convex decomposition; original asset files are unchanged. Newton generates its own decomposition, so this is not a physical-equivalence claim. A 330-step run then showed live normal peaks approximately [2.42, 9.59, 6.89, 0.90] N and friction peaks [2.27, 4.45, 3.14, 0.90] N. **The grasp failed acceptance:** zero valid four-pulp occupancy and up to 4.57 N non-pulp support, despite no drops. The shared PhysX fixture requires backend-specific calibration. Ordinary Newton task creation is guarded; `leap_grasp_newton_smoke.py` explicitly opts into this unvalidated setup.
- **OVPhysX:** the installed `isaaclab_ov` ContactSensor explicitly rejects friction-force and contact-point tracking. The task refuses it rather than silently dropping required signals.

Newton probe:

```bash
/home/andy/env_isaaclab/bin/python scripts/smoke/leap_grasp_newton_smoke.py \
  --num_envs 4 --steps 330 --output /tmp/leap-grasp-newton.json
```

The probe uses official kit-less `launch_simulation` and `presets=newton_mjwarp`; it does not start Isaac Sim. It sets `allow_unvalidated_backend=True` only for explicit validation. Current failure is reported with a nonzero exit, not accepted as a training-ready backend. Missing collider coverage and missing complete force channels fail clearly. No installed-package patches or geometry-only fallbacks are used.

A two-epoch/four-environment PPO integration run completed and wrote a 128-frame checkpoint; full restore continued to epoch 3 / 192 frames with learned exploration still inside its bounds. These runs are too short to contain a completed ten-second episode and are training/checkpoint execution tests, not learning results or population-exploitation demonstrations. A separate 21-epoch/four-environment run reached 1,344 frames and published real metrics for four completed episodes: support-free retention 0%, valid-four occupancy 3.83%, overload incidence 0%. This deliberately tiny run verifies that PBT consumes the new metrics; it does **not** demonstrate policy acquisition. The fixed-target fixture's 99.7% occupancy must not be confused with this early policy result. The final simulator-free regression pass completed 149 tests and 36 subtests (15 pre-existing legacy-RNG/deprecation warnings). The final PhysX smoke also verified real contact-loss termination, finite pre-reset terminal observations, cleared tactile state after autoreset, and exactly one target increment per control step.
