# Original DexPBT reproduction: implementation and validation

Date: 2026-09-28. Scope: original Allegro–KUKA single-arm reorientation, not Dexsuite.

## Outcome

Implemented a separate source-pinned task, recurrent PPO/PBT integration, original-stack command generator, immutable source oracles, and validation tools. Existing Dexsuite/LEAP tasks and running remote experiments were not changed.

**Exact physical reproduction on the installed modern PhysX stack is blocked.** The original robot contains six finger inertias with negative eigenvalues. Original Isaac Gym CPU simulation retains them, whereas modern PhysX makes their spectra positive when loading the USD and rejects an attempt to write the source signed tensors through its native setter. The modern port is therefore explicitly experimental; train/play require `--allow-physics-mismatch`.

No billion-transition run was launched. No learned-policy acquisition or paper-level wall-clock result is claimed.

## Source and scope

- Original public release: `isaac-sim/IsaacGymEnvs`, commit `665c32170d84b4be66722eea405a1e08b6e7f761` (July 2023 PBT release).
- Original task: `AllegroKukaLSTM` with `task/env=reorientation`.
- Port ID: `Isaac-DexPBT-Allegro-Kuka-Reorientation-Direct-v0`.
- Original robot: `assets/urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf`.
- Robot SHA-256: `52dc7ab9013ed46ddfa31ad5ff39df7365ad8698e6abe41e1480d762f47c7727`.
- Runtime: local Isaac Lab 3.0.0rc1, Isaac Sim 6.1, Torch 2.11/cu128, RL-Games 1.6.5, RTX 5070 Ti Laptop.
- Original reference CPU importer: Isaac Gym Preview4 in existing Python 3.8.20 environment, GPU pipeline disabled, no viewer. No package installation or installed-package patch was needed.

Paper/source differences are enumerated in [the operator guide](../dexpbt-original.md): source observation 117 versus paper 110, mixed relative-arm/absolute-hand control, 1.5-scaled keypoint predicate, and released versus paper PBT recipes. The default follows the actual released code, not a mixture of interpretations.

## Implemented boundaries

- Torch-only task core tested against AST-extracted original source methods.
- Original 947-filename cuboid catalog, including repeated dimensions and filename overwrite behavior.
- Separate DirectRLEnv with original deferred resets, zero startup observations, reward-state ordering, local random forces and tolerance curriculum.
- Original LSTM-768 before MLP `[768,512,256]`; source PPO configuration.
- Source bottom-only PBT and separately named paper top/middle/bottom recipe.
- Original live reward/PPO mutation keys; no invented effect for source `fallDistance`.
- Atomic post-update checkpoint transactions, strict population configuration/cadence, recipient frame accounting, fresh recurrent/episode state and tolerance-only original PBT restore.
- Independent original-simulator launch-command generator, dry-run only, distinct explicitly assigned GPUs and external logging disabled.

## Numerical and process checks

Final full repository suite: **607 passed, 4 skipped, 72 subtests passed**. Before this implementation the baseline was 554 passed, 4 skipped, 72 subtests passed. Skips are pre-existing environment-dependent checks. New source/test files passed Black, isort, flake8 and Python syntax checks; a typechecker was not installed and was not run. No lint tools were installed or upgraded.

Mandatory parity tests preserve full original source bytes under `tests/fixtures/dexpbt_original/`; their hashes are checked. Pre-commit excludes that fixture subtree so formatters cannot rewrite the reference. Derived implementation files remain normally formatted.

The checks include:

- Multi-step reward values and state transitions against extracted original functions.
- Every observation channel and its pre/post-reward ordering.
- Arbitrary normalized quaternions and keypoint geometry.
- Eight-step mixed arm/hand action target oracle, bitwise comparison.
- Exact catalog ordering against the actual original generator and template.
- Tolerance curriculum, source mutation distributions, source PPO architecture/configuration.
- Native CPU recurrent PPO training and optimizer/model/normalizer restore.
- Four independent CPU processes with deliberately synthetic objective values exercising real PBT selection and `os.execv`; donor optimizer/model inheritance, recipient frame/epoch identity and mutation/restart argument handling verified. This is plumbing evidence, not four learned manipulation agents.
- Invalid tolerance states, incompatible cadence/fixed PPO/asset contracts, and no-PBT periodic recovery checkpoints.

## Simulator and learner execution

### PhysX task smoke

A four-environment 620-control-step smoke passed after correcting two EA importer incompatibilities: a malformed importer world joint and a referenced Newton root mirror that blocked clone-safe root welding. Corrections are limited to the newly generated asset. Original source assets and inertias were not altered.

The final 64-environment 620-control-step smoke passed:

| Metric | Measured result |
|---|---:|
| Completed episodes | 73 |
| Timeouts | 48 |
| Goal success events | 0 |
| Timed rollout | 11.278 s |
| Transitions per second | 3,518 |
| Maximum sampled joint speed | 145.98 rad/s |
| Maximum sampled object velocity component | 89.43 |

The throughput includes per-step assertions/synchronization and random-action behavior. It is not an optimized PPO throughput benchmark and is not comparable to the paper's dedicated V100 training throughput.

Verified fixed base, source joint ordering, per-environment object masses, hybrid targets, current-reward observation, pending terminal resets, full diagnostic task-state restore and live mutation readback. Velocity limits are solver constraints, not hard tensor clamps; the high-speed tails are not evidence of safe or faithful grasp dynamics.

### Recurrent training, resume and evaluation

- Original-size LSTM/MLP, four environments, explicit smoke minibatch 64 and one PPO epoch per rollout.
- Fresh training through epoch 1 / 64 transitions completed and wrote the dedicated full checkpoint.
- Resume completed epoch 2 / 128 transitions with recorded architecture/optimizer state.
- Policy-only stochastic evaluation, nominal tolerance 0.01, four environments, 620 steps: **7 completed episodes, zero mean successes**.
- Mean terminal meta-objective is **1.0**, not a success: at final tolerance the original equation is `1 + successes`.
- Evaluation checkpoint SHA-256: `dc742132b3ec0162167ea58fecb4938e17cf14827d54ea6ff4d83226e92b6168`.

These are execution/restoration checks; two tiny training epochs do not establish learning.

## Physical fidelity blocker

### Original versus imported properties

Original SDK actor queries measured material friction **1.0**, restitution zero, contact offset 0.002 m and rest offset zero for robot/table/cube. The port uses those values rather than guessing modern defaults. Original rigid-shape filter masks were zero with `create_actor(..., collision_filter=-1)`; intrinsic articulation exclusions still require separate consideration.

Masses and most center-of-mass/inertia components agree closely. Remaining center-of-mass differences measured at runtime include approximately 86 mm on the immovable base and 46 micrometers on thumb link 3. The immovable-base difference is not itself evidence of moving-link dynamics error.

The decisive difference is the six index/middle/ring `link_2` and `link_3` inertia tensors:

| Body group | Original negative principal moment [kg m²] | EA runtime treatment |
|---|---:|---|
| Index/middle/ring link 2 | −4.3946245e−5 | Positive counterpart |
| Index/middle/ring link 3 | −2.3866286e−5 | Positive counterpart |

Original Gym retained all six negative tensors after actor creation, `prepare_sim`, and physics steps 1, 2 and 10. Maximum tensor change from its original no-step query was **zero**. The isolated original robot stayed finite and quiet (maximum joint speed 0.1865 rad/s at step 1, 0.0424 rad/s at step 10).

The modern runtime spectra agree with the absolute values of original spectra to approximately 3.68e−11 kg m². Eigenvalue sign changes cannot be explained by a coordinate rotation.

A separate one-environment modern PhysX experiment attempted `set_inertias_index` with the original tensors. Direct native getters were used, bypassing the Isaac Lab setter cache. PhysX emitted six errors:

```text
PxRigidBody::setMassSpaceInertiaTensor(): components must be non-negative
```

Native readback remained positive immediately and after ten steps. A successful Python return alone was not accepted as evidence that the native solver applied the values. The probe made no persistent task change.

### Consequence

Matching the task equations is possible on the modern stack; matching these original physical parameters is not supported by the tested native PhysX path. Silent absolute-value/clamp substitutions would violate the reproduction goal. The modern-port experimental opt-in and explicit report prevent a long run from being mislabeled as a faithful replication.

The native original-simulator route avoids this known migration mismatch, but original GPU execution, Blackwell compatibility, GPU-versus-CPU original solver parity, and learning quality remain unvalidated. Full reproduction requires a compatible original stack and a measured population training/evaluation experiment.

## Artifacts

Ignored local validation artifacts under `logs/dexpbt_original_validation/`:

- `smoke4.json`, `smoke64.json`.
- `train_final/000/final.pth`, effective YAML and manifest, TensorBoard data.
- `evaluation4.json`.
- `original_gym_cpu_oracle_full.json`, `original_gym_cpu_stepped.json` and stepped probe source.
- `live_asset_comparison.json`, `signed_inertia_probe.json` and isolated setter probe source.

Portable no-step reference query and authored-USD comparison: `scripts/probes/dexpbt_original_gym_asset_oracle.py`. The measured CPU asset fixture is clearly labeled separately from the pristine upstream source manifest.

No remote job was stopped/restarted, no long training was deployed, and no commit or push was made. Pre-existing FlashSAC changes were preserved.
