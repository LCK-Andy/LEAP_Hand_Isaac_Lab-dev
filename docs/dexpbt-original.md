# Original DexPBT single-arm reproduction

## Current status: native original-stack training is running; modern port remains experimental

The original-task numerical contract, recurrent PPO training/resume and PBT process plumbing are implemented and tested. **The Isaac Lab port is not an exact physical reproduction:** the modern PhysX runtime changes six finger inertia tensors to positive-eigenvalue counterparts, whereas the original Isaac Gym CPU runtime retains the signed tensors even after ten physics steps. An isolated native-setter test on modern PhysX rejected the source values with `PxRigidBody::setMassSpaceInertiaTensor(): components must be non-negative`.

Therefore long Isaac Lab training is not presented as a faithful replication. The dedicated train/play entrypoints require explicit `--allow-physics-mismatch` for experimental port runs. No such opt-in is implied by an asset loading successfully.

**Native route validated on2026-09-28:** original GPU PhysX, GPU tensors, task PPO training/resume and an8-member8192-env pilot passed on RTX6000D. A fresh original-task8×8192population is running with5Btransitions/member. An isolated official Python3.8 Torch nightly with forward-compatible PTX and disabled unsupported GPU fusion was required; original task/physics/PBT source remains unchanged. The original GPU solver was measured retaining its signed inertias. See [run provenance and limitations](research/2026-09-28-native-dexpbt-weste.md). This is a launched reproduction experiment, not a reproduced learning result.

## Scope and evidence standard

This is a separate port of the released DexPBT **Allegro–KUKA single-arm reorientation** task. It is not the Dexsuite task and does not change existing Dexsuite, LEAP grasp, FlashSAC, or TD-MPC2 environments.

Reference: [Petrenko et al., DexPBT, RSS 2023](https://arxiv.org/abs/2305.12127). Source is the initial public [IsaacGymEnvs PBT release, `665c32170d84b4be66722eea405a1e08b6e7f761`](https://github.com/isaac-sim/IsaacGymEnvs/tree/665c32170d84b4be66722eea405a1e08b6e7f761), not a moving branch. Unmodified BSD-3-Clause source fixtures and SHA-256 provenance live in `tests/fixtures/dexpbt_original/`; tests extract numerical reference methods without importing Isaac Gym into Isaac Lab.

The target runtime is the existing validated Python 3.12 / Isaac Lab 3.0.0rc1 / Isaac Sim 6.1 / PhysX stack. A numerical MDP test, a simulator smoke, a training/checkpoint execution check, and a successful learned policy are **different evidence levels**. This port must not be described as reproducing the paper's learning curves or 30-hour training time without an appropriately sized, evaluated training experiment.

Only single-arm reorientation is in scope. Regrasping, throwing, and dual-arm tasks are not implemented by this task ID.

## Paper versus released implementation

The released implementation is the default reference. It contains material differences from the paper's prose; silently combining them would produce a third experiment.

| Component | Paper | Pinned released implementation |
|---|---|---|
| Single-arm observation | 110 dimensions in Table I | 117 dimensions, including a current-reward channel |
| Actions | Prose describes joint-angle targets | Arm integrates normalized actions; hand uses absolute joint-angle targets |
| PBT middle-ranked agents | Middle 40% mutate | The source selection implementation treats the bottom fraction; middle agents are unchanged |
| Mutation probability | 0.2 | 0.15 |
| Post-mutation adaptation | 50M transitions | 100M transitions |

The optional paper PBT recipe changes the outer algorithm settings only. It does **not** turn the released-source environment into an independently validated 110-dimensional paper environment.

The seven extra source observation channels are one extra palm-orientation component, three extra per-finger best-distance components, log progress, log successes, and current reward. The source also uses keypoints at 1.5 times the cuboid half extents and compares their distance against 1.5 times the nominal tolerance. Thus a nominal 1 cm tolerance is **not** a universal strict 1 cm positional-error guarantee; for a pure translation it allows 1.5 cm. Cuboid dimensions extend down to 2.5 cm in the release, versus the paper's stated 3 cm lower bound.

### Source semantics deliberately retained

- Original `kuka_allegro_touch_sensor.urdf`, including its original collision meshes and inertial data; not Isaac Lab's modern `KUKA_ALLEGRO_CFG`.
- 23 joints in explicit source order. Force-drive stiffness 40 and damping 5; arm effort limits 300 Nm and hand limits 0.35 Nm. Robot gravity disabled. No Cartesian controller or contact-force reward gate.
- Action period 0.01667 s; the source has two physics substeps. The port uses two 0.008335 s physics steps with the same held target.
- Arm targets increment from their previous target by `10 * 0.01667 * action`; hand targets map normalized actions to joint limits. Targets are clamped.
- Sequential, new-best-distance reaching and keypoint rewards; sticky pickup state; lift bonus; separate arm/hand velocity penalties. Rewards are not multiplied by the Isaac Lab control timestep.
- Observation assembly before reward-state mutation, with the current reward inserted into the final observation channel afterward. This ordering is covered by source-oracle tests.
- Deferred reset at the beginning of the next control step, as in the released Isaac Gym task, rather than silently adopting a different autoreset ordering.
- Goal-only resets after successful reorientations; progress timeout resets on success; up to 50 successes per episode. The source's near-goal counter is not cleared on a miss.
- Keypoint tolerance curriculum and tolerance-aware population objective; not Dexsuite's position/orientation/contact criterion.
- The table top is at 0.53 m; object reset center is approximately 0.63 m, not an initially settled object. Goal positions cover the original sampled volume.
- Generated cuboid catalog preserves filename-based overwrites, repeated dimensions under different filenames, seed-42 catalog permutation, and fixed per-environment cyclic assignment. It is not an episode-resampled object catalog.
- Original probabilistic local-frame object disturbances are enabled even though broader domain randomization is disabled. The source uses the first object's reference mass for disturbance scaling across object shapes.

### Reward mutation details

The released PBT mutation list contains twelve environment names, but three are absent from the task configuration and filtered out by the source. The port does not invent effects for `distRewardScale`, `rotRewardScale`, or `actionPenaltyScale`.

Nine configured names are retained: `liftingRewScale`, `liftingBonus`, `liftingBonusThreshold`, `keypointRewScale`, `distanceDeltaRewScale`, `reachGoalBonus`, `kukaActionsPenaltyScale`, `allegroActionsPenaltyScale`, and `fallDistance`. The first eight affect the source reward. `fallDistance` is configured but unused by the source's actual termination implementation; retaining it is source compatibility, not an effective mutation. The source instead tests a hardcoded object height and fingertip-distance boundary.

## Original asset preparation

Obtain the public source in a separate directory and pin it before building:

```bash
git clone https://github.com/isaac-sim/IsaacGymEnvs.git /path/to/IsaacGymEnvs
git -C /path/to/IsaacGymEnvs checkout 665c32170d84b4be66722eea405a1e08b6e7f761

/home/andy/env_isaaclab/bin/python scripts/assets/build_dexpbt_original_assets.py \
  --source-root /path/to/IsaacGymEnvs --preflight
/home/andy/env_isaaclab/bin/python scripts/assets/build_dexpbt_original_assets.py \
  --source-root /path/to/IsaacGymEnvs
```

The preflight is simulator-free. The conversion verifies the source commit, original URDF hash, mesh dependencies and joint contract. It preserves licensed source data, writes a local USD and a conversion manifest, and refuses to overwrite a completed build. It does not install or patch the Isaac stack.

**Physics equivalence caveat:** the original finger inertia tensors include non-positive eigenvalues. A local CPU-only original Isaac Gym asset-import query confirmed that its actor properties retain those values; the modern importer preserves them as well. They are not silently clamped or replaced. The query also measured friction 1.0 for robot, table, and cube, and average friction combination—not the initially assumed 0.5 material.

Original imported mass, center-of-mass, inertia and collision-filter properties are retained as measured validation evidence. Modern convex cooking, importer fixed-joint merging, center-of-mass fallback and collision exclusions are not proven dynamically equivalent merely by loading the same URDF. Any correction or unresolved difference belongs in the validation report. The EA importer produced a malformed world joint with a non-rigid `body0`; the builder removes that exact joint and lets the spawn configuration author a clone-safe root weld. It does not bypass the fixed-base assertion or change inertial tensors.

## Native original-simulator reproduction route

For physical reproduction, use the unmodified original simulator/task together rather than the experimental modern port. The following prints eight source-pinned commands; it does not launch them, install packages, allocate GPUs, or enable external experiment logging:

```bash
/home/andy/env_isaaclab/bin/python scripts/dexpbt_original/launch_native.py \
  --source-root /path/to/IsaacGymEnvs \
  --python /path/to/original-sdk-compatible-python \
  --workspace /absolute/new/native-dexpbt \
  --devices 0,1,2,3,4,5,6,7 --population 8 --max-frames 5000000000
```

The generator checks the pinned source checkout is clean, demands a distinct explicit GPU per member, selects `AllegroKukaLSTM` / `reorientation`, uses original reward mutation and PBT cadence, and disables WandB. Run each printed command concurrently in its own session on the indicated hardware. It uses explicit seeds 42–49 and the paper-comparison budget of 5B/member rather than the source launcher's random seeds/10B default.

The local original SDK CPU oracle used Python 3.8.20. That environment's Torch 2.4.1/CUDA 12.4 and the old Isaac Gym binaries do **not** establish Blackwell/RTX PRO 6000 compatibility. Original GPU execution and full-population training remain unvalidated; do not overwrite the validated Isaac Lab stack to attempt them.

## Experimental Isaac Lab training and evaluation

The dedicated entrypoint does not import the Dexsuite PBT engine or its exploration clamps. The source recurrent PPO architecture is LSTM 768 followed by MLP `[768, 512, 256]`, with horizon and sequence length 16, source minibatch 32,768 and two PPO epochs. The released YAML's maximum frame budget is 10 billion; set `--max-frames 5000000000` for the paper's main five-billion-per-member comparison.

A bounded local execution check explicitly reduces the rollout batch without changing the network:

```bash
/home/andy/env_isaaclab/bin/python scripts/dexpbt_original/train.py \
  --headless --allow-physics-mismatch --device cuda:0 --workspace /absolute/new/original-smoke \
  --num_envs 4 --no-pbt --minibatch-size 64 --mini-epochs 1 --max-epochs 1
```

Resume that same experiment, restoring the recorded minibatch and PPO configuration rather than overriding them:

```bash
/home/andy/env_isaaclab/bin/python scripts/dexpbt_original/train.py \
  --headless --allow-physics-mismatch --device cuda:0 --workspace /absolute/new/original-smoke \
  --num_envs 4 --no-pbt --checkpoint /absolute/new/original-smoke/000/final.pth \
  --max-epochs 2
```

The original PBT restart transfers model, optimizer, normalization/scaler state and success tolerance. As in the released task, physical episodes, recurrent hidden states, curriculum timers and success history start fresh. It is **not** exact simulator continuation. The separate full environment diagnostic state API must not be confused with this tolerance-only PBT transfer.

Policy-only evaluation pins the nominal tolerance and does not restore donor curriculum:

```bash
/home/andy/env_isaaclab/bin/python scripts/dexpbt_original/play.py \
  --headless --allow-physics-mismatch --checkpoint /absolute/new/original-smoke/000/final.pth \
  --num_envs 4 --steps 620 --tolerance 0.01 \
  --output /absolute/new/original-smoke/evaluation.json
```

Source playback samples actions; `--deterministic-policy` selects an explicit alternate evaluation protocol; the launcher's separate `--deterministic` flag controls simulation settings. Add `--viz kit` instead of `--headless` for GUI playback on this EA stack. Reports include checkpoint identity, completed episodes and consecutive-success metrics; a short untrained-policy evaluation is not a learned result.

### Population allocation

Generate a script for eight **distinct** GPUs without launching jobs:

```bash
/home/andy/env_isaaclab/bin/python scripts/dexpbt_original/launch.py \
  --workspace /absolute/new/original-population \
  --devices 0,1,2,3,4,5,6,7 --population 8 --num-envs 8192 \
  --max-frames 5000000000
```

The generator is dry-run only. It does not interrupt any existing training or silently put eight members on one GPU. Keep all members on the same immutable source, assets, environment count, rollout/minibatch configuration and PBT cadence. Population checkpoints are compared only under a matching experiment contract; allowed mutated parameters and member seeds differ deliberately.

## Compute budget and reporting

The paper's main eight-member single-arm comparison uses **eight separate V100 GPUs**, 8,192 environments per member, and **5 billion transitions per member**. The approximately 30-hour duration corresponds to roughly 50,000 training transitions/s/member, not a whole population fitting inside one V100's budget. The complete population collects approximately 40 billion transitions.

For a modern single GPU, the maximum fitting population is not automatically the best configuration. Report:

- Per-member and simultaneous aggregate environment transitions/second.
- Collection, policy-inference, optimizer, reset, and checkpoint time separately.
- Effective environment count, rollout horizon, minibatch size, number of PPO epochs, and recurrent sequence length.
- Consecutive successes and current tolerance, not reward alone.
- Population mutation/replacement events and donor/recipient identities.
- Fixed-protocol evaluation of selected checkpoints with source/protocol hashes.

A reduced-environment or reduced-minibatch smoke is an execution check, not the published experiment. A modern PhysX port can match task equations without reproducing the original simulator's dynamics or sample efficiency.
