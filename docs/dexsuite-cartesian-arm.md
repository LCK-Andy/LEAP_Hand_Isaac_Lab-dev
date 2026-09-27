# Dexsuite Cartesian arm control

Implemented and locally tested on 2026-09-28. This is a separate control ablation, not cuRobo integration or a physics simplification.

## Tasks

- `Isaac-Dexsuite-Kuka-Allegro-Reorient-Cartesian-v0`
- `Isaac-Dexsuite-Kuka-Allegro-Reorient-Cartesian-Play-v0`

They inherit the preserved Dexsuite/PBT environment's assets, contact sensors, reward criteria, resets, gravity/noise curriculum, six-second episodes, 120 Hz physics and 60 Hz policy decisions. The original joint-control task is unchanged. Play retains its original high-ADR convention; use explicit fixed-evaluation configuration for controlled comparisons.

Implementation: `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_cartesian/`.

## Action contract

22 normalized action channels, clipped to [-1, 1]:

| Channels | Meaning | Default bound |
|---|---|---|
| 0–2 | Palm translation increment in robot-base frame | Euclidean norm ≤5 mm per policy step |
| 3–5 | Palm rotation-vector increment in robot-base frame | Euclidean norm ≤0.02 rad per policy step |
| 6–21 | Relative finger-joint increments | Original 0.1-rad scale |

Finger order is index, middle, ring, thumb; joints 0 through 3 within each finger. Arm output targets are ordered `iiwa7_joint_1` through `iiwa7_joint_7`.

Palm increments accumulate onto the previous requested pose. Before each IK command, its error relative to measured palm pose is capped at 5 cm translation and 0.2 rad rotation. Zero input holds the bounded persistent target rather than re-centering the target every step. This was necessary to avoid small-command deadband under the inherited randomized friction; no actuator gains or physics settings were changed.

Damped least-squares IK (damping 0.05) maps the pose error to seven arm joint targets each physics substep. Existing implicit joint PD drives execute them. No added null-space posture bias is used. Position targets are clipped to joint position limits and slew-limited to 1 rad/s. A 0.2-rad target/measured-joint error cap is applied before slew limiting; when physical disturbances make these bounds conflict, position and command-slew bounds take priority.

**These bounds constrain requested targets, not actual physical velocity, acceleration, force or collision safety.** IK is local and not a collision-free trajectory planner. The physics still has all 23 robot joints, self-collision and object contact. No object/wrist teleportation or grasp weld is used by the task.

Non-finite robot state, invalid palm quaternion and non-finite Jacobian fail explicitly. Non-finite IK output with valid measured state holds the prior finite joint target and marks control invalid. Reset clears selected controller rows; first subsequent action initializes from the new measured pose/joints.

## Observations

The agent concatenates four groups in this order:

| Group | Dimension |
|---|---:|
| policy | 165 |
| proprio | 615 |
| perception | 960 |
| controller | 22 |
| Total | **1762** |

The native prefix becomes 1740 because five history entries each lose one action channel. The unhistoried, uncorrupted 22-channel controller group exposes:

- Target palm position (3) and xyzw quaternion (4), in robot-base frame.
- Applied arm joint-position target (7).
- Current palm translation and rotation-vector errors (6).
- Target-limited and control-valid flags (2).

This exposes the persistent targets that affect future control. After reset, observations substitute the new measured pose/joints and zero errors, not stale targets.

## Training and checkpoint boundaries

Use the existing member entrypoint; its PBT framework is unmodified. This task's PPO configuration disables PBT and initial population diversity by default. It uses a separate log namespace, the existing MLP architecture, and uniform normalized-action exploration std initial/min/max = 0.15/0.03/0.3. This conservative initialization is not an optimized learning setting or a perfectly isolated exploration ablation against the original task.

```bash
/home/andy/env_isaaclab/bin/python scripts/rl_games/train_dexpbt_member.py \
  --task Isaac-Dexsuite-Kuka-Allegro-Reorient-Cartesian-v0 \
  --headless --num_envs 256 \
  pbt.enabled=false pbt.workspace=cartesian_run1
```

Use a new workspace for a fresh experiment; best-checkpoint history persists separately from timestamped training directories.

Full same-control resume uses `--checkpoint PATH` with the same environment count and controller configuration. The task stamps its control version, scales, target bounds, damping and timing into environment state, rejecting incompatible contracts before restoring curriculum state. As with the parent task, this restores training/curriculum state, **not exact simulator trajectories or partial episodes**. Controller targets are reinitialized after restore.

Old joint-action checkpoints are not compatible. Existing `--policy-prior` and Stage A/B `--warm-start` restrictions are deliberately unchanged and do not authorize Cartesian conversion.

The metric protocol remains the native fixed-definition protocol; a separate Cartesian-control condition stamp prevents comparison to joint-control evidence. PBT population/exploitation behavior for the new task is not validated by this work. The shared fixed evaluator still selects the original joint task; do not use it as a Cartesian evaluator without a deliberate separate adaptation.

Raw checkpoint resume in the existing trainer uses launch/base optimizer hyperparameters, including learning rate. The local restore test used a constant schedule to make optimizer continuation unambiguous. An adaptive-LR production resume should explicitly use the intended restored LR rather than assume the raw checkpoint wins.

## Local validation

Environment: `/home/andy/env_isaaclab`, local RTX 5070 Ti Laptop GPU. Existing installed packages were not modified; remote training was not touched.

### Simulator-free tests

```bash
/home/andy/env_isaaclab/bin/python -m pytest tests/test_dexsuite_cartesian.py -q
```

- 103 focused tests passed.
- Broader CPU regression: 524 passed, 42 subtests passed, 3 GPU tests skipped, one existing native pre-gym registration integration test deselected.
- Black at 120 columns, isort, flake8 and compile checks passed. E402 is excluded for the smoke script's required post-AppLauncher imports.
- The unit action tests execute real callbacks against explicit external simulator/math boundaries; they are not physical IK convergence evidence.

### Physical control smoke

```bash
/home/andy/env_isaaclab/bin/python scripts/smoke/dexsuite_cartesian_smoke.py \
  --headless --num_envs 8 --axis-sweep \
  --output logs/cartesian_smoke/result.json
```

Passed with eight environments:

- 22-action/1762-observation dimensions, finite observations/rewards and arm/finger ordering.
- Target command slew checks, partial reset isolation, checkpoint-contract rejection and curriculum round-trip.
- Complete native six-second episodes at fixed ADR 0 and 1: eight completed episodes each, **zero task successes** for stationary commands.
- All six Cartesian axes and both signs: all eight environments moved in the requested signed axis direction over 60 policy steps.
- Median signed translation movement across the six translation probes: approximately 0.093–0.118 m. Median signed rotation movement across the six rotational probes: approximately 0.386–0.576 rad.

The mechanical axis fixture resets the robot to its native default joint posture before probing, away from randomized wrist-limit starts. It does not establish tracking accuracy at every random configuration, under a retained grasp, or against collision obstacles. Cross-axis errors exist. Native random resets are used for the separate episode checks. The initial implementation that re-centered each tiny target at measured pose failed the motion probe; the final bounded persistent-target mapping passed.

### Tiny PPO and full resume

Fresh run:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/andy/env_isaaclab/bin/python \
  scripts/rl_games/train_dexpbt_member.py \
  --task Isaac-Dexsuite-Kuka-Allegro-Reorient-Cartesian-v0 \
  --headless --num_envs 8 --seed 42 --max_iterations 13 \
  pbt.enabled=false pbt.workspace=cartesian_local_20260928 \
  params.config.full_experiment_name=cartesian_local_20260928_fresh \
  params.config.horizon_length=32 params.config.minibatch_size=256 \
  params.config.mini_epochs=2 params.config.save_frequency=0 \
  params.config.torch_compile=false params.config.lr_schedule=constant
```

The recorded fresh run completed 13 epochs / 3328 transitions; Adam step count was 26 across 17 optimizer state entries. Full resume to absolute epoch 14 completed at 3584 transitions, with Adam step count 28. Model and normalization tensors were finite, actor action dimension 22, input normalization dimension 1762, and control contracts matched. Resume retained all in-range std values without projection.

The one-epoch resumed checkpoint filename includes `rew_-inf` because no full episode completed in that resumed suffix; the model tensors are finite. Neither this filename nor the short run is a quality result. The fresh training success metric was zero.

Ignored local evidence (not shipped checkpoints):

- `logs/cartesian_local_20260928/smoke.json`
- `logs/cartesian_local_20260928/smoke_axes.json`
- `logs/cartesian_local_20260928/training_validation.json`
- `logs/cartesian_local_20260928/{unit_tests,train,resume}.log`
- `logs/rl_games/dexsuite_reorient_cartesian/cartesian_local_20260928_fresh/nn/last_dexsuite_reorient_cartesian_ep_13_rew__-0.28573942_.pth`
- `logs/rl_games/dexsuite_reorient_cartesian/cartesian_local_20260928_resume/nn/last_dexsuite_reorient_cartesian_ep_14_rew_-inf.pth`

## Limits

This validates local execution, bounded command behavior, signed-axis motion, reset/state contracts and PPO restoration. It does **not** establish learned grasping/reorientation, grasp retention during arm motion, improved FPS, better sample efficiency, cuRobo compatibility, Newton support, hardware safety, or successful population exploitation.

The native action penalties still operate on normalized actions, whose physical meaning now differs. Compare fixed task success and measured motion/retention, not raw returns alone. Whole-arm reorientation of a held object remains valid under the inherited objective; genuine in-hand relative rotation is not required.
