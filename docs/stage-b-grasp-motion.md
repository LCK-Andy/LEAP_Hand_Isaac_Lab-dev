# Stage B: safe object-motion tracking while grasping

## Scope and task IDs

- Train: `Isaac-Grasp-Motion-Cube-LEAP-Direct-v0`
- Play: `Isaac-Grasp-Motion-Cube-LEAP-Direct-Play-v0`

This is a new fixed-hand, nominal-cube task built on Stage A. The object is moved only by the existing 16 joint-target-increment actions. The reference is a command, **not** an object pose/velocity write, external wrench, moving fixture, or wrist motion. Stage A task IDs, observation/action contract, and the existing remote population are unchanged.

Stage B teaches reference tracking while preserving the instrumented grasp. It does not estimate mass/COM/inertia or maximize Fisher information. Successful tracking would still need a measurement-model and observability study before claiming inertial identification.

## Conservative reference

Local mechanical tests rejected the proposed 1-degree/1-mm motion as too aggressive for this fixed pregrasp. Initial task defaults are deliberately small:

- Rotation amplitude: **0.1 degree per axis**, in the fixed initial object frame.
- Frequencies: **0.2 / 0.3 / 0.4 Hz**.
- Translation amplitude: **zero initially**, configurable per axis.
- One-second settle followed by a two-second quintic envelope with zero initial velocity and acceleration.
- Per-episode random phase, fixed throughout an episode. References do not resample when a hold succeeds.
- Ten-second finite horizon, unchanged Stage A prefix semantics.

Rotation is `Rz(yaw) Ry(pitch) Rx(roll)`. Reference angular velocity and acceleration use the corresponding spatial angular quantities, **not Euler rates treated as angular velocity**. Analytic derivatives are checked by finite differences and nonidentity frame rotation tests.

These reference amplitudes are a training objective, not a promise that all three axes are already tracked. The fixed-hand mechanical baseline produces much stronger X motion than Y/Z; multi-axis acquisition remains open.

## Sensing, rewards, and success

All Stage A full normal-plus-friction sensing, force-history, pulp-location/alignment, support exclusion, and hard force/drop checks remain active. Force/support history is at 120 Hz; geometric/slip gates retain Stage A's final-control-sample limitation.

Stage B adds a positive, bounded pose/velocity tracking reward **only when the grasp is valid and the command is fully active**. There is no reward for arbitrary acceleration or contact noise. Tracking success requires errors under the configured thresholds continuously for one second, during active motion. Settling and ramp samples do not earn motion success.

Default error limits: 0.5 mm position, 0.0007 rad orientation, 0.002 m/s linear velocity, 0.002 rad/s angular velocity. They are simulator-development tolerances, not hardware precision claims. The small angular tolerance is intentional: a stationary grasp must not qualify merely because the requested angle is small.

The physics backend's reported instantaneous angular velocity showed a stationary bias relative to pose trajectories in the mechanical probe. Therefore, Stage B tracking uses **pose-difference velocities**, low-pass filtered with EMA coefficient0.2. The analytic reference velocity is filtered with the same response. Measurement/filter state advances exactly once per reward step and is reset from the actual reset pose; observations and terminal-observation queries do not advance it. Raw solver speed is logged separately.

The accumulated angular-acceleration Gram matrix is explicitly a **filtered pose-motion coverage diagnostic**, not a calibrated Fisher matrix, identifiability certificate, or density estimate. Bias, differentiation noise, limited duration, and contact-model error still matter.

## Observation and warm-start contract

Observation dimension is **148**:

- First124 channels: the exact existing Stage A observation, including finite-horizon remaining time.
- Additional24: position and shortest-orientation errors, filtered linear/angular velocity errors, analytic reference linear/angular acceleration, and sine/cosine phase features. Values are pre-scaled to the reference amplitudes/tolerances.

The new `--warm-start` argument is distinct from `--checkpoint`:

- `--warm-start` initializes a fresh Stage B actor/shared critic and normalizers from a compatible Stage A checkpoint.
- The first-layer weights for the extra24 channels are initialized to zero; new input-normalizer mean/variance entries are0/1.
- Optimizer, frames, epochs, environment state, population metadata and best-checkpoint history start fresh.
- Source manifests `agent.yaml` and `env.yaml` must sit beside the donor checkpoint and identify the124-d Stage A shared256/128/64 ELU actor-critic.
- The actual downloaded member002/epoch192 donor was tested: initial action means, sigmas, and denormalized critic values are **bit-exact**, even with arbitrary appended command values.
- Input RMS has one shared sample count. The donor count is preserved for exact transfer, so reference-channel normalization initially adapts slowly; deliberate feature scaling is important.
- `--checkpoint` remains a full Stage B resume. It is mutually exclusive with `--warm-start`; population exploit restarts strip the warm-start option so the original Stage A donor is not reapplied.

## Local commands

From the repository root:

```bash
# Real physics/reset check; stationary holding must NOT solve tracking.
/home/andy/env_isaaclab/bin/python scripts/smoke/leap_grasp_motion_smoke.py --headless
```

Warm-start one PPO member from a compatible Stage A checkpoint. Donor checkpoints and
validation logs are local artifacts, not distributed with this repository. Train or obtain
a Stage A checkpoint first, and copy its original `agent.yaml` and `env.yaml` into the same
directory as the `.pth` file. Set `STAGE_A_CHECKPOINT` to that file; do not substitute Stage B
or reorientation weights.

```bash
STAGE_A_CHECKPOINT=/absolute/path/to/stage_a/checkpoint.pth
/home/andy/env_isaaclab/bin/python scripts/rl_games/train_dexpbt_member.py \
  --task Isaac-Grasp-Motion-Cube-LEAP-Direct-v0 --headless --num_envs 256 \
  --warm-start "$STAGE_A_CHECKPOINT" \
  pbt.enabled=false pbt.workspace=stage_b_run1
```

Use a fresh workspace for independent warm-start runs. Population selection is optional; `grasp_motion` ranks overload-free active-window tracking, with no increased overload or reduced safe retention. It never consumes the old `grasp_hold` selection metric as though it were motion performance. Missing post-drop active samples contribute zero to the full nominal active-window denominator.

Mechanical baseline, separate from learned Stage B evaluation:

```bash
/home/andy/env_isaaclab/bin/python scripts/probes/leap_grasp_motion_probe.py \
  --headless --steps 900 --output /tmp/leap-grasp-motion-mechanical.json
```

This probe uses the unchanged Stage A environment, a settled measured reference frame, longer horizon, and a coherent contact-position damped-least-squares controller with a bounded preload adjustment. It compares a stationary arm/hand-root baseline against scripted motion. It forbids all object-state writes after initial reset. Its timing/frame/controller differ from the new task, so its metrics establish mechanical feasibility, not Stage B policy accuracy.

## Measured local results and limits

- New MDP smoke: four environments,330 control steps,148-d finite observations,99.7% valid grasp occupancy, **0% stationary motion-tracking success**, correct terminal/reset behavior and no observation-induced filter updates.
- Warm-start PPO integration:21 epochs,1,344 frames; a real completed-episode publication and final checkpoint were written. Four completed episodes had **0% safe tracking,0% safe retention and75% overload incidence**. This deliberately tiny update run is not a learning-quality result; exact initial policy transfer does not guarantee that subsequent Stage B PPO updates retain the skill.
- Full Stage B checkpoint resume continued from epoch21/frame1,344 to epoch22/frame1,408, without reapplying the Stage A donor.
- The packaged two-environment mechanical probe was rerun for900 control steps: active-motion contact validity was97.33% overall and98.27% after the ramp (stationary baseline100%). Actual XYZ amplitudes were0.0982/0.0371/0.0052deg, max full force7.313N, no support/unsafe events. This weaker repeat result is the reproducible-script baseline; it reinforces that the earlier sweep's100% steady validity is not a robustness guarantee.
- Full simulator-free regressions:183 tests and42 subtests passed. The44 warnings concern existing TorchScript deprecation and legacy-PBT RNG restore. Invalid pose/recovery intervals cannot create false velocity credit; zero-measurement episodes are not logged as zero tracking error.
- Original mechanical sweep:0.1-degree XYZ command with bounded inward preload maintained99.78% valid contact over30s and100% after the ramp. Actual amplitudes were approximately **0.0975 / 0.0341 / 0.0066 degrees**. Rotation error RMS0.0933deg, maximum0.1724deg; position drift RMS0.0832mm; max full force7.321N; no non-pulp support or hard safety event. This is **safe, X-dominant excitation, not faithful three-axis tracking**.
- Pose-frequency-derived acceleration in the sweep exceeded the stationary baseline on all axes, but this does not prove sufficient signal-to-noise or full inertial observability on hardware.

PhysX is the validated execution backend. Stage B inherits Stage A's experimental-Newton guard; no Newton Stage B validation was performed. The initial validation above did not launch Stage B remotely. The subsequent single-learner, no-PBT TD-MPC2 integration and remote run are documented in [Stage B TD-MPC2](tdmpc2-stage-b.md); the PPO warm-start contract above does not apply to TD-MPC2.
