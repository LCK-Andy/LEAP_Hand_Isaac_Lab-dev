# Stage B with standalone TD-MPC2

## Scope

One TD-MPC2 learner trains `Isaac-Grasp-Motion-Cube-LEAP-Direct-v0`, without population-based training. The native Stage B MDP is unchanged: a fixed LEAP hand tracks a small object-rotation command using its 16 joint-target-increment actions, with full normal-plus-friction sensing, four-pulp validity and non-pulp-support exclusion. No wrist motion, object pose writes outside reset, external wrench, or physics simplification is introduced.

This is a fresh TD-MPC2 model. Stage A PPO weights cannot initialize its latent world model, five-Q ensemble and policy directly. The existing measured pregrasp still initializes each physical episode; this is not learning acquisition from an open hand. See [Stage B](stage-b-grasp-motion.md) for the objective and physical limits.

## Settings

The runner reuses the pinned official TD-MPC2 core and the existing critic-autograd correction, documented in [standalone TD-MPC2](tdmpc2-standalone.md).

| Setting | Stage B run |
|---|---:|
| Learners / simulation environments | 1 / 256 |
| Seed / transition budget | 1 / 10,000,000 |
| Model preset | 5M; five Q networks |
| Observation / action dimensions | 148 / 16 |
| Physics / control | 120 Hz / 30 Hz |
| Episode | 300 control steps; finite 10 seconds |
| Planning horizon / iterations | 3 / 6 |
| Candidates / elites / policy trajectories | 512 / 64 / 24 |
| Planning chunk size | 32 environments |
| Update batch / updates per transition | 256 / 1 |
| Learning rate / encoder multiplier | 3e-4 / 0.3 |
| Discount | 59/60 ≈ 0.98333 |
| Replay | 1,000,192 transitions; ordinary CPU memory |
| Random seed phase | 1,500 transitions; crossed at 1,536 |
| Learner compilation | enabled |
| Full checkpoint interval | 100,000 transitions |

One update per transition means 256 gradient updates per vector step—not one update for the entire batch. Planning chunks bound temporary memory, not total computation. Filled Stage B replay occupies approximately 1.18 GiB, plus model/optimizer and simulator memory.

The seed-phase count follows the existing transition-based TD-MPC2 convention (five native episode lengths). With 256 asynchronous environments, this is only six physical steps per environment, before the command becomes active at step 90. Thus the initial update burst contains settling data only. Neither this configuration nor the short validation proves sample efficiency or eventual three-axis tracking.

## Native terminal and measurement contract

`scripts/tdmpc2/grasp_motion.py` supplies a separate DirectRLEnv adapter; Dexsuite's manager-based history-preview adapter is not reused.

- Native `final_obs` is a full-batch pre-reset observation, retained in the extras dictionary across steps. Stale entries are removed before each step; terminal rows are copied before reset.
- Replay stores those terminal successors, not same-step reset observations. Both failure and finite-horizon timeout disable value bootstrap. Raw timeout/failure flags remain available separately for reporting.
- The native sticky **one-second tracking success** is captured before reset. Stage A hold success and tracking occupancy are not substituted for it.
- Completed-episode tracking fraction, support-free retention, valid-four-pulp occupancy and overload are validated and logged separately. Recent metrics use the last 100 completed episodes; they are training measurements, not held-out evaluation.
- Full-state resume restores learner, optimizers, targets, replay/sampler, RNG, update debt and completed-episode statistics. Physical episodes and planner history restart; replay continuity is cut. No simulator-state restore is claimed.
- Checkpoints include the task's physical/reference condition stamp and implementation hashes. Stage A PPO, Dexsuite TD-MPC2 and incompatible-source checkpoints are rejected.

## Commands

Run from the repository root in the validated Isaac environment:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/andy/env_isaaclab/bin/python scripts/tdmpc2/train.py \
  --task Isaac-Grasp-Motion-Cube-LEAP-Direct-v0 \
  --workspace /absolute/path/to/fresh_stage_b_run \
  --num_envs 256 --seed 1 --headless --device cuda:0
```

Resume using the same settings and `--checkpoint /absolute/path/to/run/checkpoints/latest.pt`. Use a populated workspace only for an explicit resume; workspace ownership prevents duplicate writers. The maximum transition budget may be extended. `latest.pt` and `previous.pt` are retained; interrupted training saves at the next safe update boundary. There is no automatic crash retry or scheduled monitoring.

Real native-boundary test:

```bash
/home/andy/env_isaaclab/bin/python scripts/smoke/tdmpc2_grasp_motion_smoke.py \
  --headless --device cuda:0 --output /tmp/tdmpc2-stage-b-smoke.json
```

## Monitoring

Inspect `status.json`, TensorBoard `summaries/`, and checkpoint sidecars in the workspace. Watch:

- `critic/optimizer_tensors` versus `critic/parameter_tensors`, increasing optimizer steps, and nonzero online/target head weights;
- finite value/model/policy losses, actual update count and pending update debt;
- `recent/success_rate`, `recent/tracking_fraction`, `recent/retained`, `recent/valid_fraction`, and `recent/overload_fraction`;
- `recent/retention_weighted_tracking` to distinguish tracking followed by failure from tracking retained across the full episode;
- completed-episode counts and return/length, without treating survival or total reward alone as tracking acquisition.

The stationary fixture can produce a large positive grasp return while achieving zero tracking success. Current command size is 0.1° per axis, and full multi-axis tracking remains unproven.

## Validation and remote deployment

Deployment evidence is recorded separately in [the remote run record](research/2026-09-28-stage-b-tdmpc2-weste-run.md). The existing Dexsuite TD-MPC2 run is a separate experiment and is not replaced by this launch.
