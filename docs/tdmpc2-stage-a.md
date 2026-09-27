# Stage A with standalone TD-MPC2

## Scope

`Isaac-Grasp-Cube-LEAP-Direct-v0` can train with the same standalone TD-MPC2 learner used for Stage B. There is **no PBT**, no population mutation/selection, and no prior PPO checkpoint loading. This is a fresh model-based learning experiment, separate from Stage B and Dexsuite.

The Stage A task is unchanged: maintain a persistent, force-regulated four-pulp grasp of the 0.072 m cube, using full normal-plus-friction contact measurements and excluding non-pulp support. Actions are 16 bounded joint-target increments, at most 0.015 rad per 30 Hz control step. Episodes last at most 10 seconds; observations have 124 channels. The hand starts from the measured pregrasp fixture, not an open-hand grasp-acquisition state.

See [Stage A task](stage-a-grasp.md) for the physical contract. TD-MPC2 does not itself estimate the object's mass, center of mass or inertia.

## Training settings

Default standalone launch: one learner, 256 environments, seed 1, 10 million transitions, the official 5M model preset and five critics, planning horizon 3, 512 candidates/64 elites/24 policy trajectories, six iterations, batch 256, learning rate 3e-4, and one optimizer update per transition. The compiled learner and existing corrected critic-autograd implementation are reused.

Replay holds 1,000,192 transitions in ordinary CPU memory, approximately 1 GiB plus model/optimizer memory. Full checkpoints are written every 100k transitions and retain latest/previous versions. UTD1 means 256 optimizer updates per vector step, **not** one update per vector batch. The 1500-transition random-action phase crosses at 1536 transitions; with 256 environments that covers six steps per environment, not five full trajectories per environment.

The nominal GPU may have memory headroom while already being busy. Running multiple independent learners on it shares compute and can slow each experiment; this setup is not a throughput-optimality claim.

## Success is not retention

The adapter records the task's native `grasp_state.success` **before reset**. It is a sticky flag for **two continuous seconds of valid grasp**. A later drop does not erase the fact that the hold was achieved.

Separate completed-episode metrics distinguish:

- `episode/success_rate`: fraction of episodes that achieved the native two-second hold.
- `grasp_hold/retained`: full-horizon retention without failure or non-pulp support anywhere in the episode.
- `grasp_hold/valid_fraction`: valid-four-pulp occupancy divided by the full 300-step horizon; missing steps after early failure earn no credit.
- `grasp_hold/overload_fraction`: native overload occupancy over the same full horizon.
- `grasp_hold/safe_success`: native success **and** support-free full-horizon retention.
- `recent/*`: corresponding last-100-completed-episode metrics.

Metrics are training measurements, not held-out evaluation. Zero-action fixture holding can pass the task; fixture-smoke success does not prove the learned policy is robust under exploration or disturbance.

## Terminal transitions and resume

The shared finite-horizon adapter captures pre-reset observations and native success, clears stale terminal extras, and resets planner histories on both failures and timeouts. Since the task is genuinely finite, timeouts disable value bootstrapping. No object state is written outside native resets.

Full resume restores model/targets, optimizers, scaling, learner/replay RNG, replay contents, pending update debt and completed-episode statistics. Simulator episodes reset and replay continuity is cut; no exact physics restoration is claimed. Partial episode statistics are discarded. Use the same source snapshot and contract-affecting settings when resuming: implementation hashes intentionally reject incompatible source, tasks and checkpoint formats. In particular, the already-running Stage B and Dexsuite jobs retain their own frozen snapshots.

## Commands

From the repository root:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/andy/env_isaaclab/bin/python scripts/tdmpc2/train.py \
  --task Isaac-Grasp-Cube-LEAP-Direct-v0 \
  --workspace /absolute/path/to/fresh_stage_a_run \
  --num_envs 256 --seed 1 --headless --device cuda:0
```

For full resume, add `--checkpoint /absolute/path/to/run/checkpoints/latest.pt` with the same source and configuration. Workspace locks refuse duplicate writers and accidental fresh starts over existing results. SIGINT/SIGTERM request a checkpoint at a safe update boundary. There is no automatic crash retry.

Native terminal/hold smoke:

```bash
/home/andy/env_isaaclab/bin/python scripts/smoke/tdmpc2_grasp_motion_smoke.py \
  --stage a --headless --device cuda:0 --output /tmp/tdmpc2-stage-a-smoke.json
```

The smoke script keeps its existing name for Stage B command compatibility. `--stage b` remains the default.

Remote deployment, validation and live acceptance: [Stage A Weste run record](research/2026-09-28-stage-a-tdmpc2-weste-run.md).
