# Native Dexsuite with one FlashSAC learner

This entrypoint uses the actual `flash_rl.agents.flashSAC.agent.FlashSACAgent`, not
TD-MPC2 and not FlashSAC's obsolete Isaac Lab environment wrapper. It bypasses that
wrapper using the validated `DexsuiteVectorEnv` boundary. The task ID retains
`-PBT-` because that package supplies the preserved native environment and episode
measurements; **no population, selection, mutation, donor or restart engine runs**.
Only the algorithm-neutral environment adapter, episode metrics and workspace lock
are reused from `scripts/tdmpc2/`.

## Native boundary and upstream profile

- Unchanged `policy`, `proprio`, `perception` groups/history order: 1745 observations.
- Batched `(N, 1745)` observation and `(N, 23)` action spaces. Actions stay in
  `[-1, 1]`; native relative-joint scaling applies `0.1` **once**.
- Native 6 s / 360-step episodes, 120 Hz physics, decimation 2, solver32. No
  observation compression, action changes, reward replacement or physics shortcut.
- Replay receives pre-reset terminal successors/history, not next-episode states.
  Three-step returns stop at the first termination/truncation. Truncations bootstrap
  with `gamma ** actual_horizon`; true terminations do not bootstrap.
- Actual snapshot `configs/agent/flashSAC.yaml` is parsed into `FlashSACConfig`.
  `scripts/run_isaaclab.sh` recommends 1024 envs, batch 2048, two gradient updates
  per **vector step**, warmup 100,000 stored transitions, n-step 3 and gamma .99.
  Warmup is random actions. There is no transition-scaled update burst after warmup.
- Actor 2 blocks / 128, critic 2 blocks / 256, 101 bins in [-5,5], tau .01,
  actor period 2, reward normalization to G-max 5, upstream zeta-noise repetition
  and temperature settings. AMP and compile are enabled; `auto` resolves to
  `max-autotune` on torch >=2.9. The sampler override only keeps actions on GPU.
- Budget **50,000,896 transitions**, matching the upstream Isaac Lab script,
  not a billion-transition Dexsuite-specific tuning claim. LR init/peak 3e-4,
  end 1.5e-4. Schedule lengths use the nominal gradient-update budget
  `ceil(max_steps / num_envs) * updates_per_vector_step` (97,658 by default);
  upstream warmup fraction 1e-6 rounds to zero. Actor/temperature schedules advance
  only on their own updates, retaining upstream behavior.

## Source and launch

Use an immutable copied FlashSAC source tree first on `PYTHONPATH`. Do not modify
or depend silently on the dirty installed development checkout. The supplied
snapshot must expose `flash_rl.buffers.torch_buffer.N_STEP_DISCOUNT_VERSION = 1`
with the matching critic discount patch; otherwise launch refuses to train.
The final deployment snapshot also corrects the reward-normalizer episode boundary:
compute the complete discounted return including terminal reward, update its
statistics, then clear the per-environment accumulator on termination/truncation.
This prevents dropping the preceding return or leaking terminal reward into the next
episode. These are recorded source correctness patches, not an assertion that the
upstream source is unmodified; hyperparameters and network/update algorithms remain
those of the recommended profile.
`contract.json` and checkpoints hash the imported source content, integration,
configuration, adapter and preserved task code, not just an upstream git commit.
Task/pxr imports happen only after AppLauncher has initialized.

Example from the repository root (replace snapshot/workspace with deployment paths):

```bash
PYTHONPATH=/absolute/immutable/FlashSAC${PYTHONPATH:+:$PYTHONPATH} \
  /home/andy/env_isaaclab/bin/python scripts/flashsac/train.py \
  --workspace /absolute/runs/dexsuite-flashsac --headless \
  --device cuda:0 --buffer-device cuda:0 --replay-capacity 2000000
```

The CLI default replay is 4,000,000 transitions, about **56 GB / 52.4 GiB**
for two float32 1745-channel observations, actions and scalar fields. The official
10M recommendation needs about 140 GB and cannot fit an 85 GiB GPU. Start fit
validation with 2M (~28 GB / 26.2 GiB), then explicitly validate 4M under full
compiled learner plus 1024-env simulator peak before production. The remaining
memory must cover simulator, learner, sampled batches, compilation and allocator
reservation; allocation alone is not fit evidence. `--buffer-device cpu` is
available but uses pinned host RAM in upstream code and is not a free-memory fix.
No n-step-1 or observation/action reduction fallback is introduced.

Small execution probe (not a learning-quality test):

```bash
PYTHONPATH=/absolute/immutable/FlashSAC \
  /home/andy/env_isaaclab/bin/python scripts/flashsac/train.py \
  --workspace /absolute/runs/flashsac-probe --headless --num_envs 4 \
  --replay-capacity 256 --batch-size 16 --warmup 16 \
  --stop-after-vector-steps 12 --no-compile --no-amp
```

`--no-compile` disables configurable network compilation; upstream helpers decorated
unconditionally with `torch.compile` can still compile. A deployment fit probe must
retain production AMP/compile settings. Simulator-free seam tests run with
`python -m unittest discover -s tests -p test_flashsac_native.py`; snapshot-specific
configuration checks are skipped if the patched source is absent.

## Checkpoints, metrics and stop semantics

- Exclusive workspace lock; fresh launch refuses populated workspaces. Resume
  requires a matching task/agent/source contract. There is no automatic retry.
- `--checkpoint-interval 1000000` counts actual transitions, not vector iterations.
  Atomic `checkpoints/latest.pt` and `previous.pt` retain two generations; foreign
  or stale checkpoint contracts and leftover staging files are refused.
- Final budget/probe completion and SIGINT/SIGTERM save at the next completed
  vector/update boundary. Exceptions report failure, not a successful final save.
- Checkpoints include actor, critics/target, temperature, optimizers, schedulers,
  AMP, update counters, noise, RNG, reward-normalizer statistics, native curriculum
  and lifetime completed-episode accounting. **Replay is not saved**: 4M replay
  snapshots would make frequent saves prohibitively large. Resume is therefore
  **not exact off-policy continuation**. Physics resets, replay and pending n-step
  queues start empty, warmup refills, and partial episode/normalizer returns reset;
  learned statistics and gradient schedule counters remain restored.
- Resume a trusted checkpoint using the original run configuration and
  `--checkpoint /absolute/runs/dexsuite-flashsac/checkpoints/latest.pt`. Source,
  budget and effective agent settings must match. Compile/AMP changes require a
  separate validation run rather than silently changing a resumed contract.
- `summaries/`, `status.json`, `result.json`, `agent.json`, `env.yaml`, `launch.json`
  and `contract.json` expose native measured success counts/rates and lifetime
  episode-weighted returns/lengths, learner diagnostics, frames/updates and GPU
  allocated/reserved peaks. These are training episodes, not held-out evaluation.
- First updates verify actor **and** critic weight changes, complete optimizer
  parameter/state coverage, finite parameters/metrics and changing target weights.
  Periodic diagnostics continue checking liveness. Actor metrics can be absent on
  alternate updates because its upstream period is 2.

Deployment, compiled-memory fit and native learning acquisition remain separate
validation steps. This document does not claim that the Weste launch or a successful
learning run has occurred merely because the entrypoint and CPU contracts exist.
