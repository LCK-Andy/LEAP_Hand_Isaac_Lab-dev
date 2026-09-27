# Standalone TD-MPC2 on Dexsuite

## User-selected scope

On2026-09-27 the user replaced the four-member PBT experiment with **one TD-MPC2 learner without PBT**, using recommended TD-MPC2 training settings. The existing256vector environments are retained. The native task remains `Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0`: its name identifies the preserved task and measurement adapter, not an active population algorithm. The standalone runner imports no population engine, instantiates no population member, and has no selection, mutation or donor path.

Entrypoint: `scripts/tdmpc2/train.py`. Pure ownership, checkpoint and episode-metric helpers: `scripts/tdmpc2/standalone.py`.

## Algorithm configuration

Defaults follow the pinned official [TD-MPC2 configuration](https://github.com/nicklashansen/tdmpc2/blob/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/config.yaml), with the explicit5Mmodel preset and task-specific dimensions:

| Setting | Value |
|---|---:|
| Learners |1|
| Vector environments |256|
| Seed |1|
| Transition budget |10,000,000|
| Replay requested |1,000,000|
| Allocated replay capacity |1,000,192 (whole256-env rows)|
| Update batch |256|
| Gradient updates per transition |1|
| Model preset |5M, five Q networks|
| Planning horizon |3|
| Candidates / elites / policy trajectories |512 /64 /24|
| Planner iterations |6base +2 for23actions =8|
| Learning rate / encoder multiplier |3e-4 /0.3|
| Entropy coefficient / gradient norm cap |1e-4 /20|
| Target update coefficient |0.01|
| Compile learner update |True|

Task adapter choices:1745observations,23joint actions,120Hzphysics/60Hzcontrol,360-step episodes, episodic termination head, and asynchronous vector collection. Planning chunks32environments bound peak working memory; they do not remove planning samples. The native physics, rewards, curriculum and contact gates are unchanged. Fresh training does not reuse faulty PBT/PPO weights or replay.

The initial1800-transition random-action phase crosses its threshold at2048transitions. UTD1 schedules2048initial gradient updates and256updates after each256-env collection step. This is not one update per vector step. Replay uses unpinned CPU memory and GPU sample transfer; explicit observations/successors require approximately13.1GiB at full capacity, plus other process memory.

Compilation is enabled for the learner update, not a claim that every batched planning operation is one compiled graph. `--no-compile` exists for controlled diagnostics. Tests found an Inductor issue in a two-Q test fixture; the actual five-Q default compiled without a workaround. No installed package or compiler pass was patched.

## Critic autograd correction

The first integration used TensorDict `params.to_module` inside `vmap`. On the deployed TensorDict0.14 stack this disconnected the registered Q parameters from their loss: all critic weights stayed unchanged despite finite updates. That four-member run was stopped and preserved; its performance is not a valid TD-MPC2 result.

The corrected ensemble uses an explicit named parameter mapping with `torch.func.functional_call`. Tests now establish:

- CPU and CUDA critic weights actually change and learn a positive-return fixture.
- Every critic parameter tensor has optimizer state; missing state after the first update fails fast.
- Gradients match independently instantiated ordinary networks.
- Detached critics retain action/policy gradients without accumulating critic gradients.
- Actor learning works even when entropy is zero.
- Target networks have independent storage, correct Polyak updates and eval-mode dropout.
- Full checkpoint restoration preserves the next learner update.

`Agent.critic_diagnostics()` exposes optimizer coverage/steps and online/target Q-head magnitudes. Both heads start at zero, so nonzero values and advancing optimizer steps provide direct learning evidence beyond finite losses. These metrics do not prove task success.

## Running and resuming

From the repository root:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/andy/env_isaaclab/bin/python scripts/tdmpc2/train.py \
  --workspace /absolute/path/to/fresh_run \
  --headless --device cuda:0 --num_envs 256
```

Use the same contract-affecting settings and add `--checkpoint /absolute/path/to/run/checkpoints/latest.pt` for resume. The transition budget may be extended. Faulty population checkpoints and other formats are rejected, as are incompatible implementation hashes.

Checkpoints contain model/target, both optimizers, running scale, RNG, replay and sampling RNG, update debt/accounting, native curriculum/ADR and completed-episode totals. They do not restore simulator physics: episodes and planner histories restart, and replay boundaries are cut. Learner RNG restoration occurs after simulator reset.

Checkpoint cadence is100ktransitions; `latest.pt` and `previous.pt` are retained with metadata sidecars. Only initialized replay rows are serialized until the buffer fills. Saving is synchronous and atomic; a full buffer makes checkpoints much larger. SIGINT/SIGTERM requests save at a safe update boundary, including unfinished update debt. A workspace lock prevents duplicate writers. No crash-retry automation is provided.

## Monitoring

The workspace has `summaries/`, `status.json`, `launch.json`, `agent.json`, `contract.json`, `env.yaml`, `result.json` and `checkpoints/`.

Inspect:

- `critic/optimizer_tensors` equals `critic/parameter_tensors` after updates.
- `critic/optimizer_steps_min` increases.
- `critic/head_abs_mean` and `critic/target_head_abs_mean` become nonzero.
- `train/value_loss` is not stuck at the uniform baseline2.6921544.
- All losses/diagnostics remain finite; the runner raises on nonfinite metrics instead of silently dropping them.
- Native completed-episode counts, success counts, episode-weighted return and length.
- Transitions, actual updates and pending debt separately.

Gradient-only log entries do not emit zero-FPS samples or reset rollout timing. Episode statistics are training-condition measurements, not held-out policy evaluation. No separate periodic held-out simulator is launched by this runner; official evaluation frequency is not silently repurposed as a task-success claim.

## Validation and deployment

Local CPU-only relevant suite after the critic fix:292passed,3CUDA-skipped,7subtests. Final TD-MPC2 suite86passed including CPU/CUDA. Independent five-Q compiled CUDA test passed actual optimizer updates and full-state continuation without compiler workaround. Full native256-env compiled smoke and pending-debt resume passed on the remote stack.

Fresh standalone production is running as `tdmpc2-single256-r2`, PID435187 at acceptance23:47:52+08 on2026-09-27. TensorBoard6007 run: `tdmpc2-weste-20260927-single-r2/prod`. Data: `/autodl-fs/data/tdmpc2_dexsuite_20260927_weste_single_r2/train`. Accepted6912transitions withupdatesadvancing6700→6800;10/10criticAdamstatesadvanced, head/targetnonzero, value loss2.692→~0.187. Early poststartup throughput~63transitions/s is not a sustained benchmark. Task acquisition remains unproven.

See the [fix and deployment record](research/2026-09-27-tdmpc2-critic-fix-standalone.md) for preservation, exactsourcehashes, validation and limitations.
