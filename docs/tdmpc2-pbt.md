# TD-MPC2 PBT on native Dexsuite

> **Historical PBT run, stopped2026-09-27:** the first deployment had a confirmed
> critic-gradient defect that finite-update/restore smoke tests missed. The critic
> code is now corrected locally and separately validated. The user then chose a
> **single learner without PBT**; see [standalone TD-MPC2](tdmpc2-standalone.md).
> The four-member run below is preserved history, not the current training choice.
> See the [follow-up diagnosis](research/2026-09-27-tdmpc2-pbt-migration.md).

## Scope and provenance

The requested population is **four TD-MPC2 learners, 256 environments each**, on
`Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0`. This is a fresh algorithm run, not
PPO weights renamed or converted to TD-MPC2. The remote four-member population launched
on2026-09-27; all four passed the initial update burst and performed real MPC steps.
Execution checks do not establish learning quality. See the [deployment record](research/2026-09-27-tdmpc2-pbt-migration.md).

The entrypoint is `scripts/tdmpc2/train_member.py`. Its `vendor/` subtree retains the
official [TD-MPC2 main source](https://github.com/nicklashansen/tdmpc2/tree/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2)
at **`e9f59321933cbc8e11a002b842adc7d4ffae8ff1`**, including its MIT license.
See `scripts/tdmpc2/vendor/PROVENANCE.md` for exact adaptations. Qualified imports,
configurable devices, CPU Adam compatibility, and TensorDict 0.14 functional-ensemble
compatibility are local changes; learned objectives, initialization and target updates
remain upstream. There are **no installed-package patches and no TorchRL dependency**.

`agent.py` adapts MPPI batching from official `vectorized_env` pin
`97c14471997d5e58420d05d0fd6a5bd0195410dd`, but does not use that branch's
synchronous-reset collector. It supports asynchronous row resets, GPU-returned actions,
and environment chunks. It retains main's probability-taking Gumbel sampler rather
than the vector branch's erroneous extra logarithm. `eval_mode` removes final action
noise only; candidate and elite sampling remain stochastic.

## Architecture and environment contract

- `config.py` specifies the official **5M architecture preset**: 256-wide encoder,
  512-dimensional latent state, 512-wide MLPs, SimNorm groups of eight, five Q
  networks, 101-bin reward/value predictions, learned dynamics, Gaussian policy
  prior, and an episodic termination head. “5M” names the preset, not an exact
  parameter count for this observation/action size.
- `agent.py` wraps the upstream consistency/reward/value/termination objectives,
  policy update, two Adam optimizers, running Q scale and target-Q averaging.
- `env.py` concatenates native groups in stable order:
  **policy 170 + proprio 615 + perception 960 = 1745**. Native term/history ordering
  is retained, including point-cloud observations. Actions are **23** normalized
  joint commands, clamped to [-1, 1] and passed through the native action manager's
  scaling; this is not Cartesian-arm control or a physics simplification.
- Native physics remains **1/120 s**, decimation **2** (60 Hz control), **32 solver
  position iterations**, and **6 s / 360-control-step** episodes. Native rewards,
  curriculum, ADR, resets and terminations remain in the preserved Dexsuite task.

Same-step auto-reset observations are not replay terminal successors. The wrapper
captures observations immediately before reset, previews a history append without
mutating native history/cache, and restores Torch RNG and point-cloud visualization
state afterward. Acting uses post-reset observations; replay uses pre-reset successors.
`compute_final_obs=False` is required because the native non-updating history path
would return stale stacks. Unknown stateful observation terms/modifiers and recorders
are rejected rather than silently treated as safe.

## Replay, planning and update budget

`replay.py` owns bounded **CPU** sequence replay; sampled batches move to the learner
device. Default capacity is **131072 transitions/member**, not upstream's 1M.
At 1745 observations and 23 actions, explicit float32 observation and successor
storage costs approximately **1.72 GiB/member** (allow roughly **1.73 GiB** for
planning), or about 6.9 GiB across four members, before models, simulation, checkpoints
and other process memory. Loading a donor temporarily adds a full payload alongside
allocated replay; snapshots and staging also require disk space. This is not a measured
production RAM/VRAM guarantee.

Sampling returns upstream's time-major `(obs, action, reward, terminated, None)` tuple:
`obs[H+1,B,1745]`, `action[H,B,23]`, rewards/terminals `[H,B,1]`. With horizon three,
sequences never cross reset boundaries. Both truncation and termination cut continuity;
only true termination suppresses TD bootstrap. Episodes shorter than the horizon are
counted but cannot supply full sequences. Physical-condition stamps are provenance,
not extra model inputs. Replay has its own checkpointed sampling RNG.

Production planner defaults are **horizon 3, 512 candidates, 64 elites, 24 policy
trajectories, chunks of 32 environments**. Base `--planner-iterations 6` becomes
**eight** through the upstream 23-action heuristic. Batch size is 256. Eager execution
is default; `--compile` is opt-in and does not certify compiled vector planning.

**UTD means gradient updates per individual transition, not per vector step.**
`--updates-per-transition 1` therefore schedules **256 updates per 256-env step**.
The 1800-transition random-action seed threshold is crossed at 2048 transitions;
with valid replay, the first update burst pays **2048 accumulated updates**. Pending
update debt is checkpointed. This can be slower than PPO; no speedup is promised.

`--max-steps 10000000` is an implementation default, **not a user-specified budget**:
nominally 10M transitions/member, 40M/population. Collection rounds to whole vector
steps, so a fresh 256-env member reaches 10000128 transitions for that limit.

## Population exchange and resume

`pbt.py` uses `async-pbt` 0.1.0 for completed-episode evidence, donor selection,
bounded mutation and staged restart files. Code defaults retain publication every **1M**
transitions, **5M** warmup and **3M** adaptation. The remote TD-MPC2 launch overrides these
with **100k** publication, **300k** warmup, **100k** post-exploit adaptation and
**21600 seconds (six hours)** maximum evidence age, based on measured 26–31 transitions/s
for a solo production-config member. At least **128 completed episodes** are still
required per eligible group; exact physical-condition and frame-age gates remain.
Selection is success-first with fixed
pose-progress tie-breaking, compatible physical conditions and identical immutable
training contracts. Freshness gates also apply; raw unfinished-episode reward is not
the objective.

Member zero starts at the baseline. Other members receive bounded initial diversity.
Mutations affect learning rate, entropy coefficient and gradient clipping only:
probability 0.25, multiply/divide by U(1.1, 1.5), bounded respectively to
`[1e-5,1e-3]`, `[1e-5,1e-3]`, `[1,100]`.

Full snapshots include model/target networks, both optimizers, running scale, learner
RNG, replay and replay RNG, trainer/update debt, curriculum/ADR and PBT metadata.
**Donor learner/replay random streams transfer; recipient frame/update budget,
update schedule and dedicated PBT RNG remain recipient-owned.** Donor archives are
copied into recipient-owned staged files before replacement. A YAML companion commits
and identifies the `.pth`; staged metadata overrides donor accounting. Resume honors
that recipient metadata rather than accidentally inheriting donor budget.

Neither resume nor exploit restores the simulator exactly. Episodes restart, planner
means reset, replay continuity is cut without deleting complete sequences, and old
episode-selection evidence is discarded. Only two owned numeric archives are retained;
staged restarts are separate. Never treat these trusted local Torch payloads as safe
untrusted downloads. Existing PPO checkpoints are not compatible.

## Commands and observability

Run from the repository root with the validated environment. Example for **one member**;
run four separate processes with `MEMBER=0..3`, seeds 42..45, a common fresh workspace,
and operator-selected GPU placement. This is a launch recipe, not evidence of launch.

```bash
MEMBER=0
WORKSPACE=/absolute/path/to/fresh_tdmpc2_population
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/andy/env_isaaclab/bin/python scripts/tdmpc2/train_member.py \
  --task Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0 \
  --workspace "$WORKSPACE" --member "$MEMBER" --population 4 \
  --num_envs 256 --seed "$((42 + MEMBER))" --device cuda:0 --headless \
  --max-steps 10000000 --replay-capacity 131072 --plan-chunk-size 32 \
  --interval-steps 100000 --warmup-steps 300000 --adaptation-steps 100000 \
  --max-score-age-seconds 21600
```

Resume the same member using the same contract-affecting options and add
`--checkpoint /absolute/path/to/committed_or_staged.pth`; its `.yaml` must accompany it.
The maximum budget may increase. Fresh starts into populated member/archive directories
are refused, and a writer lock prevents concurrent ownership. SIGINT/SIGTERM requests
save at a safe update boundary; confirm `TDMPC2_MEMBER_DONE` and `result.json`, rather
than assuming every interrupted or crashed process saved successfully.

Under the workspace, `members/NNN/` contains `env.yaml`, `agent.json`, `contract.json`,
`launch.json`, `status.json`, `result.json`, `pbt_events.jsonl` and TensorBoard
`summaries/`; full archives are in `tdmpc2/NNN/`. Monitor transition/update counts,
pending debt, replay occupancy, completed episodes/return/length, finite model and
policy losses, termination diagnostics, gradient norms, Q scale, throughput/timing,
PBT hyperparameters, publication/selection reasons and replacement events. Timing is
host-side elapsed accounting, not a synchronized GPU profiler. Checkpoint publication
alone is not successful exploitation or policy acquisition.

CPU-only regression command (no launcher or simulator):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /home/andy/env_isaaclab/bin/python -m pytest -q \
  tests/test_tdmpc2*.py tests/test_dexpbt*.py tests/test_pbt*.py \
  tests/test_async_pbt_migration.py tests/test_allegro_pbt.py \
  tests/test_contact_pbt.py tests/test_grasp_pbt.py tests/test_grasp_motion_pbt.py \
  tests/test_ea_task_imports.py tests/test_ea_contact_forces.py
```

## Validation boundary, 2026-09-27

- Aggregate CPU suite: **280 passed, one CUDA-only test skipped, seven subtests
  passed** in 33.73 s. Log: `logs/tdmpc2_validation/full_regressions.log`.
  Separately, the implementation's new-test run passed 72 tests including CUDA
  terminal-preview coverage. `compileall` passed; no typechecker was available.
- Project launch/resume flags were checked directly against extracted argparse
  declarations without importing AppLauncher. This is parser validation, not a
  production launch or dependency/resource preflight.
- Local native training: four environments, official 5M preset, reduced planner
  (16 candidates, three effective iterations), batch eight: **32 transitions and
  32 updates**. Full resume reached **40/40**, with replay growing 32 to 40.
  See `native_train_r1.log` and `native_resume_r1.log` in the validation log directory.
- Natural terminal check: four environments, **370 vector steps / 1480 transitions**,
  **four completed native episodes**, collect-only, zero updates. Native six-second
  episodes and solver settings were not shortened. See `native_terminals_r1.log`.
- Remote production-config smoke:256env,2560transitions/2560updates, two full MPC
  collection steps after seeding; exit0, finite checked tensors and CRC-verified full
  replay checkpoint. Solo planned-step throughput26–31transitions/s; peakGPU6099MiB.
- Four production members ×256env launched and each passed its initial2048-update
  burst and real MPC steps. Early totalGPU allocation24425MiB, cgroupRAM about27.1GiB.
  These are initial measurements, not sustained benchmarks.
- **Learning quality, throughput advantage and sustained production exploitation are
  unproven. First production publication is at100k transitions/member, not at launch.**
