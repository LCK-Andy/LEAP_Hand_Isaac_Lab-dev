# Dexsuite PBT rolling evidence and reset-index fixes

Date: 2026-09-27. Implements items1 and2 approved after the [performance investigation](2026-09-27-dexsuite-pbt-profile-results.md).

## Delivered scope

1. A genuinely rolling completed-episode evidence window and explicit PBT selection diagnostics.
2. A minimal reset-index normalization optimization, preserving native reset machinery and physical behavior.
3. Regression coverage, extended profiling/acceptance helpers, local native-PhysX smoke and actual PPO checkpoint/resume validation.

**Local changes only.** The remote15-member production snapshot was not modified or restarted. No solver settings, episode/reset rules, rewards, contact gates, publication defaults, freshness defaults or installed packages changed. No commits were made. Pre-existing unrelated working-tree changes were preserved.

## Fix1: rolling, fresh and comparable PBT evidence

### Why it was necessary

The former window accumulated an entire physical-condition group and expired it according to its first sample. At450FPS,1M frames take about37minutes, exceeding the30-minute freshness limit; a slow member could publish no usable evaluations despite many recent completed episodes.

### New behavior

- Store timestamped batches of completed-episode sufficient statistics per actual physical condition.
- Prune **individual batches**, not the whole accumulated group, by time and frame freshness during ingestion, publication and donor comparison.
- Retain remaining fresh evidence across publications; keep minimum-episode requirements and physical-condition grouping unchanged.
- Preserve truthful aggregate timestamps:oldest retained `created_at` and start frame, newest end frame. Never refresh old data's timestamp to make it appear fresh.
- Serialize compact additive batch evidence with the aggregate so donor comparison can discard only batches that expired after publication. Existing aggregate-only records receive conservative validation.
- Clear evidence when restoring/replacing a policy. Checkpoint physical/optimizer/model state still restores normally, but pre-restore rollout evidence is not inherited as fresh samples.
- Validate record/container shapes and finite nonnegative sufficient statistics before ranking; successes and good-contact sums cannot exceed episode counts. Invalid evidence receives a diagnostic instead of crashing or winning selection.
- Stamp actual completed transitions inside the RL-Games rollout. `algo.frame` alone remains fixed during `play_steps`; the observer now accounts for per-step actor transitions and resets the offset at rollout/restore/exploit boundaries. Other objectives retain their previous path.

### Diagnostics

Publications include `objective_valid`, evaluation diagnostics and cadence-level `selection_diagnostics`. Logs/TensorBoard distinguish missing,expired,insufficient or invalid evidence; stale/frame-mismatched donors; physical mismatch; disabled/warmup/adaptation gates; and insufficient improvement. A numeric zero retained for existing consumers is no longer presented as a valid zero-quality estimate when evidence is missing.

The existing donor-iteration loader restriction, frame-lag limit, success-first ranking and improvement thresholds were not loosened. These can still prevent exploitation. This fix repairs freshness and observability; it does not guarantee acquisition or a donor replacement.

### Bounded metadata

Synthetic steady-state checks at a measured-style reset cadence showed approximately252KB YAML at450FPS and476KB at850FPS per record. Individual batches omit repeated condition/protocol fields, and retention is bounded by time/frame horizons rather than total run age. Serialization overhead remains publication-cadence work and has not been benchmarked on the active remote population.

Core files:

- [pbt_utils.py](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt/pbt/pbt_utils.py)
- [pbt.py](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt/pbt/pbt.py)
- [test_dexpbt_pbt.py](../../tests/test_dexpbt_pbt.py)

## Fix2: normalize reset indices once

`DexsuiteReorientPbtEnv._reset_idx` now converts incoming IDs to `torch.long` once and reuses that tensor for episode metrics and inherited reset machinery. The native step supplies int32 IDs; repeated Torch advanced indexing downstream otherwise incurs repeated conversions. Native simulator/backend writers retain their own required index conversion.

No reset was delayed, no environment was added/removed from the reset set, no events/managers were replaced, and RNG draw order is unchanged. The initial cProfile suggestion of expensive native velocity setters was disproven by finer reversible timers; broad state-writer surgery was therefore rejected.

Core file: [environment.py](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt/environment.py), `_reset_idx`.

### Performance evidence and limits

Same-scene public-reset index experiment,512env/native32solver,10 warmups and40 timed resets per block in A–B–B–A order:

| Selected environments | int32 arm | int64 arm | Reset-time reduction |
|---|---:|---:|---:|
|1|11.603ms|10.938ms|5.7%|
|4|11.562ms|11.037ms|4.5%|
|32|11.705ms|11.132ms|4.9%|
|512|12.043ms|11.485ms|4.6%|

An exact pre-change/candidate component comparison measured reset time12.055→11.382ms per control step averaged over the profile window. Separate fresh-process index-mode ABBA measured reset12.005→11.068ms,but total uninstrumented rollout62.655→63.461ms/control-step.

**No overall rollout or training-throughput speedup was demonstrated.** The reset-specific saving is modest; variability and unrelated work dominate the full-loop comparison. The reversible native-boundary selector adds one conversion in its int32 arm, so that ABBA is an index-overhead experiment,not an exact measurement of production before/after speedup. No remote performance claim is made.

### Behavioral parity

At512env,seed45,zero applied ADR,native32solver and the same frozen checkpoint:

- All four360-step deterministic-policy trajectories matched exactly for actions,observations,rewards,dones,joint positions/velocities,object poses/velocities,CPU/CUDA RNG state and completed-episode reports.
- Each completed514episodes; subsequent profile reset schedules matched:244batches,555environments reset,median batch1.
- Explicit selected-reset tests verified selected observations/physical state/RNG,unchanged unselected objects,unchanged saved curriculum and no fabricated completed episodes.
- Small-scene acceptance now bounds and deduplicates test counts;8env acceptance uses1/4/8 rather than nonexistent IDs through31.

These are parity results for the tested configurations,not a blanket guarantee of PhysX determinism or policy quality.

## Final integrated validation

### Full regression suite

Parent rerun after review repairs:

```text
202 passed, 44 warnings, 42 subtests passed in 19.73s
```

Warnings are existing TorchScript deprecations and an intentional legacy-PBT-RNG resume warning. Black/isort/flake8,Python compilation and whitespace checks passed on scoped changes. Independent standards review reported no actionable findings. Correctness review identified malformed evidence,rollout frame stamps and small-scene probe counts; all three were fixed,regression-tested and independently rechecked.

### Native PhysX smoke

Final [smoke log](../../logs/dexpbt_profile_local/fix12_final_smoke.log):

-8environments,native6-second episodes,robot32position iterations.
-8completed training episodes,8at fixed ADR1,and8at fixed ADR0 in the final run.
-Genuine completed reports exercised injected-clock freshness pruning,minimum-count checks,held-out-condition exclusion and state restore.
-No matching traceback,CUDA/OOM or contact-overflow errors.

Smoke seeds were not fixed across separate launches; episode counts from earlier stress runs are not trajectory-equivalence evidence. The dedicated parity runs above used seed45.

### Actual tiny PPO and full resume

Final fresh PPO: **13epochs / 3,328frames**,8env,horizon32,one mini-epoch. Publication256frames and minimum1episode were diagnostic overrides only; production5M warmup remained unchanged.

- Epochs1–11: `objective_valid=false`,missing evidence.
- Epoch12/publication3,072: valid evidence,warmup reason;8episodes correctly stamped **frame2,880**,not rollout-start2,816.
- Final publication3,328:the original batch retained identical timestamp/statistics plus one newly completed episode stamped3,256.

Full checkpoint resume into a separate local workspace: **epoch14 / frame3,584**. Model/normalizers,17optimizer state entries and exact curriculum state restored. Rolling evidence was empty and objective invalid after the short resumed rollout,as required; no old evaluations were relabeled fresh.

Logs:

- [Final PPO](../../logs/dexpbt_profile_local/fix12_final_ppo.log)
- [Final resume](../../logs/dexpbt_profile_local/fix12_final_resume.log)

Checkpoints:

- [Fresh final.pth](../../logs/dexpbt_profile_local/fix12_final_ppo/logs/rl_games/dexpbt_reorient/pbt_workspace/000/final.pth)
- [Resumed final.pth](../../logs/dexpbt_profile_local/fix12_final_resume/logs/rl_games/dexpbt_reorient/pbt_workspace/000/final.pth)

These are execution,metadata and restore checks,not learned-task success or population-exploitation evidence.

## Additional artifacts

The linked logs, profiles, and checkpoints are ignored local validation artifacts, not
repository-delivered files. They resolve only in the validation workspace. Reproducing
checkpoint-driven probes requires a compatible checkpoint at `<run>/nn/<file>.pth` and
its original `<run>/params/agent.yaml` and `<run>/params/env.yaml`; use the checked-in
probe and smoke scripts with your own artifact paths.

- [Reset public acceptance](../../logs/dexpbt_profile_local/reset_acceptance.json)
- [Small-scene acceptance](../../logs/dexpbt_profile_local/reset_acceptance_small.json)
- [Reset ABBA summary](../../logs/dexpbt_profile_local/reset_abba_summary.json)
- [Pre-change reset profile](../../logs/dexpbt_profile_local/reset_events_before.json)
- [Candidate reset profile](../../logs/dexpbt_profile_local/reset_int64_after.json)
- [Profiling/acceptance CLI](../../scripts/probes/dexpbt_step_profile.py)
- [Extended native smoke](../../scripts/smoke/dexpbt_smoke.py)
- [Reset regression tests](../../tests/test_dexpbt_reset.py)
- [Probe regression tests](../../tests/test_dexpbt_step_profile.py)

All owned local simulation processes exited after validation; GPU returned to15MiB/0% utilization. The user's GUI and remote training were not touched.
