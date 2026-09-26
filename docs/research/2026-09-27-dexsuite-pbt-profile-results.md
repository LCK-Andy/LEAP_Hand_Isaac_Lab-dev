# Dexsuite PBT performance investigation: measured results

Date: 2026-09-27 (UTC+8). Supersedes unmeasured optimization priorities in [the initial assessment](2026-09-26-dexsuite-pbt-training-speed.md).

## Outcome

The authorized investigation and isolated experiment are complete. **No production job, task default, package or MPS setting was changed.** All 15 remote members were verified advancing again at 00:23:20 UTC+8 on September 27. A local 32-versus-16 robot position-iteration experiment produced **1.304× rollout throughput**, but numerical/learning parity was not established, so the setting was **not deployed**.

The previously suggested physical-condition metadata cache was rejected: measured overhead is only 0.458 ms/control-step, below 1% of local measured wall time. Correctness-sensitive cache invalidation would not be justified by that ceiling.

## What was measured

### Remote production, without attaching a profiler

15 members × 512 environments, PhysX. Logs, TensorBoard records, selected checkpoint state, launch settings and process metadata were read without changing the run.

| Member group | Last-100-epoch harmonic FPS | Environment time/epoch |
|---|---:|---:|
| 0, 1, 8 | 449, 441, 466 | 34–36 s |
| Other 12 | 785–851 | 18–20 s |

Inference remained approximately 0.52–0.56 s/epoch and PPO updates 0.31–0.34 s/epoch in both groups. The environment-time difference persists in matched epochs 1001–1100 and appeared early, not after a late PBT restart.

Earlier same-session 60-second resource sample at 23:50 Sep26:

- All15 advanced; roughly10,053 logged environment steps/s over the short common interval (quantized by epoch logging).
- Sum of recent per-member effective FPS:11,166; different per-member100-epoch wall windows, not an exactly synchronized throughput sample.
- Host-timer attribution, time-weighted:96.31% environment,2.31% inference,1.39% PPO update. Environment includes physics, sensors, reset and task code.
- CPU16.69 core-equivalents used versus25-core container quota; no throttling increment.
- GPU100%busy,104–113W,36C,2%memory utilization,80,163/85,651MiB.
- RAM76.61/92GiB; no new high-memory/OOM events during that interval.

### Explanations eliminated or weakened

- Selected slow members0/1/8 and fast member2 have identical **applied** checkpoint ADR parameters:zero gravity,zero observation noise,tolerance0.2. Different ADR bookkeeping fractions do not imply different applied conditions.
- All15 retain original lineage and `last_exploit_frame=0`; no PBT donor replacements explain the difference.
- Compilation and launch controls match. Intended initial LR/gradient/KL diversity does not explain the almost-equal inference/PPO timing.
- Saved environment configuration differs only by seed. Clone allocation uses deterministic sequential round-robin over16 templates, rather than a seed-selected single object. Live collider maps were not persisted, so actual shape/collider identity was not directly measured.
-126 threads/process and identical allowed CPU masks; NUMA placement overlaps across performance groups; no swap. Accumulated main-thread scheduler waiting was too small to support ordinary CPU scheduling starvation as the main explanation.
- Fast members have substantially higher contact rewards, contradicting a simple “more contacts means slower” account.

### Reset frequency is the leading remaining workload explanation

Slow members have14.9–18.4% out-of-bounds terminations versus0% for sampled fast peers; mean episodes335–341 versus353–355 steps. Even a modest difference in mean length can desynchronize reset times and invoke reset machinery on many more control steps. Existing production telemetry does not count reset-call frequency or batch size, so the remote slowdown's exact cause is **not proven**.

Seed-dependent sampled friction/mass/gains and trajectories remain possible contributors; those realized samples are not all preserved in checkpoints. Shared-GPU synchronization and workload-dependent PhysX cost are also not isolated by read-only logs.

## Local isolated profile

A reusable checkpoint-driven probe was added:

- [scripts/probes/dexpbt_step_profile.py](../../scripts/probes/dexpbt_step_profile.py)
- [tests/test_dexpbt_step_profile.py](../../tests/test_dexpbt_step_profile.py)

Setup: local RTX5070Ti Laptop,512 environments,seed45,128 warmup control steps,real stochastic frozen PPO policy and normalizers from member03 epoch1954/frame32,014,336. Training curriculum was restored;1745 observations/23 actions. Physics timestep1/120,decimation2,robot32 position/1 velocity,object16/0,self-collision,contact sensors and native rewards remained intact. RL-Games set effective torch threads to4. No optimizer updates occur in this probe.

Uninstrumented throughput rounds run before component instrumentation. The initial360-step component profile had66.27ms/control-step wall time:

| Component | Host ms/control-step |
|---|---:|
| PhysX simulate + fetch |33.33|
| Native rewards excluding PBT measurement |8.05|
| PBT measurement |2.65|
| Observations |5.57|
| Resets averaged across all control steps |6.63|
| Policy inference |1.97|
| Condition stamp, already included in PBT measurement |0.458|

These are nested host timings, not mutually independent CUDA-kernel timings. The condition stamp must not be added to PBT again. Synchronization can shift host attribution.

### GPU evidence

Nsight Systems2023.4 injection failed because its bundled CUPTI lacked `cuptiActivityEnableDriverApi`, required by the current stack. No package or driver was altered. The installed PyTorch profiler successfully collected a CPU/CUDA trace instead.

Over32 control steps/64 physics steps:

- Summed GPU kernel durations:1,046.30ms.
- TGS-named kernels:893.19ms,**85.37% of summed kernel time**.
- Two largest articulation/self-constraint kernels:623.69ms combined.

This establishes substantial solver work in the local reproduction. Summed kernel time is not end-to-end wall time and does not measure the remote15-process concurrency case.

## Controlled solver experiment

Only robot position iterations changed from32 to16. Object solver16/0,clock,gains,self-collision,checkpoint and seed were retained. Four fresh processes were run in **A–B–B–A** order to reduce directional time-drift bias. Each process had three uninstrumented360-step rounds after warmup.

| Run | Robot position iterations | Three round times,ms/control-step |
|---|---:|---|
|A1|32|57.21,60.91,59.69|
|B1|16|44.90,46.81,49.59|
|B2|16|41.66,45.71,44.54|
|A2|32|58.96,62.49,57.06|

Aggregate mean32-iteration time:**59.388ms**. Mean16-iteration time:**45.535ms**. Ratio:**1.304× throughput**, or**23.33% less control-step time**. This is local frozen-policy rollout performance, not a demonstrated30% remote training improvement.

In the final profiled pair, PhysX simulate/fetch fell from32.40 to19.56ms/control-step, consistent with the solver change being the primary source of speedup.

### Reset-cost measurements

| Final profile | Reset batches/360 steps | Total envs reset | Median reset batch | Reset-free step | Step with reset |
|---|---:|---:|---:|---:|---:|
|32 iterations|240|555|1|54.40ms|71.29ms|
|16 iterations|234|556|1|40.67ms|53.76ms|

Tiny asynchronous reset batches are expensive in the local workload. These within-run conditional averages are observational, not a randomized causal reset test. They support instrumenting reset cadence in representative slow/fast production workloads, but do not prove that reset cadence explains their full2× difference.

### Fidelity checks do not authorize deployment

Eight360-step diagnostics were run:four fresh processes each at fixed ADR0 and1. All produced finite sampled physical quantities,live object-filtered forces,512–518 completed episodes,zero mixed-condition episodes and **zero successes**. Logs had no matching contact-buffer overflow/drop or CUDA-error messages.

This is **not a stable-grasp or physics-equivalence pass**:

- Contact/impact trajectories and force/velocity tails differed.
- The32-iteration reference itself exhibited kilonewton force tails and object angular velocities approaching100rad/s in some runs.
- Identical random seeds/checkpoint do not yield identical actions once dynamics and observations diverge.
- Penetration,stable grasp retention,actual contact-capacity high-water marks and held-out learning quality were not established.
- The test ran on one local GPU process,not15 production contexts.

**Decision: retain32 in production.** The16-iteration option is a measured performance candidate requiring separate physical and learning validation,not a safe automatic replacement.

## Separate PBT correctness finding

The slow members' evaluation accumulators expire before publication:

- Publication interval:1,000,000 member frames.
- Maximum score age:1,800 seconds.
- At450FPS,one interval takesapproximately2,222 seconds/37minutes.
- `EpisodeWindow.ready()` checks the age of a group's **first-sample timestamp**, then deletes the entire group if expired.

The approximate boundary is1,000,000/1,800=**556FPS**. Below that,an accumulator covering nearly the full publication interval cannot remain fresh. The latest slow-member metadata has `evaluations={}` and objective0 despite large completed-episode counts. Fast members publish before this age limit. This matches the implementation and observed metadata; it is a downstream selection problem,not proof of the speed bottleneck.

Relevant implementation: [pbt_utils.py:180–260](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt/pbt/pbt_utils.py),particularly239–248; donor selection also checks group/record freshness at264–288.

This was **documented,not fixed** in the performance experiment. A correction must preserve genuinely fresh,physically comparable completed episodes. Merely refreshing the creation timestamp would mislabel old data as fresh. Changing publication interval or freshness thresholds changes PBT behavior; a real sliding window is a separate implementation with tests. No production selection-policy adjustment was made.

## Validation, cleanup and production state

-48 tests plus7 subtests passed in the delegated validation,including4 tests of the actual reversible timer class.
-Parent independently reran the4 new probe tests:4 passed.
-Black,isort,flake8,Python compilation and `git diff --check` passed for the added probe/test work.
-No runtime metadata cache was added,no existing defaults edited,no installed packages patched,no commits made.
-All owned local profiling processes exited; local GPU returned to15MiB/0% utilization.
-At00:23:20 on2026-09-27,all15 production members advanced in a65-second common observation interval;recent log excerpts contained no matching fatal/CUDA-OOM/buffer-overflow errors. Epochs ranged1278–2193 and GPU use remained80,163MiB. No remote member was stopped,restarted or reconfigured.

## Evidence and reproduction

These logs, traces, and checkpoints are ignored local validation artifacts, not files
distributed with the repository. Links resolve only in the original validation workspace.
To reproduce elsewhere, supply a compatible Dexsuite checkpoint at `<run>/nn/<file>.pth`
and its original configuration at `<run>/params/agent.yaml` and `<run>/params/env.yaml`;
replace the local checkpoint and output paths in the command below.

Local raw results:

- [ABBA summary](../../logs/dexpbt_profile_local/abba_summary.json)
- [A1](../../logs/dexpbt_profile_local/solver32_a.json),[B1](../../logs/dexpbt_profile_local/solver16_b.json),[B2](../../logs/dexpbt_profile_local/solver16_c.json),[A2](../../logs/dexpbt_profile_local/solver32_d.json)
- [Component profile](../../logs/dexpbt_profile_local/components.json)
- [Kernel summary](../../logs/dexpbt_profile_local/kernel_summary.json)
- [CPU/CUDA trace](../../logs/dexpbt_profile_local/torch_trace.json)
- [Remote diagnosis](../../logs/dexpbt_profile_local/remote_diagnosis_20260927/diagnosis_metrics.json)
- [Remote checkpoint/config evidence](../../logs/dexpbt_profile_local/remote_diagnosis_20260927/final_probe.json.gz)
- [Remote score-aging source evidence](../../logs/dexpbt_profile_local/remote_diagnosis_20260927/score_aging_source.json.gz)

Full selected TensorBoard history captured during the read-only diagnosis remains at `/tmp/leap-speed-20260926-2355/metadata_tb.json`; compact durable evidence above retains the key results.

Run with a free local GPU; use a different output path per invocation and order32,16,16,32:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/andy/env_isaaclab/bin/python \
/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab/scripts/probes/dexpbt_step_profile.py \
--checkpoint /home/andy/Documents/projects/LEAP_Hand_Isaac_Lab/logs/rl_games/dexpbt_download_weste_20260926_230552_member03/nn/000000000032.pth \
--output /home/andy/Documents/projects/LEAP_Hand_Isaac_Lab/logs/dexpbt_profile_local/reproduction32.json \
--device cuda:0 --num_envs 512 --seed 45 \
--warmup 128 --steps 360 --rounds 3 --profile-steps 360 \
--diagnostic-steps 360 --solver-position-iterations 32
```
