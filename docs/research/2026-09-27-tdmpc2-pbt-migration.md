# TD-MPC2/PBT migration on Weste — 2026-09-27

> Historical failed-critic run. After the diagnosis below, the user authorized the
> fix and changed direction to one TD-MPC2 learner **without PBT**. All four r1
> trainers were stopped on2026-09-27 at23:23:40+08, with full final checkpoints
> preserved. See [standalone training](../tdmpc2-standalone.md) for the replacement.

## Authorized scope

The user requested stopping the current training and replacing it with **TD-MPC2 plus PBT, four members with 256 environments each**. The task remains `Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0`, with native 1745 observations, 23 joint actions, 120 Hz physics/60 Hz control, solver32, six-second episodes, object distribution, rewards and curriculum. No Cartesian controller or physics simplification was authorized or deployed.

See [the operator guide](../tdmpc2-pbt.md) for implementation and checkpoint semantics. TD-MPC2 starts fresh; PPO weights are not compatible initializers.

## Previous population stopped and preserved

The eight PPO trainers were verified by PID and stopped with SIGINT. All exited130; no SIGTERM was needed. Existing TensorBoard, logs, checkpoints and immutable source remain intact. SIGINT did not produce additional final checkpoints.

Independent pre/post copies and manifests are stored remotely at:

`/autodl-fs/data/tdmpc2_migration_20260927_223450_6dfacc33`

93 preserved source/config/checkpoint files passed SHA-256 checks, including 32 checkpoint copies passing ZIP CRC, totaling786MiB. The broader manifest contains117 entries including generated evidence. Authoritative records: `stop_manifest.json`, `pre_preservation_manifest.json`, `post_preservation_manifest.json`, `post_verification.json`, `final_process_verification.json` and `SHA256SUMS.json`.

Latest periodic member005 was publication053, epoch6470/frame53,002,240. Historical raw-reward best member002 reached score12.5227 at epoch6515/frame53,370,880; this is not held-out success evidence.

## Implementation and review

A separate runner under `scripts/tdmpc2/` vendors six MIT-licensed official TD-MPC2 core files from `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`. Vector planning follows the official vectorized branch with independent resets/chunking and corrected probability sampling; its synchronous-reset collector is not used.

The integration includes native terminal-history preview, explicit pre-reset replay successors, bounded CPU sequence replay, full learner/optimizer/target/scale/RNG state, and replay-aware PBT. It reuses public async-pbt0.1.0 selection/mutation rather than disguising TD-MPC2 as RL-Games PPO. Review repaired canonical-contract comparisons, recipient-owned staged-resume accounting, pending updates at terminal budget, resident restore payloads and fresh-namespace/writer-lock safeguards.

No installed Isaac/Torch packages were patched. Remote async-pbt was absent, so the exact0ddbaca92f66495990651771ada7321a351eb8be release was installed only into the new run's `deps/` via `--no-index --no-deps --target`. The public Git clone timed out; the fallback wheel's20package files matched the release. Wheel SHA-256: `1c6728a6a78db5074aa90c33d182749e54ab0beaf6cc88cb5e5cbcf0eef8547c`.

## Local validation

- Relevant CPU suite: **280 passed, 1 CUDA-only skip, 7 subtests passed**. Separate new-code suite: **72 passed**, including CUDA terminal-preview coverage.
- Two actual CPU member processes verified donor weights/Adam/replay inheritance with recipient budget and mutation-RNG preservation.
- Native four-env training with reduced planner settings:32 transitions/32updates, then full resume40/40. Resumed model tensors were finite; optimizers retained40 and10 state entries.
- Native collect-only terminal probe:370 vector steps,1480 transitions,4 natural completed episodes, native6s/solver32 retained. Published evidence was valid for the diagnostic minimum1, not the production128 threshold.
- Black/isort/flake8, compilation and whitespace checks passed. No typechecker was installed.

Ignored local evidence: `logs/tdmpc2_validation/`.

## Full production-config remote smoke

Run root:
`/root/autodl-tmp/dexpbt_tdmpc2_20260927_weste_4x256_r1`

Data root:
`/autodl-fs/data/dexpbt_tdmpc2_20260927_weste_4x256_r1`

The smoke used256env, official5M model preset, H3,512candidates,64elites,24policy trajectories,eight effective iterations,chunk32,batch256,UTD1 and131072replay capacity. It completed **2560transitions/2560updates**, ten vector steps including two real MPC calls, four completed episodes, and exited0 with no pending updates. Evidence correctly remained ineligible because4<128.

- Initial2048-update burst: approximately56seconds.
- Two planned steps:31.13 and26.52 transitions/second.
- Accumulated plan time1.711s, environment time1.371s, update time69.812s; these are host-side accounting, not a synchronized GPU decomposition.
- Total process wall136.95s; trainer elapsed110.57s including checkpoint save.
- Peak sampled GPU6099MiB; cgroup RAM11.32GiB including checkpoint save.
- Final checkpoint1,904,766,433bytes, SHA-256 `62819a8f21108ce232e198d409a244bb0603dd7401bf21fbb5ed19a3d1c62b48`; ZIP CRC and85 checked agent tensors passed.

Acceptance record: `smoke256_acceptance.json` in the run root. The first wrapper attempt exited127 before training because `/usr/bin/time` was unavailable; its files were retained and a stdlib timing wrapper explicitly corrected it. No blind retry loop was introduced.

## Production configuration and source

Production source is an independent `repo_prod/` snapshot, preserving the smoke `repo/` and old PPO sources. Base repository commit7548774698ded3efac70cd7ba03ab30c77344b63 plus new uncommitted TD-MPC2 integration is identified by336SHA-256 file entries. Final bundle SHA-256: `6b48aefd73a5d0d9d7288356c48be7ddb18a90fa021c03f416edb2ba1f368043`.

Four isolated processes, seeds42–45, each256env:

- Full planner/model configuration as smoked, eager execution.
- Replay131072 transitions/member, approximately1.72GiB persistent CPU arrays; full snapshots include replay.
- UTD1:256 gradient updates per256-env collection step, not one update per vector step.
- Initial diversity enabled; PBT mutates LR, entropy coefficient and gradient clipping.
- PBT publication100k transitions, warmup300k, post-exploit adaptation100k, minimum128completed episodes, evidence age limit21600seconds. These launch overrides replace PPO-scale1M/5M/3M gates for the measured slower learner. Physical-condition and frame-age matching remain.
- Default budget10M transitions/member (40M nominal total), chosen because the user did not specify a budget; vector rounding reaches10,000,128 per member. It is not a promised completion time.
- No automatic crash restart. Only two owned numeric full replay archives retained, plus a separate staged restart.

Production data workspace: data-root`/prod4`. TensorBoard discovery link: `/root/tf-logs/dexpbt-tdmpc2-weste-20260927-4x256-r1`, using the existing port6007 server.

## Production acceptance

All four members launched after smoke and all-member preflight, passed the initial2048-update burst and multiple learned MPC collection steps. No startup failures or automatic retries occurred.

| Member | PID | Seed | Saved transitions / updates | Recent transitions/s |
|---|---:|---:|---:|---:|
| 0 |425920|42|7680 /7680|23.99|
| 1 |426105|43|5120 /5120|24.28|
| 2 |426285|44|5120 /5120|23.72|
| 3 |426470|45|5120 /5120|25.51|

Logs advanced beyond those saved statuses. Initial aggregate rate was approximately97.5transitions/s, not a steady-state guarantee. GPU24425MiB used/60618MiB free; containerRAM27.3GiB; datafs119GiB free. All336production source hashes matched. Distinct initial hyperparameters, runtime contracts and TensorBoard scalar streams were verified.

Remote acceptance: run-root`/production_acceptance.json`. Downloaded evidence: `logs/tdmpc2_validation/remote4x256/`, including all four agent/config/launch/status/env records. Sessions are `tdmpc2-prod256-00` through`03`. Only the temporary resource monitor was stopped; trainers and existing TensorBoard remained active. No scheduled monitoring or automatic retry was added.

The first production checkpoint/publication remains pending at100ktransitions/member. No production exploit or learned task performance was established. CPU two-member tests prove the replacement mechanism, not production acquisition.

## Critical follow-up: critic learning failure confirmed, 2026-09-27 23:19 +08:00

The user's progress-review request uncovered a blocker that the launch smoke/tests missed. This supersedes any inference that finite losses and advancing updates establish a healthy TD-MPC2 learning run. All four processes were still advancing at roughly24transitions/s/member, with35,840/33,280/33,280/33,280transitions in the latest sampled statuses, but value loss stayed at2.6921544, the uniform101-bin categorical baseline weighted by rho=.5 over horizon3. Policy gradient norms shrank toward1e-7 and Q scale remained1.

A local CPU reproduction with nonzero reward showed every registered critic parameter unchanged after20updates. Independent backward established that critic parameters were in the optimizer and required gradients but received no gradients. The deployed `Ensemble._call` path in `scripts/tdmpc2/vendor/common/layers.py:32–37` uses TensorDict `params.to_module(self.module)` inside vmap; in the inspected stack this breaks the gradient connection to registered ensemble parameters. A diagnostic-only local functional_call alternative produced a nonzero critic-head gradient; no source fix was applied.

A separate CPU-only reproduction at23:19:37+08:00 used the actual immutable remote source, torch2.11 and TensorDict0.14.2. Three updates with reward2 left all10critic parameter tensors unchanged with gradientsNone, while reward loss decreased. Direct Q-loss backward reproduced the missing critic gradients. Remote/local layers.py SHA matched `a3c98a188babaa8e46d051d52c9437602df7c080fc7a71cc717b176ef4b60889`. This proves the deployed implementation defect, without claiming direct inspection of live trainer weights.

**Verdict: operationally running, but not a healthy TD-MPC2 learning run.** Earlier tests checked finite outputs, restore equivalence and planner parity, but failed to require actual critic learning. The integration needs a critic-gradient regression and correction before another validated run. PBT cannot repair a shared implementation defect. No training processes or files were changed by the review; stopping/fixing/restarting remains a recommendation pending user authorization.

Only15episodes had completed across the four members;130–140vectorsteps/env had elapsed versus360-step episode horizon. No publication existed before100k, so actual task success was unavailable, not provenzero. No OOM or runtime failure was observed. Continuing to wait for success does not address the critic defect.

## Interpretation

TD-MPC2 is **not faster in transition throughput** here: the solo smoke was26–31transitions/s, and four-process contention can lower each member's rate. The intended experiment is sample efficiency and task learning, not a guaranteed wall-time improvement. A10M-member budget may take days or longer; compare success versus both transitions and wall time.
