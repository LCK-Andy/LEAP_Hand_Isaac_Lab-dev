# Critic autograd fix and standalone TD-MPC2 — 2026-09-27

## Authorization

The user authorized fixing the confirmed critic-learning defect, then explicitly changed the replacement run to **TD-MPC2 without PBT using recommended settings on the RTX6000D**. The new design is one learner with the existing256vector environments, not four independent learners with selection merely disabled. The same native Dexsuite task/physics/observations/joint actions are retained.

Operator guide: [standalone TD-MPC2](../tdmpc2-standalone.md).

## Faulty run preserved and stopped

The four exact r1 trainer processes425920/426105/426285/426470 received SIGINT at23:23:40+08 on2026-09-27. All exited gracefully within100seconds; no SIGTERM/KILL was used. Full final checkpoints preserve incomplete update debt:

| Member | Transitions/replay | Updates | Pending updates |
|---|---:|---:|---:|
|000|44800|44663|137|
|001|43776|43624|152|
|002|43264|43207|57|
|003|43520|43381|139|

PTH/YAML hashes and CRC checks passed; CPU loading/counters matched results. Originals (~1.905GB each) remain in `/autodl-fs/data/dexpbt_tdmpc2_20260927_weste_4x256_r1/prod4/tdmpc2/`. No large duplicate copies or deletions were made. Preservation records, faulty-source warning, TensorBoard metrics, exact argv and source hashes are in:

`/autodl-fs/data/dexpbt_tdmpc2_20260927_weste_4x256_r1/r1_stop_for_critic_fix_20260927_232339`

These are diagnostic artifacts, not valid learned TD-MPC2 policies. They are not used to initialize the replacement.

## Root cause and repair

The TensorDict parameter insertion path in `vendor/common/layers.py` created new Parameter objects from batched tensors inside vmap, disconnecting the critic loss from registered ensemble parameters on the installed0.14stack. Optimizer membership and requires_grad flags looked correct; finite loss and roundtrip restoration tests therefore missed the failure.

The narrow fix passes named parameter tensors to `torch.func.functional_call` inside vmap. TensorDict remains the parameter container, checkpoint keys stay stable, and the target/detached ensemble relationships remain unchanged. No upstream learned objective or reward, actor/critic architecture, initialization, physical dynamics or installed package is changed.

New regression tests were observed failing on both CPU and CUDA before the fix. They now establish actual learning of a positive-return fixture by every critic, optimizer-state creation, detached-policy action gradients, independent target weights and correct Polyak updates. A first-update guard rejects missing critic optimizer states; live diagnostics report online/target head magnitudes and minimum critic Adam step.

Independent comparison against ordinary non-ensemble networks confirmed forward outputs and all critic/input gradients. Actor updates with entropy zero confirmed value-driven learning rather than only entropy changes. Eager CUDA checkpoint continuation matched bitwise. Default five-Q compiled CUDA continuation passed with tolerances1e-5/1e-7.

A two-Q test fixture exposed a separate torch2.11 Inductor randperm-index pattern issue. The actual five-Q default compiled correctly, so no compiler-pass workaround was added. TensorDict recompile warnings in multi-agent test processes are not claimed as fullgraph certification.

## Standalone runner

`scripts/tdmpc2/train.py` and `standalone.py` implement one learner, pure local checkpoint ownership, episode-weighted metrics and direct critic diagnostics. They contain no population imports, objects, mutation, selection or donor restore logic. The Gym ID still ends inPBT because it denotes the preserved environment/measurement class, not an active population engine.

Pinned official TD-MPC2 main: `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`. Production settings:

- Seed1;10Mtransitions;1learner;256environments.
-5Mmodel preset, five Q networks, H3,512candidates,64elites,24policy trajectories, six base/eight effective MPPI iterations.
- Batch256; one gradient update per transition; LR3e-4,encoderLRmultiplier0.3,entropy1e-4,gradclip20,tau0.01.
- Compilation enabled with upstream reduce-overhead mode for the learner update.
- Replay requested1M, allocated1,000,192transitions to preserve whole vector rows, unpinned CPU explicitobs+nextobs (~13.1GiBfull).
- Seed phase1800transitions, rounded crossing2048; planning chunks32to bound temporaryGPUmemory.
-100ktransition full checkpoint cadence, latest/previous files. No automatic crash restart.

Fresh source contract hashes reject incompatible or foreign checkpoints. Full snapshots include both optimizers, targets, scales,RNG,replay/sampler, pending update debt, curriculum and episode totals. Only initialized replay rows are serialized; filled buffers save without cloning large arrays. Simulator state is explicitly reset on resume; pending update debt is drained even when the transition budget has already been reached.

## Validation

- RelevantCPUregressions:292passed,3CUDA-onlyskips,7subtests; separateCPU/CUDAcritic tests6passed.
- Standalone/replay/schedule tests42passed, including replay compaction/RNG, safe-save interruption, pending debt and process ownership.
- Remote exactstack:Python3.12.3,torch2.11.0+cu128,TensorDict0.14.2,IsaacLab3.0.0rc1,IsaacSim6.1.0.0; no package changes. CPUcritic suite4passed/2CUDA-skipped.
- Full native production-config256env/5M/fiveQ/compiled smoke:2560transitions andupdates,10vectorsteps,exit0. All10critic parameter tensors hadAdamstate, step2560. Onlinehead meanabs0.0181724,target0.0181089; value loss2.69215firstupdate to0.254666final. Not a task-acquisition claim.
- Full-state resume1:frame2816,updates2561,pending255. Resume2 atsameframebudget drained255pending updates to2816/2816 with **zero new vectorsteps**. Onlinehead0.0183543,target0.0182805,criticAdamsteps2816,value loss0.249626.
- Compact checkpoint sizes131MBinitial/135MBresumed; ZIPCRC/load/counters passed. InitialSHA6535433cc467a28332e89a30d9622109b17a9fa3ac0a14a407678254fa6ba656; finalSHA7316ed88622bacf9b187e4e8b688ed6a7a8d0ca5411178961fc4984cc6ba7617.
- Peak sampledGPU6379MiB; containerRAM33.2GiB/92GiB during validation, including retained filesystem cache. Full replay occupancy can grow further.

## Deployment provenance

Root: `/root/autodl-tmp/tdmpc2_dexsuite_20260927_weste_single_r2`

Data: `/autodl-fs/data/tdmpc2_dexsuite_20260927_weste_single_r2`

Corrected source bundle SHA-256: `b555b00152991317bf1823d1a95c443985cd5604ba283273508cdc7149e31c00`;28file hashes verified. The old source is independently preserved. Native evidence: `native_validation_summary.json`, `native_smoke_acceptance.json`, `native_resume_debt_acceptance.json`, `native_resume_drain_acceptance.json`, `native_validation_tensorboard.json` in the new root.

## Live production acceptance

Fresh production launched23:45:30+08 on2026-09-27, using independent immutable `repo_prod`, fresh data`/train`, session`tdmpc2-single256-r2`, and the existing TensorBoard6007 discovery link`/root/tf-logs/tdmpc2-weste-20260927-single-r2` via data`/tb/prod` to training summaries. No faulty checkpoint, replay or prior was inherited; no PBT engine is active.

At23:47:52+08:

- PID435187 was the soleGPUtrainer. OldfourPIDs were absent; TensorBoard14955 remained unchanged.
- Statusframe6912,updates6700,pending212,replay6912. TensorBoard already reached6800updates; the pending count reflects sampling mid-update block, not lost work.
-10/10critic parameter tensors hadAdamstate, minimumstep6700. Onlineheadmeanabs0.021835,target0.021733; value loss2.692firstupdate to0.18672status/0.18761latestTB. Policy gradientnorm0.0002089. Real critic optimization is now demonstrated.
- Initial2048-update burst and19MPCrollouts completed. Eleven poststartup vectorsteps measured63.03transitions/s; recentTBinterval63.86/s. These are short initial samples, not sustained performance or a completion-time promise.
- GPU6215MiBused/78828free; sampledpeakcgroupRAM~21.0GiB/92GiB. Full replay occupancy grows beyond this early sample.
- Onlytwo short completedepisodes andzero measuredsuccesses so far: no task-acquisition claim.
- First100kproductioncheckpoint notyetreached. Validated smoke/resume checkpoints are preserved separately.

Sourcehashes matched afterlaunch. The temporary acceptance monitor exited normally; trainer continued with no autorestart or scheduledmonitor. Remote evidence:`production_acceptance.json` inrunroot. Localcopy:`logs/tdmpc2_validation/critic_fix/remote_single_r2/` includes acceptance, launch/source/TB/log/smoke records.

Final local TD-MPC2 test suite:86passed including CPU/CUDA critic regressions and standalone tests; formatting/lint/diff checks passed. No changes were committed or pushed.
