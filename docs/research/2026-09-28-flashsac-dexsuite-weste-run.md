# FlashSAC replacement of Dexsuite TD-MPC2 — 2026-09-28

## User request and unchanged task

The user requested stopping the AutoDL TD-MPC2 learner and restarting with recommended FlashSAC settings. This is one fresh FlashSAC learner, without PBT, on `Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0`. The suffix denotes the preserved environment/measurement adapter, not population training. Native1745observations,23joint actions,120Hzphysics/60Hzcontrol,360-step episodes,solver32,rewards,contact gates and curriculum are unchanged.

Other Stage A/Stage B trainers were found on the same GPU and explicitly left untouched. No learned TD-MPC2 state or replay is transferred to FlashSAC.

## TD-MPC2 preservation

Only exact verified Dexsuite PID435187 received SIGINT. It completed its final checkpoint and exited0 in approximately105seconds.

- Final frame591104,updates591012,pending92.
-1566completedepisodes,0successes,lifetime meanreturn−0.348859,length352.167.
- Finalcheckpoint8406053181bytes,SHA256`08c93456288615859b22c1a1048b308ee226e67b20e7f115d266284bd4f3c643`.
- Preservation records and same-filesystem checkpoint hardlinks:
  `/autodl-fs/data/tdmpc2_dexsuite_20260927_weste_single_r2/stop_for_flashsac_20260927T185548Z`.

Original logs, source and TensorBoard remain. Stage B PID465366 and Stage A PID470984 were independently identified by argv and continued advancing. They were not stopped to make room for FlashSAC.

## Recommended configuration and resource exception

Official [IsaacLab benchmark script](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/scripts/run_isaaclab.sh) supplies the starting profile, not a task-specific Dexsuite tuning result:

-1024vector environments; minibatch2048; two learner updates per **vector step**, not per transition.
- Warmup100000storedtransitions; gamma.99; n-step3.
- Actor2residualblocks,width128; twin distributional critics2blocks,width256;101bins[−5,+5];tau.01; actorupdateperiod2.
- Reward normalization withGmax5; repeated zeta exploration and learned temperature; no additional observation normalizer or reward redesign.
- AMP enabled; compilemodeauto resolves to max-autotune ontorch2.11.
- Budget50,000,896transitions, matching the benchmark script rather than a1B custom budget.

Official10Mreplay would require approximately140.6GB with1745float32observations plus successors, exceeding thisGPU. Capacity is explicitly resource-adapted:2Mvalidated first, then4Mtested against shared-GPU headroom before production. Batchsize, update ratio and n-step semantics are not reduced to accommodate memory.

## Source integration and deliberate corrections

New entrypoint `scripts/flashsac/train.py` uses the actual FlashSAC agent, while bypassing its obsolete EA environment wrapper. The shared `DexsuiteVectorEnv` adapter supplies allthree native observation groups, correct term-major terminal histories and native episode measurements. Actor actions stay in[−1,+1]; native joint-delta scale.1 is applied once.

LocalFlashSAC source base`e4436fbccb748de7c7fd273a92b1b7195797790b` descends from official`87edc9061150ae9e962dd84e6544e27a1554b3ab` with three local commits and seven tracked dirty files. A separate content-hashed source snapshot includes relevant compatibility changes. Neither the original local clone nor installed Isaac/Torch packages was changed. Preparation is reproducible with`scripts/flashsac/prepare_source.py`.

Snapshotpatchversion2 fixes:

1. Three-step replay carries actual-horizon discounts. A timeout after one/two steps bootstraps with gamma/gamma² from its genuine terminal observation, not alwaysgamma³. True termination still masks bootstrap.
2. Reward normalization includes the terminal reward in the preceding discounted-return statistic, then clears the per-environment accumulator. It no longer drops the preceding return or carries terminal reward into the next episode.
3. Replay export avoids serializing unused backing storage; old replay files without the discount field are rejected. Archive export refuses existing destinations.

Capability marker`N_STEP_DISCOUNT_VERSION=1`; source manifestpatchversion2. Flashsource archiveSHA`639571239b7420c578ede15b0d9cf6da27653c95f0601cb425604def184d5868`. Integration10-file bundleSHA`6c4bfc9d96ed2ffe1ccfceef95b7c759cebe678d2953ebcb9551ccc80fa9b535`. The runtime uses an isolatedPYTHONPATH, not a package upgrade.

Checkpoints contain all four networks, optimizers/schedulers, AMP, running reward statistics, exploration/RNG, native curriculum and counters. **Replay is intentionally not saved**; resume restores learner state but refills replay and resets physics/partial n-step streams. This is explicitly not exact off-policy continuation. Latest/previous model snapshots are retained atomically. Graceful stop completes the current bounded update block; no automatic retry loop.

## Local validation

-17FlashSACsnapshot/native contract tests passed, including actual actor/critic gradients and weight movement, n-step discount arithmetic, target projection, reward-normalizer boundaries and checkpoint restoration.
- Independent review verified nonzero dQ/da, not merely entropy-driven actor changes.
- Relevantregression suite391passed,3CUDA-onlyskipped,7subtests; expectedTorchScriptwarnings only.
- Native4envsmoke:48transitions/14updates, full actor/critic optimizer-state coverage, nonzero actor/critic/target changes, finite checkpoint.
- Learnerresume with explicit empty-replay refill reached96transitions/28updates; learned optimizer/normalizer state restored.
- Black/isort/flake8/compilation and whitespace checks passed.

Ignored local evidence: `logs/flashsac_validation/`.

## Remote 2M replay smoke

Root:`/root/autodl-tmp/flashsac_dexsuite_20260928_weste_r1`

Data:`/autodl-fs/data/flashsac_dexsuite_20260928_weste_r1`

Isolated exactFlashSACsource imported correctly; all10integration hashes verified.1024env/batch2048/n-step3/AMP/compileauto/2updates per vector step/100kwarmup.

-400vectorsteps,409600transitions,602updates,exit0; approximately497seconds including startup/autotuning.
-1044naturalcompletedepisodes,0successes,meanreturn−0.614565,length353.147. This is execution validation, not an acquisition result.
- Actor/critic optimizer coverage1; actor/critic weightdelta4.933/10.795,targetdelta9.987.
- Checkpoint55490063bytes,216checkedtensorsfinite,SHA`589aa78403ac5a38339e412be29aa898ea73b2f8e96d11a1fb8749e538601505`.
- Peak ownGPU34664MiB,total43183MiB,minfree41860MiB. Torchallocationpeak26.687GiB; cgrouppeak55.47/92GiB,noOOMevents.
- Other Stage A/Btraineridentities and forwardprogress verified throughout.
- Postcompile positive-frame logged throughputmedian7705transitions/s,range2951–11402; no steady-state or learning-speed guarantee. The final same-frame status emits0FPS and is excluded as a reporting artifact.

Localacceptance:`logs/flashsac_validation/weste_smoke2m_acceptance.json`; evidence tar nearby. Remotecheckpoint:data`/smoke2m/checkpoints/latest.pt`.

## Production gate

A4Mreplay fit probe is authorized only with identical compiled algorithm settings and shared trainers left active. Production may use4M only if it exits0 with finite learning/coverage, GPUfree≥10GiB,cgroupusage<80GiB,noOOM, and otherjobs advancing. Otherwise use the already validated2Mcapacity for memory limitations; unrelated failures require diagnosis, not blind retry.

## Live production acceptance

The4Mfit passed:143360transitions/82updates,exit0,finite216checkpointtensors,optimizercoverage1,minimumGPUfree16902MiB,cgrouppeak54.51GiB,noOOM. Both unrelated trainers advanced. Production therefore uses **4M CUDA replay**.

Fresh production is running at data`/train`, session`flashsac-dexsuite-r1`, PID513562, seed0,50,000,896transitionbudget,1Mcheckpointinterval,100klogging. ExistingTensorBoard6007 receives run`flashsac-weste-20260928-dexsuite-r1/prod`. No TD-MPC weights/replay or smoke training state was reused.

Latest accepted observation:1,507,328transitions,2746updates,4146completedepisodes,0successes. Actor/criticoptimizercoverage1; actor/critic/targetparameterdeltas6.668/20.401/20.105; finite actorloss0.11085 and criticloss3.8586. This establishes execution and optimizer activity, not task acquisition.

Thirteen positive postcompile logging intervals had median7536transitions/s,range6700–8288, a short initial window rather than a guaranteed sustained rate. SampledminimumGPUfree16836MiB,cgrouppeak54.71/92GiB,noOOM. StageA/StageBtraineridentities and continuedprogress were verified; no unrelatedjob was stopped.

Firstcheckpoint at1,000,448transitions/1756updates was verified and preserved via same-filesystem hardlink at data`/production_acceptance/first_observed_checkpoint.pt`:55490127bytes,SHA256`cf05477e814c1ee1f13d87c7021f4f1d8ba05ddb62931fcd2a090d7f85ba1a83`,216tensorsfinite. Replay is intentionally excluded from learner checkpoints.

350frozen source files were hash-verified. Local acceptance/launch/fit records and evidence tar are in`logs/flashsac_validation/remote_production/`. The temporary monitor exited after acceptance; no scheduled monitoring or retry automation remains. Training and the existing TensorBoard server continue. No code was committed or pushed.
