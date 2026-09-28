# FlashSAC Dexsuite: additional100M continuation — 2026-09-28

## Request and result

The user requested extending the completed FlashSAC run by100Msteps. This is **additional** training, not a restart from scratch or a100Mtotal budget. Original50,000,896transitions plus100M gives an absolute requested target150,000,896; complete1024-env batches finish at150,001,664. The continuation is running, not complete.

Originaltask,1024env,4M CUDA replay,batch2048,two updates/vectorstep,gamma.99,n-step3,AMP,seed0 andnetwork/normalization/exploration settings are retained. NoPBT. Replay was not stored in the model checkpoint, so replay refills for100ktransitions before updates resume; this is not exact replay continuation. StageA/Bjobs were preserved.

## Original checkpoint and exact restoration

Original50Mrun finishedexit0 at50,000,896transitions/97,460updates/48,829vectorsteps,143,206completedepisodes and0recordedsuccesses. Checkpoint55490959bytes,SHA256`3f68b99f1ddc854a545e39804d18fc04b5fa094f734117ec6771ac1fe7851207`.

All350originalsourcehashes were verified. A same-filesystem checkpoint hardlink and originalmetadata are retained in the continuation data`/prior`, never modified in place. Before any resumed update, an external validation wrapper compared allfour models, actor/critic/temperature optimizer states and scheduler states exactly to the saved payload. Restoration passed.

OriginalLRclosure: zero warmup,97658decaysteps,minimum1.5e-4. Atresume:

| Network | Scheduler epoch | LR |
|---|---:|---:|
|Critic|97460|0.00015000152140301872|
|Actor|48730|0.0002252388568885381|
|Temperature|48730|0.0002252388568885381|

Actor/temperature advance every other learner update; they must not be prematurely clamped. The continuation reconstructs the original closure before loading states, then each scheduler continues and clamps at its own endpoint. NoLRjump or restartedwarmup.

## Narrow runner change

`--extend-budget` explicitly permits a strictly larger absolute budget, requiring acheckpoint and fresh workspace. `--schedule-steps50000896` decouples originalLRschedule construction from enlarged collectionbudget. A sourcebridge accepts only the exactoriginaltrain/runtimehashpair and verifies destination hashes against executing files; allother task/agent/Flash/native source fields remain strict. Saved frame/counters must be coherent and targetgreater thanactualsavedframe, includingvectorrounding.

Originalpair:

- runtime`65a7f5a82d3e0215bd1a814c9697b550fda9dfda62f3182151bddfc0e7175a88`
- train`4d93f655a599c4a1cf0b6ff20ef394056d671deca8cce62f5ba69daba6bb9e4f`

Finalpair:

- runtime`2f9b04aafba77fb2f7183a0ed04c8083157fdd5344e2b85b4d050f5dcfb3e86d`
- train`96fe29bf2e6ce8a521421dd7b894fba589b545f093f46fbcc42ca1d51ea37626`

## CUDA-graph failure and validated workaround

The first full-size resume restoredexactly and refilledreplay but failed at firstactorbackward with`Detected1tensor(s) in the cudagraph pool not tracked as outputs`. Failurelogs/sourcewerepreserved; no checkpointwas produced andnoproductionlaunched fromthatattempt.

The corrected execution mode is`max-autotune-no-cudagraphs`. Torch2.11 reports this retains maxautotune andcoordinate-descenttuning without enablingCUDAgraphs. Compilation andAMP remain on; noinstalledpackagepatch. Only this exactmodechange fromsavedmax-autotune isallowed duringexplicitextension; otheragentdifferencesare rejected.

Corrected120-vector smoke passedexit0 at50,123,776transitions/97,502updates (+42). Allmodels/optimizer/schedulerstate matched beforeupdates. Actor/temp48751LR0.00022518819039646669;critic97502LR0.00015000094441667525, exactlyoriginalclosure. Finitelearner/coverage1. SmokecheckpointSHA`d031a03e2b193b5d28e05f9301858f985c3fbb03842ba7aea1a1f7f68e17bbba`.

Tests:30FlashSACtests and30subtestspassed, including13extensioncases, source/schema/hparam rejection, originalLRclosure/nextstep/half-rateoptimizer behavior, actualcounterguards andstrictnewcheckpointresume. Formatting/lint/diffchecks passed.

## Live continuation

Root:`/root/autodl-tmp/flashsac_dexsuite_20260928_weste_r1_extend100m`

Data:`/autodl-fs/data/flashsac_dexsuite_20260928_weste_r1_extend100m/train`

Session:`flashsac-dexsuite-extend100m`;PID676498 atacceptance. Started from ORIGINAL50Mcheckpoint, not the smoke's extraexperience. Source/dependency snapshots independent andunchangedoriginalsverified.

Verifiedproductioncheckpoint at51,001,344transitions/99,216updates (+1756updates), preserved atdata-parent`/production_first_acceptance.pt`,SHA`8f67527a2db1851db34bbe790d3dce3eb99d52e851b18b895dfbb4000ea3cdc4`.

- CriticLR0.00015 afteritsendpoint;actor/temperature49608LR0.00022312070347429836.
- Alllearnertensorsfinite;actor/criticoptimizercoverage1,weightdeltas5.414896/9.499557.
- Ninepositiveintervals approximately6.8–7.8ktransitions/s; earlymeasurements notcompletionpromise.
- TensorBoard continues from50M, observed through51,207,168.
- GPU17008MiBfree;cgroup57.2/92GiB. StageB465366/StageA470984stillalive.
- Boundedacceptancemonitorexited; noautomaticretryornewmonitorautomation.

Localfailure/retry/productionevidence:`logs/flashsac_validation/extend100m/remote/`.

## Launch/resume recipe

Initialextension:

```bash
python scripts/flashsac/train.py --workspace NEW_WORKSPACE \
  --checkpoint ORIGINAL_LATEST --extend-budget \
  --max-steps 150000896 --schedule-steps 50000896 \
  --compile-mode max-autotune-no-cudagraphs \
  --num_envs 1024 --replay-capacity 4000000 --batch-size 2048 \
  --warmup 100000 --updates-per-vector-step 2 --headless --device cuda:0
```

Future normalresume usescontinuationcheckpoint, sameabsolute target/schedule/compilemode andoriginalsettings, but **omits`--extend-budget`**. It doesnotaddanother100M. Replayrefillremainsrequired. This extensiondoesnotestablishsuccessfultasklearning;original143206episodeshadzerorecordedsuccesses andrunninglongerdoesnotguaranteeacquisition.
