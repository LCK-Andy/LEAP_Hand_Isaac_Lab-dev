# Stage A TD-MPC2 on Weste RTX 6000D — 2026-09-28

## Request and isolation

After launching Stage B with TD-MPC2, the user requested Stage A training as well. The same **TD-MPC2** implementation is used, not the older TD-MPC algorithm. This is a separate single learner with no PBT, preserving the active Stage B and Dexsuite experiments. The request did not authorize stopping them or replacing their source snapshots.

Remote endpoint: `root@connect.weste.seetacloud.com:26065`, accessed using SSH key authentication. GPU reports NVIDIA RTX6000D, 85,651 MiB. At preflight the two existing processes used approximately10.3GiB GPU memory and the container used32.5GiB of its92GiB memory cap. Memory headroom does not imply unused compute: multiple learners share the same GPU and may slow one another.

No installed packages, system settings, backend assets, native task rewards/actions/observations or existing run data were changed. The new snapshot is copied into its own workspace, including untracked TD-MPC2 implementation files. Its source/asset manifest contains **509 SHA-256 entries**.

## Task and learner

Task: `Isaac-Grasp-Cube-LEAP-Direct-v0`. Native124observation channels,16bounded joint-target actions, 120HzPhysX/30Hzcontrol,300-step finite episodes,0.015rad target increment, full normal-plus-friction tactile sensing and non-pulp-support exclusion. The measured pregrasp fixture initializes each episode; this is not grasp acquisition from an open hand.

Success is native **two continuous seconds of valid grasp**, captured before reset. It is reported separately from full-horizon support-free retention. A hold followed by a drop can count as native success but not retained or safe success. `safe_success` means native success AND support-free retention; valid/overload fractions retain their native300-step denominator.

Settings: one learner×256environments,seed1,10Mtransitions,5Mmodel preset/fivecritics,H3,512candidates/64elites/24policy trajectories,sixplanner iterations,batch256,UTD1,LR3e-4,compiled learner. Replay requested1M/allocated1,000,192CPU transitions; discount59/60. Random seed1500transitions crosses1536, onlysixcontrol steps/environment. No PPO or other-task weights/replay reused.

Checkpoints are full learner/replay/RNG/accounting snapshots every100ktransitions, latest/previous retention. Physics resets on resume, partial episodes are discarded, replay continuity and planner histories reset. No exact simulator-state restore is claimed. Native finite timeouts suppress value bootstrapping.

## Implementation and review

- `scripts/tdmpc2/grasp.py`: native StageA adapter/metrics and shared finite-horizon capture/accounting.
- `scripts/tdmpc2/grasp_motion.py`: retains only StageB-specific motion validation, success and report semantics over the shared code. Its25original regressions remain unchanged.
- `train.py`/`standalone.py`: explicit StageA dispatch and seed defaults, no population imports.
- `scripts/smoke/tdmpc2_grasp_motion_smoke.py`: `--stage a` added, default remainsStageB.
- The existing remote StageB and Dexsuite source trees remain unchanged, so their source-hashed checkpoint contracts remain resumable with their original snapshots. New local changes intentionally require the new source contract.

Independent spec review found no introduced correctness gaps. Standards review found no hard-rule violations and one low-severity heuristic: a short tensor-report validator remains duplicated between the stage-specific report readers. That extraction was not required for correctness and was not added after validation source freeze.

## Validation

-52new StageA tests plus25unchanged StageB tests passed. Tests include native2shold interrupted/reached/then dropped, retention versus success, finite replay masks, native terminal observation capture, stale reports, malformed/nonfinite tensor rejection, reset interception cleanup, recent/full statistics restore and124-input learner continuation.
- Full repository suite: **425passed +42subtests**,144existingTorchScript/legacy-RNG warnings.
- Scoped Black/flake8/isort, byte-compilation and whitespace checks passed. Pyright/basedpyright are not installed; static typechecking was not run.
- Local real PhysX smoke forStageA and sharedStageB regression both passed,4env×330steps each. StageA recorded4real finite timeouts and4native hold successes,100%support-free retention,approximately99.7%validoccupancy,zerooverload. StageB stationary tracking remainedzero. These are fixed-fixture MDP checks, **not learned-policy acquisition**.
- Remote StageA smoke independently passed the same terminal/hold/retention criteria with the existing curatedEA experience. The adapter forbids nonfinite input and checks pre-reset observations; the smoke forbids object writes outside resets and checks that observation queries do not advance hold state.

- Remote production-size compiled probe: **2560transitions/2560updates**, tenvectorsteps, all10criticparameter tensors hadAdamstate atstep2560. Online/targethead meanabsolute weights0.018752/0.018693;value_loss0.224069. Losses finite, real MPC actions collected. No episode completed in thisshortprobe.
- Fullcheckpoint resume: **2816transitions/2816updates**, exactlyone newvectorstep,no remainingdebt. Criticoptimizersteps2816;value_loss0.362052. ZIPCRC passed for the full checkpoint. Validation files were not reused as production priors.
- ActualEAversions verified before launch:lab3.0.0rc1,Sim6.1.0.0,torch2.11.0+cu128,TensorDict0.14.2. Fresh production launched only after allnative/compiled/resume/source/resource gates passed.

## Production acceptance

At **2026-09-28 01:45:03+08:00**, freshStageA PID**470984** was advancing in session`tdmpc2-stagea-train-r1`. Two observations progressed from**5120transitions/5100updates** to**5376transitions/5300updates**, with76pendingupdates correctly accounted. All10criticparameter tensors hadAdamstate atstep5300; online/targethead means were0.019715/0.019710;lossesfinite. ActualMPC collection and criticoptimization are established, not graspquality.

The existing AutoDL TensorBoard discovered the new run with25scalar tags. Initial throughput report approximately**45.9transitions/s** is a short startup-era sample, not a steady benchmark or completion estimate. No productionepisode hadcompleted yet and no productioncheckpoint existed; thefirst isdueafter100ktransitions. Separate validatedtiny/resume checkpoints exist.

StageB PID465366 advanced41984→47872transitions andDexsuitePID435187 advanced407808→413952 duringproduction acceptance. Source/checkpoints/TensorBoard ofboth werepreserved. GPUprocess memory:StageA4164MiB,StageB4230MiB,Dexsuite6318MiB. Totalcontainermemory37.7GiB/92GiB; no new high/OOMevents (historicalhigh58808unchanged). Allthree nowshareGPUcompute.

Finalhand-off check at**01:46:23+08:00**:samePID470984active,**9216transitions/9100updates/116pending**,criticoptimizersteps9100,zero completedepisodes atonly36controlsteps/environment. Latestshort throughput47.7transitions/s. This is startup/learning-pipeline evidence only.

## Remote locations

- Source and launch root: `/root/autodl-tmp/tdmpc2_stage_a_20260928_weste_r1`
- Source snapshot: `repo/`
- Production data: `/autodl-fs/data/tdmpc2_stage_a_20260928_weste_r1/train`
- Production session: `tdmpc2-stagea-train-r1`
- Console log: `/root/autodl-tmp/tdmpc2_stage_a_20260928_weste_r1/train.log`
- AutoDL TensorBoard: port6007, run `tdmpc2-stage-a-weste-20260928-r1/train`
- Production checkpoints: data`/checkpoints/latest.pt` and`previous.pt`, first due after100ktransitions.

The existing TensorBoard server and history are preserved. Curated experience:
`/root/autodl-tmp/leap_grasp_stage_a_20260926/ea_runtime/isaaclab.python.headless.ea-curated.kit`.
Each process uses a separate Kit user configuration. Validation workspaces are separate from production.

Authoritative records: `preparation.json`, `source_manifest.json`, `tiny_acceptance.json`, `resume_acceptance.json`, `prelaunch.json`, `train_launch.json`, `production_acceptance.json`. Local copies/logs are under `logs/tdmpc2_validation/stage_a_20260928/`.

No automatic crash retry, recurring monitoring, shutdown or checkpoint pruning was scheduled. No commit or push was made. See [the StageA operator guide](../tdmpc2-stage-a.md).
