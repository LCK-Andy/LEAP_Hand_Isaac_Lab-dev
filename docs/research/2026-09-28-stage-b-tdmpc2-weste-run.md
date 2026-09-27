# Stage B TD-MPC2 on Weste RTX 6000D — 2026-09-28

## Authorization and isolation

The user requested Stage B training with TD-MPC2 on the RTX Pro 6000D, without PBT. The request did not authorize stopping the existing Dexsuite TD-MPC2 experiment. A separate single-learner Stage B run was prepared on the same GPU; the Dexsuite trainer and its source/checkpoints/TensorBoard history were preserved.

The server reports **NVIDIA RTX 6000D, 85,651 MiB**, with an effective container memory cap of 92 GiB. Verified stack: Isaac Lab **3.0.0rc1 EA**, Isaac Sim **6.1.0.0**, torch **2.11.0+cu128**, TensorDict **0.14.2**. No installed packages, global configuration, compiler passes, or physics assets were modified.

The existing curated EA experience is retained:

`/root/autodl-tmp/leap_grasp_stage_a_20260926/ea_runtime/isaaclab.python.headless.ea-curated.kit`

Each validation/production process uses a separate Kit user configuration. SSH key authentication was used; no credentials were written to source or deployment records.

## Implementation and configuration

The preexisting corrected standalone TD-MPC2 implementation is extended with the import-safe `scripts/tdmpc2/grasp_motion.py` boundary. The original Stage A/B task code and TD-MPC2 core are unchanged. The learner has no population selection, mutations, donor or PPO initialization path.

Task: `Isaac-Grasp-Motion-Cube-LEAP-Direct-v0`. Native 148 observations, 16 joint-target actions, full normal-plus-friction force sensing, finite 300-step episodes, 120 Hz physics/30 Hz control, hand solver 8/0. Motion command remains 0.1° XYZ at 0.2/0.3/0.4 Hz, no translation. Both failure and finite-horizon timeout are terminal for TD targets; native pre-reset successor observations are preserved. Tracking success is the native continuous one-second hold, not Stage A grasp success.

One learner, 256 environments, seed 1, 10M transition budget, 5M model preset, five Q networks, horizon 3, 512 candidates/64 elites/24 policy trajectories, six planner iterations, batch 256, UTD1, LR 3e-4, compiled learner. CPU replay requested 1M, allocated 1,000,192 transitions, approximately 1.18 GiB plus model/optimizer. Checkpoint interval 100k transitions, retaining latest/previous full states. Random-action seed phase 1500 transitions crosses at 1536; it covers only six control steps per environment, not a full commanded-motion episode.

Fresh Stage B model/optimizer/replay; no Dexsuite or Stage A PPO checkpoint is reused. The physical pregrasp fixture is still supplied at every reset. This run tests tracking from an initialized grasp, not grasp acquisition from an open hand.

## Validation

- New adapter/runner tests: **29 passed**. Full repository suite: **372 passed + 42 subtests**, 144 existing TorchScript/legacy-RNG warnings.
- Critic-learning/runner tests: 25 passed including CPU/CUDA gradient updates and continuation; compiler execution is tested separately on the target server.
- Scoped Black/isort/flake8, byte-compilation and whitespace checks passed. Pyright was not installed; static typechecking was not run.
- Independent spec review found no introduced correctness gaps. Standards review found smoke exception-status/ownership issues; both were repaired and rechecked.
- Local and remote native boundary smoke: four environments, 330 control steps, four real 300-step timeouts, correct no-bootstrap terminal masks and pre-reset successors. Approximately **99.7% valid grasp occupancy**, full-horizon retention, **zero stationary tracking success**. This is fixture/MDP validation, not a trained result.
- Remote production-size compiled learner probe: **2560 transitions/2560 updates**, ten vector steps, finite losses and all ten Q parameter tensors with Adam state at step 2560. Online/target head mean absolute weights **0.018884/0.018658**; value loss **0.174646**. Real MPC collection and critic optimization occurred; no episode completed in this short probe.
- Full checkpoint resume: **2816 transitions/2816 updates**, exactly one new vector step, no pending debt. All critic optimizer steps reached 2816; both checkpoint archives passed ZIP CRC checks. Physics intentionally reset; learner/replay/RNG/statistics restored.
- **508 source/asset SHA-256 hashes** matched the isolated remote snapshot. Deployment source is not inferred from git status: the preexisting TD-MPC2 files were untracked and explicitly included.

An initial local smoke was inadvertently started while the adapter helper was still being completed and failed with a missing `_report` method. The finalized source was rerun successfully locally and remotely; no failed smoke result was treated as acceptance.

## Production acceptance

At **2026-09-28 01:29:05 +08:00**, PID **465366** was accepted in session `tdmpc2-stageb-train-r1`. Independent samples advanced from **5120 transitions/5000 updates** to **5376 transitions/5300 updates**, with the remaining 76 updates correctly recorded as pending debt. All ten critic parameter tensors had optimizer state at step 5300; online/target head mean absolute weights were **0.020250/0.020240**. Losses were finite.

The AutoDL TensorBoard run was discovered with 25 scalar tags. The short initial throughput report was approximately **70.5 transitions/s**, including startup effects; it is not a steady-state speed or completion estimate. No training episode had completed yet, and no task-acquisition claim is made. The first production checkpoint is due after 100k transitions; it was **not yet present** at acceptance. Separate smoke/resume checkpoints exist.

The preexisting Dexsuite PID435187 remained alive and advanced from 366080 to 371968 transitions during this acceptance interval. Stage B used **4164 MiB VRAM**, Dexsuite 6318 MiB, and total container memory was approximately **32.3 GiB/92 GiB**. No new cgroup high/OOM events were observed; the old high counter 58808 was unchanged. GPU sharing can affect both experiments' speed.

Final live check at **01:30:27 +08:00**: **10240 transitions, 10100 updates, 140 pending**, same PID alive, no exit record. All critic optimizer states advanced to step10100; latest short throughput report approximately60.4transitions/s. Zero completed episodes and no production checkpoint yet remain expected at only40 control steps per environment, not evidence of tracking acquisition.

## Run locations

- Source/launch/log root: `/root/autodl-tmp/tdmpc2_stage_b_20260928_weste_r1`
- Source snapshot: `repo/`
- Production data: `/autodl-fs/data/tdmpc2_stage_b_20260928_weste_r1/train`
- Production tmux session: `tdmpc2-stageb-train-r1`
- Console log: `/root/autodl-tmp/tdmpc2_stage_b_20260928_weste_r1/train.log`
- AutoDL TensorBoard port: **6007**, existing server preserved.
- TensorBoard run: `tdmpc2-stage-b-weste-20260928-r1/train`
- Full checkpoints: production data `checkpoints/latest.pt` and `previous.pt` after each 100k interval.
- Validation data are separate in `tiny/` and `smoke.json`, not production priors.

Records in the run root include `preparation.json`, `source_manifest.json`, `tiny_acceptance.json`, `resume_acceptance.json`, `prelaunch.json`, `train_launch.json` and `production_acceptance.json`. Local evidence is under `logs/tdmpc2_validation/stage_b_20260928/`.

No automatic retry, periodic monitoring, shutdown or checkpoint deletion was scheduled. No code was committed or pushed. Operator instructions and metric definitions: [Stage B TD-MPC2](../tdmpc2-stage-b.md).
