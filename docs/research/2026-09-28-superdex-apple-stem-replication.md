# Case 7: Dexterous apple-stem grasp — source audit and replication guide

Research date: 2026-09-28. Scope: public-source research and static artifact inspection; **no installation or fresh physics rollout was performed**. The existing Isaac Lab environment and robot tasks were not modified.

## Executive finding

**The released demo is reproducible in principle, but it is not a demonstrated online Astra policy.** Its author states:

> “Astra assisted with writing and debugging the controller. The videos and validation records on this page use the known-pose baseline.”

> “Control: Known object poses, inverse kinematics, and joint targets. No model API at runtime.”

The best first replication is the unmodified **DexLab + SuperDex FP64 + OpenArm/Wuji** baseline in its own Python environment. Replacing the hand with LEAP, replacing physics with Isaac Lab/PhysX, or inserting runtime Astra decisions are separate experiments, not faithful reproductions.

The contact simulation is more substantive than a welded apple replay: the controller advances SuperDex dynamics, and the apple is supported by measured contacts. However, known poses, an unbreakable rigid stem, selected self-collision exclusions, and uncapped native pose-target actuation substantially limit the claim. Success is not evidence of hardware-ready manipulation or general-purpose language-model dexterity.

## 1. Sources, dates, and what can actually be verified

Inspected DexLab revision:

`eab7e0114858fc96bfe54821590a0e3a8031e113`

This is the initial public release committed on **2026-09-20**. A read-only source checkout was inspected at `/tmp/dexlab-apple-trace.86wmFO/Dexlab`; it remained clean. Public robot assets were downloaded into temporary research storage and their pinned archive SHA-256 was checked. This was not an installation.

The [Awesome collection][index] labels the original Rednote demonstration as published **2026-09-10** and places it under zero-shot simulator control. The original Rednote link redirected to login during research, so that date, the original video’s precise behavior, and its equivalence to the released baseline were not independently established. They are verified **index statements**, not verified original-post metadata.

The author’s [index PR #5][pr] added the DexLab link on September 20–21 and explicitly preserved the previous credit and technical description. That explains how a broad “Astra controls” summary can coexist with the much narrower first-party runtime disclosure. It does not establish that the original video used an online model.

No model-session transcript, original prompt, model tool-call log, or iterative development history was found in the inspected release, asset listing, or public issue/PR records. Simulation trajectories are available; they are not language-model interaction logs.

Evidence hierarchy for this report:

1. Source code and versioned native API definitions for implementation claims.
2. Author-packaged trajectories, numerical logs, manifests, and check results for that particular recorded run.
3. Author descriptions and media provenance for attribution.
4. The Awesome index as a discovery pointer, not a benchmark or proof of online inference.

## 2. What the system actually is

| Component | Released implementation |
|---|---|
| Robot | Fixed-base OpenArm V20 dual arms with two Wuji Hand 2 Beta1 hands |
| Moving side | Right arm and hand; left arm parked |
| Degrees of freedom | 54 revolute joints in the combined robot: 7 arm + 20 hand per side |
| Object | Apple fruit and stem combined into one free rigid body, mass 0.2 kg |
| Physics | SuperDex FP64; CPU execution in the documented demo |
| Display | Official unmodified MuJoCo, pose display/rendering only |
| Controller | Known-pose inverse kinematics, smooth joint-target paths, native pose tracking |
| Physics interval | 0.002 s, i.e. 500 steps/s of simulation time |
| Main recording | 40 simulated seconds, 20,000 dynamics steps |
| Observation access | Simulator ground-truth poses, not camera pose estimation |
| Runtime LLM/RL | Neither required by the released baseline |

Actual execution structure:

```text
Known apple pose + robot geometry
    → numerical inverse kinematics and grasp planning
    → smooth joint-target trajectory + gravity-compensation offsets
    → native SuperDex articulated pose controller
    → SuperDex scene.step(0.002)
    → measured contacts, object dynamics and robot poses
    → MuJoCo mocap pose synchronization + mj_forward + rendering
```

MuJoCo is not computing a second independent set of dynamics here. Offline video scripts replay saved SuperDex transforms. A rendered replay is useful evidence of the recorded motion, but it is not a new physics trial.

### No hidden runtime language-model requirement

The runtime needs no model API key or checkpoint. Astra’s documented role is controller development and debugging. There is no released prompt that can be supplied to a fresh Astra instance to recreate the original development process exactly.

This differs from the preceding humanoid experiment in this project: that experiment genuinely used successive model-selected effort commands while simulation waited. The apple release instead delegates execution to a controller authored in advance, making detailed planning and stable joint tracking much easier than token-by-token free-floating humanoid balance.

## 3. Controller sequence and physical authenticity

The released [controller][controller] prepares the scene, settles the apple for approximately 3 simulated seconds, solves the initial grasp, and then runs the 40-second sequence. Planning temporarily sets robot configurations to evaluate forward kinematics and gravity compensation before the live trial. These planning writes must not be confused with object teleportation during a grasp.

The live sequence is approximately:

| Simulation time | Action |
|---|---|
| 0.5–2.5 s | Arm approaches the planned grasp pose |
| 3–6 s | Fingers close along the planned stem-pinch path |
| 7–10 s | Arm follows the lift path |
| 11–14 s | Stem-hold validation window |
| 14–17 s | Place the apple back down |
| 17.5–19.5 s | Release the pinch |
| 20–22 s | Retreat |
| 23 s | Replan the fruit grasp using the measured released apple pose |
| Later sequence | Approach, close around fruit, and lift |
| 37–40 s | Fruit-hold validation window |

The second inverse-kinematics calculation uses an isolated planning scene so it does not move the live robot while evaluating candidates.

Within the inspected main simulation loop, actions are applied using `actor.set_articulated_target_pose(pose=target)` followed by `scene.step(DT)`. No apple pose overwrite, palm/root teleportation, supporting weld to the apple, gravity reduction, or object-velocity overwrite was found in that loop. The combined robot is mounted to the world through fixed joints. Those fixed mounting joints are expected; they are not apple attachments.

### Important actuator limitation

The demo provides stiffness/damping and pose targets but no explicit motor-effort cap. In the inspected UniLab backend reference, `effort_limit` in robot prefab data is advisory metadata rather than an automatically enforced physics constraint. The native pose-tracking saturation default is unbounded, and this demo does not supply saturation or a hardware-specific torque limiter.

Therefore, the asset’s nominal arm/finger effort limits do **not** establish that the grasp obeys real motor capabilities. The defensible description is:

> Contact-driven manipulation under native, potentially idealized pose-target actuation.

This is a central reproduction caveat, not a reason to dismiss all contact evidence. Preserve the original settings for the first replication, then add torque-limited actuation as a separate ablation.

## 4. Why the stem geometry and collision model matter

The released apple comes from converted NVIDIA Isaac Sim 5.1 assets; task-specific derived meshes are versioned, so reproducing it does not require rerunning USD conversion.

Static inspection of the packaged geometry found:

- Fruit bounding dimensions approximately **89.2 × 87.4 × 94.7 mm**.
- Combined fruit-plus-stem height approximately **104.7 mm**.
- Stem cross-section widths approximately **2.05–3.65 mm**, depending on section.
- The stem’s own bounding-box height is approximately 29.3 mm, but it extends only approximately **10.0 mm above the fruit’s maximum height**. Do not interpret the full stem bounding box as freely exposed stalk length.
- The apple and stem form one rigid SDF collider with target maximum voxel size **0.2 mm**.
- Stem bending, fracture, detachment, tissue deformation, and material strength are not modeled.

The configured contact parameters include approximately 0.1 mm penalty threshold, 0.15 mm smoothing half-distance, and penalty coefficient `1e10 Pa/m`. The contact friction coefficient reported as 0.5 is not set explicitly by the demo controller; it matches the inspected native backend default. Preserve that backend/version relationship rather than assume the report field actively configures friction.

At 0.2 mm target resolution, a 2–4 mm stem spans only roughly 10–20 voxels in width. That is enough to explain why collision geometry fidelity is consequential, but it is not a resolution-convergence proof. A coarse convex hull of the whole apple would erase the stem pinch geometry and invalidate the comparison.

The released right-hand prefab contains **140 explicit disabled self-collision pairs**. The demonstration is therefore not an all-pairs hand self-collision test. Retain these exclusions when reproducing, document them, and test changes separately.

SuperDex also exposes differentiable-simulation capabilities in its source API, but the inspected demo does not use differentiable trajectory optimization or backpropagation through physics. Smooth contact and the presence of a differentiable API must not be presented as evidence that this controller learned the grasp.

## 5. What the released numerical evidence establishes

The author supplies a historical source-build recording and a separate official-PyPI validation. These are different provenance paths:

| Record | Runtime | Solver status summary |
|---|---|---|
| Packaged historical trajectory/media | Earlier UniLab source build | 22 `STOPPED` steps, no reported divergence |
| Published installation check | Official SuperDex 1.0.0 FP64 wheels | 26 `STOPPED` steps, no reported divergence |

Both report 40 simulated seconds and all 23 independent checks passing. Here “23” counts **assertions**, not seeds, trials, or success cases. The live controller computes 21 sequence checks; the standalone verifier adds fixed-base and finite-transform checks.

Selected historical metrics from [summary.json][summary]:

- Stem-hold minimum clearance: **115.216 mm**.
- Stem-hold vertical hand-support/weight ratio: **0.999727**.
- Stem-hold table support: **0**.
- Reported non-stem contact force during stem phase: **0 N**.
- Thumb and index pad contact magnitudes around **7 N each**; combined recorded scalar approximately 14 N.
- Fruit-hold minimum clearance: **139.727 mm**.
- Maximum reported penetration: approximately **0.452 mm**.

A 0.2 kg apple weighs approximately **1.96 N**. The roughly 14 N sum of opposed contact magnitudes is not 14 N of net upward lifting force; opposing components cancel, and friction/contact directions determine support. The measured net vertical support, zero table support, and object motion together are more informative than contact magnitude alone.

Static inspection of the packaged artifact found 800 finite pose frames, each containing 93 transforms of 7 values, and a stationary robot base. In that recording, the stem-hold robot contacts were confined to the right thumb/index pads. These are checks of released bytes, **not a new SuperDex run**.

### Gaps in the verification

- Pose/contact classification is sampled at 20 Hz; force/momentum data are recorded at 500 Hz. This is not a proof of every intermediate contact geometry.
- A sequence check named `no_other_robot_support` excludes non-right-side robot links, but by itself does not exclude every other right finger or palm. The packaged contact data provide narrower evidence for the recorded hold; the broad assertion name should not be trusted as a complete support-exclusion specification.
- No randomized multi-seed success rate, held-out object pose test, or hardware calibration is established.
- Three-second hold windows do not establish indefinite retention.
- The result is not a proof of full wrench closure or robustness to disturbances.
- Maximum penetration is small in absolute terms but nontrivial relative to a millimetre-scale stalk; a contact-resolution/timestep sensitivity study is still warranted.

### `STOPPED` is not `CONVERGED`

The official documentation distinguishes an iteration stop before the residual tolerance (`STOPPED`) from invalid divergence (`DIVERGED`). The demo records status and continues; it does not restore/retry stopped steps. In the inspected native implementation, scene time advances before the solve and is not rolled back merely because the result is `STOPPED`.

The verifier rejects divergence but does not reject every stopped step. The reported PyPI hold windows were converged, but the full trajectory was not all-converged. Preserve and inspect these counters; do not rewrite “23 checks passed” as “every solver step converged.”

## 6. Recommended exact reproduction

### 6.1 Use an isolated checkout and environment

Do not install SuperDex or downgrade packages inside `/home/andy/env_isaaclab`. The existing environment contains MuJoCo 3.11.0 and the validated Isaac Lab stack. DexLab provides its own dependency pins and `.venv`.

Verified local prerequisites during this research:

- Linux x86_64, Python 3.12 available.
- `uv`, CMake, Ninja, C++ compiler, and FFmpeg available.
- RTX 5070 Ti Laptop GPU with approximately 12 GB VRAM and NVIDIA driver 595.91.07.
- SuperDex was not installed in the Isaac environment.

The documented physics run is CPU-based and slower than its 40 seconds of simulation time. The GPU is useful for rendering, but it does not imply CUDA-accelerated SuperDex execution in this demo. No local wall-time or memory benchmark has been measured. This machine appears suitable for an isolated attempt; success remains to be tested.

### 6.2 Pin the released code

The commands below are a **future execution recipe**, not commands run during this research. Use an unused checkout path; do not overwrite an existing project.

```bash
git clone https://github.com/huangkiki/Dexlab.git \
  /home/andy/Documents/projects/Dexlab

cd /home/andy/Documents/projects/Dexlab
git checkout --detach eab7e0114858fc96bfe54821590a0e3a8031e113
```

### 6.3 Install the official FP64 wheel path

```bash
env -u SUPERDEX_ROOT -u SUPERDEX_NATIVE_BIN -u SUPERDEX_PYTHON \
  bash scripts/setup.sh
```

The versioned installer:

- obtains Python 3.12 using `uv` and creates a project-local `.venv`;
- installs the pinned Python dependencies;
- installs **both** `superdex-physics[double]==1.0.0` and `superdex-robotics[double]==1.0.0`;
- checks the FP64 runtime;
- downloads the versioned robot asset archive and the apple albedo texture with SHA-256 verification.

The umbrella `superdex` package is not required. The default path requires no engine compilation or API key. Clear the three source-override variables as shown to avoid accidentally selecting an unrelated native build. The run wrapper sets `SUPERDEX_PRECISION=fp64` before imports.

The inspected direct dependency pins include NumPy 2.5.2, SciPy 1.18.1, trimesh 5.1.0, MuJoCo 3.11.0, Pillow 11.3.0, and rtree 1.4.1. Use the versioned installer rather than manually combining these with Isaac Lab's older NumPy/SciPy stack. Pinned direct dependencies do not, by themselves, imply every transitive package is frozen or that installation on this machine has been tested.

The source-build alternative uses UniLab commit `f216dace36464d70f224caa4253074ec365ed14f`, Clang 18, and the project build scripts. It is useful for reproducing the historical source baseline, but is not the recommended first run. The current official Meta repository layout differs; substituting its latest branch for the pinned UniLab tree is not an equivalent build.

### 6.4 Run fresh dynamics with a unique output directory

```bash
env -u SUPERDEX_ROOT -u SUPERDEX_NATIVE_BIN -u SUPERDEX_PYTHON \
  bash demos/apple-stem-grasp/run.sh \
  --headless \
  --output demos/apple-stem-grasp/runs/local-replication-01
```

Remove `--headless` for the interactive MuJoCo viewer. The viewer opens only after SDF preparation, settling, and planning, so a delayed window is not necessarily a hang. Headless mode still constructs the MuJoCo display model; it does not remove the MuJoCo dependency.

The wrapper runs the full sequence. Its default destination is `runs/latest`; the explicit unique output avoids accidental confusion between runs.

Expected recording files include `plan.json`, `body-plan.json`, `display.xml`, `trajectory.npz`, metrics and dynamics JSON, and `summary.json`. Video encoding is separate, not automatic.

### 6.5 Verify the new recording before making video

```bash
.venv/bin/python demos/apple-stem-grasp/src/verify_wuji_sequence.py \
  demos/apple-stem-grasp/runs/local-replication-01
```

Check the fresh output, not just the bundled `evidence/` directory. Confirm:

1. Intended wheel/source version, FP64, asset hashes, and parameter settings.
2. Full 40-second/20,000-step sequence completed.
3. All 23 independent assertions pass.
4. No divergence; record every `STOPPED` step and whether it occurred in a hold window.
5. Apple support comes from appropriate robot-object contacts, with zero table support during the hold.
6. Base is stationary and all recorded states are finite.
7. No accidental state writes, engine substitution, or visual-only replay is being mistaken for fresh dynamics.

Do not demand bitwise-identical trajectories across different native builds or CPU environments. Compare the documented check conditions and metric ranges first; investigate material deviations without tuning them away silently.

### 6.6 Render from that new recording

```bash
MUJOCO_GL=egl .venv/bin/python \
  demos/apple-stem-grasp/src/render_wuji_two_views.py \
  demos/apple-stem-grasp/runs/local-replication-01

MUJOCO_GL=egl .venv/bin/python \
  demos/apple-stem-grasp/src/render_stem_focus.py \
  demos/apple-stem-grasp/runs/local-replication-01
```

This needs FFmpeg and functional EGL/OpenGL drivers. Video duration is simulation time, not elapsed compute time. Inspect the stem close-up alongside contact and force logs; a persuasive rendering alone is insufficient.

### Licensing

DexLab code is Apache-2.0. Robot and apple assets retain separate terms: OpenArm Apache-2.0, Wuji MIT, and NVIDIA asset licensing for apple-derived geometry/textures. Media do not relicense the underlying assets. Preserve the source notices and [asset provenance][assets] if distributing a reproduction.

## 7. Three different replication goals

### A. Reproduce the released result — recommended first

Keep robot, assets, physics precision, contact settings, exclusions, controller, and sequence unchanged. Run the official wheel baseline, verify it, then render. This is the lowest-confounding-variable route and requires no LLM API or training.

### B. Build an actual online Astra apple-grasp policy

This is additional work, not something the release already implements. Keep the validated dynamics scene and expose a bounded action/observation interface:

- observations: joint states, object pose or camera estimate, fingertips, contacts, previous action outcomes;
- actions: short palm/fingertip targets or bounded joint targets;
- local execution: the explicitly disclosed IK/tracking layer;
- timing: freeze simulation while waiting for a model decision;
- logging: exact model messages, images, selected actions, durations, state transitions, and failures.

Evaluate known-pose and vision-only conditions separately. Preserve the script baseline as a comparison, and disable precomputed whole-task trajectories in the online condition. Measure success over varied apple poses/stem orientations rather than one favorable scene. This would align with the user's original interest in direct model control, but should be labeled **model planning over IK/joint tracking**, not model-generated motor torques unless that is truly the interface.

### C. Port the task to LEAP + Isaac Lab

This is an MDP/control/physics port, not a hand-name change:

- Wuji has 20 hand joints; LEAP has 16, with different fingertip geometry and reachable pinch configurations.
- The original robot has a moving arm. A fixed LEAP palm cannot reproduce the same lift trajectory without an arm or a deliberately different task.
- The millimetre-scale stem demands separately validated collision geometry; a whole-object hull is unacceptable.
- SuperDex SDF contact settings cannot simply be copied into PhysX parameter names.
- Contact normals, friction forces, actuator effort limits, self-collision exclusions, and supporting-contact identity all require measurement after the port.

For the inertial-estimation work in this repository, this apple demonstration is not equivalent to a support-free four-pulp grasp: its stem stage deliberately uses thumb and index only, and its object/stem inertia is known, rigid, and simplified.

## 8. A stronger follow-on validation study

After the exact baseline works, vary one factor at a time:

- timestep and SDF resolution;
- friction and contact smoothing;
- realistic effort/speed limits;
- self-collision masks;
- initial position and stem orientation;
- mass and external perturbations;
- perception noise versus simulator-ground-truth inputs.

Report failed trials, nonconverged steps, inappropriate supporting contacts, penetration, and per-finger forces. If the aim is physical plausibility of lifting by a real stem, add measured stem strength/compliance or clearly retain the rigid-stem limitation. Do not silently make the scene easier and call it a reproduction.

## Sources

[index]: https://github.com/zjwzcx/Awesome-Astra-Embodied-AI/blob/ba9fd3b305dcd53cdfcb3960207555849bfbb5a8/README.md
[pr]: https://github.com/zjwzcx/Awesome-Astra-Embodied-AI/pull/5
[controller]: https://github.com/huangkiki/Dexlab/blob/eab7e0114858fc96bfe54821590a0e3a8031e113/demos/apple-stem-grasp/src/wuji_stem_grasp.py
[summary]: https://github.com/huangkiki/Dexlab/blob/eab7e0114858fc96bfe54821590a0e3a8031e113/demos/apple-stem-grasp/evidence/summary.json
[assets]: https://github.com/huangkiki/Dexlab/blob/eab7e0114858fc96bfe54821590a0e3a8031e113/docs/ASSETS.md

1. [DexLab demo disclosure and usage](https://github.com/huangkiki/Dexlab/blob/eab7e0114858fc96bfe54821590a0e3a8031e113/demos/apple-stem-grasp/README.md).
2. [Installation and version provenance](https://github.com/huangkiki/Dexlab/blob/eab7e0114858fc96bfe54821590a0e3a8031e113/docs/installation.md).
3. [Controller source][controller], including pose-controller setup and live stepping.
4. [Standalone verifier](https://github.com/huangkiki/Dexlab/blob/eab7e0114858fc96bfe54821590a0e3a8031e113/demos/apple-stem-grasp/src/verify_wuji_sequence.py).
5. [Historical metrics][summary], [PyPI check](https://github.com/huangkiki/Dexlab/blob/eab7e0114858fc96bfe54821590a0e3a8031e113/demos/apple-stem-grasp/evidence/pypi-check.json), and [release check](https://github.com/huangkiki/Dexlab/blob/eab7e0114858fc96bfe54821590a0e3a8031e113/demos/apple-stem-grasp/evidence/release-check.json).
6. [Media provenance](https://github.com/huangkiki/Dexlab/blob/eab7e0114858fc96bfe54821590a0e3a8031e113/demos/apple-stem-grasp/media/stem-focus.json).
7. [Asset terms and sources][assets], [versioned asset release](https://github.com/huangkiki/Dexlab/releases/tag/apple-stem-assets-v1).
8. [Official Meta SuperDex repository](https://github.com/facebookresearch/project_superdex), and [historical UniLab reference](https://github.com/unilabsim/project_superdex/tree/f216dace36464d70f224caa4253074ec365ed14f).
9. [SuperDex physics 1.0.0](https://pypi.org/project/superdex-physics/1.0.0/) and [robotics 1.0.0](https://pypi.org/project/superdex-robotics/1.0.0/).
10. Pinned UniLab native definitions: [advisory effort limits](https://github.com/unilabsim/project_superdex/blob/f216dace36464d70f224caa4253074ec365ed14f/superdex_robotics/include/superdex_robotics/superdex_robotics.h#L81-L88), [pose-tracking saturation](https://github.com/unilabsim/project_superdex/blob/f216dace36464d70f224caa4253074ec365ed14f/superdex_physics/libraries/mochi/mochi_physics/mochi_physics_structs.mochi_gen#L1617-L1623), [friction default](https://github.com/unilabsim/project_superdex/blob/f216dace36464d70f224caa4253074ec365ed14f/superdex_physics/libraries/mochi/mochi_core/include/mochi_core/contact/contact_params.h#L183), and [scene time advancement](https://github.com/unilabsim/project_superdex/blob/f216dace36464d70f224caa4253074ec365ed14f/superdex_physics/libraries/mochi/mochi_physics/src/mochi_scene.cpp#L641-L659).
11. [Official solver-status FAQ](https://projectsuperdex.com/physics/docs/faq/#does-a-nonlinear-solver-status-of-stopped-mean-the-step-failed), live documentation rather than an immutable snapshot.

This research establishes a source-backed replication recipe and identifies important fidelity boundaries. It does **not** certify that a new run succeeds on the local machine or that the original Rednote video used the same implementation.
