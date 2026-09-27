# Dexsuite KUKA–Allegro physics simplification assessment

Date: 2026-09-27

Scope: source and asset inspection, official documentation, and re-analysis of existing measurements. No new simulator or training runs, runtime-code changes, configuration changes, package changes, or remote changes were made. This document is the only added file.

## Conclusion

**Physics simplification is possible, but the robot already has inexpensive collision geometry: 22 primitive colliders, comprising 13 boxes and 9 capsules, with no collision meshes.** There is no obvious large triangle-mesh or duplicate-collider cleanup available.

The most substantiated speed lever remains whole-articulation solver effort. A previous local robot-only requested change from 32 to 16 position iterations produced **1.304× rollout throughput**. Physical parity and faster learning were not established. A 24-iteration intermediate is a conservative experimental candidate, not a measured result.

For structural simplification, the focused candidates are selective collision-pair filtering, analytic capsule colliders for capsule-shaped objects, and possibly consolidation of internal fixed attachments. Benefits are unmeasured and the available geometry reduction is modest. Removing the seven-joint arm is a more substantial simplification, but creates a different task.

## 1. Task identity and evidence scope

The running lineage is the repository's `Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0`, which imports preserved legacy Dexsuite definitions. It uses PhysX, a 1/120-second physics timestep, and decimation 2: **120 Hz physics and 60 Hz policy control**. The installed native EA task is now `Isaac-Reorient-KukaAllegro`; it changes backend defaults, timing, rewards and reset machinery and is not a drop-in alias for the old task name. [S1–S2]

The last verified remote deployment was 8×256 on immutable commit `a3d6b19`, according to the preceding session; remote liveness was not rechecked here. Local HEAD is `7548774`, after the async-pbt migration. Relevant preserved Dexsuite and PBT environment sources have no changes between those commits. Installed package and asset versions remain separate dependencies.

The cached 6.1 robot asset was inspected as USD data, including selected payloads and instance proxies. Public asset HEAD checks verified the cache hashes. An archived remote member configuration selects the same 6.1 USD with no variant override. This is asset/configuration evidence, not a fresh runtime physics census. [A1]

## 2. Actual environment complexity

### Robot

| Component | Inspected count |
|---|---:|
| Authored rigid-body prims | 30 |
| Revolute joints | 23: seven arm, sixteen hand |
| Fixed joints | Seven: world root plus six internal attachments |
| Robot collision shapes | 22: thirteen boxes, nine capsules |
| Arm collision shapes | Six |
| Hand collision shapes | Sixteen |
| Collision meshes | Zero |

The counts describe **authored topology**, not verified runtime solver-body counts or independent constraint rows. Backend fixed-link fusion was not established.

The root is already fixed to the world. Several arm links already have no collider. Distal boxes and capsules form compound finger geometry; no duplicate parent/child collision applications or identical redundant shapes were identified. The four separate biotac-tip bodies have no colliders, but carry mass and fixed attachments.

The 26 visual meshes contain 205,624 triangles, **none of which is an active robot collision triangle**. Decimating those meshes would not simplify the present collision model. Self-collision is enabled by the Python override, and the whole robot requests **32 position / 1 velocity solver iterations**. No gravity-disable opinion was found in the active robot USD; the archived override is null. [S3, A1]

### Objects, table and floor

There are **16 object prototypes, but only one object body and one object collider per environment**:

- Six boxes, using `boundingCube` collision approximation.
- Two spheres, using `boundingSphere`.
- Six capsules and two cones, using `convexHull`.

The recorded clone plan distributes prototypes sequentially round-robin. Reducing shape diversity therefore does not remove 15 bodies per environment; they were never simultaneously present. Source-mesh triangle counts are not cooked hull counts. In particular, boxes and spheres are already primitive collision approximations. The cones do not use expensive custom-cone callbacks in this path. [S4, A1]

Each environment contains a kinematic table box. The scene has **one shared ground-plane collider**. These are already simple supports, not detailed scenery. Removing them changes reset, support or drop behavior. [S5]

### Why simply freezing the arm changes the task

This is workspace object-pose manipulation, not only rotation at a stationary wrist. All 23 joints receive actions. Commands span robot-base-frame positions `x=[-0.7,-0.3]`, `y=[-0.25,0.25]`, `z=[0.55,0.95]`, along with orientation. [S2, S5]

Four object-filtered fingertip sensors provide observations and contact-gated rewards. Contact reporting is distinct from physical constraints: removing a sensor does not remove the underlying contact solve, and would remove required task data. [S6]

## 3. What the performance evidence actually establishes

The existing local 360-step instrumented profile measured:

| Measurement | Time per control step |
|---|---:|
| Total wall time | 66.273 ms |
| Host PhysX simulate/fetch | 33.327 ms, 50.29% |
| Four sensor update timers | 1.545 ms, 2.33%, nested |
| Observation generation | 5.566 ms, 8.40% |

Sensor timings overlap reward/observation work; the rows are not additive. A separate GPU capture attributed **85.37% of summed GPU-kernel time** to TGS. This is not 85.37% of end-to-end wall time. Kernel names containing “self” or “articulation” do not isolate self-collision cost. [M1]

In the existing 512-environment ABBA test, changing **only requested robot position iterations from 32 to 16**, while retaining object16/0, reduced aggregate mean control-step time from **59.3878 to 45.5351 ms**: 23.33% less time, or 1.3042× transitions/second. [M2]

Limitations:

- The probe recorded baseline verification before applying the override; its saved `verified_settings` still says robot32. No effective post-override solver readback was saved.
- Closed-loop trajectories and subsequent actions diverge even with the same checkpoint and seed.
- All eight diagnostic run/ADR combinations recorded zero successes, with kN-scale force spikes and high velocities in both baseline and candidate. This does not establish grasp or physics parity.
- No actual PhysX contact/patch counts or capacity high-water marks were captured.
- These are frozen-policy rollout results, not measured PPO wall-clock time to a success target.

## 4. Ranked simplification options

### A. Reduce solver effort: strongest existing speed evidence

Test requested robot **24/1**, then **16/1**, retaining object16/0, geometry, self-collision, timestep and randomization. The 24-iteration case is unmeasured; the 16-iteration speed result needs physical validation.

This reduces numerical work, not model geometry. Penetration, frictional holding, drive tracking and measured forces can change. The robot is one articulation, so arm and hand links do not offer independent solver-iteration budgets. PhysX also documents iteration coupling within islands. [S7, D2]

### B. Filter only demonstrably unnecessary collision pairs

Fewer eligible pairs can reduce collision detection and contact constraints. However, adjacent linked-body collisions are already excluded by joint defaults, no redundant colliders were found, and no blanket list of safe additional exclusions has been established.

Inspect relevant poses and active contacts before excluding pairs. Keep meaningful inter-finger, hand/object and arm/environment interactions. **Disabling all self-collision is not recommended**: the policy could learn impossible finger interpenetration or arm/hand configurations. Official guidance explicitly identifies dexterous hands as a case where self-collision can be necessary. Benefit is currently unmeasured. [D1, D3]

### C. Replace capsule-object convex hulls with analytic capsules

Six of the sixteen object types are capsules currently represented by convex hulls. An analytic capsule collider is a plausible cheaper narrow-phase representation. Boxes and spheres already use primitive approximations, so there is no analogous broad conversion win for them.

Preserve dimensions, frames, visual appearance, mass/inertia, material randomization, contact offsets and object assignment. Smooth analytic capsules can change contact patches relative to polygonal hulls. The observation point-cloud sampler collects all supported geometry children, without a visual-only filter: retaining the mesh and adding a new capsule can sample both unless observation geometry is deliberately preserved. This is not a blind configuration substitution. Speed benefit is unknown and limited to one object collider per affected environment. [S4, S10]

### D. Consolidate internal fixed attachments, if not already fused

The six internal fixed joints include mount/palm and four biotac-tip attachments. Fusion might reduce articulation bookkeeping, but must preserve combined mass/inertia, transforms, body-name consumers and sensor frames. First establish how the backend already handles these attachments. They are zero-DOF joints, not six additional actuated joints; do not assume a large speedup from the authored count alone. [A1, D2]

### E. Reduce physics frequency: larger fidelity change

Changing **120 Hz physics / decimation2 to 60 Hz / decimation1** preserves 60 Hz policy decisions and halves physics calls per decision. It does not guarantee twice the throughput.

There is an additional controller difference: `RelativeJointPositionAction.apply_actions()` adds the action to current joint positions on **each physics substep**. The baseline re-centers the target twice per policy action; the coarser version does so once. Thus this changes control evolution as well as integration accuracy. Smaller contact/rest-offset gaps should not be combined blindly with a larger timestep; PhysX notes the latter may require larger margins for stability. [S8, D4]

### F. Hand-only/fixed-wrist model: substantial structural change, different task

Removing or restructuring the seven arm DOFs is a plausible major reduction in articulation complexity. Merely sending zero arm actions or increasing PD gains does **not** remove those DOFs from the solve.

A true fixed-wrist task must change reachable goals, resets and reference frames, and normally the action/observation contract. The existing 1745-observation/23-action policy is not a drop-in equivalent. This is reasonable for a deliberately narrower in-hand research task, not as a transparent speed optimization of workspace pose manipulation.

## 5. Knobs that are not free optimizations

- **Visual mesh reduction:** does not reduce the inspected robot collision geometry. Training has no active camera workload; geometry-derived point-cloud observations are separate from camera rendering.
- **Removing contact reporting:** sacrifices required observations/rewards; existing measured reporting overhead is small compared with solver work.
- **Lowering friction or disabling gravity:** changes grasp mechanics and the curriculum, rather than just solver complexity.
- **Reducing object diversity:** changes training coverage, not body count. Any collision-type speed benefit must be measured separately.
- **Increasing GPU buffers:** addresses capacity failures, not work per contact. The inspected EA backend does author these fields; historical statements that all such fields are unread should not be carried forward to this version. [S7]
- **Shrinking contact offsets blindly:** may reduce early contacts but create jitter or missed contacts. The wrapper replaces child collision properties, so the child declaration `contact_offset=0.002` is not sufficient proof of the effective value. Read back the spawned configuration first. [S4, S7, D4]

## 6. Acceptance criteria

Use isolated one-change experiments with pinned source/package/asset identity, object assignment, checkpoint, seeds and physical randomization. Check effective settings after modification, then compare warmed-up interleaved wall-clock measurements separately from GPU kernel sums.

Physical acceptance must cover joint tracking/limits, penetration, slip/retention, force distributions and spikes, abnormal/drop terminations, and held-out pose/contact success across object variants and gravity/noise conditions. Execution without NaNs is not sufficient. Finally compare learning time to a fixed success threshold, not FPS alone.

Keep physics variants in separate experiment workspaces: current PBT comparison stamps include ADR/tolerance but not arbitrary solver/timestep/geometry changes. No framework modification is needed for isolated comparisons. [S11]

**Recommendation:** for the current task, prioritize validated solver24/16, followed by selective pair filtering and capsule-object substitution. Structural simplification has limited low-distortion headroom because the asset is already primitive. A genuinely hand-only environment is a separate scientific/task choice.

## Sources

Repository paths below are relative to `/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab`.

Installed roots:

- `P = /home/andy/env_isaaclab/lib/python3.12/site-packages`
- `IL = P/isaaclab/source/isaaclab/isaaclab`
- `C = source/LEAP_Isaaclab/LEAP_Isaaclab/compat/dexsuite`
- `B = source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt`

### Primary source code

- **S1:** `P/isaaclab_tasks/core/lift/config/kuka_allegro/__init__.py:16–24`; `kuka_allegro_env_cfg.py:55–118` in that directory; `P/isaaclab_tasks/core/lift/lift_env_cfg.py:319–413,424–519,563–578`.
- **S2:** `B/__init__.py:13–30`; `B/dexsuite_reorient_pbt_env_cfg.py:6–10,28–45`; `C/dexsuite_env_cfg.py:493–507`; `C/config/kuka_allegro/dexsuite_kuka_allegro_env_cfg.py:60–62`.
- **S3:** `P/isaaclab_assets/robots/kuka_allegro.py:31–53`.
- **S4:** `C/dexsuite_env_cfg.py:41–117`; `IL/cloner/cloner_strategies.py:29–46`; `IL/sim/spawners/wrappers/wrappers.py:46–89`; `IL/sim/spawners/meshes/meshes.py:483–507`.
- **S5:** `C/dexsuite_env_cfg.py:32–38,104–157,308–323,408–418`.
- **S6:** `C/config/kuka_allegro/dexsuite_kuka_allegro_env_cfg.py:26–93`; `C/mdp/rewards.py:33–113,140–225`.
- **S7:** `IL/sim/spawners/from_files/from_files.py:380–425`; `IL/sim/schemas/schemas.py:221–271,280–378,870–919,1017–1068`; `P/isaaclab_physx/physics/physx_manager_cfg.py:49–114`; `P/isaaclab_physx/physics/physx_manager.py:788–831`.
- **S8:** `IL/envs/manager_based_rl_env.py:229–255`; `IL/envs/mdp/actions/joint_actions.py:203–233`; `P/isaaclab_physx/physics/physx_manager.py:509–525`.
- **S9:** `C/dexsuite_env_cfg.py:230–252`; `IL/envs/mdp/events.py:209–251` (runtime material randomization).
- **S10:** `C/mdp/utils.py:52–56,119–168,193–210`.
- **S11:** `B/environment.py:118–140`; `B/evaluation.py:111–120`.

### Asset provenance

**A1:** Public selected root: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Robots/KukaAllegro/kuka.usd

The following cache files were composed in memory; hashes were checked against public ETags:

| Layer | Cache file under `/home/andy/.cache/ov/cache/hub/hub-files/` | MD5 |
|---|---|---|
| Root | `0AB/00000000000000AB.usd` | `18b63d4fbe42babde76706b3e34a2e3d` |
| Primitive arm | `0B3/00000000000000B3.usd` | `abee4916dc6dd7f6d42f7edec4fcca2d` |
| Primitive hand | `0B1/00000000000000B1.usd` | `053203d9e6d61fbdffaf6c9d09b59c15` |
| Collision geometry | `0AD/00000000000000AD.usd` | `db691c0331ba5833345575e4c09b03e9` |

Selected variants: `/kuka` `Arm=Kuka_Primitive_Collider`, `EE_link=Allegro_Primitive_Collider`. Sole root: `/kuka/root_joint`. Representative compounds: `/kuka/iiwa7_link_3/collisions/{collisions,collisions1}` and `/kuka/ee_link/index_link_3/collisions/{collisions,collisions1}`.

Archived remote config: `logs/rl_games/dexpbt_download_weste_20260926_230552_member03/params/env.yaml:139,164–176,201–203,1095–1112`. These ignored logs and cache files are local evidence, not distributed artifacts.

### Existing measurements

- **M1:** `logs/dexpbt_profile_local/components.json:31–41,63–67,83–102,133–137`; `logs/dexpbt_profile_local/kernel_summary.json:2–24`.
- **M2:** `logs/dexpbt_profile_local/abba_summary.json:198–201` (aggregate **means**, not medians); `:12–40,52–80,109–137,166–194` (eight diagnostic run/ADR combinations); `logs/dexpbt_profile_local/solver16_c.json:9–33`; `scripts/probes/dexpbt_step_profile.py:309–317` (pre-override verification); `docs/research/2026-09-27-dexsuite-pbt-profile-results.md:109–119`.

### Official guidance

- **D1:** [Isaac Sim 6.1 performance optimization handbook](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/reference_material/sim_performance_optimization_handbook.html).
- **D2:** [PhysX 5.4.1 island management](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Simulation.html#island-management) and [articulations](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Articulations.html). General solver semantics; not a fresh readback of this simulator's effective settings.
- **D3:** [Isaac Sim 6.0.1 selective collider-pair tutorial](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/openusd_tuning_tutorials/tutorial_04_collider_pairs.html). General hand-filtering guidance, not this asset's effective pair census.
- **D4:** [PhysX 5.4.1 contact/rest offsets and timestep tradeoffs](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html).
