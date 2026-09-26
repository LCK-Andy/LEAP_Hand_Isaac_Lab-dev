# Speeding up Dexsuite Reorient PBT training

Date: 2026-09-26
Task: `Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0`
Deployment: 15 independent PPO/PBT members × 512 environments on RTX 6000D.

## Conclusion

Physics is a plausible bottleneck, but the available evidence does **not** establish that the PhysX solver dominates this run. RL-Games environment-step timing includes Python task logic, observations, contact reads, resets and additional PBT measurements. It is not a pure physics timer. [1, 4, 5]

Recommended order:

1. Obtain a representative timing breakdown and check CPU thread contention across the 15 processes.
2. Optimize measured environment-pipeline overhead without changing observations, rewards or physical conditions.
3. If the solver dominates, test lower articulation solver iterations in an isolated fidelity study.
4. Measure actual PhysX buffer demand and recover memory before increasing environments per member.
5. Treat kit-less backends and a shared-simulator population as separate parity/architecture projects, not automatic speed switches.

No training settings, simulator packages or running jobs were changed for this assessment. The only deliverable written is this report. No speedup benchmark was run and no numerical speedup is promised.

## Evidence and limitations

The local Isaac Lab 3.0.0rc1 source and the downloaded production member-03 configuration were inspected. Downloaded checkpoint/configuration hashes were checked against the transfer manifest. These saved settings are stronger evidence for this run than historical environment notes. [1, 2, 3]

Two attempts to read the remote run failed at TCP connection, with 15-second and 30-second timeouts, around 2026-09-26 15:30 UTC. This does **not** indicate whether training has stopped. Current rollout/PPO timings, CPU pressure and GPU kernel overlap could not be measured.

Earlier launch-time observations from this conversation, **not fresh measurements**:

| Configuration | GPU memory | Container RAM |
|---|---:|---:|
| 15 × 512 production | 80,087 / 85,651 MiB | 78.03 / 92 GiB |
| Eight-member benchmark | 42,762 MiB | 42.52 GiB |

Launch records on the remote machine:

- `/root/autodl-tmp/dexpbt_reorient_20260926_weste_r1/prod15/verification_final.json`
- `/root/autodl-tmp/dexpbt_reorient_20260926_weste_r1/prod15/launch_record.json`
- `/root/autodl-tmp/dexpbt_reorient_20260926_weste_r1/prod15/storage_migration.json`

The 15-member launch had 5,564 MiB GPU headroom (6.50%) and 13.97 GiB container-RAM headroom. Memory scales almost linearly with member count in those observations, but this does not distinguish Kit, physics buffers, CUDA contexts and PPO allocations. The older 4090 run's 86–87% rollout share is **not** a measurement of this deployment.

## Verified production configuration

| Component | Saved setting |
|---|---|
| Physics/control | `dt=1/120`, decimation 2: 120 Hz physics, **60 Hz control** |
| Episode | 6 seconds; 360 control steps |
| Robot solver | TGS, **32 position / 1 velocity iterations**, self-collision enabled |
| Object solver | **16 position / 0 velocity iterations** |
| Scene | 512 environments/member; physics replication, collision filtering and lazy sensor updates enabled |
| Rendering | No active visualizers; both cameras null; zero reset rerenders |
| Contact sensors | Four object-filtered fingertip sensors; no contact-point/friction/air-time/pose tracking or sensor history |
| Observations | 1,745 inputs; five-step history; includes 64 geometry-derived 3D object points |
| Policy | Separate actor/critic MLPs `[512, 256, 128]`; 23 actions |
| PPO | Horizon 32, minibatch 1,024, five mini-epochs; mixed precision and model compilation disabled |
| PBT | 15 members; publication every 1M member frames; 5M warm-up; 3M post-exploit adaptation |

Sources: [1, 2, 4, 5]. The source's old “50 Hz” comment is inconsistent with its actual timestep and decimation.

### Rendering is not an obvious remaining optimization

The point cloud is not rendered from a camera. Surface points are sampled from object geometry during initialization, cached and transformed using live poses. Its per-step tensor/history work is real, but removing the channels changes the policy's information about object position and shape. [4]

The installed headless experience omits RTX-renderer dependencies, disables Replicator, and disables Fabric transformation, velocity, force and joint-state output. The saved production configuration has no active visualizer or camera. The remote uses a curated experience, so exact extension behavior remains a remote-verification item; nevertheless, there is no evidence for an expensive per-step rendered-camera workload. [1, 6]

## 1. Profile the entire population, then budget CPU/GPU scheduling

For a representative interval, report both **aggregate population frame delta / wall time** and per-member progress. Printed training FPS alone can omit logging, checkpoint and process-restart overhead. Split time into:

- PhysX simulation and result synchronization;
- contact/tensor reads;
- observations, rewards, resets and PBT measurement;
- policy inference;
- PPO updates;
- publication and member replacement.

Use lightweight existing logs first; use a short representative Nsight Systems trace only as a separately authorized profiling experiment. CUDA synchronization can move apparent cost between host timers, so do not equate a high `env.step` share with solver time. [5]

### CPU thread budgets

There are 15 independent simulators, not one distributed PPO process group. Existing OMP/MKL thread limits do not automatically control every Kit/TBB/physics thread pool. Check the actual launch overrides and effective container CPU quota before choosing limits.

NVIDIA documents these controls: [7]

- `/plugins/carb.tasking.plugin/threadCount`
- `/persistent/physics/numThreads`
- `/plugins/omni.tbb.globalcontrol/maxThreadCount`

Fewer threads per process can help if the processes oversubscribe their available CPU allocation; it can also hurt if set too low. This is a benchmark candidate, not a known current defect.

### CUDA MPS

MPS may allow kernels from different processes to overlap when each process submits too little work to saturate the GPU. It does not merge simulator scenes, Python execution or policy state. It will not fix a CPU bottleneck or a GPU already saturated with useful computation. A 99–100% utilization reading measures busy time, not useful occupancy. Consider MPS only if traces show a relevant concurrency opportunity, and account for its resource/fault-isolation implications. [8]

The current population was chosen to maximize safely fitting members. **Maximum member count, maximum aggregate FPS and minimum time to learning success are different objectives.** Retaining all 15 is the default recommendation here. A comparison against fewer, larger members could be informative, but it changes population diversity and must be evaluated as a learning-system trade-off—not silently substituted.

## 2. Optimize concrete environment-step work without changing physics

The PBT environment adapter performs several operations every control step: additional pose/contact measurements, conversion of a CUDA ADR value to a Python float, traversal/deep-copy of physical randomization settings and JSON serialization of a condition stamp. Separate reward functions also repeatedly reduce the same four contact-force vectors. These are identifiable optimization candidates, but their runtime share has not been profiled. [9]

Candidate changes:

- Recompute physical-condition metadata only when actual ADR parameters, tolerances or restored state change.
- Share contact-force magnitudes and pose intermediates within the same post-physics phase.
- Fuse measured tensor-heavy reward/observation operations where useful.

Correctness constraints are essential: retain completed-episode statistics, physical-condition comparisons, random-number ordering, reset timing and the distinction between pre-reset reward state and post-reset observations. Never cache forces across physics updates or resets.

The installed sensors already cache reads within a data generation and CUDA-graph their postprocessing. Their underlying PhysX tensor reads remain eager; blindly capturing the entire environment step in an outer CUDA graph can produce stale sensor data and is explicitly guarded against. [10]

### Do not assume a Direct port fixes the solver

A Direct environment gives more control over fused/JIT/Warp task computation. Both Manager-based and Direct loops still call the same physics backend. A complete Direct rewrite therefore does not automatically accelerate PhysX or eliminate process memory. Optimize a measured hot path first. [11]

There is an additional parity caveat: the saved environment resolves some contact rewards through `tasks/kuka_leap_dexsuite/force_utils.py`, not just the nominal compatibility module. Its contact-count helper differs in normalization from the original compatibility helper. Any refactor must preserve the **actual saved/runtime functions**; correcting reward semantics would be a separate change. [9]

## 3. Direct physics optimization: test solver iterations cautiously

The clearest exposed solver-cost lever is the robot's **32 position iterations**. If a profile shows solver-heavy execution, compare 32 versus 16, then 8 only if the 16-iteration configuration passes physical tests. Keep timestep, decimation, gains, velocity iterations, collision settings and observations fixed during that comparison. [1, 12]

This changes numerical fidelity; it is not a guaranteed semantics-preserving optimization. Reducing the object's 16 iterations alone may not help while the articulation still requests 32.

Acceptance criteria:

- valid object-filtered forces and stable grasp/contact behavior;
- no increased penetration, explosive velocities or dropped contact patches;
- matching joint tracking and reset stability;
- tests at later ADR/full-gravity conditions, not just easy zero-gravity rollouts;
- held-out task success versus both wall time and sample count.

Do not remove self-collisions or fingertip geometry simply to improve FPS: those changes can make an easier, different manipulation problem. Current robot collider counts and broadphase/narrowphase costs were not measured, so geometry simplification is a conditional investigation, not an established opportunity. [7, 12]

## 4. Recover memory before increasing environments/member

The saved 512-environment process requests large PhysX capacities: [1]

| Capacity | Requested |
|---|---:|
| Rigid contacts | 8,388,608 |
| Rigid patches | 655,360 |
| Found/lost pairs | 67,108,864 |
| Found/lost aggregate pairs | 33,554,432 |

These are requested capacities, not measured active-contact counts or proven actual allocations. Patch capacity is four times the installed default, and found/lost-pair capacity is 32 times that default. The current rc1 source authors valid GPU-capacity USD attributes; historical beta reports that these fields were never consumed must not be generalized to rc1. Installed PhysX exposes scene statistics and high-water demand counters for contacts, patches, pairs and memory categories. [12]

Measure effective allocation and stressful contact/reset demand, then reduce only excessive reservations while retaining a safety margin. Undersizing can drop contacts and invalidate force-gated rewards. Memory savings are not automatically an FPS gain, but could permit larger per-member batches without deleting observations or sensors.

Do not double all 15 members immediately: launch-time GPU headroom was only 6.5%. Also, the PBT checkpoint's per-environment ADR state requires an exact environment-count match; changing 512 to 768/1,024 requires a deliberate state-migration or new-run policy. It is not a transparent full resume. [9]

### PPO consequences of batching

Current rollout size per member is `512 × 32 = 16,384` transitions. At minibatch 1,024 and five mini-epochs, there are **80 optimizer minibatches per rollout**. [2, 5]

- More environments at unchanged horizon/minibatch can amortize fixed simulator-call costs, but proportionally increases PPO work and memory per rollout.
- Each transition is still reused five times; optimizer steps/frame stay constant, but policy updates occur after larger collections.
- Increasing minibatch size instead reduces optimizer steps/frame and changes optimization behavior.
- Longer horizons do not amortize each physics step; they change trajectory length, GAE and update cadence.

Model compilation or mixed precision is worth testing if inference/PPO is substantial. Neither accelerates the PhysX solver. Include compilation warm-up, extra memory and repeated PBT replacement costs before enabling compilation across the population. [2, 5]

## 5. Kit-less backends and shared simulation: larger projects

### Newton/MuJoCo-Warp

The compatibility task has a `newton_mjwarp` preset, and current Newton can graph simulation and combine decimation when actuators support it. However, the current PBT entrypoint launches AppLauncher before resolving task configuration. Changing a preset alone therefore does not make this launcher Kit-less. [13]

Potential benefits are less Kit/process overhead and a different GPU execution pipeline, **not a proven Dexsuite speedup**. Current Newton supports filtered contact reporting; the old blanket claim that its sensors always return zero is no longer appropriate. [13, 14]

Concrete parity issues:

- The task uses `force_matrix_w`: current PhysX exposes normal-only forces through it, whereas Newton exposes normal-plus-friction forces. Identical task source can therefore change tactile observations and threshold rewards.
- PhysX gravity randomization is scene-wide; Newton's implementation can update selected worlds. Member-wide ADR behavior must be reproduced deliberately.
- Collider import, joint drives/limits, randomization, reset state, sensor filtering and PBT restore all need validation.

### Kit-less OVPhysX

Dexsuite's normal-force-only sensor requirements are less restrictive than Stage A's full-force/contact-point requirements. Installed OVPhysX exposes a compatible basic filtered-force interface, but this task's complete assets, pose readback, ADR and learning behavior have not been validated on it. Launcher/process isolation is required. Do not infer task compatibility from the existence of a sensor API. [15]

### One simulator for many policies

A vectorized population could remove duplicated overhead, but requires separate policies, optimizers, normalizers, episode statistics and curricula inside a shared runner. Current independent PhysX members also have independent **scene-wide gravity**, so one shared scene cannot express all member conditions unchanged. Observation-noise parameters likewise need member-specific handling. This is an architectural redesign, not a launch option. [9, 13]

## Speed versus learning progress

For component fraction `f` made `k` times faster, total speedup is:

`S = 1 / ((1 - f) + f / k)`

For example, **if** physics is 80% of runtime, doubling physics throughput yields only **1.67× overall throughput**, not 2×. This is illustrative arithmetic, not a measured fraction for this run.

The downloaded member-03 publication at 32,014,336 frames / epoch 1,954 reports **zero successes in 2,589 completed episodes**, applied zero gravity/noise, and about **3.55% good-contact occupancy**. This is a single saved member/window, not current population performance. It does show why throughput alone is insufficient: compare held-out success and contact behavior against wall time and frames. [3]

## Sources

Paths and line references are from the files inspected on 2026-09-26.

1. [Saved production environment](../../logs/rl_games/dexpbt_download_weste_20260926_230552_member03/params/env.yaml): lines 1–54 physics/rendering, 108–175 robot solver, 244–355 drives/object solver, 1288–1306 cameras/sensors, 2058–2230 observations, 3272–3291 resolved rewards, 3463 episode length.
2. [Saved production PPO/PBT configuration](../../logs/rl_games/dexpbt_download_weste_20260926_230552_member03/params/agent.yaml): lines 1–121.
3. [Publication metrics](../../logs/rl_games/dexpbt_download_weste_20260926_230552_member03/nn/000000000032.yaml): lines 1–45; [transfer manifest](../../logs/rl_games/dexpbt_download_weste_20260926_230552_member03/transfer_manifest.json).
4. [Observation implementation](../../source/LEAP_Isaaclab/LEAP_Isaaclab/compat/dexsuite/mdp/observations.py):61–200; [geometry sampler](../../source/LEAP_Isaaclab/LEAP_Isaaclab/compat/dexsuite/mdp/utils.py):29–50,111–176. Installed `isaaclab/.../managers/observation_manager.py`:385–436.
5. Installed `/home/andy/env_isaaclab/lib/python3.12/site-packages/rl_games/common/a2c_common.py`:49–63,249–280,780–841,1229–1295,1381–1440; `rl_games/torch_runner.py`:184–235. These define batch sizes, host-timer FPS and model compilation.
6. Installed `/home/andy/env_isaaclab/lib/python3.12/site-packages/isaaclab/apps/isaaclab.python.headless.kit`:26–42,58–59,85–89,161–174; `isaaclab_physx/physics/physx_manager.py`:509–527; `isaaclab/source/isaaclab/isaaclab/markers/visualization_markers.py`:273–301.
7. NVIDIA, [Isaac Sim Performance Optimization Handbook](https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html). CPU pools, physics, Fabric and collision recommendations; generic rendering advice must be checked against the already-headless configuration.
8. NVIDIA, [When to Use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html) and [MPS Architecture](https://docs.nvidia.com/deploy/mps/architecture.html).
9. [PBT environment adapter](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt/environment.py):53–61,97–139,184–230; [physical-condition evaluation](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt/evaluation.py):184–207,218–246; [ADR implementation](../../source/LEAP_Isaaclab/LEAP_Isaaclab/compat/dexsuite/adr_curriculum.py):17–122; [resolved force helpers](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/kuka_leap_dexsuite/force_utils.py):26–65,94–106; [original contact helpers](../../source/LEAP_Isaaclab/LEAP_Isaaclab/compat/dexsuite/mdp/rewards.py):66–113.
10. Installed `isaaclab_physx/sensors/contact_sensor/contact_sensor.py`:421–470,490–508; `isaaclab/source/isaaclab/isaaclab/scene/interactive_scene.py`:519–545; `isaaclab/source/isaaclab/isaaclab/sensors/sensor_base.py`:203–223,407–419. Paths are under `/home/andy/env_isaaclab/lib/python3.12/site-packages/`.
11. Isaac Lab, [Task Workflows](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html); installed manager-based loop `isaaclab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`:207–308 and direct loop `direct_rl_env.py`:399–488.
12. Installed `isaaclab_physx/physics/physx_manager_cfg.py`:27–35,77–115,188–218; `physx_manager.py`:807–831. Installed PhysX schema `isaacsim/extscache/omni.usd.schema.physx-110.3.2+110.3.0.lx64.r.cp312.u7f4/plugins/PhysxSchema/resources/schema.usda`:327–385; statistics interface `isaacsim/extscache/omni.physx-110.3.2+110.3.0.lx64.r.cp312.u7f4/omni/physx/bindings/_physx.pyi`:1522–1538,2694–2705,2708–2801,2820–2829. See also [PhysX Rigid Bodies and Articulations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html).
13. [PBT entrypoint](../../scripts/rl_games/train_dexpbt_member.py):49–68,109–159,207–237; [Newton preset](../../source/LEAP_Isaaclab/LEAP_Isaaclab/compat/dexsuite/dexsuite_env_cfg.py):425–450,493–507. Installed `isaaclab_newton/physics/newton_manager.py`:3480–3567; `isaaclab/source/isaaclab/isaaclab/app/sim_launcher.py`:237–243,381–399; `isaaclab/source/isaaclab/isaaclab/envs/mdp/events.py`:1386–1440.
14. Isaac Lab, [Newton Integration](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html) and [Limitations](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/limitations-and-known-bugs.html). Installed force definitions: `isaaclab/source/isaaclab/isaaclab/sensors/contact_sensor/base_contact_sensor_data.py`:233–253 and `isaaclab_newton/sensors/contact_sensor/contact_sensor_data.py`:111–125. Public experimental docs are contextual; installed rc1 source is authoritative for these version-specific observations.
15. Installed `isaaclab_ov/sensors/contact_sensor/contact_sensor.py`:42–88; `isaaclab/source/isaaclab/isaaclab/app/sim_launcher.py`:409–429, under `/home/andy/env_isaaclab/lib/python3.12/site-packages/`.
