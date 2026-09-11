# mjlab as an Isaac Lab replacement — feasibility research

**Date:** 2026-09-11
**Claim under test:** "mjlab is easier to set up than Isaac Lab, so we could replace Isaac Lab with it."

## Verdict

**The setup claim is TRUE — emphatically.** The replacement is **feasible but not free**: mjlab
deliberately mirrors Isaac Lab's manager-based API (concepts transfer 1:1), but we would need to
convert assets to MJCF, retune actuator gains, and write a small FlashSAC adapter. Performance is
somewhat below Isaac Lab at large env counts but ample at our scale. Recommendation: adopt
incrementally (zero-install demo → one pilot task), keep Isaac Lab running in parallel.

## What mjlab is

[mujocolab/mjlab](https://github.com/mujocolab/mjlab) — arXiv
[2601.22074](https://arxiv.org/abs/2601.22074) (Zakka, Liao, Yi, Le Lay, Sreenath, Abbeel;
UC Berkeley/Sorbonne, Jan 2026). A GPU-accelerated robot-learning framework that combines
**Isaac Lab's manager-based API** (observation/reward/event/command/curriculum/termination
managers — the same architecture as our manager-based tasks) with **MuJoCo Warp**
([google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp)), a GPU backend
that keeps the MjModel/MjData paradigm with a leading world dimension and captures the sim step
as a CUDA graph. NOT MJX (the JAX backend) — it is PyTorch-native via a zero-copy `TorchArray`
wrapper. Apache-2.0. Backed by an active team (nightly benchmarks on CI, fast issue turnaround),
deployed in Berkeley's graduate robotics course, and adopted by Unitree
([unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)), ROBOTIS
([cyclo_mjlab](https://github.com/ROBOTIS-GIT/cyclo_mjlab)), MyoSuite, and WBC research
([wbc-mjlab](https://github.com/wbc-mjlab/wbc-mjlab)).

Shipped assets: Unitree G1, Unitree Go1, i2rt YAM arm. Shipped tasks: velocity tracking, motion
imitation (BeyondMimic-style), **cube lifting** (`tasks/manipulation/lift_cube_env_cfg.py`), cartpole.
No dexterous hand ships.

## The setup claim, verified

| | Isaac Lab (our lived experience) | mjlab |
|---|---|---|
| Install | conda py3.12 + Isaac Sim 6.0.1 pip wheels (multi-GB), Vulkan ICD env-var prefix, Kit user configs | `pip install mjlab` (Py 3.10+), or `uv add mjlab`; **zero-install** `uvx --from mjlab --refresh demo`; Colab notebook; prebuilt Docker |
| Startup per launch | ~1–2 min (Omniverse Kit) | seconds |
| Headless/GUI | headless≠GUI; needs `visualizer=["kit"]` kwargs; Wayland quirks | native MuJoCo viewer + web (Viser) viewer, offscreen renderer |
| Beta-bug surface | many 3.0.0b2 pitfalls we hit: broken `set_coms/inertias_index`, deadlocking instance_segmentation, unread PhysX buffer cfg fields, double-spawned scenes, deprecated-fields crashes, registration kwarg renames | configs are plain typed dicts (paper explicitly avoids the dataclass `__post_init__` silent-override class of bug we hit), fully typed, 40+ test files, tyro CLI |
| The paper's own words | "requires the Omniverse runtime, which adds installation complexity and startup latency"; PhysX was closed-source until recently | "installable with a single command, requiring minimal dependencies" |

The paper lists "minimal installation friction" as a stated design goal, and the
[installation guide](https://mujocolab.github.io/mjlab/main/source/installation.html) confirms it:
Linux + NVIDIA GPU (CUDA 12.4+) + Python 3.10+ is the entire requirement list. Our
`env_isaaclab` saga (Vulkan prefixes, visualizer kwargs, beta API landmines documented across
AGENTS.md) is exactly the pain mjlab is designed to remove.

## Performance (honest read)

From [discussion #220](https://github.com/mujocolab/mjlab/discussions/220) (Go1 velocity, RTX 4090,
corrected after a maintainer-confirmed broken install made mjlab look 6–7× slower):

| envs | Isaac Lab | mjlab (fixed) |
|---|---|---|
| 4096 | 240k steps/s | 190k steps/s (~25% slower) |
| 8192 | 340k | 160k (drops) |
| 16384 | 420k | 100k (drops further) |

Isaac Lab scales better at very large env counts; mjlab throughput *decreases* past ~4096 envs on
24 GB. At our working point (512–2048 envs on 12 GB) both are far above what we consume, and the
comparison is not apples-to-apples (different PPO sizes, obs groups, collision geometry). Note the
comparison tasks are locomotion; contact-rich dexterous numbers will differ for both.

## Fit for OUR stack

**Transfers well:**
- Manager-based API is deliberately Isaac-Lab-shaped: our manager-based task knowledge and even
  task structure (obs/reward/event/command/curriculum terms) map over.
- **Tactile sensing**: `sensor/contact_sensor.py` with force history; MuJoCo contact pairs/exclusions
  give filtered per-fingertip sensing (our LEAP pulp pattern).
- **Domain randomization**: broad (body/joint/geometry/camera/light/material/tendon) including
  **physically consistent inertial randomization via pseudo-inertia parameterization** and per-world
  model fields with CUDA-graph rebuilds — directly relevant to the (m, COM, I) estimation project
  (our v1 manifest-inertial task would be *easier* here).
- Depth/RGB GPU cameras across all worlds — our DP3-style point-cloud variant (v2) would port;
  obs noise/curriculum patterns match.
- RSL-RL integration is native (`rl/runner.py`), multi-GPU via torchrunx. Our reward-design
  research line (Eureka-on-RSL-RL) plugs in directly.
- Sim2real: hardware deploy is sim-independent (Dynamixel SDK), and MuJoCo position actuators
  (kp/kv/forcerange/armature/frictionloss) map cleanly onto our joint-tuning needs.

**Does not transfer / gaps:**
- **Photorealistic rendering is out of scope** (paper says so explicitly) — the RGBD camera variant
  (v1) loses its purpose; the sanctioned path is privileged-state training + distillation. Depth
  for point clouds is fine.
- **No LEAP hand, no Franka, no KUKA assets.** We must convert our URDFs to MJCF (MuJoCo imports
  URDF; the LEAP community has MuJoCo models floating around, and leaphand.com historically shipped
  MuJoCo resources) and **retune actuator gains/armature/frictionloss** — our Isaac tuning does not
  transfer numerically. This is real, careful work (our Franka+LEAP merge history shows why).
- **FlashSAC has no mjlab env adapter.** flash_rl supports `isaaclab` and `mujoco_playground`;
  mjlab envs are torch-native gym envs with an RSL-RL VecEnv wrapper, so an adapter is a
  ~100–150-line job modeled on our `IsaacLabVectorEnv` (ACTION_BOUNDS dict included) — but it is
  new surface area, and the hold-to-confirm/EMA hooks would need re-testing there.
- Our six Isaac tasks and all their verified mechanics stay Isaac-only. mjlab is explicitly
  single-backend (cross-simulator portability is a stated non-goal).

**Hardware check (RTX 5070 Ti, Blackwell sm_120):** low risk. The one alarming report
([NVIDIA/warp#1405](https://github.com/NVIDIA/warp/issues/1405), sm_120 kernel build failure) was in
Newton's FEM/MPM *examples* — the reporter noted other solvers worked — and the issue is closed.
mujoco_warp's core worked on sm_120 even then, and warp PTX-JITs at runtime. The zero-install demo
below settles it on our machine in ~5 minutes.

## Recommended path (staged)

1. **Today, zero risk, zero install:** `uvx --from mjlab --refresh demo` — validates Blackwell
   compat and gives a feel for the viewer/CLI.
2. **Pilot task (~a few days):** port ONE task — best candidate is the Dexsuite-style lift
   (mjlab ships a cube-lift reference to crib from) or LEAP cube repose — including URDF→MJCF
   conversion of the LEAP hand and gain retuning. Train with RSL-RL (native) to keep FlashSAC out
   of the critical path initially.
3. **If the pilot trains well:** write the flash_rl mjlab adapter and move the tactile/inertial
   randomization research line over, where mjlab's pseudo-inertia DR and per-world model fields are
   a genuine capability upgrade. Keep Isaac Lab for the vision-variant line and anything already
   frozen.

## Sources

- [mujocolab/mjlab](https://github.com/mujocolab/mjlab) (README, source tree)
- [mjlab paper, arXiv 2601.22074](https://arxiv.org/html/2601.22074v2)
- [Installation guide](https://mujocolab.github.io/mjlab/main/source/installation.html)
- [Performance comparison discussion #220](https://github.com/mujocolab/mjlab/discussions/220)
- [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab),
  [cyclo_mjlab](https://github.com/ROBOTIS-GIT/cyclo_mjlab),
  [wbc-mjlab](https://github.com/wbc-mjlab/wbc-mjlab)
- [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp),
  [NVIDIA/warp#1405](https://github.com/NVIDIA/warp/issues/1405)
- [leaphand.com](https://leaphand.com/),
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
