# AGENTS.md — LEAP Hand Isaac Lab

## Environment & setup

- Current local EA validation (2026-09-25): **Isaac Lab 3.0.0rc1 + Isaac Sim 6.1**, Python **3.12**, torch **2.11/cu128**, Warp **1.16**, Newton **1.5.2**, RSL-RL **5.4.1**, NumPy **2.3.1**, `/home/andy/env_isaaclab`. Legacy 2.1/Sim 4.5 instructions below are historical, not the validated environment.
- EA migration scope: LEAP direct repose/contact PhysX smoke, tiny contact/Dexsuite/Allegro RL-Games training and checkpoint checks. Other tasks/backends are not blanket-certified. Validation details and provenance follow below.
- `LEAP_Isaaclab.compat` preserves the 3.0.0b2.post1 task definitions because EA's relocated native tasks also changed MDP semantics. Do not replace these with new EA bases without a deliberate policy/MDP migration. No installed-package patches or global namespace aliases are used.
- EA spawn expressions must use segment-safe `env_[^/]+`, not `env_.*`: the latter can spawn objects under every robot descendant. Direct LEAP cloning uses the explicit EA clone-plan API. EA no longer adds `--headless` to argparse; scoped RL-Games entrypoints retain it explicitly.
- Tactile force correction: direct LEAP uses the object-filtered force matrix, not `net_forces_w` (which includes self/ground contacts). Historical checkpoints may see different tactile observations/rewards; missing filtered data is an error, not a silent zero.
- The repo root is NOT the Python package root. The package lives at `source/LEAP_Isaaclab/`.
- Install with: `python -m pip install -e source/LEAP_Isaaclab`
- All Isaac Lab / Isaac Sim / torch / numpy deps must be pre-installed in the environment. `install_requires` includes `psutil` and the public `async-pbt` v0.1.0 Git dependency; do not install its trainer extra over the validated Isaac stack.
- PBT engine implementation is supplied by `async-pbt`; local `tasks/dexsuite_reorient_pbt/pbt/` modules are compatibility adapters. Preserve the LEAP `dexpbt` default, legacy workspace paths, raw-checkpoint precedence, and pre-AppLauncher import safety. Task-specific warm-start/prior validation remains local. See `docs/async-pbt-integration.md`.

## EA migration validation and provenance (2026-09-25)

- Preserved upstream source: `isaaclab-3.0.0b2.post1-py3-none-any.whl` from
  `/home/andy/isaaclab-upgrade-backup-20260925/`; read as an archive only, never
  executed the relocated backup environment. Original BSD-3-Clause headers remain.
  `compat/inhand_direct` comes from `isaaclab_tasks/direct/inhand_manipulation`,
  `compat/inhand` from `manager_based/manipulation/inhand`, `compat/dexsuite` from
  `manager_based/manipulation/dexsuite`, and `compat/allegro` from `direct/allegro_hand`.
  Registration-only initializers were replaced with inert package initializers;
  the original Allegro id is registered locally only when absent. EA native ids remain intact.
- Local snapshot adaptations: EA backend import `isaaclab_ov`, segment-safe spawn
  paths, explicit cloning, `cloner.query.iter_sources`, and Dexsuite's missing
  `compute_final_obs=False` (EA checks it on terminal transitions). The reward,
  reset, observation ordering, curriculum and PBT algorithms remain legacy.
- Passed local PhysX checks: direct repose (4 envs, 136-d obs, 60 steps), direct
  contact (4 envs, 184-d obs, cube half extent 0.03 m, 60 steps plus actual
  four-pulp cube contact and native three-step hold/resampling). Object-filtered
  measured force maxima were approximately 9.5/7.6/15.8/21.4 N. The smoke uses a
  seeded opposing-thumb fixture; all-ones curl previously counted self-contact.
- Tiny training: contact PBT two epochs/128 frames plus full checkpoint resume
  through epoch 3/192 frames; Dexsuite PBT two epochs/128 frames; legacy Allegro
  two PPO epochs plus real deterministic checkpoint evaluation (one completed
  episode). Dexsuite fixed checkpoint evaluation also completed. These are
  execution/restore checks, not learning-quality or population-exploitation evidence.
- Simulator-free regressions include installed native-import layout, segment-safe
  spawning and filtered tactile force selection. Historical checkpoints trained
  with unfiltered net forces are loadable but **not behavior-identical** after the
  documented tactile correction. Missing enabled-sensor partner data raises.
- Not validated here: Newton/ovphysx, camera/point-cloud variants, objects variants,
  other robots, hardware, GUI playback, or large populations. Several unrelated
  direct tasks still use removed `clone_environments`; migrate them separately
  before claiming EA support. Kit-less launcher imports now point to
  `isaaclab.app.sim_launcher`, but their end-to-end behavior was not tested.

## Stage A persistent grasp (2026-09-25)

- New `Isaac-Grasp-Cube-LEAP-Direct-v0` / `-Direct-Play-v0` tasks live in
  `tasks/leap_hand_grasp/`; existing contact-point task IDs/MDPs are unchanged.
  See `docs/stage-a-grasp.md` for commands, criteria and validation limits.
- Full object-filtered **normal + friction** force vectors, force-history checks,
  measured pulp-region gate, non-pulp-support exclusion, bounded joint-target
  increments, persistent 2 s hold and finite 10 s episodes. Observation is 124-d,
  including remaining time; no random contact goals or resampling on success.
- Pregrasp is a measured support-free fixture, not a learned policy. Loaded
  DexCube is **0.072 m** per side at scale 1.2 (new task half extent 0.036 m);
  mass explicitly pinned to 0.216 kg. Earlier contact smoke's permissive bbox
  check did not establish the old 0.03 m half extent; old task remains unchanged.
- PhysX passed 4-env/330-step fixture holding with ~99.7% valid occupancy and
  live shear forces; tiny PPO/checkpoint checks are not learning evidence.
  Use `train_dexpbt_member.py` for exploration bounds; `grasp_hold` PBT selection
  is opt-in, uses support-free retention/valid occupancy and physical condition stamps.
- Newton current full-force API works after a new-task-only collision-schema
  adaptation (session layer, no asset/package patches). Its 330-step shared-fixture
  probe retained the cube but failed four-pulp validity with non-pulp support;
  training is guarded, explicit `leap_grasp_newton_smoke.py` opts into experimental
  validation. OVPhysX's installed sensor lacks friction/point tracking and is refused.
  Do not generalize historical Newton zero-force reports: this probe initially
  had **zero imported hand colliders**, not a missing full-force reporting API.

## Stage B grasp-motion tracking (2026-09-26)

- New `Isaac-Grasp-Motion-Cube-LEAP-Direct-v0` / `-Direct-Play-v0` tasks in
  `tasks/leap_hand_grasp_motion/`; see `docs/stage-b-grasp-motion.md`. Stage A
  remains unchanged, including its running remote source snapshot.
- Fixed-hand joint actions track a smooth commanded object pose: initial0.1deg
  XYZ at0.2/0.3/0.4Hz, zero translation,1s settle+2s quintic ramp. No object
  writes outside resets or external object forces. Full Stage A force/support
  validity gates the tracking reward;148obs = exact124prefix+24 command channels.
- Pose-difference/EMA velocities, not biased raw solver velocities, drive
  tracking. Filter updates once per reward step; reset/recovery handling excludes
  invalid intervals. Angular Gram diagnostics are NOT a Fisher matrix or estimator.
- `train_dexpbt_member.py --warm-start <StageA.pth>` strictly initializes only
  model/normalizers (124->148 zero padding), fresh optimizer/frame/best; colocated
  Stage A agent/env YAML required. `--checkpoint` is full Stage B resume instead.
  `grasp_motion` PBT selection ranks safe tracking, not Stage A contact occupancy.
- Local PhysX smoke passed:4env330steps,99.7% holding but0% stationary tracking.
  Warm-start21epoch/1344frame and resume22/1408 execution checks passed, NOT
  acquisition. Separate scripted mechanical probe preserved~97.3% contacts but
  achieved X-dominant motion only;1deg/1mm rejected by physical tests. Full
  multi-axis tracking and inertial-identification accuracy remain unproven.

## Commands

| Purpose | Command |
|---|---|
| Train (rl_games) | `python scripts/rl_games/train.py --task Isaac-Reorient-Cube-Leap --headless` |
| Play checkpoint | `python scripts/rl_games/play.py --task Isaac-Reorient-Cube-Leap --num_envs 1 [--checkpoint PATH]` |
| Train (rsl_rl) | `python scripts/rsl_rl/train.py --task Isaac-Reorient-Cube-Leap --headless` |
| Play (rsl_rl) | `python scripts/rsl_rl/play.py --task Isaac-Reorient-Cube-Leap --num_envs 1` |
| Train (rsl_rl, kit-less Newton) | `python scripts/rsl_rl/train_kitless.py --task <ID> --headless presets=newton_mjwarp` |
| Hardware deploy (Python SDK) | `cd source/LEAP_Isaaclab/LEAP_Isaaclab/deployment_scripts && python reorient_z.py` |
| Hardware deploy (ROS2) | `cd source/LEAP_Isaaclab/LEAP_Isaaclab/deployment_scripts && python reorient_z_ros.py` |

Key CLI flags (shared across train/play scripts):
- `--task` — Gym environment id (always `Isaac-Reorient-Cube-Leap`)
- `--num_envs` — overrides the default 8192
- `--headless` — no GUI (AppLauncher arg)
- `--checkpoint` — path to a model checkpoint
- `--video` / `--video_length` / `--video_interval` — recording

Scripts use `AppLauncher.add_app_launcher_args(parser)` which adds `--headless`, `--device`, and other Omniverse flags.

## Architecture

```
scripts/rl_games/          ← training & playback entrypoints (primary RL framework)
scripts/rsl_rl/            ← alternative RSL-RL entrypoints
source/LEAP_Isaaclab/
  config/extension.toml    ← Isaac Lab extension metadata (version, deps, module name)
  setup.py                 ← pip packaging, reads version/title from extension.toml
  LEAP_Isaaclab/
    __init__.py             ← wildcard-imports tasks + optional UI extension
    assets/leap.py          ← LEAP_HAND_CFG (articulation: 16 DOF, relative position control)
    tasks/
      __init__.py           ← calls import_packages() to auto-register all task sub-packages
      leap_hand_reorient/
        __init__.py         ← `gym.register(id="Isaac-Reorient-Cube-Leap", ...)`
        leap_hand_env_cfg.py ← LeapHandEnvCfg (scheduling, rewards, ADR, scene)
        reorientation_env.py ← ReorientationEnv(DirectRLEnv) — RL loop, rewards, sim2real indices
        agents/
          rl_games_ppo_cfg.yaml  ← rl_games PPO config (GRU 256, MLP [512,256,128], lr 5e-4)
          rsl_rl_ppo_cfg.py      ← RSL-RL PPO config class
    deployment_scripts/
      load_model.py         ← RLGamesPolicy wrapper
      reorient_z.py         ← Python SDK hardware controller (Dynamixel via USB)
      reorient_z_ros.py     ← ROS2 hardware controller
    utils/
      adr.py / adr_utils.py ← automatic domain randomization scheduling
      obs_utils.py          ← observation history buffer & latency emulation
      leap_hand_utils/      ← Dynamixel client + joint coordinate conversion
```

## Task details

- **Observation**: 96-d (16 joint positions + 16 targets, 3-timestep history).
- **Action**: 16-d joint position deltas (relative control), smoothed with a moving average (1/24).
- **Sim2real joint index mapping**: `[4, 0, 8, 12, 6, 2, 10, 14, 7, 3, 11, 15, 1, 5, 9, 13]`
- **Control**: 30 Hz effective (120 Hz physics, 4-step decimation).
- **Rewards**: position distance (-10), rotation alignment (+1.0), action penalty (-0.0002), goal bonus (+250), fall penalty (-10).
- **ADR**: 25 increments. Degenerate initial ranges expand linearly. Applied per `consecutive_successes / z_rotation_steps / mean_episode_length`.

## Code conventions

- **Line length**: 120 (Black, flake8, isort, pyright all agree).
- **Docstrings**: Google-style (configured in `.flake8`).
- **Pre-commit hooks**: `pre-commit run --all-files` (20 hooks across 7 repos: black, flake8, isort, pyupgrade, codespell, pygrep-hooks, generic checks).
- **isort**: custom section `ISAACLABPARTY` between THIRDPARTY and FIRSTPARTY for `isaaclab*` imports. Scientific libs (numpy, torch, gymnasium, etc.) are treated as `extra_standard_library`.
- **pyright**: `typeCheckingMode = "basic"` (not strict). Many checks disabled (`reportMissingImports`, `reportGeneralTypeIssues`). Not enforced by pre-commit.
- **Wildcard imports**: `from .tasks import *` and `import_packages()` are the Isaac Lab registration pattern — never remove them.
- **`__init__.py` files**: F401 suppressed globally in `.flake8` (`per-file-ignores=*/__init__.py:F401`).
- Some config files have `# noqa: F401` / `# noqa: F811` on imports kept only for reference — do not remove these.
- Git LFS tracks binary/asset files: `*.usd`, `*.pt`, `*.jit`, image/video formats.
- `.vscode/` in git only contains template files; run `.vscode/tools/setup_vscode.py` to generate settings.
- Ignored tasks (not yet published): `leap_crawler/`, `leap_three_axis_reorient/`.

## Gotchas

- **The Gym task id is `Isaac-Reorient-Cube-Leap`** — note "Reorient" not "Reorientation".
- `import LEAP_Isaaclab.tasks` is required **after** `import isaaclab_tasks` in entrypoint scripts so the custom environment gets registered before `hydra_task_config` resolves it.
- The package name in the import is `LEAP_Isaaclab` (capital L, capital I), matching the directory name exactly.
- The extension module name in `config/extension.toml` (`[[python.module]] name = "LEAP_Isaaclab"`) must match the import name.
- `scripts/utils/list_envs.py` lists gym task ids containing "leap" (case-insensitive), so it covers both camelCase `Leap` tasks and all-caps `LEAP` Repose ids; missing from the table ≠ unregistered.
- `scripts/rsl_rl/train_kitless.py` resolves the env cfg BEFORE launching and hands it to
  `isaaclab_tasks.utils.sim_launcher.launch_simulation`, which starts Kit ONLY when the task
  needs it (PhysX physics or Kit-rendered cameras); a kit-less physics cfg (e.g. the official
  tasks' `newton_mjwarp` NewtonCfg preset) runs with NO Isaac Sim at all -- measured 13 s
  wall for launch+5 iters+checkpoint on `Isaac-Humanoid-v0` vs ~2 min with Kit. Select
  backend presets via a hydra-style `presets=newton_mjwarp` argument (the classic tasks ship
  matching physics+observation `newton_mjwarp` PresetCfg alternatives). The script also strips
  the deprecated rsl_rl<4 model fields from the serialized agent cfg generically (the
  rsl_rl 5.0.1 `MLPModel` crash that otherwise needs per-cfg `__post_init__` fixes).
- No CI workflows exist in this repo.

## Additional tasks: Franka + LEAP (welded)

A single 23-DOF articulation (Franka 7-DOF + LEAP 16-DOF) is available as
gym task `Isaac-Franka-Leap` (DirectRLEnv, `tasks/franka_leap/`).

| Purpose | Command |
|---|---|
| Rebuild merged URDF | `python scripts/assets/build_franka_leap_urdf.py` |
| Rebuild pre-built robot USD | `python scripts/assets/build_franka_leap_usd.py` (required after URDF changes) |
| Headless smoke test | `python scripts/smoke/franka_leap_smoke.py` |
| Render demo video | `python scripts/demos/franka_leap_demo.py` |

Notes:
- The merged URDF lives at `source/LEAP_Isaaclab/LEAP_Isaaclab/assets/urdf/franka_leap.urdf`
  with bundled meshes (`urdf/franka/meshes/`, `urdf/leap/meshes/`).
- The runtime asset is the PRE-BUILT `assets/franka_leap/franka_leap.usd`
  (`UsdFileCfg`), NOT a live URDF conversion: the Isaac Lab 3.0.0b2 URDF
  import pipeline emits a floating-base articulation nondeterministically
  (root API on a parent Xform) plus a world `root_joint` whose clones are
  broken by the env cloner -- the robot free-fell and exploded. The build
  script converts once with the classic converter (asset-transformer and
  multi-physics conversion disabled, fixed joints merged so visual prims
  stay physics-driven) and text-patches the file to the fixed-base encoding
  used by the working `leap_hand_v1_right` USD: a single ArticulationRootAPI
  on `panda_link0` and no importer world joint (a live Isaac Sim stage
  listener re-applies removed root APIs, so USD-API surgery at runtime is
  unreliable -- edit the file text). `fix_root_link=True` in the cfg then
  creates the cloner-safe world joint at spawn, exactly like the LEAP hand.
  `NewtonArticulationRootAPI` inherits from `UsdPhysics.ArticulationRootAPI`
  -- leaving it on the old root Scope counts as a second articulation root.
- Actions are 23-d relative joint position deltas (arm first, then hand).
- The importer fuses the flange (panda_link8), LEAP base, and palm into the
  wrist; the weld is still physically rigid (smoke: fingertip-flange distance
  constant to <0.5 mm).
- `panda_link7` is the wrist body used for world-space flange reporting.
- Arm gains are strict position control (400/80): the official 80/4 assume the
  official Franka USD's real masses; with this URDF's mesh-density-derived
  masses they leave ~0.2 rad gravity sag. Self-collisions are OFF -- the
  converter's coarse convex hulls (palm/wrist region) overlap the straight
  fingers at the open pose (~0.3 rad contact deflection); revisit with the
  fine LEAP-team collision meshes.
- `scene.replicate_physics=True` is REQUIRED: with `False`, this Isaac Lab
  3.0.0b2 build never binds simulated poses to the rendered prims (the robot
  renders frozen at spawn / looks exploded mid-motion) even though physics is
  correct.

## Additional tasks: LEAP in-hand cube reorientation (manager-based)

`Isaac-Repose-Cube-LEAP-v0` (`Isaac-Repose-Cube-LEAP-Play-v0` for eval) is the official
manager-based `Isaac-Repose-Cube-Allegro-v0` in-hand task with the Allegro hand swapped
for the LEAP hand (`tasks/leap_hand_repose/`). EMA joint-position actions, 72-d proprio +
object obs, randomization, and terminations are inherited unchanged from
`isaaclab_tasks.manager_based.manipulation.inhand`; the rewards extend the two official
task terms (success bonus, inverse-orientation) with the five shaping terms of the
7-term reference in-hand reorientation reward (`mdp/rewards.py`): fingertip-object
distance (-1.0), joint energy |q̇|ᵀ|τ| (-1e-3, replaces `joint_vel_l2`), object-away
indicator (-1.0 at 0.25 m, below the 0.3 m termination), object-drop indicator
(weight 0 — no table; dropping already terminates), and fingertip over-curl below the
palm-base plane (-1.0 per fingertip; the LEAP palm is vertical, so the paper's
"fingertip above height threshold" maps to tips tucked *below* the hand-root plane —
tips reach -0.017..+0.01 m rel root at the joint limits, cube at +0.07 m).

| Purpose | Command |
|---|---|
| Train (RSL-RL) | `python scripts/rsl_rl/train.py --task Isaac-Repose-Cube-LEAP-v0 --headless` |
| Play (RSL-RL) | `python scripts/rsl_rl/play.py --task Isaac-Repose-Cube-LEAP-Play-v0 --num_envs 1` |
| Headless smoke test | `python scripts/smoke/leap_repose_smoke.py` |

Direct variant (`Isaac-Repose-Cube-LEAP-Direct-v0`) — the official
`Isaac-Repose-Cube-Allegro-Direct-v0` with the hand swapped and the reference
reward (`leap_direct_env.py`: local `LeapHandDirectEnv` subclass overrides
only `_get_rewards`; official terms + reference shaping + the tactile contact
term — see the objects variant below; `dist_reward_scale` defaults to 0.0 — set
-10.0 to restore the official dense position term; term-8 knobs are
`fingertip_contact_*`; terms 6/7/9 are zero-scaled, removed at user request).
v1/v2 keep the official reward (their
cfgs re-pin `dist_reward_scale = -10.0`; to give them the reference reward,
parent their envs on `LeapHandDirectEnv` and drop the pin).

| Purpose | Command |
|---|---|
| Train (RSL-RL) | `python scripts/rsl_rl/train.py --task Isaac-Repose-Cube-LEAP-Direct-v0 --headless` |
| Headless smoke test | `python scripts/smoke/leap_repose_direct_smoke.py` |

Vision variants (both keep the 124-d "policy" obs unchanged; rsl_rl only):

| Task | Added observation | Smoke test |
|---|---|---|
| `Isaac-Repose-Cube-LEAP-Direct-v1` | `"image"` (4, 128, 128) RGBD from a downward camera (CNN encoder) | `scripts/smoke/leap_repose_direct_v1_smoke.py` |
| `Isaac-Repose-Cube-LEAP-Direct-v2` | `"points"` (256, 4) object/hand point sets, DP3-style | `scripts/smoke/leap_repose_direct_v2_smoke.py` |

Objects variant (`Isaac-Repose-Objects-LEAP-Direct-v0`) — the v0 task (same
`LeapHandDirectEnv` MDP + reference reward) with the in-hand object
RESAMPLED EVERY EPISODE from a palm-size-filtered YCB catalog (spawn-all-and-toggle
design, cf. Chen et al. 2022; `leap_objects_direct_env.py`). The reward extends the
official terms with the reference shaping terms plus three tactile terms (shared
`compute_leap_rewards`, applies to the cube tasks too), ALL gated on MEASURED pulp
contact forces from the per-tip ContactSensors (filtered vs the in-hand object —
exactly the signal the real tactile pulps will report): term 8 rewards each pulp
with contact force above `fingertip_contact_force` (0.1 N; +0.1/tip/step), term 9
penalizes the object palm-pinned (−0.1/step within `palm_contact_dist` 6 cm of the
hand root), and term 10 rewards each pulp PRESSED above `goal_press_force` (3.0 N,
calibrated — random-action contacts are multi-N: nonzero p10 ≈ 2 N, median ≈ 8.5 N;
+0.5/tip/step, max +2.0) while the repose is DONE (rotation within
`goal_press_tolerance` 0.2 rad of the goal) AND the object is not palm-pinned — the
c9 penalty + c10 palm gate together make palm-crushing earn zero press bonus, so a
symmetric fingertip grasp strictly dominates; the gate uses its OWN tolerance knob
because the FlashSAC hold-hook overwrites `success_tolerance` (positional arg 14)
to -1 inside the base reward. The OBSERVATIONS carry the same pulp forces as 12
extra channels (observation_space 124 → 136). Terms 6 (object drop) and 7
(fingertip over-curl) are computed but ZERO-SCALED (removed at user request
2026-09-08; set their cfg scales nonzero to restore).

| Purpose | Command |
|---|---|
| Train (RSL-RL) | `python scripts/rsl_rl/train.py --task Isaac-Repose-Objects-LEAP-Direct-v0 --headless` |
| Train (FlashSAC) | `scripts/launch/flashsac.sh train --task Isaac-Repose-Objects-LEAP-Direct-v0` (`--exp` / env vars for the profile) |
| Headless smoke test | `python scripts/smoke/leap_repose_objects_direct_smoke.py` |

Objects v1 variant (`Isaac-Repose-Objects-LEAP-Direct-v1`) — the v0 task with the
in-hand object's inertial properties drawn from the sibling
`~/Documents/Estimate_weight_distribution` ground-truth manifest instead of
density x collision volume (`leap_objects_direct_v1_env.py`, subclasses
`LeapObjectsDirectEnv`; MDP and 136-d obs unchanged — the ground truth is
deliberately NOT observed, per the estimation premise). Mechanics:
- The manifest is BAKED to `assets/ycb/inertial_manifest.json` (committed; 13
  real-YCB objects x 6 density strategies = 78 variants; mass [kg], COM [m],
  full 3x3 inertia about the COM [kg*m^2], principal moments/axes + their
  quaternion, per-entry friction) by
  `python scripts/assets/generate_ycb_inertial_manifest.py` (env_isaaclab + `pip
  install defusedxml`; imports the sibling's `synthetic_density/inertial.py`
  STANDALONE via importlib — the package `__init__` pulls `gmsh`; ~13 min for
  the 500k-tet graphs). Values are in the YCB canonical mesh frame == the
  wrapper-USD body-local frame (identity root transform, metersPerUnit 1).
- Template prims get the FULL MassAPI state at spawn (mass/centerOfMass/
  diagonalInertia/principalAxes — explicit mass OVERRIDES density and explicit
  inertia decouples dynamics from the collision-proxy volume; the deterministic
  "uniform" variant is used). v0's density authoring lives in the
  `_author_object_mass` hook that v1 overrides.
- Every `_reset_idx` resamples (m, COM, I) per env from the active member's
  manifest variants via the `RigidObjectCollection` runtime writers (full 3x3
  inertia about the COM, body-local, row-major 9-flatten; coms = 7-float COM
  pose, identity quat). `env.active_inertials` exposes the sampled ground
  truth. NOTE: `set_coms_index` AND `set_inertias_index` are BROKEN in this
  isaaclab 3.0.0b2 physx backend (coms: a transformf-typed warp array reaches
  the omni.physics frontend → KeyError; inertias: the (I,B,9) buffer goes
  through the 2-D view-order reshape → "Incompatible size ... expected 648,
  received 72"); the env replicates both setters with the fixes
  (`_write_com_poses` adds `.view(wp.float32)`, `_write_inertias` uses
  `reshape_data_to_view_3d`). `set_masses_index` works as-is.
- Catalog restricted to manifest-covered objects: strawberry, lemon, orange
  (pear is covered but 10.05 cm long — dropped by the 0.09 m palm filter; plum
  has manifest data but no USD here; the other five v0 objects have no manifest
  data). The synthetic densities (200-5000 kg/m^3) make objects MUCH heavier
  than v0's uniform 400 kg/m^3 (up to ~0.9 kg orange) — a fresh dynamics
  regime. The manifest's per-object friction pair is not yet applied.

| Purpose | Command |
|---|---|
| Train (RSL-RL) | `python scripts/rsl_rl/train.py --task Isaac-Repose-Objects-LEAP-Direct-v1 --headless` |
| Train (FlashSAC) | `scripts/launch/flashsac.sh train --task Isaac-Repose-Objects-LEAP-Direct-v1` |
| Headless smoke test | `python scripts/smoke/leap_repose_objects_direct_v1_smoke.py` |

Contact-point variant (`Isaac-Contact-Cube-LEAP-Direct-v0`,
`...-Direct-Play-v0` for eval) — the repose direct MDP with the orientation
goal REPLACED by 4 contact points sampled uniformly on the cube surface (all
6 faces incl. palm-facing regions — reachable only by REGRASPING; cube-local
frame; min pairwise separation 2.5 cm, rejection-resampled; DexCube at scale
1.2 is a 6 cm cube, `cube_half_extent = 0.03` pinned from the smoke-measured
bbox). A point is TOUCHED when its one-to-one MATCHED pulp (exact assignment,
argmin over all 4! = 24 tip-permutations) is within `touch_distance` (1.5 cm)
AND reports > `touch_force` (0.1 N) of cube-filtered contact force — the
repose task's per-pulp ContactSensors, so hovering never counts. Success uses
a NATIVE hold timer (per-env duration from `hold_time_range`, v2 default
(0.1, 0.5) s = 3-15 steps): all 4 touched continuously → +250
(`reach_goal_bonus`) and points resample mid-episode (the base task's
goal-resample pattern). Reward v3 (2026-09-21): −0.05 × Σ matched distances +
0.5/touched point + **+0.1 alive bonus/step within 0.10 m of the anchor**
(`alive_bonus_scale`/`alive_distance`) + −0.25 × ‖cube − anchor‖ drift signal
+ the base action/joint-energy/object-away penalties + −50 `fall_penalty` on
the out-of-reach termination. DESIGN RULE (learned the hard way): keep every
per-step COST stream under the drop-exploit break-even
`fall_penalty / max_episode_length ≈ 0.083` and make holding EARN — both
GRU-PPO (peak 77 → 14) and FlashSAC (peak 294 → 15, return IMPROVING as it
collapsed) found the drop-early optimum of v2's −1.0/−2.0 streams and v1's
free fall. v3 result (FlashSAC 20M local): episode length stable at 284-289
for the whole run, returns positive (+17), exploit gone; successes still 0
(the 4-point hold is the open frontier). `reset_rot_scale` (default 0.0)
scales the reset cube rotation: 0 = axis-aligned spawn; raise toward 1.0 as
curriculum. rot, goal-dist, fingertip-to-center, palm, and press terms
dropped. Obs = the 136-d repose obs + the 4×4 tip→point relative-vector
matrix in env frame (48 d) = 184 d; point order fixed per goal (matching is
reward-side). The reward fn is deliberately named `compute_contact_rewards` —
the FlashSAC hold-hook patches `compute_leap_rewards` by name and must stay
inactive (hold is native; RSL-RL and FlashSAC would get identical semantics).
Debug: 4 colored sphere markers glued to the points, poses recomputed every
step (`/Visuals/contact_points`; no-op headless). Backends: **physx** =
reference; **newton** allowed ONLY in geometry-only mode (`touch_force <= 0`
drops the force gate — pulp sensors read zero under newton, so the gate must
drop out entirely; `>` against all-zero forces would never fire); **ovphysx**
experimental — sensors re-plumbed to `isaaclab_ovphysx.sensors` and forces DO
report live there (probe: 87% of readings > 0.1 N, max 48 N), but the
isaaclab-3.0.0b2 x ovphysx-0.4.13 combo has a frozen/garbage articulation
pose readback (root/body positions constant ~1e13 while physics-side contacts
are real) — NOT trainable until that is fixed. ovphysx wheel notes: only
0.4.13 has the `add_usd` API the beta's manager calls (0.5.11/0.6.3 dropped
it); the manager also needs a 2-line `device`-kwarg patch (see the seetacloud
memory). rsl_rl only; FlashSAC wiring (ACTION_BOUNDS + TUI id) deliberately
deferred. SCALE CEILING on Isaac Sim 6.0.1/PhysX (measured 2026-09-20): the
GPU patch buffer caps at 163840 and this isaaclab generation has NO way to
raise it (PhysxCfg gpu_* fields unread anywhere in the tree; physxScene USD
attrs ignored whether authored in _setup_scene or by pre-creating
/physicsScene before SimulationContext creation) — peak demand is ~15.4
patches/env, so >~10.6k envs drops contact patches and corrupts the
force-gated touch test; 10240 envs is the clean default for big runs. GPU
utilization stays ~0% at any env count (CPU-serialization-bound pipeline,
iteration time nearly flat in env count — env doubling raises throughput
until the patch cap); parallel independent runs are the working lever for
aggregate output. Large-scale history: contact-cube-r1 (2026-09-20, seetacloud
4090, newton + geometric mode, 8192 envs — superseded same day) and
**contact-physx-r6/r6b** (2026-09-20, same box, FULL force-gated task on
PhysX: `/root/autodl-tmp/env_isaaclab` venv ships isaacsim 6.0.1 + an
isaaclab 6.1.18 editable and runs this repo's 3.0.0b2-targeted code
unchanged — smoke passed, pulp forces multi-N; 2 parallel runs, 10240 envs,
seeds 42/43, tmux `contact-physx`/`contact-physx-b`, logs
`/root/contact-cube-physx-r6{,b}.log`).

| Purpose | Command |
|---|---|
| Train (RSL-RL, MLP) | `python scripts/rsl_rl/train.py --task Isaac-Contact-Cube-LEAP-Direct-v0 --headless` |
| Train (RSL-RL, GRU) | `python scripts/rsl_rl/train.py --task Isaac-Contact-Cube-LEAP-Direct-GRU-v0 --headless` (same env/MDP + PPO hyperparams; GRU-256 recurrent actor-critic for temporal/force history — clean MLP-vs-GRU A/B; Play twin `-GRU-Play-v0`) |
| Headless smoke test | `python scripts/smoke/leap_contact_direct_smoke.py` (also measures the cube bbox) |

Notes:
- `scripts/agents/random_agent.py` mirrors the official Isaac Lab random-agent API
  (`scripts/environments/random_agent.py`: --task/--num_envs/--disable_fabric +
  launcher args, Kit visualizer on by default, runs until the window closes) with
  opt-in `--verify` for the correctness checks. Two deviations required by this
  build: `--headless` clears the kit-visualizer default (the combo crashes Kit
  startup), and the task cfg is resolved INSIDE the launch context (the LEAP cfg
  import chain pulls pxr pre-app, which corrupts Kit in the pip Isaac Sim wheels).
- Every env spawns the whole filtered catalog as a `RigidObjectCollection`;
  inactive members are parked in a ground row beyond the FULL env-grid span
  (a per-env local offset only half the grid lets edge envs park inside other
  envs' cells), and each `_reset_idx` randomly activates ONE member per env at
  the palm. `self.object` is a duck-typed view of the active member.
- The GOAL MARKER mirrors the active object (not the inherited DexCube): the
  marker cfg is replaced in `_setup_scene` with one `VisualizationMarkers`
  prototype per catalog object (wrapper USDs; markers-dict order == collection
  member order), and `_reset_target_pose` passes `marker_indices ==
  _active_idx` (the active-object resample runs BEFORE the goal update). A
  second marker group (`goal_axes_markers`, built in `__init__`) draws the
  official `frame_prim` orientation triad (red/green/blue x/y/z arrows, size
  via `goal_axes_scale`) at the goal pose. The Kit backend is a USD
  `PointInstancer` — visual-only, physics-stripped, invisible to camera
  secondary rays, and not spawned at all in headless runs without rendering
  (the UI-element USDs load from the ISAAC_NUCLEUS_DIR S3 host, like the
  DexCube marker).
- DEFAULT `scene.num_envs` is 1024 (not the official 8192): this task has 8 rigid
  bodies per env, and this Isaac Lab beta never applies the PhysX GPU buffer cfg
  fields (`gpu_max_rigid_patch_count` etc. are defined but unread), so larger
  counts overflow the default patch buffer (errors + degraded physics; 8192 also
  OOMs the 12 GB GPU). The env authors `physxScene:maxRigid*Count` attrs directly,
  but they land after physics init and are only partially effective.
- Catalog filtering measures each USD's bounding box and drops objects larger
  than `max_object_extent_m` (default 0.09 m ≈ LEAP palm; 8 objects pass:
  tuna can, strawberry, apple, lemon, peach, orange, foam brick, Rubik's cube).
  Candidates are listed in `object_names` (`leap_objects_direct_env_cfg.py`).
- Isaac Lab 3.0.0b2's heterogeneous clone plan (`MultiUsdFileCfg` in the scene
  cfg) is BROKEN in the DirectRLEnv flow (empty prim path in `usd_replicate`);
  the env spawns per-env prims itself via `spawn_from_usd` before cloning.
- Object masses follow density (400 kg/m^3) x collision-shape volume (convex-hull
  volume for the default hull proxies; the sphere-union volume for sphere proxies,
  where overlapping spheres double-count somewhat). The density is
  authored via USD Physics MassAPI on each template prim in the env: the YCB wrapper
  USDs carry no mass attributes, so `MassPropertiesCfg(density=...)` in the spawn cfg
  is silently ignored (logged as "Could not perform 'modify_mass_properties'").
- Collision proxies: DEFAULT is `collision_proxy = "hull"` -- the wrapper USDs
  with the env de-duplicating their double CollisionAPI (root + mesh child →
  PhysX built TWO identical hulls per object: ~2x mass and doubled contacts;
  the env strips the root API and authors an explicit
  `UsdPhysics.MeshCollisionAPI physics:approximation = "convexHull"` on the
  mesh, which also silences the fallback parse-error spam). MEASURED cost at
  1024 envs (`scripts/assets/bench_collision_proxy.py`, 300 random-action steps):
  hull 32.5 ms/step (2.8 GiB, 0 patch overflows), 14 spheres 106.8 ms/step
  (4.6 GiB, 3 overflows), 50 spheres 222.5 ms/step (11.4 GiB -- near the
  12 GiB limit -- and 245 patch overflows, i.e. DROPPED contact patches).
  Set `collision_proxy = "spheres"` to
  use the MorphIt sphere sets (`assets/ycb/spheres/`, 14-sphere fits; the
  50-sphere variant lives in `assets/ycb/spheres50/`, select via
  `sphere_proxy_dir`). Sphere fitting is cheap: MorphIt-S 500 iterations ≈ 2 s/object
  even at 50 spheres. Regeneration pipeline (patches its own venv at
  ~/Documents/projects/MorphIt; needs `PYTHONPATH=src` — the script's built-in
  path insert is one dir too high):
  1. `cd ~/Documents/projects/MorphIt && PYTHONPATH=src ./venv/bin/python run_ycb.py
     --num-spheres 14 --out-dir results/ycb` (the venv's torch was upgraded to
     2.9.1+cu128 -- the pinned cu126 build has no kernels for the RTX 5070 Ti)
  2. `~/env_isaaclab/bin/python scripts/assets/generate_sphere_proxy_usd.py
     --results ~/Documents/projects/MorphIt/results/ycb --out <YCB_DIR>/spheres`
     (strips the mesh CollisionAPI -- merely disabling it still makes PhysX
     parse-and-warn -- and adds the sphere colliders; visual mesh unchanged)
  Outputs live in `assets/ycb/spheres*/<name>_spheres.usd`.
- Per-pulp PhysX CONTACT SENSORS (default on, `fingertip_contact_sensors`):
  each of the four fingertip links gets a `ContactSensor` filtered against
  the in-hand object(s) (the cube prim for cube tasks; every catalog member
  for the objects task — only the active one is ever in reach). Read
  `env.fingertip_contact_forces` — (num_envs, 4, 3) net forces in N; this
  emulates the tactile pulp sensors planned for the real hand and does NOT
  change the observation space. Mechanics: `PhysxContactReportAPI` is authored
  on the four pulp links of the env_0 template PRE-clone (the cloner replicates
  applied schemas — same trick as `assets/kuka_leap.py`), then one single-body
  `ContactSensor` per tip is created post-clone (filtered reporting requires a
  single prim per env); sensors are registered in `scene.sensors` so
  `scene.update(dt)` refreshes them. MEASURED cost at 1024 envs: +4.1 ms/step
  (36.6 vs 32.5 hull-only), VRAM unchanged. v1/v2 camera variants do not have
  them (own `_setup_scene`s; the property returns zeros there).
- FlashSAC: the task id is registered in the patched clone's ACTION_BOUNDS; the
  hold-to-confirm/EMA hooks apply unchanged (env subclasses `LeapHandDirectEnv`).

Notes:
- Requires the `env_isaaclab` environment (Isaac Lab 3.0 / Isaac Sim 6.0 / Python 3.12).
- The manager-based cube spawns at `(0.0, -0.10, 0.56)` — the LEAP palm center
  (Allegro uses `-0.19`); the direct variant uses the same offset (Allegro: `-0.17`)
  with the official direct-task cube scale `(1.2, 1.2, 1.2)`.
- The LEAP hand joints are `a_0`..`a_15` (16 DOF, same action dim as Allegro).
- The direct task uses the local `LeapHandDirectEnv` (`leap_direct_env.py`, subclass of
  the official shared `InHandManipulationEnv` from
  `isaaclab_tasks.direct.inhand_manipulation`) — resets/goal bookkeeping are
  inherited; `_get_observations` appends 12 tactile channels (4 pulps x 3
  filtered contact-force components, scaled by `force_torque_obs_scale`:
  observation_space 124 -> 136) and `_get_rewards` differs (see the
  reference-reward note above). 16-d actions.
- v1 (`leap_direct_v1_env.py`) subclasses it with a top-down RGBD camera; the CNN PPO
  runner is `agents/rsl_rl_ppo_direct_v1_cfg.py` (it re-attaches the RGBD runner's
  deprecated-field-stripping `to_dict` after the class body — `@configclass` resets
  `to_dict` on subclasses).
- v2 (`leap_direct_v2_env.py`) backprojects the depth camera to a sparse point cloud,
  split into object/hand sets by color-coded `semantic_segmentation` (the object prim is
  tagged post-spawn via `add_labels` on the template prim; tags on PresetCfg spawns do
  not survive, and `instance_segmentation*` annotators DEADLOCK this Isaac Sim build).
  The model is the DP3-style `LeapPointSetModel` (`point_model.py`: shared per-point
  MLP + masked max-pool per set), referenced by dotted `class_name` from
  `agents/rsl_rl_ppo_direct_v2_cfg.py` (`point_split` must equal
  `num_object_points` in the env cfg).
- `agents/rsl_rl_ppo_cfg.py` and `agents/rsl_rl_ppo_direct_cfg.py` strip the deprecated
  rsl_rl<4 model fields (`stochastic`, `init_noise_std`, ...) in `__post_init__`:
  `isaaclab_rl` 3.0.0b2 still serializes them and rsl-rl-lib 5.0.1's `MLPModel`
  rejects them (this also breaks the official Allegro tasks' stock RSL-RL cfgs in
  this env).

## Additional task: DexPBT replication on DexSuite Reorient

`Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0` (+ `-PBT-Play-v0`; task pkg
`tasks/dexsuite_reorient_pbt/`): the OFFICIAL manager-based
`Isaac-Dexsuite-Kuka-Allegro-Reorient-v0` MDP with a DexPBT-inspired
population wrapper (NOT a faithful paper replication). Reworked 2026-09-24
after the first population was diagnosed as four identically-configured PPO
seeds (selection gate dead at the observed score scale, broken restart
plumbing, unbounded learned exploration — checkpoint sigma implied >=68-71%
of action components clipped; see
docs/research/2026-09-24-dexpbt-training-strategy.md). Key pieces:
- **PBT** (`pbt/`): one PROCESS per member
  (`scripts/rl_games/train_dexpbt_member.py`, repo's proven AppLauncher flow),
  shared workspace, publication every `interval_steps` env steps. Selection
  uses COMPLETED-EPISODE stats (`objective: dexpbt`, success-first with a
  fixed pose-progress tie-break — never the soft success reward), requires
  >=128 completed episodes and fresh scores, and compares only
  physically-matching conditions (actual gravity/noise parameter stamp; the
  ADR scheduler's bookkeeping fraction is ignored below the native <0.1
  NO_CHANGE hysteresis). Underperformers re-exec from a random leader with
  lr/grad_norm/kl_threshold mutated ×U(1.1, 1.5) within explicit bounds
  (entropy PINNED 0 — not mutated); 5M-step warm-up, 3M post-exploit
  adaptation; bounded initial diversity (member 0 baseline). Restarts
  preserve full argv identity via sys.executable, restore optimizer/adaptive
  LR/normalizers + recipient frame accounting, and normalize torch.compile's
  `_orig_mod.` prefixes; `pbt.enabled=false` keeps the restore path for
  exploitation-disabled diagnostics.
- **Exploration** (`pbt/exploration.py`): learned logstd bounded to
  [.05, 1.0] (initial std 0.3, arm 0.15/.5 cap when the 23-joint action order
  is verified) after restore, pre-inference, and every optimizer step;
  logs actual sampled clipping/saturation fractions. Runner `--sigma` is
  RAW LOGSTD, not std.
- **Curriculum** (`environment.py`, `evaluation.py`): the 0.2 m start
  tolerance is actually applied at spawn (the old code stored .2 but never
  wrote it), annealed from COMPLETED-episode sticky success flags (the old
  code read a log dict the rl-games wrapper pops), 1024-episode weighted
  windows; curriculum+ADR state rides checkpoints via get/set_env_state
  (strict num_envs match).
- **Fixed evaluation** (`scripts/rl_games/evaluate_dexpbt.py`): held-out
  report at pinned ADR fraction + 0.1 m/0.5 rad/0.01 N thresholds
  (centralized constants), policy-only restore (no donor curriculum), JSON
  report with checkpoint hash; `scripts/smoke/dexpbt_smoke.py` covers
  metrics/curriculum/fixed-ADR in-sim.
Launch via `scripts/launch/dexpbt_reorient.sh` (POP/ENVS/EPOCHS/INTERVAL/
BASE/PYTHON/RESUME/DRY_RUN knobs; preflights ALL tmux sessions before
launching any, refuses fresh-start into a populated workspace, NO automatic
retry of crashed members — relaunch explicitly with RESUME=1). Tests:
`tests/test_dexpbt_{pbt,exploration,curriculum}.py` (42 + 7 subtests,
simulator-free). Scaled run (2026-09-24, westb 2×4080S 32 GB): 8 members ×
1024 envs, seeds 42-49, ~35k steps/s aggregate, ~21 h for 2.62B steps —
see docs/research/2026-09-24-dexpbt-scaled-westb-run.md. Gotchas: rl-games
1.6.5 yaml needs algo.name/model.name/mlp.units/reward_shaper (the westb box
ships 1.6.1 — prepend a `pip install --no-deps --target` 1.6.5 dir on
PYTHONPATH); 3.0.0b2 wrapper wants obs_groups `{"obs": [...], "states":
[...]}`; hydra drops the yaml's top-level pbt block and silently reverts
early agent-cfg writes — the member script fishes `pbt.*`/`params.*` CLI
tokens and applies params.* directly before `runner.load`; simulation_app
`.close()` internally `os._exit`s, so entrypoints must catch/traceback and
pass an explicit exit_code or failures read as success. Phase 2 (evolving
reward coefficients, DexPBT's signature) deliberately deferred until fixed
evaluation demonstrates acquisition.

## Additional task: Direct port of the official Dexsuite KUKA+Allegro Lift task

`Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0` (`...-Direct-Play-v0` for eval,
`tasks/dexsuite_lift_direct/`) is a DirectRLEnv re-implementation of the official
manager-based `Isaac-Dexsuite-Kuka-Allegro-Lift-v0`, built so FlashSAC can train it:
the official task splits its state across the `policy`/`proprio`/`perception` obs
groups and flash_rl only reads `"policy"` (34-d, no proprioception — unlearnable).
The port keeps the official MDP (23-d relative joint-position actions, scale 0.1;
official Lift reward terms/weights x step dt; goal pose in the robot-base frame
resampled on a 2-3 s timer; ADR difficulty 0..10 driving scene gravity 0 -> -9.81
via the MEAN difficulty — the robot asset has `disable_gravity=True`, so only the
object feels it — and per-env obs noise; official reset/out-of-bound/velocity-limit
terminations). Observation is one flat 800-d vector (160 x 5-step history:
object quat 4, object pos 3 (in the robot base frame -- a DELIBERATE addition:
the official actor sees the object position through the perception group's
point cloud, which this port drops; the first lift-r1 run plateaued at
avg_return ~0.18 with ZERO successes over 600M env steps without it), goal
pose 7, last action 23, joint pos/vel 46, palm+tip body
states 65, per-tip contact forces 12 -- the four fingertip ContactSensors
filtered against the object, official prim paths
`{ENV_REGEX_NS}/Robot/ee_link/<finger>_link_3`, net force per tip in the robot
base frame clipped to +-20 N). Contact gating uses the official force
thresholds: fingers_to_object's contact bonus at 1.0 N, good_finger_contact /
position_tracking at 0.1 N, success (+ sticky flag) and contact_count at
0.01 N; "good contact" = thumb AND at least one finger above the threshold.
Sensor liveness VERIFIED 2026-09-10 (object at fingertip centroid + curling
actions -> multi-N readings; at ADR difficulty 0 gravity is zero so contacts
flicker as the object floats). V1 parity gaps (startup randomization NOT
ported): friction buckets, object mass scaling,
inertia offset, actuator-gain scaling, joint-friction scaling; the object is the
fixed default cuboid (the multi-shape spawner's `random_choice` is a no-op in
this build anyway).

| Purpose | Command |
|---|---|
| Train (FlashSAC) | `scripts/launch/flashsac.sh train --task Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0` |
| Headless smoke test | `python scripts/smoke/flashsac_dexsuite_lift_smoke.py` |

Notes:
- Action bounds 0.1 (the RelativeJointPositionAction scale) are registered in the
  patched flash_rl clone's ACTION_BOUNDS; HOLD_TIME stays 0 (in-hand-only hook),
  ACT_SMOOTH works (the cfg declares `act_moving_average`, default 1.0 = off).
- This Isaac Lab 3.0.0b2 API surface differs from 2.x in several places the port
  had to absorb: gym registrations need the kwarg `env_cfg_entry_point` (NOT
  `cfg_entry_point` — `load_cfg_from_registry` reads only the former); quaternion
  utils are xyzw (`quat_from_euler_xyz` returns (x,y,z,w), root poses follow warp
  `transformf`); asset writers are keyword-only `*_to_sim_index` variants; the
  compiled quat helpers need FLAT (-1,3)/(-1,4) inputs (mid-dim broadcasting
  fails inside TorchScript); the base DirectRLEnv builds `InteractiveScene(cfg.scene)`
  ITSELF and `_setup_scene` is for extras only (creating a scene there double-spawns
  prims); fingertip bodies are named `*_tip` (`index_tip` etc.), not `*_link_3`;
  there is no `num_actions`/`num_observations` property (use `cfg.action_space`).
- KUKA+Allegro is the official `KUKA_ALLEGRO_CFG` (isaaclab_assets): arm gains
  300/45..25/15, finger gains 3/0.1, finger effort 0.5 Nm, armature 0.01 — much
  softer fingers than the LEAP hand.
- Default `scene.num_envs` 1024 with the same unread-PhysX-buffer caveat as the
  LEAP repose tasks; the train script's laptop profile is 1024 envs.
  MEASURED VRAM at 512 envs: sim ~4.6 GB (self-colliding 23-DOF arm + 4 contact
  sensors/env + Isaac Sim base); each 1M transitions of the 800-d obs costs
  ~6.0 GB for obs+next_obs in float32. The script's BUFFER_DEVICE knob picks the
  replay-buffer placement: `cpu` (default) hosts the buffer in PINNED system RAM
  (flash_rl allocates pin_memory=True and the agent copies each 2048-row
  mini-batch to GPU with non_blocking=True -- ~13 MB/update, measured NO
  slowdown at UTD 1); `cuda` needs ~4.6+6.0x(buffer/1M) GB of the 12 GB. RAM
  budget on the 30 GB laptop (pinned memory cannot swap; Isaac Sim's host
  process takes ~8-10 GB): the 2M script default pins ~12.8 GB -- the practical
  ceiling, fall back to 1.5M on host-OOM; the LEAP-style 6M would need ~36 GB
  and OOMs (measured exactly `17.55 GiB` at the old 785-d obs = 6M x 785 x 4 B;
  the 800-d obs sizes to 6M x 800 x 4 B = 17.9 GiB).
  TB cadence gotcha: upstream derives logging/metrics cadence from the TOTAL
  budget (logging = num_interaction_steps/100, metrics = /10), so a ~1B-step
  budget logs actor/critic scalars only every ~10M env steps (~30 min) and
  avg_return every ~100M — the script pins LOG_EVERY (~1M) and
  METRICS_EVERY (~0.5M env steps) instead; the x-axis is env steps (iteration x num_envs).
  The script's `${VAR:-default}` tunables honor stale exported values (a leftover
  `BUFFER_LEN=6000000` from a LEAP launch silently overrode the default
  2026-09-10) — the launch banner prints the effective profile. One-off startup
  flakes (`CreateJoint - no bodies defined` / `no active physics scene`) can
  happen in this beta with the KUKA USD cloning — just relaunch.

## FlashSAC training (off-policy)

`Isaac-Repose-Cube-LEAP-Direct-v0` can be trained with [FlashSAC](https://github.com/Holiday-Robot/FlashSAC)
inside `env_isaaclab`. FlashSAC pins Isaac Lab 2.3/Python <3.12 upstream, so a patched clone
lives at `~/Documents/projects/FlashSAC` with these local modifications:

- `flash_rl/types.py` — jax import made optional (only mujoco-playground needs it; the
  torch path runs jax-free).
- `flash_rl/envs/isaaclab.py` — custom task packages registered before the task id is
  resolved via `FLASH_RL_EXTRA_TASK_MODULES=LEAP_Isaaclab.tasks`; action bounds added for
  `Isaac-Repose-Cube-LEAP-Direct-v0`; `FLASH_RL_PRESET=newton_mjwarp` switches the
  wrapper to hydra preset resolution + `isaaclab_tasks.utils.sim_launcher.launch_simulation`,
  which starts Kit ONLY for PhysX/camera tasks — this is what `scripts/launch/flashsac.sh ... --backend newton`
  uses to train WITHOUT Isaac Sim; when `headless=False` the AppLauncher is given
  `visualizer=["kit"]` (Isaac Lab 3.0 forces headless unless a Kit visualizer is requested);
  `install_hold_to_confirm` is generalized to also patch
  `LeapHandDirectEnv.compute_leap_rewards` (it resolves the function from the globals of
  the env's `_get_rewards` IMPLEMENTATION — subclass modules like the objects env never
  re-import it, and resolving via the subclass module silently patched the unused
  official `compute_rewards` instead, i.e. hold-to-confirm was inactive for the objects
  task until 2026-09-08; check the banner says `(compute_leap_rewards)`; it relies on
  both reward signatures keeping `success_tolerance` at positional index 14,
  passing trailing args through — the tactile terms 8/9 are appended after it)
- `flash_rl/agents/utils/network.py` — `load()` remaps torch.compile's `_orig_mod.` state
  dict prefix in both directions, so compiled-saved checkpoints load into eager modules;
  `load(..., pad_new_inputs=True)` zero-pads grown input dims (first-layer weights 0,
  BatchNorm running_var 1) so a checkpoint trained on a SMALLER obs space plays
  unchanged in the current env (the new channels contribute nothing).
- `flash_rl/agents/flashSAC/agent.py` — `load(..., pad_new_inputs=True)`: the actor
  pads exactly; the critics take obs||action (new channels sit MID-concatenation, not
  positionally pad-able) and are SKIPPED with a warning — playback only samples
  actions. Used to play the pre-tactile-obs 124-d objects-r2 checkpoint in the 136-d
  env.
- `play_isaaclab.py` — `parse_known_args()` so extra AppLauncher flags pass through;
  loads checkpoints with `pad_new_inputs=True` (legacy-obs compatibility).
- `train.py` — always writes a `final` checkpoint after the last eval (upstream saves
  only via the periodic gate, which counts CONTROL ITERATIONS — a run shorter than
  `save_checkpoint_per_interaction_step` iterations silently discards its weights;
  e.g. SAVE_EVERY=204800 at 1024 envs first saves at ~210M env-steps).
- `pyproject.toml` — `requires-python >=3.10` (3.12 allowed) + explicit package discovery,
  so `pip install -e ~/Documents/projects/FlashSAC --no-deps` works.

| Purpose | Command |
|---|---|
| Train (menu TUI; also `train|play` positional + `--task <full-gym-id> --backend physx|newton`) | `scripts/launch/flashsac.sh` |
| Train (laptop-GPU profile) | `NUM_TRAIN_ENVS=512 BUFFER_LEN=1000000 scripts/launch/flashsac.sh train --task Isaac-Repose-Cube-LEAP-Direct-v0` |
| Play with Isaac Sim GUI | `scripts/launch/flashsac.sh play --task Isaac-Repose-Cube-LEAP-Direct-v0` (checkpoint picker; forwards `--hold_time`, `--action_smooth`) |
| Monitor | `tensorboard --logdir ~/Documents/projects/FlashSAC/runs` |
| Checkpoints | `~/Documents/projects/FlashSAC/models/<group>/<exp>/Isaac-Repose-Cube-LEAP-Direct-v0/` |

`scripts/launch/flashsac.sh` (2026-09-11) REPLACES the four former wrappers
(`flashsac_{train,play}_{leap_direct,dexsuite_lift}.sh`): one menu TUI covering
task (FULL gym ids; the four known ones are listed, a free-text "other" entry
accepts any id with generic defaults — group `flashsac`, exp `run1`, play env =
train env, 1024-env/2M-cpu profile, newton allowed but flagged UNVERIFIED) x
mode (train/play) x backend (physx = Isaac Sim+PhysX, newton = Newton, no Isaac
Sim; the old `kit`/`kitless` values print a migration hint). Old nicknames
(cube/objects/objects-v1/lift) print a migration hint. Every menu choice also has a CLI form (`train|play` positional + `-t/-b` flags; `--mode` prints a migration hint),
`--dry-run` prints the exec line, unknown args forward verbatim (train: extra
`--overrides`; play: `play_isaaclab.py` args), and all former env-var knobs
(`NUM_ENV_STEPS`, `NUM_TRAIN_ENVS`, `BUFFER_LEN`, `BUFFER_DEVICE`,
`UPDATES_PER_STEP`, `SAVE_EVERY`, `LOG_EVERY`, `METRICS_EVERY`, `HOLD_TIME`,
`ACT_SMOOTH`, `GROUP`, `EXP`, ...) still win over the built-in defaults (a stale
exported `FLASH_RL_PRESET` is deliberately UNSET in kit mode). Play picks the
checkpoint from a listed run menu (GROUP + optional EXP filter, newest step*/
default) and auto-switches lift to the `-Direct-Play-v0` env (ADR max difficulty).

Kit-less (Newton) training status, measured 2026-09-11
(`scripts/probes/leap_repose_direct_newton_probe.py --task <id> --curl`, 8-16 envs):
- cube / objects / objects-v1 BUILD and STEP on `newton_mjwarp` — hand anchored
  by a pxr `UsdPhysics.FixedJoint` root weld (`_fix_root_link_kitless`), object
  held (z stable under curl), rewards finite, obs 136-d, and v1's manifest
  inertial writes go through the newton collection's NATIVE
  `set_coms_index`/`set_inertias_index` (COM position (n,K,3) + full 3x3
  body-local inertia (n,K,9) — exactly the manifest semantics).
- BUT the pulp contact sensors READ ZERO in both contact modes — an
  isaaclab_newton beta REPORTING gap (physics itself has contacts): with
  `use_mujoco_contacts=False` the Newton collision pipeline's model ships only
  8 precomputed `shape_contact_pairs` (ground<->object; NO intra-world pairs),
  so `collide()` reports nothing; with `use_mujoco_contacts=True` the solver's
  reported `nacon` stays 0 (freshly-reset buffers). Until fixed upstream, the
  12 tactile obs channels + reward terms 8/9/10 are dead under kit-less. The
  TUI prints this warning at newton-backend launch.
- Objects-task fixes that made kit-less possible (also required for the
  PhysX path, verified by `scripts/smoke/leap_repose_objects_direct_smoke.py`):
  objects spawn NESTED at `/World/envs/env_*/obj/<name>` — the newton
  `RigidObjectCollection` builds ONE fnmatch pattern over all members whose `*`
  CROSSES `/`, so flat sibling prims made the pattern `/World/envs/env_*/*`
  swallow the hand articulation (view `is_floating_base=False` from the hand's
  root weld -> `wp.clone(None)` crash in `RigidObjectCollectionData`); and the
  shared `PhysicsCfg.newton_mjwarp` pins
  `collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=2**20)` because
  the pipeline's auto-sized Contacts (1000) < the solver's total naconmax
  (nconmax 70 x num_envs) crashed `SolverMuJoCo.update_contacts` at >= 15 envs.
- `lift` has NO Newton support (PhysX-only `PhysxContactReportAPI` sensors by
  design); the TUI refuses newton for the lift task.

Training-time behavior knobs (env vars, also honored at play):
- `HOLD_TIME` (leap tasks default `"0.2:1.5"`, lift `0`) — hold-to-confirm goals:
  the object must keep the target orientation this long before the goal counts
  and resamples. `"lo:hi"` is randomized per env and per goal; `"1.0"` is fixed;
  `"0"` disables.
- `ACT_SMOOTH` (leap default `0.2`, lift `1.0` = off) — joint-target EMA
  (`act_moving_average`).
- If training crashes with `These live storage data ptrs are in the cudagraph pool`
  (intermittent torch 2.10 cudagraph-trees bug on this GPU), retry with
  `--overrides agent.compile_mode=default` (cudagraphs off, inductor still on).

Notes:
- Built-in train profiles: leap tasks 4096 envs / 6M CUDA buffer / UTD 1
  (12 GB+ GPU, e.g. the 4090 remote); lift 1024 envs / 2M pinned-RAM buffer with
  pinned TB logging cadences (12 GB laptop GPU — see the lift section's RAM
  budget). Override per launch with the env vars above; `--dry-run` shows the
  effective profile.
- `flash_rl` is installed editable into `env_isaaclab`; re-run the pip command after
  pulling upstream changes into the clone.
- The Vulkan env-var prefix (`VK_ICD_FILENAMES`, `__NV_PRIME_RENDER_OFFLOAD`) is set by
  the script itself.

## Agent skills

### Issue tracker

GitHub Issues on `leap-hand/LEAP_Hand_Isaac_Lab`. Use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
