# AGENTS.md — LEAP Hand Isaac Lab

## Environment & setup

- **Isaac Sim 4.5.0 + Isaac Lab 2.1.0**, Python **3.10**, Ubuntu 22.04.
- The repo root is NOT the Python package root. The package lives at `source/LEAP_Isaaclab/`.
- Install with: `python -m pip install -e source/LEAP_Isaaclab`
- All Isaac Lab / Isaac Sim / torch / numpy deps must be pre-installed in the conda environment; `install_requires` lists only `psutil`.

## Commands

| Purpose | Command |
|---|---|
| Train (rl_games) | `python scripts/rl_games/train.py --task Isaac-Reorient-Cube-Leap --headless` |
| Play checkpoint | `python scripts/rl_games/play.py --task Isaac-Reorient-Cube-Leap --num_envs 1 [--checkpoint PATH]` |
| Train (rsl_rl) | `python scripts/rsl_rl/train.py --task Isaac-Reorient-Cube-Leap --headless` |
| Play (rsl_rl) | `python scripts/rsl_rl/play.py --task Isaac-Reorient-Cube-Leap --num_envs 1` |
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
- **Pre-commit hooks**: `pre-commit run --all-files` (9 hooks: black, flake8, isort, pyupgrade, codespell, pygrep-hooks, generic checks).
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
- `scripts/list_envs.py` lists gym task ids containing "leap" (case-insensitive), so it covers both camelCase `Leap` tasks and all-caps `LEAP` Repose ids; missing from the table ≠ unregistered.
- No CI workflows exist in this repo.

## Additional tasks: Franka + LEAP (welded)

A single 23-DOF articulation (Franka 7-DOF + LEAP 16-DOF) is available as
gym task `Isaac-Franka-Leap` (DirectRLEnv, `tasks/franka_leap/`).

| Purpose | Command |
|---|---|
| Rebuild merged URDF | `python scripts/build_franka_leap_urdf.py` |
| Headless smoke test | `python scripts/franka_leap_smoke.py` |
| Render demo video | `python scripts/franka_leap_demo.py` |

Notes:
- The merged URDF lives at `source/LEAP_Isaaclab/LEAP_Isaaclab/assets/urdf/franka_leap.urdf`
  with bundled meshes (`urdf/franka/meshes/`, `urdf/leap/meshes/`); the USD cache is
  generated on first spawn (hash-based; delete `urdf/` cached dirs or set
  `force_usd_conversion=True` in `assets/franka_leap.py` to regenerate).
- Actions are 23-d relative joint position deltas (arm first, then hand).
- The importer fuses the flange (panda_link8) and LEAP base into their neighbors;
  the weld is still physically rigid (verified: fingertip-flange distance constant to 0.5 mm).
- `panda_link7` is the wrist body used for world-space flange reporting.

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
| Headless smoke test | `python scripts/leap_repose_smoke.py` |

Direct variant (`Isaac-Repose-Cube-LEAP-Direct-v0`) — the official
`Isaac-Repose-Cube-Allegro-Direct-v0` with the hand swapped and the same 7-term
reference reward (`leap_direct_env.py`: local `LeapHandDirectEnv` subclass overrides
only `_get_rewards`; `dist_reward_scale` defaults to 0.0 — set -10.0 to restore the
official dense position term; term-6/7 thresholds are `object_drop_z_offset` /
`fingertip_overcurl_z_offset`). v1/v2 keep the official reward (their cfgs re-pin
`dist_reward_scale = -10.0`; to give them the reference reward, parent their envs on
`LeapHandDirectEnv` and drop the pin).

| Purpose | Command |
|---|---|
| Train (RSL-RL) | `python scripts/rsl_rl/train.py --task Isaac-Repose-Cube-LEAP-Direct-v0 --headless` |
| Headless smoke test | `python scripts/leap_repose_direct_smoke.py` |

Vision variants (both keep the 124-d "policy" obs unchanged; rsl_rl only):

| Task | Added observation | Smoke test |
|---|---|---|
| `Isaac-Repose-Cube-LEAP-Direct-v1` | `"image"` (4, 128, 128) RGBD from a downward camera (CNN encoder) | `scripts/leap_repose_direct_v1_smoke.py` |
| `Isaac-Repose-Cube-LEAP-Direct-v2` | `"points"` (256, 4) object/hand point sets, DP3-style | `scripts/leap_repose_direct_v2_smoke.py` |

Notes:
- Requires the `env_isaaclab` environment (Isaac Lab 3.0 / Isaac Sim 6.0 / Python 3.12).
- The manager-based cube spawns at `(0.0, -0.10, 0.56)` — the LEAP palm center
  (Allegro uses `-0.19`); the direct variant uses the same offset (Allegro: `-0.17`)
  with the official direct-task cube scale `(1.2, 1.2, 1.2)`.
- The LEAP hand joints are `a_0`..`a_15` (16 DOF, same action dim as Allegro).
- The direct task uses the local `LeapHandDirectEnv` (`leap_direct_env.py`, subclass of
  the official shared `InHandManipulationEnv` from
  `isaaclab_tasks.direct.inhand_manipulation`) — obs/resets/goal bookkeeping are
  inherited; only `_get_rewards` differs (see the reference-reward note above).
  124-d "full" obs, 16-d actions.
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

## FlashSAC training (off-policy)

`Isaac-Repose-Cube-LEAP-Direct-v0` can be trained with [FlashSAC](https://github.com/Holiday-Robot/FlashSAC)
inside `env_isaaclab`. FlashSAC pins Isaac Lab 2.3/Python <3.12 upstream, so a patched clone
lives at `~/Documents/projects/FlashSAC` with these local modifications:

- `flash_rl/types.py` — jax import made optional (only mujoco-playground needs it; the
  torch path runs jax-free).
- `flash_rl/envs/isaaclab.py` — custom task packages registered before the task id is
  resolved via `FLASH_RL_EXTRA_TASK_MODULES=LEAP_Isaaclab.tasks`; action bounds added for
  `Isaac-Repose-Cube-LEAP-Direct-v0`; when `headless=False` the AppLauncher is given
  `visualizer=["kit"]` (Isaac Lab 3.0 forces headless unless a Kit visualizer is requested);
  `install_hold_to_confirm` is generalized to also patch
  `LeapHandDirectEnv.compute_leap_rewards` (it detects the function via the task module
  and relies on both reward signatures keeping `success_tolerance` at positional index 14,
  passing trailing args through — otherwise hold-to-confirm goals would be silently
  bypassed by the local reward override).
- `flash_rl/agents/utils/network.py` — `load()` remaps torch.compile's `_orig_mod.` state
  dict prefix in both directions, so compiled-saved checkpoints load into eager modules.
- `play_isaaclab.py` — `parse_known_args()` so extra AppLauncher flags pass through.
- `pyproject.toml` — `requires-python >=3.10` (3.12 allowed) + explicit package discovery,
  so `pip install -e ~/Documents/projects/FlashSAC --no-deps` works.

| Purpose | Command |
|---|---|
| Train (laptop-GPU profile) | `EXP=run1 scripts/flashsac_train_leap_direct.sh` |
| Play with Isaac Sim GUI | `scripts/flashsac_play_leap_direct.sh` (opts: `--hold_time`, `--action_smooth`) |
| Monitor | `tensorboard --logdir ~/Documents/projects/FlashSAC/runs` |
| Checkpoints | `~/Documents/projects/FlashSAC/models/<group>/<exp>/Isaac-Repose-Cube-LEAP-Direct-v0/` |

Training-time behavior knobs (env vars of the train script, also honored at play):
- `HOLD_TIME` (default `"0.5:1.5"`) — hold-to-confirm goals: the object must keep the
  target orientation this long before the goal counts and resamples. `"lo:hi"` is
  randomized per env and per goal; `"1.0"` is fixed; `"0"` disables.
- `ACT_SMOOTH` (default `0.3`) — joint-target EMA (`act_moving_average`); `1.0` disables.
- If training crashes with `These live storage data ptrs are in the cudagraph pool`
  (intermittent torch 2.10 cudagraph-trees bug on this GPU), retry with
  `--overrides agent.compile_mode=default` (cudagraphs off, inductor still on).

Notes:
- The script profile (512 envs, 1M CUDA buffer, UTD 1, n-step 3) is sized for the 12 GB
  RTX 5070 Ti laptop GPU; upstream uses 1024 envs / 10M buffer / UTD 2 — scale via the
  env vars documented in the script header.
- `flash_rl` is installed editable into `env_isaaclab`; re-run the pip command after
  pulling upstream changes into the clone.
- The Vulkan env-var prefix (`VK_ICD_FILENAMES`, `__NV_PRIME_RENDER_OFFLOAD`) is set by
  the script itself.

## Agent skills

### Issue tracker

GitHub Issues on `leap-hand/LEAP_Hand_Isaac_Lab`. Use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.