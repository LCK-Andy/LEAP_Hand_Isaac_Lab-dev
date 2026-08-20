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
- `scripts/list_envs.py` searches for `Template-*` prefix — it is an unused artifact from the extension template.
- No CI workflows exist in this repo.