# Consuming the public async-pbt framework

LEAP's PBT entrypoints consume [async-pbt v0.1.0](https://github.com/LCK-Andy/async-pbt/releases/tag/v0.1.0), rather than maintaining a second population engine. The release commit is `0ddbaca92f66495990651771ada7321a351eb8be`.

## Installation

The LEAP package declares the public Git-tag dependency. Normal editable installation installs it:

```bash
python -m pip install -e source/LEAP_Isaaclab
```

For an already validated Isaac environment, install only the framework without upgrading the simulator/trainer stack:

```bash
python -m pip install --no-deps 'async-pbt @ git+https://github.com/LCK-Andy/async-pbt.git@v0.1.0'
```

PyYAML must already be available. Isaac Lab, Isaac Sim, Torch, NumPy, and RL-Games remain preinstalled environment requirements; the framework integration targets RL-Games 1.6.5. Do not install the framework's CPU tutorial dependency recipe into the Isaac environment. A sibling editable framework checkout is not required or used.

## Ownership

The external package owns population coordination, bounded mutation, checkpoint publication, generic restore/re-exec helpers, exploration controls, and historical best-checkpoint storage. Its simulator-independent metric adapters supply the existing Allegro/contact/grasp/grasp-motion measurement helpers.

LEAP retains task definitions, environment measurements, physical-condition stamps, curriculum and environment state, simulator startup, warm-start observation padding, and policy-prior validation. Existing imports under `tasks/dexsuite_reorient_pbt/pbt/` remain available through small compatibility adapters.

## Compatibility choices

- An omitted objective remains **`dexpbt`**, not the standalone framework's `success_return` default. Existing `infos['dexpbt']` and other task protocols are not converted to generic return scoring.
- The legacy `pbt.directory` launch prefix and trainer-relative population workspace layout remain unchanged. Existing launch scripts still find member publications at their established paths.
- A raw checkpoint without a companion retains LEAP's launch/base-hyperparameter precedence. Companion-based resume retains population parameters, recipient age and mutation RNG. Missing companion data cannot reconstruct an original random stream or omitted metadata.
- The prelaunch runtime facade remains safe to load without importing LEAP task registration or `pxr` before AppLauncher.
- Observer order remains logging, exploration, optional warm-start/prior, historical best, then PBT.
- Constant/adaptive schedules are supported by the external release. Linear schedules are rejected rather than silently reinterpreted.

Task IDs, reward functions, observation order, physics, curriculum definitions, initialization contracts, and checkpoint schemas are not changed by this migration. Warm-start and policy-prior initialization remain distinct from full checkpoint resume.

## Existing and remote experiments

No saved checkpoint or workspace conversion is performed. Existing running processes continue using their already loaded code. This change does not deploy to remote machines, restart populations, or migrate source snapshots automatically. Before using new code on another machine, install the pinned dependency into that machine's validated environment and verify a copied checkpoint/workspace.

## Validation

Migration regressions check installed-package delegation, import safety, legacy defaults and paths, parameter precedence, warm-start/prior behavior, checkpoint state, and completed-episode selection.

Validated locally on 2026-09-27 with the unchanged Isaac environment: Isaac Lab 3.0.0rc1, Isaac Sim 6.1.0.0, Torch 2.11.0+cu128, RL-Games 1.6.5, NumPy 2.3.1, and PyYAML 6.0.3. The installed framework's `direct_url.json` resolves the public tag to the release commit above.

- **220 tests and 7 subtests passed** across migration, PBT, RNG, best history, task objectives, curriculum, warm-start and policy-prior suites. Expected TorchScript deprecations and a legacy RNG-recovery warning remain.
- The framework's **three actual-process CPU tests passed in the Isaac Python environment**, including donor model/optimizer inheritance, real re-exec, fresh evidence, terminal budget, and child cleanup.
- Final-code native PhysX Dexsuite training with four environments reached **epoch 2 / 128 transitions**, then full companion-based resume reached **epoch 3 / 192 transitions**. Terminal schema-v2 checkpoints retained optimizer and `env_state['dexpbt']`; model tensors were finite. Output: `/tmp/leap-async-pbt-validation-au2o67o8/`.
- Two concurrent native Dexsuite members, four environments each, reached **epoch 24 / 1,536 transitions per member**. Each published four completed native episodes with `objective_valid=true`, separate identities, and the legacy shared-workspace path. Output: `/tmp/leap-async-pbt-validation-7sm80gs8/pair/`.
- Changed-file Black, isort, flake8, and `git diff --check` passed.

The two-member Isaac check disabled selection, so it proves shared publication and native task measurements, not an Isaac-process exploitation event. Actual re-exec was checked in the framework's CPU fixture; LEAP-specific restore/default/path contracts are covered by migration regressions. Tiny simulator checks are not learning-quality evidence or blanket validation of other tasks/backends. Temporary outputs are local validation artifacts, not committed training results.

Reproduce the simulator-free matrix from the repository root:

```bash
python -m pytest -q tests/test_dexpbt*.py tests/test_pbt*.py \
  tests/test_async_pbt_migration.py tests/test_allegro_pbt.py \
  tests/test_contact_pbt.py tests/test_grasp_pbt.py \
  tests/test_grasp_motion_pbt.py tests/test_grasp_warm_start.py
```

The native checks use `scripts/rl_games/train_dexpbt_member.py` with task `Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0`, `--headless --device cuda:0 --num_envs 4`, horizon 16, minibatch 64, one mini-epoch, a fresh `pbt.directory`, `pbt.interval_steps=64`, `pbt.min_completed_episodes=1`, and initial diversity/selection disabled. Resume keeps the same task, architecture, member identity and environment count and passes both `--checkpoint` and its `--pbt-resume` companion; `--max_iterations` is the absolute epoch budget.
