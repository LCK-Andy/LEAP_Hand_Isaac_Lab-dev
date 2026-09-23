# DexPBT acquisition and evaluation changes — 2026-09-24

This implementation addresses the mechanisms identified in [the strategy assessment](2026-09-24-dexpbt-training-strategy.md). It is a DexPBT-inspired experiment on Dexsuite, not an exact reproduction of the paper.

## Operating rules

- Keep the existing remote runs and their checkpoints unchanged until an explicit deployment/relaunch decision. A local code change does not update an already running Python process.
- Use a **new population directory** for a changed scoring protocol. Never compare old soft-reward scores with the new completed-episode metrics.
- Initially evolve only learning rate, gradient clipping and KL target within configured bounds. Entropy remains zero; reward-weight evolution is deliberately deferred until the fixed evaluation demonstrates acquisition.
- A `fixed_sigma` RL-Games policy has learned, state-independent log standard deviation. New policies start at normalized-action std 0.3; std is bounded to [0.05, 1.0], including after checkpoint restore and each optimizer update. These are initial experimental settings, not established optimal values. The action scale and PPO minibatch settings are unchanged.
- In-training metrics use fixed success criteria but are explicitly labeled **training-condition metrics**. They are not held-out evaluation. PBT compares only compatible condition groups with sufficiently populated, fresh completed-episode windows. Episodes spanning condition changes do not qualify for these comparisons.
- The independent evaluator freezes difficulty and tolerances, does not evolve or clamp the saved policy, and excludes the checkpoint's training curriculum from restoration.

## Independent checkpoint evaluation

Run from the repository root, in the Isaac Lab environment. The checkpoint must be a trusted file produced by the training system; loading it uses Python serialization.

```bash
/home/andy/env_isaaclab/bin/python scripts/rl_games/evaluate_dexpbt.py \
  --checkpoint '/absolute/path/to/model.pth' \
  --num_envs 64 --episodes 512 --seed 1042 --adr_fraction 0.0 \
  --headless --output '/tmp/dexpbt-acquisition-eval.json'
```

Use `--adr_fraction 1.0` and a different report filename to evaluate at full difficulty. Use `--stochastic` for a separate assessment of the actual noisy policy; default evaluation uses the policy mean. Do not conflate these two modes. On the rental machine, replace the Python path with `/root/autodl-tmp/env_isaaclab/bin/python`.

For comparisons, hold environment count, seed, difficulty, action mode and episode budget fixed. Use several fixed seeds before selecting a winner. The report includes checkpoint SHA-256, evaluation settings, completed episodes, actual success rate, timestep-averaged pose errors aggregated per episode, contact fraction and episode length. The final vector step may slightly exceed the requested number of episodes; the report records the actual denominator. Existing reports are never overwritten. If the step cap or application exit prevents collecting the requested sample, the report is marked incomplete and the process fails.

Success requires the thumb and another finger in contact (>0.01 N), position error <0.1 m, and orientation error <0.5 rad at least once during an episode. This is not a sustained hold or consecutive-goal-completion metric.

## Diagnostic comparison before a long extension

Run a bounded four-condition acquisition comparison with exploitation disabled: current versus controlled exploration, crossed with inherited versus corrected curriculum. Keep all other settings identical. Repeat the promising comparison across additional seeds. Compare successful episodes per environment step and per wall-clock hour, not total shaped return or GPU utilization alone.

Restoring an old high-noise policy into bounded training intentionally changes its exploration. Preserve the old checkpoint and compare against it with the independent evaluator, which does not clamp it. A fresh-start comparison answers a different question from repairing an existing policy.

## Launching a new population

After deploying the files, use a new workspace rather than resuming the old soft-reward-scored population. Environment-variable prefixes below set launcher options for that command only.

```bash
PYTHON=/root/autodl-tmp/env_isaaclab/bin/python \
  BASE=/root/dexpbt_reorient_v2 POP=4 ENVS=512 \
  scripts/launch/dexpbt_reorient.sh
```

This command has **not** been run on the remote machine. Existing `dexpbt-*` tmux sessions are not replaced automatically. `DRY_RUN=1` previews the launch; `RESUME=1` loads each member's checkpoint and companion metadata from the selected population directory. Forward `pbt.enabled=false` for exploitation-disabled diagnostics. CLI `--sigma` retains RL-Games' **raw log-standard-deviation** semantics, not standard deviation; prefer the explicit `params.config.exploration.*` controls.

Fresh members start with bounded diversity in the three mutable hyperparameters; member zero keeps the baseline. New defaults use a 5M-step warm-up and 3M-step adaptation period after exploitation, with 1M-step publication cadence and at least 128 completed episodes per comparison window. These timings are starting settings, not tuned optima. Selection is success-first; pose progress only distinguishes zero-success members.

Physical condition matching uses actual gravity/noise parameter values, not inactive ADR scheduler bookkeeping below the native interpolation threshold. Genuine changes still exclude mixed-condition episodes. If members diverge into incompatible physical conditions, online PBT deliberately declines comparisons; use the independent fixed-difficulty evaluator rather than lowering this guard. Held-out reports are separate artifacts, not automatically ingested into online tournaments.

## Validation

- Simulator smoke: eight environments; actual initial tolerance, fixed ADR 1 then 0, finite completed-episode metrics, and curriculum-state round-trip passed.
- PPO integration: three updates at eight environments; model/optimizer/curriculum checkpoints and PBT metadata produced.
- Real checkpoint resume: compiled checkpoint restored into the pre-compile model, advanced from epoch 3 to 4 with monotonic frame counts, and effective optimizer learning rate matched published metadata after adaptive scheduling.
- Independent evaluator: the eight-environment checkpoint evaluated at four environments, fixed ADR 0, for four completed 360-step episodes; valid JSON report with policy hash and fixed conditions. This nearly untrained policy had zero successes, as expected; no learning improvement is claimed.
- Exploration: the fresh smoke policy stayed near std 0.3; its last logged rollout had sampled-action clipping fraction approximately 0.048. This is a small integration sample, not a performance benchmark.
- Disabled-PBT diagnostic resume: the same compiled checkpoint resumed successfully with exploitation disabled, without a PBT workspace/publication.
- Simulator-free regressions: 42 tests plus seven subtests passed, covering forced tournament/restart transactions, mutation/config overrides, disabled selection, actual Gaussian exploration/optimizer hooks, episode accounting, curriculum state, retention, and physical-condition identity. The forced tournament test intercepts process exec; it is not a claim of a production multi-process simulator tournament.
- Black (120 columns), flake8, Python compilation, launcher syntax and population dry-run checked. Pyright is unavailable in the installed environment.

The four-condition learning study and production population relaunch have not been run. Proposed experiments above are not claims of improved task success.
