# Next PBT improvements — 2026-09-25

The initial assessment below was read-only. The approved local implementation is recorded at the end; no remote training, evaluation, deployment, commit, or push was performed. The user has chosen PBT as the preferred framework; improvements remain hypotheses rather than assumed performance gains.

## Completed experiment evidence

Read-only inspection of `/root/allegro_pbt_20260925/{pbt,control}` on westb compared native completed-goal publications at the same nominal budget, 33,554,432 transitions/member. These are **online training-window measurements**, not held-out evaluations of frozen checkpoints.

| Final publication metric | Four-member PBT population | Four independent controls |
|---|---:|---:|
| Mean across members, goals/episode | 10.7791 | 6.7242 |
| Best member | 10.9995 | 10.8031 |
| Worst member | 10.5276 | 2.0629 |

Pooling each member's final three windows by its episode count gives member means of 10.7383 versus 6.5999. Best pooled member scores are 10.9743 versus 10.6408. Control seed45 outperformed PBT seed45. The strongest evidence is therefore **reliability and rescue of weak initial configurations**, not a large improvement over the strongest independent policy. The four PBT members are interacting relatives, not four independent PBT replications.

There were 45 exploit events: recipients0/1/2/3 had18/8/14/5; initial leader3 supplied23 of45 donations. Logged inter-segment gaps sum to1684seconds, including restart/initialization/first rollout—not isolated startup time or directly attributable GPU-hours. GPU sharing and changing contention prevent a clean PBT overhead estimate from elapsed times alone.

### Newly identified terminal-budget defect

Member1 exploited at exactly33,554,432steps and then ran `epoch:2049/2048`, writing a terminal checkpoint with `rew_-inf` because no episode had finished after restart. Publication64 is the clean nominal-budget artifact; the later last checkpoint is a different, unscored policy. Local current `pbt.py` has no max-epoch/frame guard before exploitation. Fix before another run: final publication/checkpoint allowed, final-budget exploitation forbidden; resume at/above budget must stop without another update. Also consider preventing a late exploit when less than one adaptation period remains. The latter is a method choice, separate from strict budget correctness.

## Already fixed locally

Commits bf7d42a/c398d68 include recipient-owned persistent mutation RNG and historical/branch-best checkpoint protection. They were **not deployed to the completed run**. Do not recommend these as unfinished fixes or infer corrected behavior from that old run. Historical best tracks raw reward; it is not automatically the held-out native-goal champion. Model, normalizers and optimizer transfer already exist.

## Recommended order

### 1. Make selection harder to trigger on a small lucky difference

Current selection compares the highest eligible donor using `max(.001,1×combined standard error)` in native goals/episode. Repeated selection of a maximum across peers is not a controlled significance test. Pilot **2SE instead of1SE**, changing no other method settings; report useful improvements per replacement, not only number of replacements. If necessary, next require superiority across two fresh nonoverlapping donor/recipient windows. A 2SE threshold is still heuristic under dependent episodes and repeated testing.

Donor maturity is currently separate from recipient adaptation: a new branch may become a donor before its96-rollout recipient protection expires. Consider requiring64rollouts of branch maturity for donor eligibility; test separately, as this delays dissemination of genuinely useful discoveries.

### 2. Determine whether mutation adds value beyond copying

Run matched-budget conditions with the corrected infrastructure: independent PPO, copying without hyperparameter mutation, and full PBT. A fourth useful baseline runs multiple seeds with the strongest initial configuration from the completed comparison. This tests whether much of the benefit came from simply propagating seed45's configuration/early policy.

**Do not assume `mutation_rate=0` means copy-only:** the current mutation helper forces at least one change. Copy-only requires an explicit tested no-mutation mode.

For mutation itself, first pilot probability .33 rather than1.0 per parameter, keeping factor range[1.1,1.5]. With forced-at-least-one behavior this changes about1.29 of3 parameters per exploit away from bounds. Test gentler factors[1.05,1.2] later and separately. Changing the factor range also affects initialization diversity; preserve the actual starting configurations in such comparisons.

Learning rate and adaptive KL target both control update aggressiveness. An eventual ablation can evolve KL/gradient clipping while inheriting, but not separately perturbing, adaptive LR. Mutation LR bounds are not continuous scheduler bounds: the installed scheduler can move beyond the mutation interval.

### 3. Preserve diversity while rescuing weak branches

The current highest-donor rule rapidly spread one early strong lineage. That helped consistency, but frequent replacement of near-leading branches can narrow search. Pilot replacing only the worst member and leaving the strongest member untouched; if two leaders qualify independently, selecting among them can retain more ancestry. Four members are too few to copy larger-paper percentile splits mechanically. Measure lineage diversity and whether mutated children beat their parents under a fixed evaluation protocol.

### 4. Reduce replacement cost only after measuring it

Continue recording raw rewards honestly and mark the first19rollouts after cold restart as a distinct measurement period. These are differently distributed training samples, not automatically wasted transitions. Current adaptation96rollouts already exceeds this period.

A cadence-only pilot can change publications/checks32→64rollouts, keeping adaptation unchanged. If startup cost remains material, implement in-process transplant at a clean PPO update boundary, initially resetting environments and clearing old-policy rollout/metric buffers. This saves Kit startup without also attempting physics-state transplantation. Keeping in-flight environments is a later experiment because episodes would mix policies.

### 5. Evaluate frozen policies and replicate populations

Evaluate nominal-budget final checkpoints under held-out seeds, identical environment settings and episode budgets. Select candidates using a validation set, then report an untouched test set; do not choose and evaluate a champion on the same data. Report mean, best and worst policy, learning-curve area, time to a goal-count threshold, equal-transition cost, and wall-clock cost including restarts. More evaluation episodes cannot replace independent training populations.

Keep final native-goal score as the scientific metric even if reward shaping later evolves. Compare historical candidate search only with equal candidate allowances in controls.

## Later options, not immediate priorities

- Evolve one or two reward-shaping magnitudes only after selection/mutation ablations, leaving native success criteria and evaluation unchanged. This follows DexPBT's separation of training reward and task meta-objective, but would be a new MDP experiment.
- Consider PB2 (surrogate-guided hyperparameter proposals) for a small population only after ordinary PBT baselines and score reliability are established.
- Do not enlarge the model/population, change exploration bounds, and alter rewards simultaneously. That would obscure which intervention helped.

## Primary sources

1. Jaderberg et al., *Population Based Training of Neural Networks* (2017): https://arxiv.org/abs/1711.09846 — asynchronous copying/tuning, selection variants and small-population limitations.
2. Petrenko et al., *DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population Based Training* (2023): https://arxiv.org/abs/2305.12127 — top/bottom selection, independent parameter mutation, reward evolution, warm-up/adaptation. Paper-scale timings/population sizes are not validated defaults for our shorter run.
3. Parker-Holder et al., *Provably Efficient Online Hyperparameter Optimization with Population-Based Bandits* (2020): https://arxiv.org/abs/2002.02518 — GP-guided proposals and small-population experiments, not a guarantee for this task.

## Local and remote evidence anchors

- `scripts/launch/allegro_pbt_experiment.py`
- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt/pbt/{pbt,allegro,mutation,runtime,best_checkpoint}.py`
- `docs/research/2026-09-25-allegro-pbt-experiment.json`
- `docs/research/2026-09-25-pbt-rng-best-checkpoint-fixes.md`
- Remote `/root/allegro_pbt_20260925/{pbt,control}/member_*.log`
- Remote `{pbt,control}/logs/rl_games/allegro_hand/pbt_workspace/{000..003}/0000000000{62,63,64}.yaml`
- Remote `.../member_*/summaries/`, read with all scalar samples and restart event files included.

All numerical proposed settings are pilot hypotheses, not measured optima.

## Approved local implementation

### Budget and checkpoint contract

The post-update observer checks actual `algo.frame` / `epoch_num` against native
`max_frames` / `max_epochs` (`-1` remains unlimited). It publishes an unpruned
`pbt_workspace/<member>/final.pth` plus atomic `final.yaml` at the terminal update,
including off-cadence budgets, and does not select/mutate/re-exec. Native final
saves still run. An instance-local public `train()` guard runs after Runner restore,
preventing even init/reset/update when a restored checkpoint is already complete.
The regression executes the installed rl-games continuous loop, not a copied loop.
The launcher pins both limits to equal, rollout-aligned budgets. Arbitrary nonaligned
`max_frames` outside this launcher retains native full-rollout rounding; the heldout
helper rejects overshot artifacts rather than calling them exact-budget finals.
A separate late-exploit/adaptation-remaining restriction is NOT silently enabled.

### Selection, mutation and diversity controls

Allegro's resolved defaults are 2SE, mutation probability .33, factor range [1.1,1.5].
Other task objectives retain their former defaults. Four new typed controls are:

| PBT key | Default | Meaning |
|---|---|---|
| `copy_only` | false | Copy effective donor parameters exactly; bypass mutation, clamping and RNG draws |
| `evidence_windows` | 1 | Optional 2 requires fresh nonoverlapping donor AND recipient windows on unchanged branches |
| `recipient_role` | `any` | `weakest` ranks only comparable fresh windows; strongest/tied-all population protected |
| `donor_maturity_rollouts` | 0 | Optional 64 uses donor rollout size and `frame - last_exploit_frame`; initial donors also pass training warm-up |

Recipient-owned evidence is persisted in publications and cleared on replacement.
Reusing the same donor record cannot supply a second evidence window. Donor age is
not recipient age: latest source publications remain usable across asynchronous
lifetime-frame differences; score/window wall time and source-relative sample age
still gate eligibility. Recipient cooldown stays 96 rollouts in every pilot.
`[pbt-selection]` JSON records acceptance/rejection reason, sample counts, score gap,
combined SE and margin when measurable. These are heuristic margins, not controlled
significance tests. Maturity/weakest roles are optional experiments, not new proven defaults.

Allegro initial diversity uses the fixed seeded [1.1,1.5] initialization recipe,
independent of copy-only, mutation probability and subsequent perturbation range.
Dedicated persisted recipient RNG and historical/branch raw-reward best protection
are retained. `mutation_rate=0` still has forced-at-least-one behavior; only explicit
`copy_only=true` means no mutation. Smaller late factors remain a separate future
pilot (range is configurable; no unvalidated time schedule was added).

### Isolated launch pilots

`scripts/launch/allegro_pbt_experiment.py --root <new-root> --arms <arms...>` generates
plans/scripts only; `--launch` is explicit. Use an existing code snapshot in
`<new-root>/code` and the intended `--python`/`--deps`. Generated manifests include
seeds, GPU assignment, exact transition/epoch budgets and final checkpoint paths.

| Arm | Difference from corrected baseline |
|---|---|
| `baseline` | Prior 1SE, mutation 1.0, 32-rollout cadence, corrected RNG/best/budget infrastructure |
| `control` | Selection disabled; same realized initial population |
| `selective` | 2SE only |
| `copy-only` | No mutation; `--gate-se 2` (default) pairs with selective; `--gate-se 1` pairs with baseline |
| `gentler` | Mutation probability .33 only |
| `diversity` | Weakest recipients only |
| `maturity` | 64-rollout donor maturity only |
| `evidence` | Two-window evidence only |
| `cadence` | 64-rollout publication/check cadence only |

Every arm keeps 192-rollout warm-up, 96-rollout adaptation, [1.1,1.5] factors and
fixed initial diversity. Default 1024 envs × 16 steps × 2048 epochs = 33,554,432
transitions/member. Seeds and GPU assignment are paired across arms. Launch at most
two arms together; use equal GPU allocation/contention and a separate root for each
paired pilot. `--seed-base` supports independent population replications. Do not pool
interacting members as independent populations. Two GPU ids default to 0/1; `--gpus`
selects the intended devices. None of these pilots has been launched by this change.

### Heldout finals and profiling roadmap

`scripts/rl_games/evaluate_allegro_population.py --manifest <experiment.json> --split
validation|test` preflights every final's actual torch counters and companion metadata,
then prints serial evaluation jobs. Only `--run` executes them. Defaults are validation
seeds 1042/1043 and untouched test seeds 2042/2043; training/validation/test seed sets
must be disjoint. All final members get identical requested episode/env budgets;
no test-based champion selection is performed. Reports are never overwritten and
checkpoint hashes are checked after evaluation. Candidate selection beyond these
predeclared finals must be done on validation data with equal candidate allowances;
do not use test outcomes to tune settings. Frozen-policy runs were not executed here.

`[pbt-replacement]` records parent/child lineage, changed parameters and checkpoint
copy time. `[pbt-profile]` distinguishes process-to-first-update, observer-to-first-update
and replacement-to-first-update wall time. These include startup/first rollout/update
and contention, NOT isolated Kit cost. `pbt/cold_first_19` and `pbt/segment_rollout`
label the first 19 rollouts, with text transitions at 1/19/20. Raw reward statistics
and samples are unchanged; the label does not remove data, assert wasted samples,
or establish a performance gain. Measure these costs before an in-process transplant
at an update boundary. That rewrite is explicitly deferred; retaining environments
across policy swaps remains an even later experiment.

Local validation includes an eight-environment headless Allegro measurement smoke:
forced native goals counted 8, observation shape [8,124], finite step/reward checks
passed. Unit tests cover native loop terminal behavior, copy-only/RNG persistence,
selection windows/roles/maturity, isolated plans and heldout budget/seed preflight.
Final local suite: **110 tests + 7 subtests passed** (one expected legacy-RNG warning
and 14 upstream TorchScript deprecations). Focused isort, Black at 120 columns,
flake8, compilation of 11 touched Python files, and `git diff --check` passed.
Independent review found an evidence-expiry extension; a failing boundary regression
was added and the fix now retains the oldest original sample/publication timestamp.
No performance improvement is inferred from these correctness tests.
