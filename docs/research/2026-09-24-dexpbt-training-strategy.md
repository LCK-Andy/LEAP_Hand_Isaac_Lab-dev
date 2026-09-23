# DexPBT training strategy assessment — 2026-09-24

## Bottom line and evidence scope

**Do not interpret this run as a failed DexPBT experiment yet.** The observed four members currently behave like identically configured PPO seeds, with an inactive selection gate, defective restart plumbing, and very large Gaussian exploration. First make those mechanisms testable; then compare exploration/curriculum changes under a fixed evaluation protocol. Simply extending the run or copying the paper's population size is not the highest-value next step.

Suggestions only: no training code, remote machine, or running job was changed by this assessment. The requested note did not exist before writing. Local installed sources were inspected without launching Isaac Sim. Remote observations below were supplied by the coordinating agent, not independently collected here; source parity is confirmed only where explicitly noted.

**Supplied live evidence:** four × 512 environments on RTX 4090D, about 9,800 aggregate transitions/s and 87% rollout time; resumed around 190M transitions/member. Recent frames 192–197M: selection scores `[1.965e-7, 0, 0, 0]`; each member's last 50 retained metadata records has exactly the same parameter tuple `(lr=.001, entropy=.005, grad_norm=1, KL=.01)`. Largest scores in those records are .0001699 and .0001632 for members 0/3, zero for 1/2. Approximately 300 logged samples/member since resume have success_rate=0 and ADR=0. Recent mean episode length 341–345/360, timeout fraction .86–.90, position error .71–.77 m, rotation error 2.19–2.23 rad. Contact-count reward .64–.79 dominates; members 1/2 have almost no good-finger-contact/tracking reward. These are current-window observations, **not proof of zero successes throughout the full training history**.

## Source keys (absolute local paths)

Line references below use these prefixes, with normal one-based line numbers.

- **R** = `/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab`
- **T** = `/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab/source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt`
- **D** = `/home/andy/env_isaaclab/lib/python3.12/site-packages/isaaclab_tasks/manager_based/manipulation/dexsuite`
- **I** = `/home/andy/env_isaaclab/lib/python3.12/site-packages/isaaclab/source/isaaclab/isaaclab`
- **W** = `/home/andy/env_isaaclab/lib/python3.12/site-packages/isaaclab_rl/rl_games/rl_games.py`
- **G** = `/home/andy/env_isaaclab/lib/python3.12/site-packages/rl_games`

Primary research sources:

1. [Petrenko et al., DexPBT, RSS 2023, paper](https://arxiv.org/html/2305.12127): success definition, tolerance curriculum, Eq. 5 meta-objective, PBT algorithm and experimental hyperparameters.
2. [Authors' project page](https://sites.google.com/view/dexpbt), which identifies NVIDIA IsaacGymEnvs as the implementation repository.
3. [Original repository PBT observer](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/pbt/pbt.py): completed-environment objective aggregation, initialization mutation, warm-up/adaptation gates, asynchronous comparison and restart handling.
4. [Original tolerance/objective helpers](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/allegro_kuka/allegro_kuka_utils.py).
5. [Original PPO configuration](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/train/AllegroKukaPPO.yaml) and [LSTM override](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/train/AllegroKukaLSTMPPO.yaml).

Repository links are current `main` sources inspected on this date, not a claim that they are an immutable paper-release commit. The paper and current repository differ in some population-management details.

## Prioritized findings and recommendations

### P0 — Repair selection semantics and plumbing before enabling more exploitation

**Confirmed selection blockage for the supplied score range.** `T/agents/rl_games_ppo_pbt.yaml:103–118` selects `episode.Episode_Reward/success`, with absolute margin .001. `T/pbt/pbt.py:150–175` requires an underperformer below `min(mean−.2×std, mean−.001)` and leader above the corresponding upper cut. All reported scores are nonnegative and below .00017: the lower cut is negative, so none can be an underperformer. This explains the lack of parameter diversity without invoking broken checkpoint saving. Lowering only the interval does nothing to fix it.

**The selected quantity is not a success count or success rate.** `D/mdp/rewards.py:140–178` returns a contact-gated product of smooth position/rotation kernels; `I/managers/reward_manager.py:116–125,145–154` logs its weighted, dt-integrated episode sum averaged over resetting environments and divided by the **configured maximum episode duration**, not actual duration. It can be positive without any successful episode, and it changes scale with success weight/tolerance. The observer retains only the latest available reset-batch scalar (`T/pbt/pbt.py:67–91`), not an episode-count-weighted evaluation window. Small batches, stale values and curriculum differences can bias tournaments.

**Recommendation:** log actual completed-episode counts and aggregate fixed-definition success/pose errors over a sufficiently populated window; evaluate at shared difficulty, tolerances and seeds. Use an explicitly normalized, fixed progress tie-breaker while success is identically zero, not raw total reward dominated by contact-count. Once measurable success emerges, prioritize it. Calibrate selection margins against score uncertainty and freshness rather than replacing .001 with an arbitrarily tiny number. Do not mutate success reward weight/tolerance while using that weighted term as the cross-member ranking objective. The paper's Eq. 5 combines **tolerance progress and consecutive successful manipulations**, not soft success reward [1,4]. It is an example of metric/task co-design, not directly compatible with the current MDP.

**Confirmed local implementation defects; coordinator also confirmed the key argument/ordering defects remotely:**

- `R/scripts/rl_games/train_dexpbt_member.py:63–69,277–282` strips `params.` when parsing but applies the remaining path from `agent_cfg`, not `agent_cfg['params']`. An override intended for `params.config.learning_rate` instead creates root `config.learning_rate`; the printed “applied” message does not establish that PPO consumed it.
- The observer snapshots mutation parameters before these overrides (`...train_dexpbt_member.py:243–245,268–286`; `T/pbt/pbt.py:40–44`). Fixing only the root path still leaves stale PBT metadata.
- `...train_dexpbt_member.py:59–87` removes PBT overrides from `sys.argv`; the restart reconstructs from that stripped argv (`T/pbt/pbt.py:189–204`). Member identity/workspace/interval and other launch choices must survive re-exec explicitly. `EnvArgs` restores task/seed/env count/headless, not PBT identity (`T/pbt/pbt_utils.py:33–46`).
- The vendored restart hardcodes a relative `_isaac_sim/python.sh` (`T/pbt/pbt.py:219–252`); locally this resolves to `/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab/source/_isaac_sim/python.sh`, which does not exist. Remote interpreter-path status was not checked here.
- Launcher `POP` is not forwarded as `pbt.num_policies` (`R/scripts/launch/dexpbt_reorient.sh:17–20,40–42`); YAML remains 8. Missing slots are excluded from statistics, so this alone does **not** explain inactivity, but obsolete workspace members can enter comparisons and missing slots affect cleanup (`T/pbt/pbt_utils.py:168–212`). Coordinator confirms remote banner/config still says eight with four active.
- Manual resume selects `.pth` only, not its companion evolved parameter metadata (`R/scripts/launch/dexpbt_reorient.sh:24–41`). Current identical parameters make this latent; after real evolution it matters.

**Acceptance test before any new training budget:** force a short, isolated exploit/restart; verify unique identity/workspace, donor weights, actual runner/scheduler/optimizer parameters, observer metadata, resume parameters and frame accounting agree. Check curriculum/observation-normalizer restoration as part of the same test. This is a proposed test, not performed here.

### P1 — Excessive pre-clipping exploration is measured; its role in stagnation is a strong hypothesis

`fixed_sigma: True` means **state-independent learned log-standard-deviation**, not frozen noise: `G/algos_torch/network_builder.py:293–318` creates a trainable parameter, and `G/algos_torch/models.py:295–299` exponentiates it and sums Gaussian entropy. Initial logstd=0 means std=1 (`T/agents/rl_games_ppo_pbt.yaml:28–45`). Actions are clipped to ±1 (`W:302–304`), then used as relative joint-position changes with scale .1 (`D/config/kuka_allegro/dexsuite_kuka_allegro_env_cfg.py:60–62`). The bounds penalty constrains **means only**, not sigma (`G/algos_torch/a2c_continuous.py:250–257`). Positive entropy reward can therefore keep buying preclip variance with little additional executed-action diversity.

Coordinator's checkpoint inspection: mean coordinate std 6.85, 13.95, 13.22, 7.70; maxima 26.3, 52.8, 53.0, 42.5; median std around 4.2–5.0. From the individual sigmas, the coordinator calculated average `P(|N(0,sigma²)|>1)` of .713, .692, .681, .707. For any Gaussian mean, this is a **lower bound on expected coordinate saturation** because zero mean maximizes mass inside [-1,1]. It is not an actual rollout clipping measurement. Rising logged summed entropy 64–71 is consistent with these parameters.

**Recommendation:** instrument per-joint sigma, preclip/action clipping fraction, mean-action saturation, joint-limit occupancy, thumb-contact duty cycle and actual joint motion. Compare an entropy-zero/reduced-entropy condition and bounded or lower arm-versus-hand noise with the current condition. For an already high-sigma checkpoint, changing entropy alone need not undo accumulated variance quickly; separate an explicit sigma intervention from a clean-start comparison. Do not change action scale simultaneously. The paper's zero entropy coefficient [1,5] makes a useful ablation, **not proof that zero is universally optimal**. Smoothing or temporally correlated exploration are later hypotheses if motion remains destructive.

### P1 — The advertised tolerance curriculum is not initialized and its feedback channel is wrong

`T/dexsuite_reorient_pbt_env_cfg.py:45–59` sets an internal `std=.2` but does not apply it to the reward until three successful windows. Thus the task starts at the inherited **.1 m**, not advertised .2 m; the first hypothetical update would set .18 m and initially make the task easier. Moreover it reads `env.extras['log']['Metrics/success_rate']`, whereas curriculum computes before reward reset emits that metric (`I/envs/manager_based_rl_env.py:369–389`), and the RL-Games wrapper removes `log` via `extras.pop('log')` (`W:321–324`). Under this wrapper the previous-step log is normally unavailable, not a reliable current success signal. These are source-level defects; coordinator confirmed the corresponding remote curriculum code.

**Metric semantics are otherwise real:** `D/mdp/rewards.py:132–138,170–176` records sticky “at least once this episode” success, requiring thumb plus any other finger >.01 N, position error <.1 m and rotation error <.5 rad. It does not count held goals or consecutive target completions. ADR reads the sticky flags directly before reset (`D/mdp/curriculums.py:78–86`), independently corroborating the zero logged success; coordinator confirmed remote parity. A large contact-count reward does not imply the thumb requirement is met (`D/mdp/rewards.py:72–112`).

**Recommendation:** initialize the actual tolerance, drive curriculum from current finished environments' sticky flags with a minimum sample count/window, and record/checkpoint curriculum state. Prefer making position/rotation goal difficulty reachable first rather than just raising a reward weight. Fix one curriculum dimension at a time. Retain fixed-tolerance evaluation so relaxed training successes cannot masquerade as progress. Official ADR already interpolates gravity from zero to −9.81 based on mean difficulty (`D/adr_curriculum.py:16–18,111–120`); at reported difficulty zero, this is already a zero-gravity task, not a full-gravity lift failure. Consider an explicit reach→thumb opposition→transport→orientation progression or narrower initial goal/reset ranges only after inspecting trajectories; those are hypotheses, not established fixes.

### P2 — Restore purposeful PBT diversity and reward evolution after those prerequisites

This is a **DexPBT-inspired PPO wrapper**, not a faithful replication. Original reorientation uses keypoint pose error, consecutive successes, tolerance 7.5→1 cm and progress-based reach/pick/target rewards [1,4]. Current Dexsuite instead uses contact-gated kernels, periodically resampled commands (2–3 s), six-second episodes and a gravity/noise curriculum (`D/dexsuite_env_cfg.py:363–401,489–503`). Its reward design creates a plausible “touch without transport” local optimum given the observed .7 m position error and contact dominance, but trajectories/controlled ablation are needed to demonstrate exploitation.

Local PBT mutates only lr, entropy, grad norm and KL; all initial members share the same values; only classified underperformers mutate (`T/agents/rl_games_ppo_pbt.yaml:113–118`; `T/pbt/pbt.py:160–177`). Four independent 25% mutation coins also give a 31.6% chance of no change. No explicit initialization diversification, initial learning delay or post-mutation adaptation delay exists. The paper used 200M-transition warm-up, 20M ordinary intervals and 50M adaptation, 8–32 members, and reward/PPO evolution [1]; the original repository also aggregates completed-environment metrics and introduces initialization mutation and eligibility gates [3]. At the current supplied throughput a 1M/member interval is approximately 6.8 minutes at equal shares, so process restarts and noisy short-horizon ranking would become significant if the gate were merely lowered.

**Recommendation:** keep four members within the measured hardware budget; explicitly diversify a few bounded settings and preserve an unmodified baseline. Schedule tournaments based on enough new completed episodes, stable evaluation and measured restart cost; preserve recipient training-age accounting when copying older donors. Introduce a small set of bounded reward coefficients only after fixed scoring and override round-trips work. Prioritize transport/pose progress relative to contact-only shaping, rather than unrestricted reward mutation. Prefer progress or potential-based shaping experiments with explicit reset/goal-resample handling over simply amplifying dense contact bonuses.

### P2 — PPO/observation changes are secondary, and current settings do not exactly mirror official PPO

The actor correctly includes policy, proprioception and perception (`T/agents/rl_games_ppo_pbt.yaml:18–23`); **there is no evidence of the missing-proprioception bug seen in the separate FlashSAC port**. Installed state observations contain 5-step histories, 64 object surface points, hand/joint state and contact force (`D/dexsuite_env_cfg.py:165–222`; `D/config/kuka_allegro/camera_cfg.py:193–204`). A point-set encoder or explicit object-position/pose-error features could improve inductive bias, but replacing observations changes the experiment. Verify actual post-wrapper shapes, point transforms and sensitivity to object motion first. `seq_length: 4` does not create recurrence: the local network has no RNN block.

At 512 environments, horizon32 gives 16,384 transitions/update; minibatch1024 × five epochs means **80 minibatch updates/rollout**. Official RSL-RL has four minibatches × five epochs =20, and value-loss coefficient1, not local critic coefficient4 (`D/config/kuka_allegro/agents/rsl_rl_ppo_cfg.py:45–64`; `T/agents/rl_games_ppo_pbt.yaml:84–94`). Cross-library loss normalization must be checked before equating coefficients. A minibatch4096/3–5 epoch comparison is reasonable after exploration is controlled; measure KL, clip fraction, explained variance and success per sample, not just fps. The reported startup KL spikes 2.3–14.9 warrant checking restored optimizer LR/normalizers and first-update handling; they do not by themselves identify a cause.

The paper's 8192 environments, horizon16, minibatch32768, two PPO epochs, large LSTM/MLP and entropy0 [1,5] operated at a different batch/compute scale. Do not copy its batch size into a 16,384-transition rollout. With 87% rollout time, even eliminating all remaining time would yield only about 1.15× throughput; a larger model is not the first remedy. Profile sensors/observation construction/physics only after learning validity is established.

## Proposed next experiment (not launched)

1. Preserve the current checkpoints and evaluation baseline; prove one isolated restart/override/metric round-trip and correct curriculum initialization first.
2. Use the four available slots for a controlled 2×2 comparison: current versus reduced/bounded exploration; inherited versus corrected/meaningful easy-to-hard curriculum. Hold reward weights, action scale and PPO batch settings fixed; disable exploitation during this diagnostic window so conditions remain identifiable. Repeat promising comparisons with additional seeds before treating one member as conclusive.
3. Evaluate all conditions with identical fixed success criteria/difficulty and report successful-episode counts, pose errors, thumb opposition, action saturation and wall-clock/sample efficiency. Also evaluate at current training difficulty to distinguish transfer from acquisition failure.
4. Only then enable calibrated PBT selection and bounded reward evolution; retain a non-evolving baseline. If all four still merely touch, investigate reward/goal geometry and observation correctness before spending another long blind extension.
