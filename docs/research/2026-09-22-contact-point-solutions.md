# Contact-Point Task at Zero Success: Curricula, Assignment Freezing, and Reachability Resets

**Date:** 2026-09-22
**Scope:** Why `Isaac-Contact-Cube-LEAP-Direct-v0` (`tasks/leap_hand_repose/leap_contact_direct_env.py`) has an exactly-zero success rate after 200M+ env steps despite solved retention, and what the literature says to change. Runners: FlashSAC (200M steps) and GRU-PPO, episode length stable ~287/300 under reward v3.

**Provenance convention.** Three evidence layers, kept distinct throughout:
- **[code]** read in this repo during this session (`leap_contact_direct_env.py`, `leap_contact_direct_env_cfg.py`).
- **[abs✓]** spot-checked by me against the arXiv abstract page / GitHub README on 2026-09-22 — confirms the paper exists and its top-level claims only.
- **[full✓]** verified against full text (ar5iv/arXiv-HTML), repo code, or rendered PDF by the verification pass whose verdicts accompanied this report; quotes below are from that pass.
Claims that FAILED verification are marked **[failed✗]** and stated as such — two rationale claims in the top recommendation did not survive, and are flagged where they occur.

## 1. Problem statement

The task replaces the repose orientation goal with 4 contact points sampled uniformly on the cube's surface (all 6 faces, min pairwise separation 2.5 cm, cube-local frame, resampled on success). A point is TOUCHED when its one-to-one matched pulp is within 1.5 cm AND reports > 0.1 N of cube-filtered contact force; success = all 4 touched continuously for 0.1–0.5 s (+250, points resample). Code anchors, all read this session:

| Mechanism | Where |
|---|---|
| Uniform sampling over all 6 faces (incl. palm-facing) | `leap_contact_direct_env.py:212-226` (`torch.randint(0, 6, ...)` at :215) |
| 24-permutation argmin one-to-one assignment, recomputed EVERY step | `leap_contact_direct_env.py:298-303` |
| Touch test: distance AND force gate; `touch_force <= 0` = geometry-only | `leap_contact_direct_env.py:310-314`; cfg `touch_distance`/`touch_force` at `leap_contact_direct_env_cfg.py:35-36` |
| Hard all-4 gate + native hold timer | `leap_contact_direct_env.py:315-319`; `hold_time_range = (0.1, 0.5)` at cfg :41 |
| Linear unbounded distance cost −0.05·Σ matched dist | `leap_contact_direct_env.py:436,450`; cfg :43-47 (with the 0.083 drop-exploit break-even rule) |
| v3 retention: +0.1 alive, −0.25 drift, −50 fall | cfg :51-57; `reset_rot_scale = 0.0` (axis-aligned spawn) at :55 |
| Obs: 184-d = 136 repose + 4×4 tip→point matrix, point order fixed per goal | `leap_contact_direct_env.py:333-341`; cfg :31 |

**Failure signature.** Reward v3 solved retention (episode length 284-289/300, positive returns, drop exploit gone — per this repo's AGENTS.md contact-point section), but successes are EXACTLY ZERO after 200M+ env steps for both optimizers (numbers from the research brief that commissioned this report; not re-measured here). The policy holds and repositions the cube but never gets all four matched pulps onto their assigned points simultaneously.

**Suspected causes.** (a) the simultaneous 4-contact configuration is too rare under uniform exploration; (b) NON-STATIONARY per-finger targets — the argmin assignment is recomputed every step, so which pulp "owns" which point can flip whenever distances cross; (c) the force gate; (d) palm-facing points are reachable only by regrasping/reorienting the cube.

**Convergent diagnosis (from 23 merged findings across three research lenses).** The task fixes every difficulty knob at its hardest setting from step 0 — a 4-of-4 AND gate at 1.5 cm + 0.1 N + 0.1-0.5 s hold, uniform goals over all 6 faces, per-step re-assignment, linear unbounded distance costs — while every published system that solved an analogous problem annealed at least one of those. On suspicion (b) specifically: no source directly studies per-step Hungarian/argmin churn (both lenses searched; disclosed honestly), but no successful system found recomputes a hard one-to-one assignment every step either.

## 2. Ranked recommendations (with verification status)

### 2.1 Success-gate curriculum: start k-of-4, loose tolerance, no force gate, zero hold — anneal toward the spec gate on measured success — **HOLDS**

**Changes.**
- New cfg knobs in `leap_contact_direct_env_cfg.py`: `required_touch_points: int = 4`, `touch_distance_start: float = 0.04`, `hold_time_range_start: tuple = (0.0, 0.0)`, `touch_force_start: float = 0.0` — the geometry-only force mode already exists (`leap_contact_direct_env.py:311-314` handles `touch_force <= 0`).
- Replace the hard-coded all-4 test at `leap_contact_direct_env.py:315` with `>= required_touch_points_cur` so partial-credit success (k=1..4) fires, resamples points, and populates `successes` — the signal that is currently exactly zero.
- Success-keyed annealer: per-env (or global) `touch_distance_cur`/`required_points_cur`/`hold_target_cur`, tightened ×0.9 toward (0.015 m, 4, (0.1, 0.5) s, 0.1 N) when a rolling success window exceeds a threshold; the env already maintains a `consecutive_successes` EMA (`leap_contact_direct_env.py:467-475`) to drive it.
- Scale `alive_bonus_scale` (cfg :51) and `cube_pull_scale` (cfg :53) ~10× lower at curriculum start, ramping with the success EMA.
- Tie `reset_rot_scale` (cfg :55, currently 0.0) to the same annealer.
- Optional: promote the two parallel contact-physx remote runs to a poor-man's PBT population (shared checkpoint dir, periodic tournament, mutate reward scales + PPO lr).

**Evidence and verification.** DexPBT [abs✓] is PBT for dexterous manipulation (regrasping, grasp-and-throw, reorientation, Isaac Gym; abstract: PBT "significantly outperforms regular end-to-end learning"); [full✓] its full text gives the annealing rule verbatim — `ε ← max(0.9ε, ε*)`, checked when `N_succ > 3`, 7.5 cm → 1 cm — and states that "without PBT none of the eight single-GPU training sessions reached the target tolerance ε*". RotateIt [full✓] Appendix B states verbatim the "only learn to stably hold the objects" failure at full-strength penalties, fixed by ramping from 0 — our exact symptom class. mjlab [abs✓ exists: LEAP-hand in-hand cube rotation on MjLab, HORA-style rewards, README: "Penalties scale with rotation progress"; full✓] drives every penalty via `reward_weight_by_metric_progress` (curriculums.py:37) from exactly 10×-smaller weights over progress 0.05→0.25 (env_cfg:308-361). Touch Dexterity [full✓] confirms binary touch at a low threshold suffices as the contact criterion.
**Failed rationale claims [failed✗]:** (1) HORA (2210.04887) contains NO phase-A/B staging or deferred hard gate — reward weights fixed, fall termination active from step 0; (2) AnyRotate does NOT tighten tolerance 0.25→0.15 — it LOOSENS `d_tol` during student training and its ablation shows tighter hurts (its keypoint reward, 50.0 fall penalty, and ungated contact shaping do match). Minor: mjlab's 3-of-4-saturation contact term exists with the quoted docstring but is commented out in the shipped cfg; the Rubik's/ADR claim in this recommendation had no cited source (the Rubik's mechanism itself IS verified, under 2.3). These flaws are in supporting color, not the load-bearing evidence (DexPBT + RotateIt + mjlab).

**Effort/impact.** Medium / highest single lever: converts the exactly-zero success signal into a graded one that fires from step 0, giving both optimizers a gradient toward the 4-contact manifold and an automatic path to the spec gate.

### 2.2 Freeze the pulp→point assignment at goal resample (fix the non-stationary 24-perm argmin) — **HOLDS (implement with option 3)**

**Changes.**
- Compute the 24-perm argmin ONCE per goal: set a `goal_fresh` flag per env in `_reset_target_pose` (`leap_contact_direct_env.py:245-255`); in `_compute_intermediate_values` run the argmin (:298-303) only for fresh-goal envs, store `frozen_tip_for_point (N,4)`, gather `matched_dist`/`matched_force` with frozen indices thereafter. Zero obs change — the policy already sees the full 4×4 matrix (:333-341), only the reward-side owner is stabilized.
- If per-step recompute is kept, add hysteresis (accept a new permutation only on ≥20% total-cost improvement).
- **Implement WITH option 3** (verdict's requirement): keep the exact per-step argmin ONLY inside the final success confirmation — frozen assignment for shaping streams, exact for the all-4-hold test — so a bad frozen assignment cannot corrupt the success gate.
- Diagnostic: log per-env assignment-switch count per goal episode; if frozen assignments drastically change touched-rate at the loose curriculum level, churn was a real contributor (currently unmeasurable).

**Evidence and verification.** [full✓] D-Grasp (2112.03028): contact reward parameterized by a FIXED per-episode grasp label; the Table-4 ablation removing it gives success 0.0 vs 0.89. RECIPE (2502.20396 [abs✓ exists — "a generalized reward formulation based on contact and object goals"]; full✓): fingertips scored to the NEAREST contact marker, `1/(1+α·min-dist)`, no Hungarian/thresholds/force gate. CordViP (2502.08449, a 16-DOF LEAP-hand pipeline [full✓]): dense soft contact-map correspondence, deliberately avoiding discrete one-to-one assignment; the ablation quote "significantly reduces success rates" is verbatim (real but modest: avg 90→84, Flip-Cup 85→75). DHER ([full✓] via github.com/mengf1/DHER — the OpenReview forum page itself is bot-gated) is the canonical RL treatment for goals that change mid-episode. Honest caveats: no source directly studies per-step matching churn (disclosed), and "cannot hurt" is mildly overstated — which is exactly why option 3 is required.

**Effort/impact.** Small / removes the suspected moving-target noise in every reward stream at near-zero cost; the diagnostic settles suspicion (b) with data.

### 2.3 Goal-region curriculum: sample point sets on pulp-exposed faces first, widen toward palm-facing regions on success — **HOLDS**

**Changes.**
- New cfg `goal_palm_face_prob: float = 0.0`: in `_draw_face_points` (`leap_contact_direct_env.py:212-226`), sample palm-facing faces with probability p (0 = exposed/side faces only — at `reset_rot_scale = 0` the palm axis in cube-local frame is fixed and known; 1 = today's uniform).
- Drive p with the SAME success-keyed annealer as 2.1.
- Optional second mode per AnyRotate: `goal_sampling_mode = "uniform" | "exposed" | "incremental"`, where "incremental" resamples the next goal by rotating the CURRENT contact configuration by a small angle (~30°, AnyRotate's ablated optimum), chaining goals from achieved states.
- Log success split by whether the goal set contains palm-facing points — the measured difficulty signal, and the direct test of suspected cause (d).

**Evidence and verification.** [full✓] Rubik's cube (1910.07113 [abs✓ exists; ADR = "ever-increasing difficulty"]): Appendix C.1 Algorithm 4 alternates single face-rotations and whole-cube flips, one subgoal class per goal, with ADR-style boundary expansion gated on performance thresholds — the claim's substance holds ("regrasp-class"/"fine-motion" is the recommender's paraphrase of the paper's "flip"/"face rotation"). POISE (2609.13761 [full✓ abstract verbatim]): palm-relative 6D goal reaching with adaptive goal curriculum + diverse grasp resets, 6.2%→59.5% full-range success. AnyRotate [full✓]: incremental goal chaining, Appendix L ablation 30° best (5.26 successful goals vs 4.36/3.86 at 40°/50°). GoalGAN [full✓]: intermediate-difficulty band GOID, Rmin=0.1, Rmax=0.9 — uniformly-sampled goals fed to a zero-success policy are all "too hard".

**Effort/impact.** Small / directly attacks suspected cause (d): master goals reachable in the current grasp first; widen into the regrasp-required region only when it is the frontier.

### 2.4 Reshape the per-point streams: saturating distance kernel or best-so-far progress, plus a graded (not gated) force bonus — **HOLDS**

**Changes.**
- Replace the linear drain `point_dist * point_dist_reward_scale` (`leap_contact_direct_env.py:436,450`; cfg :43) with EITHER (a) a bounded saturating kernel `+w_k · Σ_i 1/(1 + α·d_i)` per matched pulp (RECIPE's exact form; same inverse-error family as the official Isaac Lab in-hand orientation term `1/(dθ+eps)`), OR (b) a best-so-far progress term: per-env per-point min-distance buffer, reward `c · ReLU(prev_min − new_min)` so each unit of approach is paid exactly once.
- Graded contact bonus in SHAPING: `+w_f · min(‖f_matched,i‖, f_cap) · 1(d_i < touch_distance)` per point; keep the hard distance-AND-force gate (:310-314) ONLY inside the success test.
- Optional unintended-contact penalty per CHORD (matched-pulp force NOT near any goal point), so palm-crushing cannot ride the graded force bonus.
- Kernels (a)/(b) are bounded/pay-once → stay under the 0.083 drop-exploit break-even by construction rather than by hand-tuned scale (cfg :43-47).

**Evidence and verification.** [full✓] RECIPE: per-fingertip `1/(1+α·d)` nearest-sticker kernel, no thresholds/force gating. DextrAH-G (2407.02274): "stateful function minimize(e)=max(e_smallest−e,0)" — exactly the best-so-far mechanism. CHORD (2607.00033): 4,739-task benchmark, 1,831 evaluated, 82.12% success, trains WITH FlashSAC, `exp(−penalty/var)` tolerance-band wrench reward; ablation Position-Only 0.862 vs 0.918 AUC — the graded-vs-gated contact supervision numbers, exact. AnyRotate: `r_kp = d/(e^{ax}+b+e^{−ax})` verbatim, contact quality (≥2 tips) as a shaped non-gate term (binary in count, not force-graded — nuance). Ng et al. 1999: the PDF's text layer is broken (Type-3 fonts), so the pass rendered page 1 to PNG and read it visually — abstract verbatim confirms potential-difference shaping preserves the optimal policy (our raw per-step cost is NOT potential-based). Minor unverified bits, none substantive: RECIPE's "CoRL 2025" venue label (search results only), an incidental Dactyl `d_t − d_{t+1}` mention not among the listed sources.

**Effort/impact.** Small / smooth bounded gradient toward each point at all distances (not just inside the 1.5 cm gate) and converts the force gate from a learning obstacle into a shaped signal — provably without re-opening the drop exploit.

### 2.5 Hindsight/reachability resets: start a fraction of episodes ON the 4-contact manifold — **HOLDS**

**Changes.**
- New cfg `hindsight_reset_fraction: float = 0.0` (target ~0.3): on that fraction of `_reset_idx` (`leap_contact_direct_env.py:257-278`), after the standard reset, (a) script a settle — drive the 16 joints toward a stored power-grasp target ~0.5 s (in-repo precedent: the curl scripting in `scripts/probes/leap_repose_direct_newton_probe.py:36-38,125-127`), then (b) overwrite `contact_points_local` by transforming current pulp positions into the cube frame and projecting onto the nearest surface point (rejection-enforce the 2.5 cm separation, cap at `max_point_resamples`).
- Optional precompute: a `scripts/assets/collect_contact_grasp_cache.py` settling several grasp states to .npz (mjlab's cache pattern).
- Anneal the fraction DOWN as curriculum success rises; randomize which stable grasp per reset.
- Ordering: `_sample_contact_points` for hindsight envs runs AFTER the settle; reset `hold_timer`/`goal_fresh` in `_reset_target_pose`.

**Evidence and verification.** [abs✓ + full✓] Khandate et al. (2303.03486): my abstract spot-check confirms the core mechanism verbatim — "we use states found via sampling-based exploration to generate reset distributions that enable training control policies"; [full✓] the full text adds that fixed-init and self-exploration baselines never learned finger gaiting, with 10^5-node trees, a ≥10^4-node ablation, and ~2×10^4 reset states. MyoChallenge Baoding winners [full✓]: Reference State Initialization along the target trajectory + curriculum, "RSI was only needed at slower speeds" (matches the anneal-down change); caveat: the fetched text does not explicitly call the Baoding reward an "AND-gate". mjlab [full✓ via repo code read by the pass]: grasp-cache resets verbatim — `reset_from_grasp_cache` event, .npz cache, `scale_list=(0.95,0.9,1.0,1.05,1.1)`, `scale_jitter=0.025`, collect script with 240 settle steps. POISE [full✓]: diverse stable-grasp initialization, 40.1→51.5% held-out success. HER (1707.01495) [abs✓ — I fetched the abstract directly: "sample-efficient learning from rewards which are sparse and binary", "combined with an arbitrary off-policy RL algorithm", "a form of implicit curriculum"].

**Effort/impact.** Medium / the strongest exploration lever: puts the policy physically inside the 4-contact region every episode so it first learns to HOLD the configuration that never occurs under uniform exploration — the documented unlock for exactly our never-encountered-success structure, and the prerequisite that makes the other curricula trainable. "Near-goal" for Khandate is a loose paraphrase (their resets are task-relevant largest-rotation path nodes).

### 2.6 HER for the FlashSAC arm: relabel replay goals with achieved contact points, prioritized by contact energy — **HOLDS (FlashSAC-only, largest effort)**

**Changes.**
- In the flash_rl clone's replay path: on insertion, when ≥1 pulp reports > `touch_force` of cube-filtered force (the env exposes `fingertip_contact_forces (N,4,3)` per step), compute that pulp's cube-frame contact location and store it as an achieved-goal candidate; on minibatch sampling, relabel a fraction HER-"future"-style with 4-point goal sets from achieved locations plus rejection-sampled fillers meeting the separation constraint.
- CEBP-style priority weighted by a contact-energy proxy Σ‖f_i‖·step_dt × TD-error.
- Scope: FlashSAC runs only (GRU-PPO is on-policy and skips this unchanged).
- Cheaper first step: reuse the same achieved-goal bookkeeping to drive the hindsight RESETS of 2.5 — most of HER's benefit without touching flash_rl internals.

**Evidence and verification.** [abs✓] HER: verified directly (see 2.5). CEBP (2312.02677, ICRA 2024 [full✓; venue via search]) does prioritize HER replay using touch-sensor forces — but its actual priority is EPISODE-LEVEL contact energy `Σ(f_left+f_right)·‖Δd‖` with TD-error only in PER-style IS weights, NOT the "TD-error × contact energy" product the recommendation states; the `Σ‖f_i‖·step_dt` proxy is the recommender's own (defensible) adaptation, since CEBP's displacement term reads ~0 for a held cube. DHER (openreview Byf5-30qFX [full✓]) exists and is on-topic (HER degrading under time-varying goals — relevant to the per-step argmin churn) but is never cited in the rationale. Two omissions flagged by the verdict: "touched holds by construction" is strictly true only at timesteps where all four pulps report contact (filler points are untouched by definition), and goal-dependent shaping streams (the −0.05 matched-distance term) must also be recomputed at relabel time.

**Effort/impact.** Large / FlashSAC-only: converts 200M steps of "failed" episodes already in the buffer into dense near-boundary supervision for the goal-conditioned critics.

## 3. Appendix: other findings (compact)

| Work | What it shows [all full✓ unless noted] | Lever for this task |
|---|---|---|
| [NeuralTouch](https://arxiv.org/abs/2510.20390) (2025) | NDF-style neural contact descriptors + tactile servoing; zero-shot real transfer | If obs redesign is ever on the table: smooth surface descriptors instead of the raw 48-d tip→point vectors — architecture change, not a fix for zero-success |
| [GenDexGrasp](https://arxiv.org/abs/2210.00722) (2023) | Generative contact maps → hand-specific grasp realization | Feasibility screen for `_sample_contact_points`: our 2.5 cm-separated uniform sets may be unsatisfiable by ANY 16-DOF grasp of a 6 cm cube — no reward change can fix that |
| [Contact Map Transfer w/ diffusion](https://arxiv.org/abs/2511.01276) (2025) | Diffusion contact maps + grasp-recovery stage that discards unreliable points and re-optimizes | Template for a repair stage: resample points conditioned on current cube-vs-palm orientation so a valid finger configuration exists by construction during curriculum |
| [Contact-Grounded Policy](https://arxiv.org/abs/2603.05687) (2026) | Diffusion contact forecasting → compliance controller (Allegro V5 + Digit360) | Supports a decaying tolerance zone wider than the 1.5 cm gate; true impedance control out of scope for our position-controlled hand |
| [Reward-Machine-Guided Self-Paced RL](https://proceedings.mlr.press/v216/koprulu23a.html) (UAI 2023) | Reward machines in policy + self-paced context updates; up to 75% shorter curricula | The 4-bit touched-vector is a natural reward-machine state — an alternative scheduler for the 2.1 knobs if the plain success-threshold annealer oscillates |
| [Task-Oriented Grasping w/ Contextual Reward Machine](https://arxiv.org/abs/2512.10235) (2025) | Per-subtask rewards, action spaces, abstractions; 95% over 1,000 sim tasks | Stage decomposition for the "all 4 simultaneously" conjunction; suggests contact-freezing (damp joints of an already-touching pulp) once k≥2 exists |
| [Touch-based Curiosity](https://arxiv.org/abs/2104.00442) (2021) | Prediction error on force/torque feedback as intrinsic reward; beats curiosity baselines on sparse contact tasks | Cheap exploration bonus on exactly the 12 tactile channels we already observe — concentrated on novel finger-cube configurations incl. regrasps |

## 4. Proposed next experiments (in order)

**Experiment 0 — feasibility probe (hours, no training).** Before any curriculum: script a settle-grasp with the existing curl probe (`scripts/probes/leap_repose_direct_newton_probe.py:36-38,125-127`) against sampled point sets and measure how many points a realized static grasp can touch. Settles whether the uniform+2.5 cm sampler emits structurally unsatisfiable sets (the GenDexGrasp concern) — if even scripted power grasps touch ≤2 points of typical sets, no reward change will ever reach 4, and the sampler needs the repair stage first.

**Experiment 1 — freeze assignment + k-of-4 gate curriculum (one training run).** Ship 2.2 (with its option 3: frozen assignment for shaping, exact per-step argmin for the success test) and 2.1 (start at k=1..2, `touch_distance` 0.04, force gate off, hold 0; anneal ×0.9 on a rolling success window via the existing `consecutive_successes` EMA) together with both diagnostics: the assignment-churn counter and success-split-by-palm-face. These are mutually reinforcing and both cheap; success criterion = nonzero `successes` and annealer progression beyond k=2 within a budget comparable to the 200M steps already spent at zero.

**Experiment 2 — goal-region curriculum + kernel shaping (if Experiment 1 stalls before k=4).** Add 2.3's `goal_palm_face_prob` on the same annealer (palm-facing success is the measured frontier), and swap the linear distance stream for 2.4's saturating `1/(1+α·d)` kernel (bounded by construction, so the 0.083 break-even rule of cfg :43-47 needs no re-tuning). Keep the graded force bonus out unless the force gate itself proves to be the blocker at the tightened levels.

**Experiment 3 — hindsight resets (contingency if successes remain zero at loose levels).** Implement 2.5 at `hindsight_reset_fraction ≈ 0.3` with the settle-then-project point sampling, annealing down as success rises; the achieved-goal bookkeeping should also feed Experiment 1's diagnostics. This is the strongest exploration lever and the documented unlock for never-visited-success structures; it is ordered last only because it is the most invasive change to the reset path.

Not scheduled: 2.6 (replay surgery in the flash_rl clone) — revisit only if Experiment 3 lifts FlashSAC above zero but sample efficiency becomes the binding constraint.
