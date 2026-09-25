# Mamba and state-space models in reinforcement learning — evidence and fit for LEAP

**Date:** 2026-09-25  
**Question:** Is a Mamba/state-space policy a justified next step beyond our PPO GRU-256 and FlashSAC MLP?  
**Scope:** Primary papers and author-linked code; research only. No training, dependency changes, or implementation.

## Verdict

**Mamba is a credible memory architecture, but not an evidence-backed general upgrade to GRU or MLP for dexterous manipulation.** The strongest positive case is efficient learning from *long, genuinely informative histories*. The evidence is much weaker for short-horizon continuous control, and weaker still for online off-policy SAC with contact-rich hands.

Three distinctions determine how to read the literature:

1. **Offline Decision Mamba is not recurrent PPO/SAC.** Predicting actions from return-conditioned demonstration trajectories tests a different learning problem from discovering a grasp online.
2. **DRAMA is a real Mamba-2 world-model agent.** Its results support compact learned dynamics, not replacing a FlashSAC actor with Mamba. It does not beat larger DreamerV3/STORM on aggregate Atari100k performance.
3. **There is direct online PPO evidence**, particularly S5 on POMDPs and a later Mamba/Mamba-2 benchmark. It is mixed, and the latter has important training-state/reset caveats.

**Recommendation:** Keep GRU-256 as the recurrent reference and MLP/history-stacking as controls. A small, correctness-gated Mamba-2 or S5 comparison is warranted *if* controlled experiments demonstrate a memory bottleneck—especially adaptation to hidden mass/COM/inertia. Do not migrate FlashSAC or build a world model on the strength of Decision Mamba results alone.

## Search and evidence boundaries

This is a targeted literature review, not a systematic meta-analysis. Searches covered Decision Mamba, MambaDreamer, Mamba world models, PPO/POMDP selective SSMs, S5 in-context RL, recurrent off-policy baselines, and negative/short-context findings. Primary arXiv text and metadata, conference proceedings, and author-linked repositories were used for substantive claims. Search snippets were discovery aids, not numerical evidence.

The reporting cutoff is **2026-09-25**, but accessible search results are not an exhaustive index through that date. Some searches returned truncated/root-domain links or irrelevant results; OpenReview's R2D2 page returned a browser-verification screen. Publication dates below distinguish first submissions from revisions. No post-cutoff work is used; a newer-looking search timestamp is not proof of a newer experiment. Failure to locate a source means **unverified in this search**, not nonexistent.

Numbers are **author-reported**, not locally reproduced. Comparisons within a table retain that paper's metric; normalized returns, raw rewards, prediction errors, and steps/second must not be pooled across papers. GitHub `main` was inspected as accessible during this review, not pinned to the exact experimental commit. Code observations therefore identify reproducibility concerns, not a retrospective proof of how every published run executed.

## 1. What Mamba changes—and what it does not

### Mamba-1 versus Mamba-2

**Mamba-1** is Gu and Dao's *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, first submitted **2023-12-01** [1]. A simplified selective recurrence is

`h_t = A_bar(x_t) h_(t-1) + B_bar(x_t) x_t; y_t = C(x_t) h_t`.

The actual block adds input projections, a causal convolution, gating, and an output projection. Input-dependent selection lets the model retain or discard information; it is not merely a fixed linear filter. Unlike a GRU's nonlinear hidden-to-hidden update, the affine recurrence permits an efficient parallel scan over a known input sequence.

**Mamba-2** is Dao and Gu's *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*, first submitted **2024-05-31** [2]. Its structured state-space duality (SSD) formulation reorganizes the selective SSM into hardware-efficient, chunked computations with more matrix-multiplication-friendly structure. It changes the architecture and computation, not just a version number. The reported **2–8×** acceleration is for its core layer relative to Mamba-1—not an Isaac Lab, PPO, SAC, or whole-training guarantee.

**S4/S5 are related, not synonyms for Mamba.** S5 predates input-selective Mamba and provides a useful independent test of whether a parallelizable recurrent state-space model helps RL. Positive S5 evidence cannot automatically be attributed to Mamba's selectivity.

### Parallel training is not parallel environment interaction

With a stored trajectory, sequence positions can participate in a parallel scan. This can make long-sequence backpropagation substantially faster than sequential RNN unrolling. During interaction, however, the next observation depends on the action just taken: the environment-policy loop remains sequential in time. Vectorized environments parallelize the **batch**, not causally dependent timesteps.

Recurrent inference has a cache whose size is constant in *history length*, unlike an ever-growing Transformer KV cache. But **constant does not mean small**. The official Mamba-1 cache contains both convolution and SSM states [3]. For one block with `d_model=256`, expansion 2, `d_state=16`, and convolution width 4, that is

`512 × (16 + 4) = 10,240` values per environment,

versus 256 values for a one-layer GRU-256. At 10,240 environments, the illustrative FP32 cache is about **400 MiB per block**, versus **10 MiB** for that GRU hidden state; separate actor/critic copies and additional blocks multiply the difference. This is an architectural calculation, **not a measured candidate policy footprint**. Matching hidden width does not match parameters, state size, or compute.

Short sequences, narrow proprioceptive inputs, kernel-launch overhead, convolution/cache updates, precision, and GPU occupancy can erase scan advantages. Measure single-environment deployment latency separately from thousands-of-environments throughput. The language-model headline of 5× Transformer inference throughput [1] is not a GRU comparison.

## 2. Offline sequence modeling: useful results, wrong direct comparison for PPO

Several different papers use “Decision Mamba.” They must not be conflated.

### 2.1 Ota's Decision Mamba: mixed results from a relatively direct substitution

Toshihiro Ota, *Decision Mamba: Reinforcement Learning via Sequence Modeling with Selective State Spaces*, **2024-03-29**, arXiv:2403.19925 [4]. Author code: [Toshihiro-Ota/decision-mamba](https://github.com/Toshihiro-Ota/decision-mamba).

This replaces Decision Transformer's attention with Mamba in a return/state/action token sequence. Training is supervised action prediction on offline trajectories, not PPO or a Bellman-trained SAC policy. A history of K environment steps becomes 3K tokens.

Selected D4RL expert-normalized returns, five Decision Mamba seeds:

| Dataset | Decision Transformer | Decision ConvFormer | Decision Mamba |
|---|---:|---:|---:|
| HalfCheetah medium | 42.6 | 43.0 | 42.8 ± 0.08 |
| Hopper medium | 68.4 | 92.5 | 83.5 ± 12.5 |
| Hopper medium-replay | 85.6 | 94.2 | 82.6 ± 4.6 |
| Walker medium-replay | 71.2 | 76.6 | 70.9 ± 4.3 |
| Hopper medium-expert | 109.6 | 110.4 | 111.1 ± 0.3 |

Atari uses **500,000 offline transitions**, 1% of a DQN replay dataset, and three seeds. Gamer-normalized scores include Breakout **239.2 ± 26.4** versus DT **242.4 ± 31.8**, and Pong **63.2 ± 102.1** versus DT **105.6 ± 2.9**. Decision ConvFormer exceeds Mamba on all four reported games.

Longer history is not uniformly beneficial: Qbert falls from **56.4 ± 16.6 at K=10** to **15.7 ± 5.5 at K=60**. Baseline numbers are taken from the ConvFormer paper rather than a fully controlled rerun. The paper explicitly does **not** establish an efficiency gain from simple substitution; CPU/GPU interaction remains a bottleneck.

**Implication:** Mamba can work, but neither longer memory nor architecture novelty guarantees better control.

### 2.2 Huang et al.'s hybrid: strong long-context results, not online gradient learning

Sili Huang et al., *Decision Mamba: Reinforcement Learning via Hybrid Selective Sequence Modeling*, first submitted **2024-05-31**, NeurIPS 2024 [5]. Author-linked code: [SiliHuang-ai/hybrid_mamba](https://github.com/SiliHuang-ai/hybrid_mamba). The current README uses “Hybrid Mamba”/“HDM”; the paper uses DM-H. The paper-to-code link establishes provenance, but the small visible repository listing does not establish complete benchmark reproduction.

DM-H uses a Mamba history model to provide intermediate-goal information to a short-window Transformer. It trains on offline trajectories, then adapts in context during evaluation **without gradient updates**. “Online evaluation” is not online PPO training.

The reported nine-task D4RL mean is **87.53**, versus **84.83 TD3+BC**, **82.84 AD–Transformer**, and **77.69 DT**. TD3+BC still exceeds DM-H on all three HalfCheetah datasets. Its method changes both backbone and conditioning, so gains cannot be assigned solely to Mamba.

The often-quoted **28×** is long-context **evaluation time** against AD–Transformer. On HalfCheetah, ten evaluation episodes take **173.11 versus 6.21 minutes**, with a 4,000-step context. AD–Mamba takes **6.16 minutes**: the hybrid is not 28× faster than a recurrent baseline. Corresponding training times for 10,000 updates are **37.10 versus 20.96 hours**, a much smaller difference. Tmaze tests extend to 20,000-step contexts for the hybrid; exact plotted returns are not transcribed here.

### 2.3 DeMa: a particularly relevant warning about short windows

Yang Dai et al., *Is Mamba Compatible with Trajectory Optimization in Offline Reinforcement Learning*, arXiv:2405.12094; numerical evidence here is **v2, 2024-10-27** [6]. Code: [AndssY/DeMa](https://github.com/AndssY/DeMa).

The successful DeMa configuration uses short fixed windows and parallel scanning rather than carrying recurrent state over whole episodes. Reported means:

| Comparison | DT | Windowed DeMa | Recurrent DeMa |
|---|---:|---:|---:|
| Eight-game Atari | 62.2 | 111.8 | 67.3 |
| Nine-task MuJoCo | 76.4 | 78.5 | Not the same task subset |

The separate six-task MuJoCo recurrent comparison is **61.1 recurrent versus 66.0 windowed**. Recurrent Atari runs use only one seed, so the recurrent disadvantage is suggestive rather than a precise population estimate.

Atari training is **50 ms/step versus DT's 55**, but the listed MuJoCo configurations show **57.6/58.8 versus 56/58 ms**—no speed advantage. Smaller parameter counts do not guarantee less memory or lower runtime. This is evidence against assuming full-episode recurrence is intrinsically superior, not evidence that Mamba generally fails.

### 2.4 A third Decision Mamba, and a search lead not promoted to evidence

Qi Lv et al., *Decision Mamba: A Multi-Grained State Space Model with Self-Evolution Regularization for Offline RL*, first submitted **2024-06-08**, revised **2025-01-22**, NeurIPS 2024 [7], is another distinct offline method. Code: [aopolin-lv/DecisionMamba](https://github.com/aopolin-lv/DecisionMamba). It adds multi-grained processing and self-evolution regularization. Its existence/method are verified, but its numerical tables were not extracted for this review; it is not counted as an independent PPO success.

A search also surfaced a 2025 repository/thesis-index item titled *Online Decision Mamba*. A sufficiently stable primary paper/code pair was not verified here, so no effectiveness claim relies on it. Even a verified online return-conditioned method would need to be distinguished from recurrent model-free actor–critic learning.

## 3. Directly relevant evidence: online recurrent PPO and POMDPs

### 3.1 S5: the clearest positive case for memory-demanding RL

Chris Lu et al., *Structured State Space Models for In-Context Reinforcement Learning*, first submitted **2023-03-07**, revised **2023-11-23**, NeurIPS 2023 [8]. Author repositories: [s5rl](https://github.com/luchris429/s5rl) and the PureJAX implementation [popjaxrl](https://github.com/luchris429/popjaxrl).

A key contribution is **associative handling of resets and carried initial hidden states**, not merely swapping the neural layer. This matters because RL training batches contain episode boundaries.

For five hard POPGym environments, their online PPO implementation reports the following **maximum epoch-average episodic rewards**, mean ± SD over eight seeds:

| Task | S5 | GRU | Memory-free MLP |
|---|---:|---:|---:|
| Stateless CartPole | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.26 ± 0.00 |
| Noisy Stateless CartPole | 0.28 ± 0.00 | 0.27 ± 0.00 | 0.22 ± 0.00 |
| Stateless Pendulum | 0.79 ± 0.01 | 0.75 ± 0.00 | 0.41 ± 0.02 |
| Noisy Stateless Pendulum | 0.55 ± 0.01 | 0.61 ± 0.01 | 0.34 ± 0.01 |
| Repeat Previous | **0.91 ± 0.01** | **−0.46 ± 0.01** | **−0.48 ± 0.00** |

The delayed-repetition result is substantial; the noisy Pendulum result favors GRU. “SSM beats RNN” is therefore too broad.

These runs use 15M steps, 64 environments, 1,024-step rollouts, and approximately parameter-matched architectures; the S5 is four layers wide 256 and the GRU width is 256. The paper reports roughly **5–6× faster** training than its GRU implementation, with slightly different wording across sections. PureJAX compilation and long unrolls differ materially from PyTorch Isaac Lab. Its 33-minute-versus-22-hour comparison with an earlier benchmark additionally changes hardware and software and must not be treated as an architecture-only speedup.

Other experiments include memory-length tasks and meta-RL with randomized observation/action projections. The multi-task DMControl experiments use **Muesli, not PPO**, and much larger compute. Some held-out tasks transfer; pendulum swingup fails. These support learned temporal adaptation, not demonstrated physical inertial estimation on a hand.

### 3.2 RLBenchNet: Mamba-2 helps one memory task, but is not a universal controller

Ivan Smirnov and Shangding Gu, *RLBenchNet: The Right Network for the Right Reinforcement Learning Task*, **2025-05-21**, arXiv:2505.15040v1 [9]. The accessible arXiv record does not establish a conference acceptance. Author code resolves to [SafeRL-Lab/BenchNetRL](https://github.com/SafeRL-Lab/BenchNetRL).

All models use PPO, making this much closer to our architecture question. Parameter counts are approximately matched; GRU/LSTM have one recurrent layer. CartPole/LunarLander hide velocities, and MiniGrid visibility is reduced. Selected final-return mean ± SD:

| Task | MLP, 1 observation | GRU | Mamba-1 | Mamba-2 |
|---|---:|---:|---:|---:|
| MemoryS11 | Not reported | 0.49 ± 0.02 | 0.49 ± 0.03 | **0.96 ± 0.01** |
| Masked CartPole | 49.4 ± 2.1 | 449.4 ± 152.8 | 272.8 ± 104.7 | 434.2 ± 90.5 |
| Masked LunarLander | 16.6 ± 28.9 | **204.8 ± 23.2** | −313.2 ± 235.1 | 42.5 ± 182.2 |
| HalfCheetah | 3918.8 ± 312.8 | 2745.0 ± 1118.1 | 2238.7 ± 627.2 | 3718.0 ± 528.5 |
| Hopper | 1194.1 ± 200.4 | 1676.6 ± 724.5 | 1409.2 ± 635.7 | 1390.2 ± 639.3 |
| Walker2d | **3379.3 ± 1039.8** | 3170.3 ± 355.4 | 2512.6 ± 1000.5 | 3038.0 ± 505.2 |

MemoryS11 trains for 15M steps; MuJoCo and masked classic-control tasks for 5M. Most use eight seeds; classic control uses ten. GTrXL obtains **0.93 ± 0.11** on MemoryS11, showing that the task rewards memory rather than uniquely validating Mamba. Four-observation PPO also provides an important cheap baseline: it scores **458.7 ± 28.9** on masked CartPole and **372.7 ± 20.3** on Breakout, above Mamba-2's **239.2 ± 39.5** on the latter.

**Compute is split:** on HalfCheetah, the profiling table reports **2,429 steps/s Mamba-2 versus 259 GRU**, but inference is **1.263 versus 0.747 ms**. Thus the faster overall training implementation has *slower policy inference*. It uses only eight parallel MuJoCo environments and 2,048-step rollouts, unlike our thousands-of-environments/shorter-unroll setting. No direct speed transfer is justified.

#### Important implementation qualification

The paper itself calls for proper Mamba training-state resets. Inspection of its current `ppo_mamba.py` [10] shows:

- Rollout collection clears both Mamba cache components at episode termination/truncation.
- Optimization processes whole rollout sequences without equivalent episode-boundary masks.
- Saved initial states are put into an inference cache with `seqlen_offset=0`.

In the current official Mamba-1 and Mamba-2 wrappers [3], that zero-offset full-sequence path **does not pass the existing cache as the scan's initial state**; it overwrites the cache after scanning. The code paths inspected therefore raise both a boundary-reset mismatch and an initial-state mismatch. Version pinning and a minimal equivalence test are required before reproducing or reusing this implementation. This is not a complete audit or proof that a specific archived run had identical code.

**Weight of evidence:** encouraging for Mamba-2 on a long-memory benchmark, but insufficient to conclude it is a better online continuous-control backbone than a correctly implemented, tuned GRU.

## 4. World models: DRAMA is real; “MambaDreamer” was not verified

### DRAMA

Wenlong Wang, Ivana Dusparic, Yucheng Shi, Ke Zhang, and Vinny Cahill, *Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient*, first submitted **2024-10-11**, ICLR 2025; latest checked **v4, 2025-05-16** [11]. Code: [realwenlongwang/Drama](https://github.com/realwenlongwang/Drama).

DRAMA uses Mamba-2 in a learned world model and trains behavior through model-based learning. The repository offers Transformer/Mamba/Mamba-2 dynamics and actor–critic/PPO behavior options. This does **not** isolate the effect of a Mamba actor under ordinary model-free PPO. Diversified replay sampling (DFS) is another algorithmic ingredient.

Atari100k: 26 games, 100,000 environment interactions (400,000 frames with action repeat), five DRAMA training seeds. Reported human-normalized aggregates:

| Agent/configuration | Mean | Median |
|---|---:|---:|
| DRAMA XS + DFS | 105% | 27% |
| DRAMA XS + uniform replay | 80% | 28% |
| DreamerV3 XS | 37% | 7% |
| DreamerV3 | 125% | 49% |
| STORM | 127% | 58% |
| IRIS | 105% | 29% |
| TWM | 96% | 51% |

**Honest interpretation:** compact DRAMA is competitive, especially against the small Dreamer configuration, but does not beat full DreamerV3/STORM. DFS improves the mean while slightly reducing the median. Mean-only reporting hides uneven game-level gains.

The paper gives **7M world-model parameters**, while an appendix figure labels DRAMA XS **10M**; those counting scopes are not explicitly reconciled. Do not compare 7M world-model parameters against another method's entire agent without checking scope.

A separate grid-world dynamics test is more informative about sequence efficiency than about policy returns:

| Architecture | Sequence length | Training step | Reported prediction error |
|---|---:|---:|---:|
| Mamba-2 | 208 | 25 ms | 15.6 ± 2.6% |
| GRU | 208 | 75 ms | 21.3 ± 0.3% |
| Mamba-2 | 1,664 | 214 ms | 14.2 ± 0.3% |
| GRU | 1,664 | 628 ms | 34.7 ± 25.4% |

The Transformer runs out of memory at the longer setting on the reported 8GB GPU. This error metric measures grid/movement consistency, not task success. Timing experiments include an RTX 2000 Ada Mobile laptop GPU; the final appendix acknowledges short-context Mamba-2 overhead. There is no verified end-to-end “DRAMA is X× faster than Dreamer” number here.

### Name collision: MAMBA meta-RL is not selective Mamba

Zohar Rimon et al., *MAMBA: an Effective World Model Approach for Meta-Reinforcement Learning*, **2024-03-14**, ICLR 2024 [12], expands MAMBA as **MetA-RL Model-Based Algorithm**. It is a DreamerV3-based recurrent state-space world-model method, not Gu/Dao's selective SSM architecture. Official code: [zoharri/mamba](https://github.com/zoharri/mamba).

An exact search for **“MambaDreamer”** did not verify a corresponding primary paper/code pair. It should not be cited as established evidence. DRAMA and the unrelated MAMBA acronym are the verified nearby sources.

## 5. Recurrent alternatives deserve a fair baseline

Tianwei Ni, Benjamin Eysenbach, and Ruslan Salakhutdinov, *Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs*, ICML 2022 [13], reports improved efficiency/performance in **18 of 21 environments** against methods from six earlier specialized approaches. This is not a Mamba comparison. Its lesson is that weak recurrent baselines can produce misleading architecture conclusions.

The author repository [pomdp-baselines](https://github.com/twni2016/pomdp-baselines) includes recurrent SAC and TD3. Its implementation notes recommend separate actor/critic networks, support GRU/LSTM, and allow observation/action/reward history and tunable sampled sequence lengths. This is a more relevant starting reference for recurrent FlashSAC semantics than offline Decision Transformer code.

Useful comparison set:

- **MLP:** cheapest; may already observe enough current state.
- **MLP + short history:** tests whether missing velocity, delay, or short contact transients explain the deficit.
- **GRU-256/LSTM:** mature recurrent controls; training sequence length and state handling matter as much as cell choice.
- **S5:** compelling online-POMDP evidence and explicit reset-aware scans, but porting a JAX research result to our stack is not free.
- **Mamba-2:** worthwhile selective-SSM candidate if long memory is demonstrated.
- **GTrXL:** useful long-memory reference, but likely unnecessary as the first high-cost hand experiment.

## 6. Correctness requirements before any architectural conclusion

These are experimental requirements, **not an implementation request**.

### PPO and truncated backpropagation

1. Maintain separate state for each vectorized environment and reset **both convolution and SSM memory** on actual episode resets. State does not reset merely because a rollout buffer fills.
2. Train on contiguous sequences with correct initial states and boundary masks. Randomly shuffling individual transitions destroys the intended memory problem.
3. Detaching the initial state at a truncated-BPTT boundary limits the **gradient horizon**, not necessarily the inference memory horizon. A model can carry a long history while receiving only short-range credit for learning what to preserve.
4. Require sequence-versus-step numerical agreement, including nonzero initial states, internal resets, padding, and batch reordering. An inference cache is not automatically a differentiable training-state interface.
5. Keep termination and time-limit semantics distinct. Reset memory when the environment resets, but bootstrap a time-limit value from the correct final observation/state rather than the auto-reset observation. A value-only query must not silently advance the live actor cache.
6. Repeated PPO epochs and changed weights can make cached states stale. Match the chosen recomputation/carry strategy across GRU and SSM rather than giving only one model fresh history.

A useful timing distinction is **physics episode reset versus goal resampling**. For in-hand manipulation, an object can retain its inertials while the orientation/contact goal changes. Clearing all physical memory on every new goal would discard useful identification. Conversely, retaining it after a new object/inertial draw creates cross-episode contamination. Meta-RL across episodes is legitimate only when explicitly designed and signaled.

### FlashSAC and recurrent replay

A feed-forward replay tuple is not sufficient to train an arbitrary recurrent policy correctly. A controlled recurrent SAC variant needs episode-aware contiguous replay, history inputs, masks, and state reconstruction for the actor, critics, and target calculations.

**Burn-in** runs a prefix through the current recurrent network, typically without a learning loss and often without gradient tracking, to initialize the learning segment. Alternatives include stored collection-time states or reconstruction from episode start. Stored states become stale as parameters change; burn-in reduces mismatch but cannot guarantee recovery of information older than its prefix. These are known recurrent replay issues, not special benefits of Mamba.

The R2D2 primary record [14] is a relevant follow-up, but its page was access-blocked in this review; no paper-specific R2D2 numerical claim is used. Ni et al.'s accessible recurrent SAC/TD3 code is directly relevant. In FlashSAC, replay sampling, next-state alignment, twin critics, target networks, normalization, and compilation must all remain comparable. **A model-class substitution alone is not a valid recurrent SAC experiment.**

### Kernels and systems

Official Mamba provides specialized scan/convolution kernels and different whole-sequence versus single-step paths [3]. Before interpreting speed, establish which path actually runs, supported CUDA/PyTorch versions, precision, compile behavior, warmup cost, and cache allocation. Do not assume a fallback Python path represents the intended architecture or that language-model generation benchmarks match control.

The accessible DRAMA README, for example, specifies a recent Python 3.12/PyTorch 2.7/CUDA 12.8 installation with pinned Mamba components. That is reproducibility context, **not advice to modify our Isaac environment**. Benchmark in isolation before considering any dependency changes.

## 7. Fit for our 136/184-dimensional LEAP tasks

Local configuration confirms **136-d** direct repose observations, **184-d** contact observations, and separate one-layer **GRU-256 → [512,256]** actor/critic networks in the contact GRU runner. These are structured state/proprioception/object/tactile vectors, **not just raw proprioception or pixels**. Input channel count is not a reason by itself to use a sequence model: time steps, not the 136/184 channels, are the natural sequence axis.

### Most plausible benefit: hidden physical parameters

The manifest-inertial objects task changes mass, COM, and inertia per episode while deliberately withholding that ground truth from the policy. Action-conditioned histories of motion and force could support an implicit belief about these parameters. This is a better scientific motivation for recurrence than “Mamba is newer.”

But memory does not create identifiability. Different COM/inertia/friction/contact configurations can generate similar observations under a static grasp. Excitation, sufficiently informative motion, action history, sensor quality, and the task's reward incentives determine whether useful parameters can be inferred. A latent state predicting successful control need not encode an accurate physical parameter estimate.

Separate **control adaptation** from **identification**: evaluate held-out inertial variants and transient recovery; if studying identification itself, probe frozen hidden states against held-out mass/COM/inertia labels without feeding those labels into the deployed actor. Good probe accuracy alone does not prove the controller uses that information.

### Contact timing may need only short memory

The force channels are impulse-like, actuator smoothing creates temporal dynamics, and the contact task's hold predicate is temporal. A GRU or short history can plausibly handle these. Multi-thousand-step language-model contexts are not automatically relevant. A bounded hidden hold timer can favor memory without requiring full inertial identification.

Task histories in AGENTS.md also document reward-driven dropping and contact-sensor/backend failures. **An SSM cannot repair an exploitative reward or recover contact forces that the backend reports as zero.** Use the validated force-reporting physics path, identical rewards, and identical sensing in all comparisons. Keep the contact-goal task and the manifest-inertial objects task as separate experiments rather than silently changing the MDP to combine them.

### Systems fit is uncertain

Our large vectorized simulations may be dominated by physics/host synchronization, not sequence-network training. Large batches help GPU utilization, but short PPO unrolls reduce the opportunity for scan acceleration; per-environment SSM caches can add significant memory. If simulation dominates, even a faster optimizer may barely change end-to-end throughput.

There is **no verified result in this review showing Mamba beating GRU-256 or MLP on our LEAP tasks, tactile dexterous regrasping, or FlashSAC inertial randomization**. Any expected gain is a hypothesis.

## 8. A bounded experiment is justified—only after a memory diagnostic

### Stage A: establish that memory is the bottleneck

Compare the existing MLP and GRU-256 with a modest history-stacked MLP under identical rewards, sensing, action smoothing, randomization, and compute budgets. Evaluate both a fixed-dynamics control and a hidden-inertial variant. Avoid changing exploration or reward coefficients alongside architecture.

If GRU/history does not improve held-out adaptation or task success, there is little reason to invest in a more complicated recurrent stack. If all methods score zero, first diagnose exploration, reachability, reward balance, and learning signals; zero success is not evidence that a longer-memory model will solve the task.

### Stage B: one SSM candidate, correctness first

Start with **PPO**, not a simultaneous recurrent FlashSAC rewrite. Mamba-2 is the most directly supported selective-SSM candidate; S5 is an alternative if reset-aware, long-history learning is the main research question. Choose one initially.

Keep observation encoders, policy distribution, action scale, actor/critic separation, and PPO tuning budgets comparable. Match parameter counts approximately, then separately report state size and wall-clock cost. Include at least five seeds if feasible; compare equal environment steps **and** equal elapsed time. Sweep only a small, predeclared range of temporal training lengths that covers the suspected memory need—do not interpret one arbitrary unroll as a model verdict.

Prerequisite tests: step/scan agreement, nonzero-state initialization, mid-sequence resets, padding invariance, no state leakage across envs, correct time-limit bootstrapping, finite gradients, and deterministic checkpoint/state restoration.

### Stage C: judge the result with task-relevant metrics

Primary outcomes should be completed-task success/goal rate, continuous hold duration, drop rate, and adaptation to held-out inertials—not merely shaped return. Report median/IQR or confidence intervals across seeds and learning-curve area/sample efficiency, not the best seed/checkpoint alone.

Also report policy inference latency at batch 1 and production env count; collection, optimizer, and simulator time separately; GPU/host memory; and recurrent cache size. History ablation or controlled memory resets can test whether any improvement actually uses long-term information. If probing inertia, separate training and held-out objects/variants.

**Go/no-go:** proceed only if the SSM reproducibly improves success or hidden-dynamics robustness at acceptable wall-clock/memory cost. A benchmark-only optimizer speedup, larger cache, or a return gain without more successful grasps is not sufficient. Recurrent FlashSAC or DRAMA-style world modeling should be later, independently justified research projects.

## Sources and reproducibility pointers

1. Gu & Dao. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** First submitted 2023-12-01. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).
2. Dao & Gu. **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.** First submitted 2024-05-31; ICML 2024. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060).
3. **Official Mamba code:** [state-spaces/mamba](https://github.com/state-spaces/mamba); inspected [Mamba-1 wrapper](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py) and [Mamba-2 wrapper](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py). Current-main observations, not historical commit verification.
4. Ota. **Decision Mamba: Reinforcement Learning via Sequence Modeling with Selective State Spaces.** 2024-03-29. [Primary v1 text](https://arxiv.org/html/2403.19925v1); [code](https://github.com/Toshihiro-Ota/decision-mamba).
5. Huang et al. **Decision Mamba: Reinforcement Learning via Hybrid Selective Sequence Modeling.** First submitted 2024-05-31; NeurIPS 2024. [arXiv text](https://arxiv.org/html/2406.00079); [conference record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/850e8063d902e0825d3c5504d183bafe-Abstract-Conference.html); [author-linked code](https://github.com/SiliHuang-ai/hybrid_mamba). Numerical extraction uses the accessible arXiv text.
6. Dai et al. **Is Mamba Compatible with Trajectory Optimization in Offline Reinforcement Learning.** [Primary v2 text, 2024-10-27](https://arxiv.org/html/2405.12094v2); [code](https://github.com/AndssY/DeMa).
7. Lv et al. **Decision Mamba: A Multi-Grained State Space Model with Self-Evolution Regularization for Offline RL.** First submitted 2024-06-08; NeurIPS 2024. [Metadata](https://arxiv.org/abs/2406.05427); [code](https://github.com/aopolin-lv/DecisionMamba). Metadata/method verification only here.
8. Lu et al. **Structured State Space Models for In-Context Reinforcement Learning.** First submitted 2023-03-07; NeurIPS 2023. [Metadata](https://arxiv.org/abs/2303.03982); [primary text](https://arxiv.org/html/2303.03982); [PureJAX code](https://github.com/luchris429/popjaxrl).
9. Smirnov & Gu. **RLBenchNet: The Right Network for the Right Reinforcement Learning Task.** 2025-05-21. [Primary v1 text](https://arxiv.org/html/2505.15040v1); [metadata](https://arxiv.org/abs/2505.15040).
10. **RLBenchNet author code:** [repository](https://github.com/SafeRL-Lab/BenchNetRL); [inspected PPO Mamba implementation](https://github.com/SafeRL-Lab/BenchNetRL/blob/main/ppo_mamba.py).
11. Wang et al. **Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient.** First submitted 2024-10-11; ICLR 2025. [Version history](https://arxiv.org/abs/2410.08893); [primary v4 text, 2025-05-16](https://arxiv.org/html/2410.08893v4); [code](https://github.com/realwenlongwang/Drama).
12. Rimon et al. **MAMBA: an Effective World Model Approach for Meta-Reinforcement Learning.** 2024-03-14; ICLR 2024. [Primary text](https://arxiv.org/html/2403.09859v1); [conference paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/51053d7b8473df7d5a2165b2a8ee9629-Paper-Conference.pdf); [code](https://github.com/zoharri/mamba).
13. Ni, Eysenbach & Salakhutdinov. **Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs.** ICML 2022. [Proceedings](https://proceedings.mlr.press/v162/ni22a.html); [code](https://github.com/twni2016/pomdp-baselines); [implementation notes](https://github.com/twni2016/pomdp-baselines/blob/main/docs/our_details.md).
14. **Recurrent Experience Replay in Distributed Reinforcement Learning** (R2D2), ICLR 2019. [Primary record](https://openreview.net/forum?id=r1lyTjAqYX). Access-blocked during this review; follow-up reference, not a source for numerical findings here.

### Local context checked

- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/leap_direct_env_cfg.py`: 136-d observations.
- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/leap_contact_direct_env_cfg.py`: 184-d observations.
- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/agents/rsl_rl_ppo_contact_direct_gru_cfg.py`: separate GRU-256 actor and critic, one layer, [512,256] heads.
- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/leap_objects_direct_v1_env_cfg.py`: manifest-backed per-episode inertials.
- `AGENTS.md`: task/backend constraints and prior training observations. These are project context, not externally reproduced research results.

**Bottom line:** The literature justifies testing an SSM for a demonstrated temporal-inference problem. It does not justify treating Mamba as a shortcut around reward design, correct recurrent training, or a strong GRU baseline.
