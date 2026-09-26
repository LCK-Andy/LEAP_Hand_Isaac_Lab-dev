# Redesigning LEAP contact tasks for inertial identification

Date: 2026-09-25
Scope: assessment of `Isaac-Contact-Cube-LEAP-Direct-v0` and `Isaac-Contact-Cube-LEAP-Direct-GRU-v0`, with the sibling `Estimate_weight_distribution` project as the estimation reference. No task, training, estimator, or playback code was changed. No new simulator rollout or learning experiment was run.

## Executive judgment

The current environments optimize **random contact-point acquisition and repeated regrasping**, not **a persistent, force-regulated, instrumented grasp that produces identifiable object motion**. Their difficulty is therefore partly self-imposed and unrelated to the user's actual objective. Increasing policy capacity or population size is not the first intervention.

The recommended research task is:

> Establish a feasible four-pulp grasp, maintain safe compressive contact without unmeasured support, and move the object through a bounded trajectory that reduces uncertainty in its ten inertial parameters.

Four-pulp contact is a mechanical/sensing requirement of this project, not a mathematical prerequisite for ten-parameter identification. Static holding does not identify the full inertia tensor. If a wrist/arm is available, maintain the grasp and excite the object by moving the wrist; if the hand must remain fixed, start with small in-hand motion and establish that it excites the required modes before attempting general regrasping.

## 1. What the current code actually does

Repository-relative source links below are relative to this note's directory.

### Objective and success

- Four points are sampled over all six cube faces, subject only to pairwise separation. There is no inverse-kinematics, collision-free reachability, opposing-contact, friction-feasibility, or force-closure filter. A point set being geometrically valid does not establish that the hand can hold all four points simultaneously. The exact frequency of infeasible sets has not been measured.
- At each step, all 24 tip-to-point assignments are evaluated; the minimum summed distance assignment determines which fingertip force is checked against each point. Matching is recomputed without persistence and ignores force when selecting the assignment.
- A point is touched if the assigned **fingertip body origin** is within 1.5 cm and its object-filtered normal force exceeds 0.1 N. This is not a measurement of the actual contact location, nor a guarantee that the tactile pulp region rather than another part of that distal body touches the target.
- All four contacts must survive a sampled 0.1–0.5 s hold. Success produces +250 and immediately samples another point set, which may require dismantling an already useful grasp.

Sources: [contact sampling and matching](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/leap_contact_direct_env.py#L211-L313), [hold and goal replacement](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/leap_contact_direct_env.py#L336-L386), [configuration](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/leap_contact_direct_env_cfg.py#L30-L58), [body-origin fingertip positions](../../source/LEAP_Isaaclab/LEAP_Isaaclab/compat/inhand_direct/inhand_manipulation_env.py#L312-L329).

**Interpretation:** this is a defensible dexterity/regrasp benchmark, but it is the wrong primary benchmark for a stable identification grasp. Permutation matching removes an arbitrary finger assignment, but not reachability or load feasibility. It can also change the reward assignment near a tie; this is a possible learning nuisance, not a demonstrated root cause.

### What “firm” currently means

The 0.1 N gate detects contact. There is no target force band, upper force limit, load margin, shear/slip test, pulse rejection, or post-success sustained-contact requirement. The contact task does not use the parent's 3 N goal-press reward or palm-pin terms. In particular, contact with an uninstrumented palm can support the object without being excluded by this task's success criterion.

A larger lower threshold alone is not a solution: it can reward excessive squeezing and contact-location error, saturate the actuators/sensors, and reduce the freedom to excite the object. Force targets should be calibrated to the physical sensor, conservative friction assumptions, object/load range, and allowed disturbance—not copied from random-action force maxima.

### Reward landscape

The current reward is, omitting notation for weights,

`-0.05 * sum(matched distances) + 0.5 * touched_count + 0.1 * near_anchor`

`-0.25 * anchor_distance -0.0002 * ||action||² -0.001 * sum(|qdot| |tau|)`

`-1 * object_far + 250 * hold_confirmed -50 * out_of_reach`.

Source: [reward implementation](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/leap_contact_direct_env.py#L437-L466).

Important consequences:

1. **Dense approach feedback is tiny.** Improving one matched distance by 1 cm adds only 0.0005 reward per step, whereas crossing one touch threshold adds 0.5: a 1,000-fold difference. There is little gradual guidance toward an unachieved contact compared with already-earned contacts.
2. **Partial contact pays indefinitely.** Three matched contacts plus the alive term pay 1.6 per step, or 480 over a nominal 300-step episode before costs. Trying to obtain the fourth may lose those contacts and incur a fall. A large success bonus does not teach an event that is almost never sampled.
3. **Success destroys its own useful state.** Resampling random points after a good grasp incentivizes another rearrangement rather than keeping that grasp for data collection.
4. **The reward has no identification term.** No estimator uncertainty, excitation rank, wrench residual, sensor quality, or estimation error enters the objective. Optimizing it longer need not improve identification.
5. **The absolute-action penalty is not a smoothness penalty.** Because actions map to absolute joint positions, `||action||²` biases toward joint-range midpoints; it does not directly penalize action changes, abrupt target motion, or slip.
6. **Positive holding reward is helpful, but not an exploit-proof theorem.** Comments based on `fall_penalty / episode_length` are an undiscounted heuristic. Discounting, time-limit bootstrapping, uncontrolled energy costs and alternative partial-contact strategies matter. With gamma 0.99, the discounted weight of a reward stream over 300 steps is about 95.1, not 300. Diagnose reward-term distributions and behavior, not only total return.

### Initialization and mechanical difficulty

The hand is fixed at its root, starts with zero nominal joint positions plus reset noise, and the object starts near the palm under gravity. The task therefore combines grasp acquisition, retention, random contact rearrangement and sustained simultaneous contact from the beginning. Current rotation reset is made easier, but this is not a verified pregrasp curriculum.

The smoke test establishes actual four-pulp contact in a snapshot. Its timer test then puts targets at the fingertip origins and calls the reward three times **without integrating physics between calls**. This is a useful bookkeeping test, not proof that a naturally sampled target can be held dynamically or reached by a policy.

Sources: [hand initialization and fixed root](../../source/LEAP_Isaaclab/LEAP_Isaaclab/assets/leap.py#L32-L54), [reset](../../source/LEAP_Isaaclab/LEAP_Isaaclab/compat/inhand_direct/inhand_manipulation_env.py#L246-L295), [smoke fixture](../../scripts/smoke/leap_contact_direct_smoke.py#L110-L159).

## 2. Model architecture and control: secondary, but concrete problems

### Absolute target control and exploration

These tasks do **not** use the relative-delta control described elsewhere for the original LEAP reorientation task. They inherit the shared direct environment's mapping from normalized actions to **absolute joint-limit positions**, followed by saturation. Physics is 120 Hz, control 30 Hz; `act_moving_average=1.0` disables smoothing.

Both RSL-RL variants start with Gaussian action standard deviation 1.0 and entropy coefficient 0.005. At zero mean, std 1 already puts 31.7% of coordinates outside [-1,1]. They become joint-limit targets; other means increase the outside probability. The distribution has no useful grasp-specific standard-deviation bounds.

Read-only inspection of two historical local checkpoints found:

| Historical checkpoint | Mean learned std | Lower bound on mean per-coordinate target saturation probability |
|---|---:|---:|
| `logs/rsl_rl/leap_hand_contact/2026-09-20_00-51-27/model_250.pt` | 0.954 | 28.29% |
| `logs/rsl_rl/leap_hand_contact_gru/2026-09-20_15-31-28/model_3500.pt` | 1.344 | 44.37% |

The bounds are averages of `erfc(1 / (sqrt(2) * std_i))`, assuming the optimal zero mean for avoiding saturation. They are **not measured rollout clipping fractions** and do not establish causation. These checkpoints also do not constitute a current-MDP comparison.

Sources: [target mapping](../../source/LEAP_Isaaclab/LEAP_Isaaclab/compat/inhand_direct/inhand_manipulation_env.py#L120-L143), [RSL configuration](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/agents/rsl_rl_ppo_direct_cfg.py#L12-L40), [GRU configuration](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/agents/rsl_rl_ppo_contact_direct_gru_cfg.py#L14-L42).

**Recommendation:** use a calibrated, rate-limited residual/delta joint-target controller around a demonstrated pregrasp, with bounded exploration and explicit torque/force safety limits. For example, integrate `Delta_q_max * tanh(u)` into the previous target once per control step, then enforce position/rate limits and anti-windup. Determine `Delta_q_max` from grasp-retention tests. A force-regulation inner loop is a useful structured alternative if real tactile bandwidth and calibration support it. It is not necessary to move directly to raw torque control.

### MLP versus GRU is not the main diagnosis

- MLP actor/critic: `[1024,512,256,128]`.
- GRU actor/critic: GRU-256 plus `[512,256]`.
- Both see the same 184-dimensional state-rich observation; no privileged critic is configured.
- No explicit observation history, estimator belief, hold-progress observation or sampled hold-duration observation is included.
- The GRU hidden state persists across rollout boundaries, but training segments are bounded by the 16-step rollout: about 0.53 s. It is incorrect to say memory resets every 16 steps.

The default contact configurations use a fixed cube and nominal density setting; they do not sample hidden mass/COM/full-inertia variants per episode. Consequently, the default training distribution does not require inertial adaptation. Existing objects-v1 manifest machinery is a useful starting point for this later curriculum, not a claim that those broader variants are already EA-validated.

The MLP has ample capacity for a small-state grasp-control problem. Recurrence helps with noisy sensing, delays and unknown dynamics; it cannot make impossible goals reachable or restore missing shear forces. This is also not a capacity-matched recurrence experiment.

Recommended baseline: compact MLP for the state-based stable-grasp teacher; add a short history or GRU-128/256 when testing partial observability and randomized inertials. A per-finger encoder can share structure while preserving finger identity and the thumb's special geometry, but is not a prerequisite. For identification, provide commanded motion/phase and an explicit uncertainty summary when available; do not expect untrained recurrent state to automatically become an inertial estimator. Test 32–64-step training segments as an ablation, with environment count/batch size adjusted to avoid confounding compute changes.

### Confirmed evaluation and wrapper issues

1. **GRU playback reset:** repository `scripts/rsl_rl/play.py:123–130` discards dones and never resets the inference actor's recurrent state. Installed RSL-RL returns the actor itself as the inference policy; it maintains hidden state until reset. Training correctly resets actor and critic. This contaminates playback after the first reset but is not a cause of training failure.
2. **RL-Games force clipping, not RSL-RL:** tactile components are multiplied by 10, while the contact RL-Games wrapper clips raw observations to +/-5 before network normalization. Each component therefore saturates above 0.5 N. This is particularly damaging for a redesigned force-regulation task. RSL's wrapper does not perform that observation clipping.
3. **Evaluation configuration drift:** playback resolves current registry defaults rather than reconstructing saved task configuration; the Play configuration also disables the ordinary timeout. Old and current checkpoints are not comparable without pinning MDP, tactile semantics, episode budget and evaluation mode.
4. **PBT is not a change in objective:** current contact selection is held-goals first, touch occupancy only as a zero-goal tie-break. It does not select for firm grasp quality or estimation accuracy. Keep PBT as an optimizer after redesign; do not use it to compensate for a misaligned objective.

Sources: [play loop](../../scripts/rsl_rl/play.py#L94-L130), [RL-Games contact YAML](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/agents/rl_games_ppo_contact_cfg.yaml#L4-L64), [tactile scaling](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/leap_direct_env.py#L242-L247), [PBT objective](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt/pbt/contact.py#L61-L88).

Installed implementation evidence: `/home/andy/env_isaaclab/lib/python3.12/site-packages/rsl_rl/runners/on_policy_runner.py:163–166`, `rsl_rl/modules/rnn.py:35–67`, `rsl_rl/algorithms/ppo.py:157–161`, and `isaaclab_rl/rl_games/rl_games.py:334–365`.

## 3. What inertial estimation actually requires

### Measurement contract

The current PhysX property selects `normal_force_matrix_w`, not a full contact wrench. The three Cartesian components are the normal force expressed in world coordinates, **not normal plus shear force**. Sensor configuration does not enable contact locations or friction-force tracking. The physical estimate requires more than these twelve observation channels.

For each accepted measurement window, establish:

- Object-on-pulp versus pulp-on-object sign and consistent coordinate transforms.
- Normal and tangential forces on the object, with calibrated noise and bias.
- Actual application points/centers of pressure. A body origin or requested point is not a measured application point.
- Contact moments where a finite soft pulp transmits appreciable torsion. Averaging positions and summing forces does not generally preserve the total moment.
- All external object contacts. Either instrument the palm/other supports or exclude such contacts during identification. Do not infer palm contact from object-root distance alone.
- Object pose and synchronized linear/angular motion of a known body-fixed reference point, with a physically justified acceleration estimator.
- Known gravity and any other applied external wrench; no hidden reset/teleport/fixture force in identification windows.

EA exposes separate filtered friction-force and average-contact-position channels, but their physical completeness and torque reconstruction must be tested before use. A sensor API exposing a field is not evidence of an accurate hardware-equivalent wrench. Physics-step contact data should be timestamped and aggregated consistently; a 30-Hz policy need not imply 30-Hz-only identification logging.

Sources: [current force property](../../source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/leap_direct_env.py#L180-L247); installed `/home/andy/env_isaaclab/lib/python3.12/site-packages/isaaclab_physx/sensors/contact_sensor/contact_sensor_data.py:100–181`; [estimator input equations](/home/andy/Documents/Estimate_weight_distribution/measurements/estimator.py#L29-L107).

### Static contact is not dynamic identifiability

The sibling estimator uses ten inertial parameters: mass, three first moments `h = m*c`, and six independent inertia components. Its generator supplies full contact forces and accelerations under a zero-angular-velocity snapshot model.

A read-only numeric rank check of its actual design matrix found:

| Data regime | Rank out of 10 |
|---|---:|
| Repeated static hold, one gravity direction | 3 |
| Static holds at diverse orientations | 4 |
| Diverse specific-force directions with only one angular-acceleration axis | 7 |
| Generic independent snapshots, one / two / three snapshots | 4 / 7 / 10 |

These are algebraic checks of the simplified model, not claims that three noisy real samples reliably identify an object.

Interpretation:

- Static gravity loading can identify mass and the COM components visible through the gravity torque.
- Reorienting the object relative to gravity can reveal the remaining COM direction.
- Full inertia requires sufficiently independent rotational accelerations and measurable dynamic torque; zero angular acceleration and velocity provide no inertia information.
- Pure internal squeezing can cancel in both net force and torque, producing no new rigid-body excitation. It may improve contact stability, but stronger squeeze is not information gain.
- More force can amplify torque error from uncertain contact locations: `delta_tau` contains `delta_r x f`. Overgripping is not automatically better metrology.

### The sibling estimator is a reference, not a drop-in real-trajectory estimator

Main's implementation is batch Huber/least squares. It assumes omega = 0 and uses acceleration at the COM, which its generator computes with ground-truth mass/COM. A camera generally tracks a known geometric origin, not the unknown COM. Feeding geometric-origin acceleration into that model during rotation is wrong.

For real trajectories, use a known object-fixed origin O, body-coordinate inertial acceleration `a_O`, gravity `g`, angular velocity `omega`, angular acceleration `alpha`, first moment `h = m*c`, and inertia about O, `I_O`:

```
v = a_O - g
F = m*v + alpha x h + omega x (omega x h)
tau_O = h x v + I_O*alpha + omega x (I_O*omega)
```

This is linear in `[m, h, I_O]`. Recover COM and COM inertia afterwards:

```
c = h/m
I_C = I_O - (||h||² * Identity - h*h.T)/m
```

Use physical-consistency constraints for the joint mass/first-moment/inertia estimate, not just positive diagonal inertia. Model sensor bias, unmeasured moments and errors in the acceleration regressor rather than expecting Huber loss alone to absorb them.

The synthetic scenario generator additionally perturbs equilibrium forces without reprojecting them into friction cones or checking LEAP reachability. It is not evidence that all its independent force/acceleration experiments are physically realizable by this hand. A separate sequential worktree exists with Bayesian updates and active-selection experiments, but it retains the simplified dynamics and uses a ground-truth oracle. Its candidate dynamics also require correction before transferring information scores.

Sources: [generator](/home/andy/Documents/Estimate_weight_distribution/measurements/generate.py#L161-L249), [estimator](/home/andy/Documents/Estimate_weight_distribution/measurements/estimator.py#L29-L107), [sequential reference](/home/andy/Documents/Estimate_weight_distribution_seq/measurements/sequential.py#L125-L185), [candidate prediction caveat](/home/andy/Documents/Estimate_weight_distribution_seq/experiments/active_selection.py#L81-L132).

## 4. Recommended task redesign

Preserve the existing task IDs as the contact-point benchmark. Give the new MDP new task IDs and versioned configs; do not silently reinterpret existing checkpoints.

### Phase A: stable, instrumented four-pulp grasp

**No arbitrary surface points.** Reward each named pulp touching a usable object surface within a safe force band, with an opposing thumb and sufficient disturbance resistance. Four nonzero readings alone are not force closure.

- Begin from collision-valid, dynamically verified pregrasps with bounded joint-target exploration. Settle contact before collecting measurements. Do not count forces generated by artificial fixture support.
- Train persistent holding for seconds, not a 0.1 s touch event followed by resampling.
- Measure actual pulp-region contact and reject or account for non-pulp support.
- Progress from nominal object/mass/friction to moderate perturbations, then wider randomized inertials/friction/sensor dynamics.
- Mix in increasingly difficult acquisition resets only after retention is reliable. Do not demand arbitrary pickup and active identification simultaneously from the first training step.
- If no hand-executable pregrasp baseline exists, establish one first using constrained IK, a closing controller, or a small grasp search. Failure here points to geometry/contact/control, not neural-network capacity.

**Reward proposal:** a small number of bounded, interpretable terms, for example

`r = w_g * grasp_quality + w_valid * valid_hold - w_slip * slip - w_over * overload - w_smooth * target_change - w_drop * dropped`.

For each finger, define a smooth contact/preload score based on signed surface distance, pulp alignment and a finger-dependent force band `[F_low_i, F_high_i]`. Do not require equal forces: an opposing thumb may balance several fingers, and the appropriate load distribution depends on geometry and the commanded wrench. Reward approaches to the surface before force exists; saturate force reward within the band and penalize force above the safe ceiling. Initially average the scores for acquisition, then emphasize the weakest contact with a smooth minimum or similar bottleneck aggregate. Rewarding only the mean permits sacrificing the thumb/fourth finger; using only an all-finger product from the beginning recreates sparsity.

Use hysteresis or a calibrated short dropout tolerance for sensor noise, but log true contact-loss duration and exclude invalid data windows. Do not allow a long grace period to count airborne pulps as firm contact.

Weights and force bounds are **to be calibrated**, not claimed optimal. Log each term's empirical scale, normalize physical units and test whether stationary safe holding beats release, violent squeeze, palm support and partial-contact strategies.

### Phase B: informative motion while preserving the grasp

Keep the same grasp and command bounded excitation:

1. Slow multi-axis tilts: static/quasi-static mass and COM validation.
2. Smooth angular acceleration and deceleration about multiple nonparallel axes: full inertia.
3. Modest translations where useful for separating mass and first-moment effects.

Prefer sinusoidal/multisine or bounded reference motion to impact-driven exploration. Initial amplitude/frequency must come from reachability, actuator bandwidth, slip margin and force/pose signal-to-noise tests; no universal safe value follows from the current simulator configuration.

**Wrist/arm route:** keep the fingers approximately fixed relative to the object and use the wrist/arm for excitation. This is the cleanest engineering route when that hardware exists. It is a scope extension beyond these fixed-hand environments, not an implicit requirement to migrate to the currently unvalidated arm task.

**Fixed-hand route:** use small rolling/repositioning motions of the cube. Recompute contact positions, track the object and monitor the regressor's weak directions. If the reachable motions cannot identify the full tensor at the required accuracy, report that limit or enlarge the motion capability; more RL does not overcome a physically unobservable mode.

Train tracking plus grasp maintenance before adding an information objective. Compare a scripted excitation baseline with learned excitation under the same limits.

### Phase C: uncertainty-aware active identification

Represent the observable inertial model as `w_t = Y_t theta + noise`, where `w_t` is the measured nongravity object wrench. Whiten by measurement covariance and normalize parameter units:

```
Y_tilde = R_t^(-1/2) * Y_t * D
Lambda_t = Lambda_(t-1) + Y_tilde.T * Y_tilde
r_info = logdet(Lambda_t) - logdet(Lambda_(t-1))
```

`D` defines physically meaningful parameter scales; initialize a positive-definite prior in normalized coordinates. This is a D-optimal surrogate, not a guarantee of actual estimation accuracy. An A-optimal reduction in normalized posterior trace is another choice; ASID specifically uses an A-optimal inverse-information criterion.

Safeguards:

- Add information only for synchronized, dynamically valid, fully accounted-for contact windows.
- Account for temporal correlation and regressor uncertainty; repeated correlated samples are not independent information.
- Cap/robustify incentives from numerical impacts, high-frequency acceleration noise and model mismatch.
- Keep safe force/slip constraints; information reward must not buy unsafe contact loss.
- Use diminishing returns, weakest-direction diagnostics and downstream estimation error. Unscaled determinant or matrix norm can reward a few easy directions while missing others.
- Keep hidden mass/COM/inertia out of deployed actor inputs. Ground truth is appropriate for training diagnostics, privileged critic inputs and validation, not the real actor's measurement stream.
- Stop when required parameter uncertainty/accuracy is met, subject to calibrated confidence; do not repeatedly resample goals and restart the information collection.

Do not start with this complete stage from random actions. Establish the measurement model and stable grasp first.

## 5. Observations and architecture for the redesigned task

Actor observations should reflect what can actually be measured or estimated on hardware:

- Joint positions/velocities, previous targets and actions.
- Per-pulp calibrated normal/shear forces, contact confidence and locations when available, with sensor validity masks.
- Object pose/velocity estimates in a hand/object-relative frame, and the gravity direction.
- Commanded motion, phase and relevant force-band targets.
- Short synchronized history and, for active selection, estimator uncertainty/innovation summaries.

Remove the 48 arbitrary tip-to-goal displacement channels and redundant fixed-orientation-goal features. Do not remove object-state feedback without providing an alternative; tactile-only pose inference is a separate partial-observability problem. Preserve force dynamic range before normalization and expose saturation/invalid-data flags rather than silently clipping away the quantity being controlled.

An asymmetric critic can use exact simulator object state, inertials and contact information while the actor uses only feasible measurements. Begin with an MLP and one short-history/GRU ablation. Larger MLPs, transformers or Mamba are not the first-line remedy for this objective mismatch.

Keep PPO/PBT if it is operationally preferred. Bound exploration, record actual action/target saturation, and use a selection hierarchy aligned with the new stages: safety and grasp validity first, then identification accuracy/time under matched conditions. Raw episodic return is not the scientific outcome.

## 6. Validation gates and decisive ablations

Suggested engineering acceptance targets below are starting gates, not measured current performance or universal metrology specifications.

### Gate 0: measurement model

- Verify contact-force sign, coordinate transforms and timestamps.
- Check mass from static total external force; check moments using known COM and known contact geometry.
- Compare reconstructed wrench against simulator ground truth over mild trajectories.
- Check gravity, angular velocity terms, application-point errors, hidden support and contact moments.
- Complete known-ground-truth estimator checks before using information reward.

### Gate 1: grasp retention

On held-out inertials and friction, aim initially for >=95% of fixed 10 s evaluation episodes retained, high joint four-pulp-valid occupancy (e.g. >=95%), and no force/torque safety-limit violations. These criteria must include whether palm support is allowed. Report acquisition and retention separately.

### Gate 2: excitation

Report the rank and weakest normalized information directions, not only total angular speed or determinant. Measure whether low-amplitude scripted trajectories generate a wrench above sensor/model noise and preserve contact. Include force saturation, slip and unmeasured contact rates.

### Gate 3: estimation

Report mass relative error, COM error in mm, relative Frobenius inertia error, uncertainty calibration, time to the application's accuracy target and failure rate across held-out density distributions. State whether inertia is about COM or a geometric origin, and in which axes. For near-isotropic objects, eigenvectors are ill-conditioned even when the tensor estimate is accurate; do not overinterpret principal-axis error.

Minimum comparisons:

1. Current random-point objective versus contact-anywhere force-regulated grasp, under matched dynamics/control where possible.
2. Open/noisy initialization versus verified pregrasp starts.
3. Absolute broad-range exploration versus bounded residual/delta targets.
4. Compact MLP versus matched-budget history/GRU.
5. Static hold versus scripted safe excitation versus learned safe excitation.
6. Noisy deployable observations versus privileged-state teacher, without hidden-inertia leakage to the deployed actor.

Use multiple seeds, identical evaluation windows, saved environment configs and explicit sensor semantics. Fix recurrent playback state resets before comparing GRU results. Historical net-force checkpoints and post-correction object-filtered checkpoints are different observation/reward regimes; do not treat them as controlled A/B evidence.

## 7. Evidence limits

- Source and installed-library inspection supports the objective/control/sensor findings.
- Historical local event files show zero success and shortening episodes, but saved configs predate the current reward. They do not establish the performance of today's exact MDP.
- Two historical checkpoint standard deviations were inspected without running policies; the saturation numbers above are analytic lower bounds.
- Algebraic rank checks and existing estimator caches/tests are not hardware or contact-dynamics validation. The sibling's batch/physics/metrics tests (9) and sequential tests (11) passed without cache writes; that does not close the sensor/model gaps.
- No new training, real-physics feasibility study, remote population evaluation or hardware test was performed for this assessment.

## 8. Primary research anchors

- Sundaralingam and Hermans, **In-Hand Object-Dynamics Inference using Tactile Fingertips**, [arXiv:2003.13165](https://arxiv.org/abs/2003.13165): close precedent using tactile forces/contact locations plus object pose and active physical motion; not evidence that contact-count maximization alone identifies inertials.
- Memmel et al., **ASID: Active Exploration for System Identification in Robotic Manipulation**, [arXiv:2404.12308](https://arxiv.org/abs/2404.12308): information-driven exploration principle; not a direct full-ten-parameter four-pulp LEAP implementation.
- Wensing, Kim and Slotine, **Linear Matrix Inequalities for Physically Consistent Inertial Parameter Identification**, [arXiv:1701.04395](https://arxiv.org/abs/1701.04395): physical-consistency constraints for inertial estimation.

See the companion [primary-source note](2026-09-25-inertial-grasp-primary-sources.md) for paper scope, equations and limitations.

**Priority order:** align the task with persistent grasping; validate the measured object wrench and dynamics model; make grasp control/exploration gentle and feasible; demonstrate scripted informative motion; then optimize information with PPO/PBT. Architecture scale is downstream of these decisions.
