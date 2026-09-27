# Critical assessment: learned palm targets with conventional arm control

Date: 2026-09-28

Assessment only. No controller implementation, package installation, simulation, training, or remote changes were made. This note is new; the existing `2026-09-27-dexsuite-cartesian-arm-hierarchy.md` and other concurrent workspace changes were left untouched. Local controller/task sources and primary cuRobo/DextrAH sources were checked for this assessment.

## Verdict

The hierarchy is feasible and well motivated as a **learning abstraction**, not automatically as a simulation optimization. It moves known arm kinematics, posture selection and some motion constraints out of the learned policy. It leaves the same 23 robot degrees of freedom and contact dynamics in PhysX, while adding controller/planning computation.

The recommended first baseline is **bounded six-dimensional palm-pose increments plus sixteen learned finger commands, with batched differential IK and existing joint PD for the arm**. A global trajectory planner on every policy decision is not a prerequisite. cuRobo V2 global planning is a plausible later component for longer collision-aware transitions; its MPC is a separate reactive-control option whose cost and failure behavior require measurement.

## 1. Define the proposed interface correctly

Here, “hand target pose” means the rigid palm/wrist pose, not the sixteen internal finger coordinates. A palm pose alone does not specify a grasp or contact forces. If finger actions are also removed, another explicit grasp controller, hand synergy model or learned lower-level hand policy is necessary.

The existing policy already commands relative joint-position increments to PD-controlled actuators, rather than raw motor torques. The proposed change moves the arm's inverse kinematics out of the policy:

- Existing: seven arm increments plus sixteen finger increments, 23 actions.
- Proposed minimal change: three translation increments, three rotation-vector increments, and sixteen finger increments, 22 actions.

The gain is semantic structure, not a dramatic reduction in dimension. An absolute position/quaternion interface would retain 23 numerical outputs, but would still change action meaning. Equal tensor shapes do not imply compatible policies.

For this task, goal pose is the **object's** pose. The hierarchy controls the **palm's** pose:

`T_base_object = T_base_palm * T_palm_object`.

During rolling, slipping, acquisition or regrasping, `T_palm_object` changes. The policy still has to coordinate that transform with finger contact. Reaching a planned palm goal is not proof of object-goal success.

The task resets the robot and object independently and does not guarantee a valid initial grasp or collision-free planning start. Thus an “object is attached to wrist” assumption is not valid from reset. The actual table and grasp acquisition cannot be bypassed without changing the task. [L1–L4]

## 2. Strong feasibility precedent, with a limited performance claim

[DextrAH-G](https://arxiv.org/html/2407.02274v1) uses the same KUKA–Allegro robot. Its learned policy commands three palm-position coordinates, three Euler orientation coordinates and five hand PCA coordinates; a geometric fabric produces joint-level trajectories, executed by PD controllers. Policy frequency is 15 Hz and fabric integration is 60 Hz. Its observations include fabric positions, velocities and accelerations. [D1–D2]

Appendix C reports approximately four times longer single-object learning for a joint-target fabric alternative to reach comparable grasp performance. This supports the value of structured actions, but is not a prediction for this task: palm representation, hand dimensionality and grasp prior change together. It is not an execution-speed benchmark or an isolated Cartesian-arm ablation. The five PCA components explain 98% of the retargeted grasp dataset variance, not 98% of all dexterous manipulation motions.

This precedent argues for testing the hierarchy. It does not justify simultaneously adding a planner, reducing finger dimensions and changing control frequency, because that would obscure which change helped.

## 3. What cuRobo V2 can and cannot supply

The public v0.8.0 commit is labelled “cuRoboV2 research release.” The current inspected source is pinned to `78fd485fa82d9b9a063fb4985e371814587e666a`; its README explicitly notes the public API changed from v1. [C1]

Confirmed capabilities:

- Inverse kinematics and collision-aware global trajectory optimization.
- Differentiable inverse dynamics, torque constraints, motion limits, and supplied external-wrench/payload inputs. Calling V2 purely kinematic would be incorrect.
- Reactive MPC for changing pose goals and updated obstacle geometry.
- Independent collision-world batching in both relevant planning paths and MPC; MPC exposes `max_batch_size`, `multi_env` and per-robot goal/reset selection. [C2–C4]

These do **not** establish a dexterous frictional-contact model or guarantee force closure, grip retention, slip prevention or tactile force regulation. Inverse dynamics with a supplied external wrench is different from predicting unknown multi-finger contact forces. Nor is a planning collision attachment a physical grasp.

Global planning and feedback control are distinct. A global plan returns a trajectory that must be tracked through actuators. Local differential IK updates joint targets toward a pose; it does not guarantee a collision-free path. MPC repeatedly optimizes a horizon, but execution/preemption and failure handling remain integration responsibilities.

### Scaling and latency

The V2 paper reports 48 ms at the 75th percentile for 2,600 Franka planning problems on an RTX 4090, including a 3 kg payload dynamics evaluation. This is a **global planning** result, not an MPC-step rate and not a benchmark for 256 independent KUKA–Allegro worlds co-running with PhysX and PPO. [C2]

GPU optimization seeds for one query are not the same workload as independent RL environments. Multi-world batching is supported, but measured memory, warmup/graph capture, hard-instance latency and simulator contention remain open for this application. Batch planning disables PRM seeding in its multi-environment path. [C4]

A 60 Hz simulated controller can wait for a slow planner; the cost is reduced simulation throughput. Real hardware has separate wall-clock deadlines. The planning benchmark alone cannot prove either that 60 Hz MPC is feasible or that it is infeasible.

## 4. Expected failure modes

### 4.1 Collision avoidance conflicts with intentional contact

The object must be contacted by fingers. Treating it as an ordinary forbidden obstacle can prevent grasping; ignoring all hand/object collisions can admit invalid motions. Planning exemptions need carefully scoped contact allowances and phase handling.

Finger geometry changes while the arm moves. A fixed gripper envelope may be too conservative or fail to cover actual fingers. The arm planner must account for the current hand geometry and, for longer horizons, its expected change or uncertainty.

### 4.2 A feasible arm trajectory can still drop the object

Arm acceleration, rotation and jerk impose loads on a grasp. A trajectory satisfying robot torque limits may still exceed the available frictional grip or generate damaging contact forces. Palm position tracking does not regulate individual finger normal/tangential forces.

Keep finger/contact feedback and bound wrist speed, acceleration and jerk. A compliant tracking controller may be useful, but is not interchangeable with plain IK and requires its own actuator/dynamics validation. No wrist teleportation, object pose writes or simulator weld should stand in for grasp mechanics.

### 4.3 Cartesian control does not eliminate seven-joint redundancy

A six-dimensional pose leaves elbow redundancy. A solver can select an undesirable posture, approach a singularity, jump IK branches, or prevent a later grasp even when the current target is reachable. Continuous local IK, posture preferences and explicit joint/velocity limits matter. A fixed nullspace preference can itself restrict useful configurations; an extra learned redundancy parameter is possible, but should be justified by failures.

Damped least-squares IK mitigates singularity sensitivity, not collision or hard-limit violations. Unreachable targets need bounded, deterministic handling and observable failure/projection status. Many requests mapping to the same projected target create poor exploration and credit assignment.

### 4.4 Persistent controller state changes the learning problem

A filtered target, integrator, active trajectory, time along that trajectory and command buffer affect future behavior. If hidden, identical robot observations/actions can lead to different transitions. Expose relevant applied target, tracking error, motion progress and planner status; reset per-environment controller state and prevent stale commands after reset or goal change.

Repeatedly planning long motions while the policy changes goals at every tick can cause cancellation, chattering or delayed effects. A lower-rate waypoint policy is possible, but changes decision timing and reward/discount interpretation. Initially preserve the current 60 Hz policy and 120 Hz physics to isolate action abstraction.

### 4.5 Model mismatch and privileged information

The planner and simulator must agree on tool frame, joint order, limits, masses, inertias, reflected armature, collisions and gravity. Use current installed asset values, not historical robot configuration notes. Palm and flange are not automatically the same frame; rotation conventions and root/world offsets must be explicit.

If the scientific goal includes estimating unknown object mass/COM/inertia, feeding their simulator ground truth to the low-level controller would introduce privileged information. Known robot kinematics are appropriate; use nominal or estimated object properties and account for uncertainty. Payload-model support is not permission to assume the unknown payload is known.

### 4.6 Task success may not mean in-hand reorientation

The current reward/success checks object pose and finger contact; it does not require the object to rotate relative to the palm. Holding a stable grasp and moving the whole arm can satisfy it. This is already allowed by the original joint-action task.

If the intended scientific outcome is genuine in-hand reorientation, separately measure object-to-palm rotation and sustained contact/retention. Do not interpret improved task score as proof of learned regrasping.

## 5. Concrete integration caveats

### Existing Isaac controllers

The installed `DifferentialInverseKinematicsAction` supports a selected joint subset and a body/tool offset, using batched Torch computations. It samples a target during policy-action processing and solves toward it on physics substeps. This is a workable foundation, not a tested KUKA–Allegro integration. [L5]

Current relative-joint actions instead form `q_target = q_current + processed_action` on each substep. These mappings have different dynamics even at the same frequencies. The installed IK offers DLS/adaptive DLS; its optional limit-avoidance bias is projected through the **position** Jacobian, so it must not be assumed to preserve full orientation or enforce hard bounds. [L6]

Operational-space control is also installed, but outputs joint efforts and may use mass matrices/gravity. It needs deliberate actuator configuration to avoid conflicting with existing position drives. Its current closed-loop wrench path measures only three force components; moment components are open loop. It is not an off-the-shelf four-finger tactile controller. [L7]

### cuRobo source-specific details

At the pinned revision:

- `AttachmentManager` states attach/detach operations apply to all environments, although poses/offsets can differ. Independent asynchronous grasp acquisition/loss cannot be assumed to map directly onto this shared operation. Geometry updates and payload updates are separate concerns. [C5]
- `optimize_next_action()` consumes a buffered command window; `optimize_action_sequence()` reoptimizes on each call. A goal update is not proof of immediate preemption. [C6]
- The automatic infeasibility-to-deceleration branch contains an `and False` guard. That particular fallback does not execute. This is not proof all MPC execution is unsafe, but the application must explicitly check feasibility and define a valid fallback rather than relying on the configuration flag. [C6]
- The V2 paper presents some learned-policy safety-layer/global-plan-to-MPC integrations as future directions, not turnkey validated solutions. [C2]

Exact compatibility with Python 3.12, Torch 2.11, Warp 1.16 and this Isaac stack was not tested. Broad dependency bounds are not a compatibility result. The official documentation warns against installing `usd-core` inside Isaac Sim. No package changes were made. [C7]

### Policy and reward compatibility

With six Cartesian plus sixteen finger actions, five history samples of the last-action term change the unaugmented observation size from 1745 to 1740. Controller-state observations may grow it again. Existing joint policies and the strict same-task policy-prior path are not drop-in compatible; dimensional padding alone cannot change learned action semantics. [L1–L4]

Translation in metres, rotation in radians and finger increments require separately chosen scales/exploration. Raw action magnitude and action-difference penalties no longer represent the same physical regularization. Keeping their coefficients unchanged does not make returns directly comparable. Compare fixed physical success, retention, actual motion and wall-clock time instead. [L3]

## 6. Recommended comparison and acceptance

Recommended initial architecture:

1. The learned policy supplies bounded base-frame palm translation/rotation-vector increments and the existing sixteen finger commands.
2. Batched local differential IK produces seven arm joint targets; the existing joint PD tracks them. Finger control remains unchanged.
3. Real articulation/contact physics, object distribution, reset rules and task criteria remain present. Controller-specific observation and failure handling are explicit.

Compare this with joint-action control under matched seeds and physical conditions. Do not also introduce finger PCA or reduced decision frequency. Before training, check frame/Jacobian correctness, smooth tracking, reachability, singularities, collision/limit behavior, partial resets and grip disturbance under motion. These are proposed acceptance tests, not completed results.

Record both transitions to a held-out success threshold and measured transitions per second:

`wall-clock time to target ≈ transitions required / transitions per second`.

Also measure retention, relative object/palm motion, force spikes, planner failures, applied-versus-requested pose, real joint motion and GPU memory. Lower FPS can still yield shorter training if sample efficiency improves sufficiently; higher task reward alone cannot establish that.

Add geometric fabrics/operational-space compliance if local tracking/contact quality is the observed limitation. Add cuRobo global planning for obstructed approach/transport, or evaluate its MPC when reactive collision avoidance is needed. Planner phase boundaries must follow actual grasp state, not an assumed rigid attachment.

**Recommendation:** pursue palm-pose action abstraction; do not equate it with replacing dexterous control by a planner, eliminating arm physics, or obtaining a guaranteed training-speed multiplier.

## Primary sources

### Task and installed controller sources

Repository root: `/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab`.
Installed Isaac root: `/home/andy/env_isaaclab/lib/python3.12/site-packages/isaaclab/source/isaaclab/isaaclab`.

- **L1:** `source/LEAP_Isaaclab/LEAP_Isaaclab/compat/dexsuite/dexsuite_env_cfg.py` — command ranges, observations, independent reset randomization, timing.
- **L2:** `source/LEAP_Isaaclab/LEAP_Isaaclab/compat/dexsuite/config/kuka_allegro/dexsuite_kuka_allegro_env_cfg.py:26–93` — joint-action and fingertip-sensor configuration.
- **L3:** `source/LEAP_Isaaclab/LEAP_Isaaclab/compat/dexsuite/mdp/rewards.py:23–30,149–176`; `dexsuite_env_cfg.py:364–366` — action regularizers and pose/contact success.
- **L4:** `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt/pbt/policy_prior.py` — existing same-task checkpoint contract.
- **L5:** Installed `envs/mdp/actions/task_space_actions.py:58–113,190–273` — joint masking, tool offsets, target capture, per-substep IK.
- **L6:** Installed `controllers/differential_ik.py:189–214,267–294,332–360`; `controllers/differential_ik_cfg.py:38–89`; `envs/mdp/actions/joint_actions.py:203–233`.
- **L7:** Installed `controllers/operational_space.py:375–425,486–515`; `envs/mdp/actions/task_space_actions.py:276–397`.

### DextrAH-G

- **D1:** [DextrAH-G paper](https://arxiv.org/html/2407.02274v1), sections 3–4 and appendices B, C, D, E, H.
- **D2:** [First-party KUKA–Allegro fabric deployment](https://github.com/NVlabs/DEXTRAH/blob/main/dextrah_lab/deployment_scripts/kuka_allegro_fabric.py). The wrapper uses `KukaAllegroPoseFabric` and `DisplacementIntegrator`, not a global motion-planning call per tick.

### cuRobo V2

All implementation links below pin revision `78fd485fa82d9b9a063fb4985e371814587e666a`; conclusions are version-specific.

- **C1:** [Official repository](https://github.com/NVlabs/curobo); [v0.8.0 research-release commit](https://github.com/NVlabs/curobo/commit/4ea77366ca48ee453e7df139e39fa6532af49f3b).
- **C2:** [V2 paper](https://arxiv.org/html/2603.05493v1), sections 6.3, 7.1–7.2, 7.7 and 9.
- **C3:** [Reactive-control example](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/examples/getting_started/reactive_control.py); [MPC config, independent worlds](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/_src/solver/solver_mpc_cfg.py#L40-L44).
- **C4:** [Batch motion planner](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/_src/motion/motion_planner_batch.py#L68-L76); [MPC multi-environment setup](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/_src/solver/solver_mpc.py#L287-L300).
- **C5:** [Attachment manager contract](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/_src/collision/attachment_manager.py#L24-L34).
- **C6:** [Command buffering and optimization](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/_src/solver/solver_mpc.py#L533-L624); [disabled automatic fallback branch](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/_src/solver/solver_mpc.py#L665-L690); [public MPC alias](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/model_predictive_control.py#L39-L40).
- **C7:** [Installation](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/docs/getting-started/installation.rst); [dependency metadata](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/pyproject.toml); [benchmark/Isaac USD warning](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/docs/reference/benchmarks.rst#L96-L99).
