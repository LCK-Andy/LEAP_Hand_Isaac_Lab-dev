# Cartesian arm targets with a learned dexterous-hand policy

Date: 2026-09-27. Research assessment only; no controller implementation, package installation, or training changes. Primary-source inspection was completed before the user requested checkpoint playback.

## Verdict

The hierarchy is feasible and well motivated as a **learning/action-abstraction experiment**, not an automatic physics-speed optimization. A policy can command a wrist/palm pose and finger actions while a classical controller handles arm kinematics. The robot still has seven arm plus sixteen finger degrees of freedom in physics. Planning/IK adds computation; any wall-clock benefit must come from needing fewer training transitions or safer/more effective exploration.

Start with the existing batched Cartesian differential-IK controller and joint PD, not a full global trajectory solve on every policy step. Keep learned finger control initially. Introduce cuRobo V2 planning/MPC only after measuring the simpler hierarchy and identifying a need for collision-aware longer-horizon arm motion.

## What the proposal means in this task

The current policy already commands relative joint-position increments into a PD-controlled robot, not raw motor currents. The proposed hierarchy moves arm inverse kinematics and motion coordination out of the policy.

A reasonable first action is **six Cartesian wrist increments plus sixteen finger increments: 22 dimensions instead of 23**. The numerical reduction is small; the useful change is semantic alignment between policy actions and end-effector motion. A wrist pose alone cannot specify sixteen finger joints, a grasp, contact forces, or object orientation. Removing learned finger actions would require another explicit hand controller, grasp library or synergy representation.

The installed task uses 60 Hz policy updates and 120 Hz physics. Its 1745-dimensional observation concatenates five history samples; the last-action component means changing 23 to 22 actions changes the unaugmented observation dimension to 1740. Controller-state additions would increase it again. Existing joint-action checkpoints are not drop-in Cartesian policies even if input sizes were padded.

The task commands an **object pose**, whereas the proposed action commands a **palm pose**. Their relation is

`T_base_object = T_base_palm * T_palm_object`.

The second transform changes during grasp acquisition, rolling, sliding and regrasping. A planner reaching the requested palm pose does not establish that the object reached its goal.

Native success is an episode-sticky pose-and-contact condition, not Stage A's persistent four-pulp/support-free grasp. Whole-arm rotation of a securely held object can satisfy it; relative object-to-palm rotation is not required. This is already permitted by the existing task, not a new loophole introduced by Cartesian control. If the scientific objective is genuine in-hand reorientation, measure `T_palm_object` separately.

## Strong precedent, but not proof of this task's speedup

[DextrAH-G](https://arxiv.org/html/2407.02274v1) uses the same KUKA–Allegro robot with an 11-dimensional policy command: three palm-position coordinates, three palm-orientation coordinates and five hand PCA coordinates. A geometric-fabric controller generates joint-level commands. Policy inference is 15 Hz and fabric integration is 60 Hz. The policy observes controller-internal positions, velocities and accelerations as well as measured robot state.

The paper's Appendix C reports that a joint-target fabric alternative took roughly four times longer to reach comparable single-object grasp performance. This is **not** a fabric-versus-no-fabric execution benchmark, not an isolated palm-control ablation, and not a result for this reorientation task. The proposed interface also changes hand dimensionality and its prior. The five PCA coordinates capture 98% of the retargeted grasp dataset's variance, not 98% of all dexterous motions. Such a grasp-oriented subspace can exclude useful reorientation motions.

This is convincing feasibility evidence for a hierarchy, but does not justify promising a learning-speed multiplier.

First-party implementation: [KUKA–Allegro fabric deployment](https://raw.githubusercontent.com/NVlabs/DEXTRAH/main/dextrah_lab/deployment_scripts/kuka_allegro_fabric.py). It uses `KukaAllegroPoseFabric` with `DisplacementIntegrator`; it is not equivalent to inserting a global trajectory planner at every policy step.

## What cuRobo V2 actually supplies

The official [v0.8.0 release](https://github.com/NVlabs/curobo/releases/tag/v0.8.0) was published 2026-04-18. Current documentation is https://nvlabs.github.io/curobo/latest/; curobo.org describes the legacy API. Implementation details below were inspected at main commit `78fd485fa82d9b9a063fb4985e371814587e666a` dated 2026-09-10, not asserted identical to the April release.

- `MotionPlanner.plan_pose`, `plan_cspace` and `plan_grasp` produce trajectories. A tracking controller still executes them through actuators.
- `MPCSolver` provides receding-horizon control interfaces, including `optimize_next_action`. MPC exists in V2; it is not merely a legacy feature.
- V2 includes differentiable inverse dynamics, torque constraints, and position/velocity/acceleration/jerk constraints, as well as payload/inertia and external-wrench inputs. It would be incorrect to call it dynamics-blind.
- Batched independent planning environments are supported. However, `multi_env=True` disables PRM graph seeding in the inspected batch planner. MPC has a different set of batching/seed restrictions. Parallel optimization seeds and independent RL environments must not be conflated.
- The inspected attachment manager models objects rigidly following robot links. Attach/detach operations apply across environments sharing obstacle identities, although per-environment poses are supported. Asynchronous grasp acquisition, loss and reset therefore require explicit integration design.

These capabilities do not establish a dexterous contact solver or guarantees of friction-cone feasibility, force closure, slip prevention or tactile force regulation. Robot torque feasibility and grasp stability are different requirements. Rigid collision attachment is a planning approximation, not a physical weld or proof of a grasp. During in-hand motion the relative attachment transform changes.

The [V2 paper](https://arxiv.org/html/2603.05493v1), section 7.1, reports 48 ms 75th-percentile planning time and 99.7% success for 2600 cluttered Franka 7-DOF planning instances with a 3 kg payload on an RTX 4090. That does not establish 60 Hz MPC performance for 2048 independent KUKA–Allegro worlds.

No turnkey KUKA–Allegro configuration was found in the inspected robot-config directory. Generic custom-robot support still requires validating combined kinematics, joint limits, inertial data, collision spheres, ignored collision pairs and tool-frame conventions against the simulator asset. Exact coexistence with this Isaac stack's Python 3.12, torch 2.11 and Warp 1.16 was not tested.

## Expected problems

### Contact avoidance can conflict with intended manipulation

The planner must allow deliberate fingertip/object contact without allowing arbitrary hand penetration or palm-support exploits. Treating the object as an ordinary obstacle can prevent grasping; ignoring it can permit collisions. Grasp phase, collision allowances and payload assumptions need explicit definitions. A moving sixteen-joint hand cannot always be approximated by one fixed gripper envelope.

Fast collision-free arm motion can still drop an object. Arm acceleration creates inertial loads that depend on grip strength, object mass/inertia and friction. Position tracking alone does not control contact force. Bound wrist velocity, acceleration and jerk; retain force observations and physical validation. Do not teleport the wrist/object or replace grasp contact by a simulator weld.

### Seven-joint redundancy remains

Six-dimensional Cartesian pose does not specify the elbow configuration of a seven-joint arm. Multiple inverse-kinematic branches, singularities, joint limits and nullspace posture can cause discontinuities or block useful grasps. Use continuity-aware IK and a reasonable nullspace preference. A fixed posture preference can itself restrict reachability; an optional learned redundancy parameter would restore an action dimension but could still preserve the semantic benefit.

Unreachable Cartesian goals need bounded projection/failure behavior. Mapping many requested goals to the same reachable target creates flat or discontinuous action effects and weak learning signals.

### Planning latency and state complicate RL

Replanning a long trajectory for every 60 Hz action can repeatedly cancel plans and cause chattering. Hard instances can create variable batch latency and GPU contention with PhysX/PPO. Independent-world batching must be benchmarked at the actual workload; the 8×256 population entails 2048 independent worlds, not 2048 seeds for one query.

A persistent plan, filtered target, integrator state, nullspace target or asynchronous queue becomes part of the environment state. Observations should expose relevant requested/applied targets, tracking error and plan status. Reset must clear the state of the corresponding environment without affecting others. Goal resampling and failed plans must not execute stale commands.

Projection through IK/planning is a legitimate actuator mapping for PPO. The likelihood remains that of the sampled high-level action; it should not be replaced with a fictitious probability for the generated joint command. The practical risks are many-to-one mappings, hidden state and delayed credit, not intrinsic invalidity of PPO.

### Frequency and controller semantics are confounds

The current relative-joint action re-centers its target on measured joint position at every physics substep. The installed differential-IK action samples a pose target at the policy step and solves toward that target on substeps. This is a meaningful controller change even if policy and physics frequencies are unchanged.

Lowering policy frequency to 15 Hz may be attractive, but should be a separate experiment. DextrAH-G's timing is a precedent, not proof that 15 Hz preserves this task's contact reactions.

### Existing training/checkpoint contracts change

The action and observation layouts, physical action units and exploration grouping change. Metres, radians and finger-joint increments need separately chosen scales. Existing exact-23-joint exploration grouping must not silently be assumed valid for a 22-dimensional Cartesian interface.

The existing same-task `--policy-prior` and Stage A-to-B `--warm-start` do not implement joint-to-Cartesian policy migration. Fresh training or deliberately designed distillation/transfer is required. Keep the new control task and checkpoints distinct from the running population.

## Recommended experiment

1. **Cartesian local-control baseline:** six bounded base-frame palm pose increments plus the existing sixteen finger increments. Use the installed differential-IK/DLS controller on the seven arm joints with existing joint PD. Preserve 60 Hz policy/120 Hz physics, object distribution, rewards and contact gates. Validate the palm frame, Jacobian joint mask, rotation convention, joint limits and nullspace behavior.
2. **Matched joint-control baseline:** same seeds, scene and physical conditions. Compare learning success, grasp retention, force spikes, joint-limit/singularity encounters, commanded/applied pose error and wall-clock cost. Do not change finger synergies or policy frequency at the same time.
3. **Add complexity only for an observed limitation:** test compliance/operational-space control or a fabric when contact quality requires it; test cuRobo MPC or lower-rate global planning for collision-aware approach/transport when local IK fails. Avoid assuming stage boundaries or reliable grasp detection before they are implemented and evaluated.

Judge both transitions needed to reach a fixed success target and actual training throughput:

`time to target ≈ required transitions / measured transitions per second`.

A slower environment can still train faster if learning improves sufficiently. Conversely, a more sophisticated planner can increase latency without solving the difficult finger/contact learning problem.

**Recommendation:** pursue the hierarchy, but first as an action-abstraction experiment using local Cartesian control. cuRobo V2 is a credible later component, not a prerequisite and not an automatic substitute for dexterous contact control.

## Focused primary sources

Local root: `/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab`.

- `source/LEAP_Isaaclab/LEAP_Isaaclab/compat/dexsuite/dexsuite_env_cfg.py` (object goal, timing, rewards and resets).
- `source/LEAP_Isaaclab/LEAP_Isaaclab/compat/dexsuite/config/kuka_allegro/dexsuite_kuka_allegro_env_cfg.py` (23-joint action/sensor configuration).
- `source/LEAP_Isaaclab/LEAP_Isaaclab/compat/dexsuite/mdp/rewards.py:149–176` (pose/contact success).
- `/home/andy/env_isaaclab/lib/python3.12/site-packages/isaaclab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:203–233` (relative targets).
- Installed `DifferentialInverseKinematicsAction` and `DifferentialIKController` sources were inspected; controller reuse is supported by source, but the proposed KUKA–Allegro configuration was not run.
- [DextrAH-G paper](https://arxiv.org/html/2407.02274v1), sections 3–5 and appendices C/D/H.
- [cuRobo V2 paper](https://arxiv.org/html/2603.05493v1), especially section 7.1.
- [V2 release](https://github.com/NVlabs/curobo/releases/tag/v0.8.0).
- [Pinned MPC source](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/_src/solver/solver_mpc.py).
- [Pinned batch planner](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/_src/motion/motion_planner_batch.py).
- [Pinned attachment manager](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/_src/collision/attachment_manager.py).
- [Custom robot builder](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/examples/getting_started/build_robot_model.py).
- [Dependency metadata](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/pyproject.toml).
