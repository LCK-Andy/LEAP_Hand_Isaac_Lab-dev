# Primary sources: firm four-pulp grasp and identifiable inertial excitation

Date: 2026-09-25. Scope: literature and established rigid-body equations for redesigning a LEAP-hand task; not a task-code or estimator-code audit. Repository code was not modified. This is a targeted primary-source review, not an exhaustive or systematic review. The requested filename was checked before creation; no earlier note was overwritten.

## Conclusions

1. **Stable four-pulp contact is a useful control requirement for the proposed experiment, not an identification objective or mathematical prerequisite by itself.** Static wrench balance contains mass and first-moment information but no rotational-inertia information. Increasing squeeze in the grasp's internal-force nullspace does not increase the net rigid-body wrench.
2. **The tactile anchor paper actually requires forces, contact locations, and object pose.** Sundaralingam and Hermans estimate ten inertial quantities and friction, but explicitly do not optimize excitation. Their soft fingertips are modeled as frictional point contacts, and grasps are chosen to avoid large unmeasured torsional moments. This is not evidence that four scalar normal forces alone recover a full tensor.
3. **ASID supports information-directed exploration, not a demonstrated four-finger full-tensor solution.** It uses a simulator-based Fisher-information objective and real object-pose observations for rod balancing and shuffleboard. The paper does not document unconstrained recovery of all ten inertial parameters with fingertip sensing.
4. **Use a standard body-origin regressor and enforce physically realizable inertial parameters.** Mass, first moment, and inertia about one known object-fixed origin form a linear ten-parameter representation. Positive mass plus an SPD inertia tensor is insufficient: the principal moments must also satisfy triangle inequalities.
5. **For a fixed wrist, test feasible information before promising full inertia identification.** Finger-driven translation and multi-axis rotation can produce information, but reach, contact persistence, contact-model validity, actuator authority, and sensor bandwidth bound what can be excited. If the feasible measured regressor remains deficient, report identifiable combinations or add wrist motion rather than treating a prior as recovered information.

## 1. Verified primary sources and their actual scope

### S1. Sundaralingam and Hermans — tactile inertial estimation

**Balakumar Sundaralingam and Tucker Hermans. _In-Hand Object-Dynamics Inference using Tactile Fingertips_.** arXiv:2003.13165, submitted 2020; v2 dated 2021-01-19. The arXiv record reports acceptance in IEEE Transactions on Robotics.

- Durable record: <https://arxiv.org/abs/2003.13165>
- Versioned full text: <https://arxiv.org/html/2003.13165v2>
- PDF: <https://arxiv.org/pdf/2003.13165v2>

**Direct evidence.** Section III defines the targets as mass, three COM coordinates, six independent entries of a symmetric inertia tensor about the COM, plus a separately estimated static friction coefficient. The assumptions are a known mesh, a rigid object with fixed mass distribution, and frictional point contacts. The factor graph uses **object pose, contact points, and contact forces** as observations, not forces alone (§III-B).

Hardware is an Allegro hand with four BioTac fingertips on a KUKA LBR4 arm (§IV-A). An ASUS Xtion camera and DART provide visual tracking. Because in-hand depth tracking is poor, a second evaluation initializes an object-to-thumb transform with DART and subsequently uses joint sensing and a relaxed-rigidity approximation. This is an assumption-dependent pose estimate, not direct inertial sensing. The OptoForce HEX-E wrist F/T sensor supports a **separate rigidly attached-object baseline**; the fingertip method does not require measured contact torques.

**Useful locators.**

- §III-A, Eqs. (1)–(5): friction from tangential/normal force at slip onset. One finger's normal force is reduced while tangential force is maintained; the other fingers support the object. Force data strictly inside the friction cone do not by themselves locate the slip boundary.
- §III-B1, Eqs. (6)–(9): summed fingertip forces and lever-arm moments related to rigid-body dynamics. Eqs. (10)–(17): pose-based velocity/acceleration construction. These are the relevant precedent for reconstructing a net object wrench without individual torque sensors.
- §III-B3, Eqs. (23)–(30), (35)–(37): principal second-moment parameterization, positive mass, and COM containment in the object's convex hull.
- §IV-A: GTSAM batch Levenberg–Marquardt inference; pose, force, and contact measurements are optimized jointly. Ground-truth mass is measured with a scale; COM and inertia references are obtained from CAD for the constructed test objects.
- **§V-C explicitly states that the trajectories are fixed, active excitation is not addressed, and estimating every inertial parameter requires sufficiently exciting it.**
- §V-C2: individual tactile forces contain both object-dynamics contributions and forces transmitted from the other fingertips. A reported ring-finger contact loss/recovery illustrates that the method does not require four uninterrupted contacts in every trial.
- §VI: sensor mounting-frame contacts can produce unobserved forces; sensor transients matter at high speed; penalty-based inequality factors encourage but do not guarantee physical feasibility. Robot forward dynamics/control inputs are not included because accurate hand torque feedback is unavailable.

**Limits on interpretation.** The paper itself reports numerical parameter mismatch under incomplete excitation. It should not be cited as demonstrating arbitrary high-accuracy full-tensor recovery from a static grasp. The “Constraint+Geodesic” evaluation sets the prior to ground-truth inertial parameters (§IV-B), so it is not an unknown-object prior-free benchmark. Soft-finger torsion is explicitly excluded in §III: the selected grasp avoids large frictional torques that their BioTac implementation cannot estimate.

**Equation-transcription caution.** The text after Eq. (30) prints a parallel-axis correction that omits the usual squared-distance identity term. Also, the prose preceding friction Eq. (1) reverses sine/cosine relative to Eqs. (2)–(5). Do not mechanically copy these expressions into an estimator. Section 2 below gives consistent standard notation, cross-checked against S3. The scalar “inertial error” in Eq. (38) is printed as `|4 − tr(P_gt⁻¹ P_est)|`; despite its description as a geodesic distance, this expression is not a separating metric on SPD matrices. For example, `P_gt = I₄` and `P_est = diag(1.5, 0.5, 1, 1)` give zero with unequal matrices. This is a mathematical caution, not a new assessment of their experiments.

### S2. Memmel et al. — active identification through simulation

**Marius Memmel, Andrew Wagenmaker, Chuning Zhu, Patrick Yin, Dieter Fox, and Abhishek Gupta. _ASID: Active Exploration for System Identification in Robotic Manipulation_.** arXiv:2404.12308, 2024; v2 dated 2024-06-27.

- Durable record: <https://arxiv.org/abs/2404.12308>
- Versioned full text: <https://arxiv.org/html/2404.12308v2>
- PDF: <https://arxiv.org/pdf/2404.12308v2>

**Direct evidence.** The objective is a sim-to-real-to-sim-to-real procedure: train informative exploration in a simulator family, collect real interaction data, identify simulator parameters, then synthesize a downstream controller (§§3–4). The formal protocol uses a real exploration episode; this is not a theorem that one episode identifies arbitrary dynamics.

Section 3 defines trajectory Fisher information and gives the Cramér–Rao mean-squared-error lower bound in Eq. (1). Section 4.1 chooses **A-optimal design**, minimizing `tr(I(θ*, π)⁻¹)` in Eq. (2). Equation (3) assumes Gaussian additive state-transition noise. Equation (4) replaces the unknown true parameter with an expectation over an initial parameter distribution. Simulator derivatives are approximated by finite differences where analytic derivatives are unavailable. It is not simply maximizing `tr(I)` or randomizing actions.

Section 4.2 identifies parameters by matching real trajectories to simulated trajectories that replay recorded actions. The physical experiments use CEM; simulated identification uses REPS. Successful downstream control is the ultimate objective, which is different from certifying accuracy of every physical parameter.

**Sensor modalities and parameters.** Real experiments use a Franka Emika Panda and **two calibrated Intel RealSense D455 cameras**, obtaining object position/orientation through color-thresholded point clouds (§5.5). No four-fingertip tactile or wrist-wrench identification modality is specified for these experiments.

- Sphere experiments concern friction/contact dynamics in simulation.
- Laptop experiments concern articulation identification and reconstruction in simulation.
- The rod experiment varies mass distribution and identifies “inertia” and environmental friction; real control chooses where to pick the rod to balance it. Appendix A.1.3 uses planar end-effector exploration commands `δx, δy` with a peg.
- Shuffleboard identifies sliding and torsional friction for a puck (§5.5).

**Limits on interpretation.** The rod's full inertial parameter-vector dimension and a six-component tensor-accuracy evaluation are not specified in the inspected text. Labels such as “inertia: left/middle/right” are experimental mass-distribution categories, not evidence of unconstrained ten-parameter recovery. Section 5.5 explicitly reports ambiguity when the rod COM is near its middle. The idealized theory observes an MDP state, assumes the true dynamics lie in the simulator family, and uses regularity assumptions for Fisher/CRLB reasoning. Real pose extraction is not direct access to that full state. Smooth finite-difference sensitivities around contact transitions require particular care.

### S3. Wensing, Kim, and Slotine — physical consistency

**Patrick M. Wensing, Sangbae Kim, and Jean-Jacques E. Slotine. _Linear Matrix Inequalities for Physically-Consistent Inertial Parameter Identification: A Statistical Perspective on the Mass Distribution_.** IEEE Robotics and Automation Letters 3(1), 60–67, 2018; final author manuscript arXiv:1701.04395v3, 2017.

- DOI: <https://doi.org/10.1109/LRA.2017.2729659>
- Record: <https://arxiv.org/abs/1701.04395>
- Versioned full text: <https://arxiv.org/html/1701.04395v3>

**Direct evidence.** §III distinguishes positive spatial inertia (“semi-consistency”) from realizable mass distributions. §IV-B, Eqs. (15)–(18), Proposition 1 and Corollary 1 connect central second moments to inertia triangle inequalities. §IV-C, Definition 7 and Theorem 3 give the **4×4 pseudo-inertia LMI**. §V, Theorem 4/Eq. (23) adds realizability in an enclosing ellipsoid; constraining the COM alone is weaker.

The experiments concern robot-link identification, not tactile LEAP-hand manipulation. The realizability mathematics applies to a rigid object's inertial parameters regardless of the sensing system. It restricts admissible parameters; it cannot make missing excitation observable.

### S4. Swevers et al. — statistically designed excitation

**Jan Swevers, Chris Ganseman, Dilek Bilgin Tükel, Joris De Schutter, and Hendrik Van Brussel. _Optimal Robot Excitation and Identification_.** IEEE Transactions on Robotics and Automation 13(5), 730–740, 1997.

- Institutional full-text copy: <https://lirias.kuleuven.be/retrieve/80252>

**Direct evidence.** §II-A discusses Fisher information and the CRLB; §II-B gives the weighted linear least-squares specialization. §III-A, Eq. (11) parameterizes joint excitation with finite Fourier series, allowing bounded bandwidth, periodic averaging, and analytic differentiation of fitted position responses. §III-B optimizes parameter uncertainty, with a **D-optimal covariance-determinant criterion**, under position, velocity, acceleration, and Cartesian workspace/collision constraints. This differs from minimizing condition number alone and differs from ASID's A-optimal choice.

The source also discusses errors in both measured motion and torque. Systematic kinematic errors are not repaired by a random-noise likelihood (§II). Its experiments concern the first three axes of an industrial KUKA arm. Transfer of periodic excitation to finger-held objects is a design recommendation, not a result established by this paper. Some equation glyphs in the institutional PDF's text extraction were incomplete; only section locators and the clearly identified Fourier equation number are relied on here.

### S5. Buss, Hashimoto, and Moore — external wrench versus internal grasp forces

**Martin Buss, Hideki Hashimoto, and John B. Moore. _Dextrous Hand Grasping Force Optimization_.** IEEE Transactions on Robotics and Automation 12(3), 406–418, 1996.

- Author/institutional full-text copy: <https://users.cecs.anu.edu.au/~john/papers/JOUR/144.PDF>

**Direct evidence.** §II, Eq. (1) maps contact-wrench intensities to the external balancing wrench using a grip transformation matrix. §II-A, Eq. (2) imposes compressive normal force and Coulomb tangential-force limits. §II-B distinguishes three-component frictional point contact from soft-finger contact with an additional moment about the normal, and discusses coupled shear/torsion limits and linear/elliptical approximations. §III combines wrench equilibrium, friction constraints, and a grasp-force cost.

The paper explicitly notes the trade-off between distance from friction limits and high internal grasp forces/object pressure. Its optimization assumes the resulting wrenches are realizable and does not enforce joint torque limits (§II); those assumptions should not be inherited uncritically for LEAP. This is a force-allocation foundation, not proof that a high contact count or an equilibrium solution guarantees closed-loop dynamic stability.

### S6. Ford et al. — a modern tactile grasp-regulation example

**Christopher J. Ford, Haoran Li, Manuel G. Catalano, Matteo Bianchi, Efi Psomopoulou, and Nathan F. Lepora. _Shear-based Grasp Control for Multi-fingered Underactuated Tactile Robotic Hands_.** arXiv:2503.17501v1, 2025-03-21.

- Record: <https://arxiv.org/abs/2503.17501>
- Versioned full text: <https://arxiv.org/html/2503.17501v1>

**Direct evidence.** A Pisa/IIT SoftHand uses optical microTac fingertip sensing. §§IV-A–B train tactile-image predictions of contact geometry and three force components against ATI Mini40 transducer measurements. Marker deformation is thus **not itself a force reading**; force outputs require calibration/supervision. §IV-C uses changes in shear force to regulate closure, with orientation compensation in Eqs. (2)–(5). Rotational shear/torque is omitted (§IV-A). A single actuator prevents independent fingertip-force control; without measured finger configurations, force combination uses approximate orientations (§IV-C/§VII).

This is evidence for closing the loop on tactile shear rather than just binary contact. It is not an inertial-identification study, a four-independent-finger controller, or a friction-cone guarantee. Its gains and calibration ranges should not be transplanted as LEAP settings.

## 2. Established equations and their identification consequences

This section is an explicit derivation/synthesis from rigid-body mechanics, S1's Newton–Euler formulation, S3's inertial representation, and S5's contact-wrench map. Symbols and equation labels below belong to this note, not to the source papers.

### 2.1 Choose one known body origin, not an unknown COM measurement frame

Let `O` be a known frame rigidly attached to the object, with axes fixed in the object. Express all following quantities in those axes:

- `m`: mass; `c`: COM position relative to O; `h = m c`: first moment.
- `I_O`: symmetric rotational inertia about O; `I_C`: inertia about the COM with the same axis orientation.
- `a_O`: **inertial linear acceleration of O**, expressed in body axes; `ω, α`: angular velocity and angular acceleration; `g`: gravity in those axes.
- `S(x)y = x × y`; `u = a_O − g`.
- `F, τ_O`: sum of non-gravitational external forces and their moments about O.

Then

\[
F = m u + [S(\alpha)+S(\omega)^2]h,
\tag{N1}
\]
\[
\tau_O = -S(u)h + I_O\alpha + S(\omega)I_O\omega.
\tag{N2}
\]

The acceleration relation and parallel-axis conversion are

\[
a_C=a_O+\alpha\times c+\omega\times(\omega\times c),
\qquad
I_C=I_O-m[(c^Tc)I_3-cc^T].
\tag{N3}
\]

Do not confuse `d(v_O expressed in body axes)/dt` with `a_O`: the changing basis contributes `ω × v_O`. Likewise, a simulator's truth-COM acceleration is not an independently available deployment measurement when COM is an estimation target. A known-origin pose/kinematic observation model avoids that circularity.

Use the ten-vector

\[
\pi=[m,h_x,h_y,h_z,I_{O,xx},I_{O,yy},I_{O,zz},I_{O,xy},I_{O,xz},I_{O,yz}]^T.
\]

Here the off-diagonal quantities are the **actual symmetric matrix entries**, not unsigned products of coordinates. Define

\[
L(v)=\begin{bmatrix}
v_x&0&0&v_y&v_z&0\\
0&v_y&0&v_x&0&v_z\\
0&0&v_z&0&v_x&v_y
\end{bmatrix},
\quad I_Ov=L(v)\,\mathrm{vech}(I_O).
\]

The standard linear regressor is

\[
\begin{bmatrix}F\\\tau_O\end{bmatrix}
=
\underbrace{\begin{bmatrix}
u&S(\alpha)+S(\omega)^2&0_{3\times6}\\
0_{3\times1}&-S(u)&L(\alpha)+S(\omega)L(\omega)
\end{bmatrix}}_{Y_t\in\mathbb R^{6\times10}}
\pi.
\tag{N4}
\]

The `u` in the first block is a 3×1 column. COM and `I_C` are recovered after estimating `π`; fitting `m`, `c`, and `I_C` as if all enter linearly would be incorrect.

### 2.2 Why static firm contacts cannot identify full inertia

For an object at rest, `a_O = ω = α = 0`. Thus

\[
F=-mg,\qquad \tau_O=-h\times g.
\tag{N5}
\]

**Derived consequences under known gravity and a complete calibrated net-wrench measurement:**

- One static orientation gives mass and two components of `h`; the COM component parallel to gravity is invisible. The ten-parameter regressor has rank at most **three**.
- Several static orientations with nonparallel gravity directions can identify mass and all three first moments. They still provide **zero sensitivity to all six inertia entries**.
- Pure translation with `ω = α = 0` likewise leaves the six inertia columns identically zero, although varying `u` can improve mass/COM identification.
- A fixed body-axis spin is generally insufficient for the full tensor: for `ω` and `α` along body z, only the inertia entries in the z column enter the rotational terms. Different angular directions and acceleration histories are needed to distinguish the remaining combinations.

A physically plausible posterior or a small residual on these trajectories does not establish unique full-tensor recovery. Priors can select one plausible answer from many dynamically indistinguishable answers.

### 2.3 Contact measurement and the internal-force nullspace

For ideal point contacts, with `f_i` the force **on the object** and `r_i` the contact location relative to O,

\[
w=\begin{bmatrix}F\\\tau_O\end{bmatrix}
=\sum_i\begin{bmatrix}I_3\\S(r_i)\end{bmatrix}f_i
=Gf.
\tag{N6}
\]

If sensors instead report force on the finger, reverse the sign. Transform all forces, contact positions, normals, and kinematics into consistent frames and synchronize them. All other external contacts must be measured/modeled or excluded.

Given a desired wrench, a compatible force solution has the algebraic form

\[
f=G^\dagger w+(I-G^\dagger G)z.
\tag{N7}
\]

The second term is in `ker(G)` and produces **no net object wrench**. It can supply squeeze and friction margin; arbitrarily increasing it is not inertial excitation. The pseudoinverse expression is not a complete controller: unilateral contact, friction, force bounds, and actuator feasibility still apply.

Unknown contact positions imply unknown lever arms. Scalar normal forces cannot reconstruct the full force vector when shear is present; even three-axis forces do not determine `r_i × f_i` without locations. Large canceling contact forces can make a small net inertial wrench sensitive to force biases and contact-position errors. Four pulp contacts also do not certify that there are no unobserved palm, side-link, mounting-frame, or table forces.

For a soft contact, the wrench includes a patch moment `m_i`:

\[
\tau_O=\sum_i[r_i\times f_i+m_i].
\tag{N8}
\]

In a common soft-finger model `m_i` includes torsion about the contact normal. It is not recoverable from a three-axis resultant force alone. A pressure centroid/center of pressure can represent some distributed-contact moments but not an independent pure torsional couple. Either sense/model the moment, validate an approximation that keeps it negligible, or include model uncertainty. S1 explicitly chooses the approximation rather than establishing that torsion never matters.

### 2.4 “Firm” should mean bounded compressive load with friction margin

Choose contact normal `n_i` so positive force into the object is compressive:

\[
f_{n,i}=n_i^Tf_i>0,\qquad
f_{t,i}=(I_3-n_in_i^T)f_i,\qquad
\|f_{t,i}\|\le\mu_i f_{n,i}.
\tag{N9}
\]

This explains why `||f_i|| > threshold` is not a normal-force requirement and can reward shear near slip. A useful load band is `f_n,min ≤ f_n,i ≤ f_n,max`, with a reserve on `μ_i f_n,i − ||f_t,i||`. These quantities require normals and appropriately resolved forces.

For soft fingers, shear and torsion share a contact limit; an illustrative elliptical approximation is

\[
\left(\frac{\|f_t\|}{\mu f_n}\right)^2+
\left(\frac{\tau_n}{\tau_{\max}(f_n)}\right)^2\le1.
\tag{N10}
\]

This is a model family, **not a universally calibrated law for LEAP pulps**. Patch geometry/material and normal load affect torsional capacity. Treating independent maximal shear and maximal torsion as simultaneously available is unjustified (S5 §II-B). A no-slip force reading only supplies a lower bound on the friction coefficient under the Coulomb model; S1's friction experiment deliberately approaches slip to locate the boundary.

## 3. Physical consistency and information criteria

### 3.1 A realizable tensor, not merely an SPD matrix

With the same origin convention as above, define

\[
\Sigma_O=\tfrac12\operatorname{tr}(I_O)I_3-I_O,
\qquad
J(\pi)=\begin{bmatrix}\Sigma_O&h\\h^T&m\end{bmatrix}.
\tag{N11}
\]

S3's Theorem 3 uses `J ≻ 0` for a nondegenerate mass distribution. Its central covariance formulation also describes the semidefinite boundary: with `m > 0`, `J ≽ 0` is equivalent to `Σ_C = Σ_O − hhᵀ/m ≽ 0`, permitting degenerate mass measures. For an ordinary volumetric object, a carefully scaled strict interior constraint is appropriate; indiscriminate numerical eigenvalue floors have units and can distort small-object inertia.

For principal moments `λ₁, λ₂, λ₃`, physical realizability requires

\[
\lambda_i\ge0,\qquad
\lambda_1+\lambda_2\ge\lambda_3
\quad\text{and cyclic permutations}.
\]

SPD alone fails this test: `diag(1,1,3)` is positive definite but violates a triangle inequality. Conversely, physical consistency does not identify an unexcited direction. Known geometry can tighten constraints, but COM inside the convex hull alone does not ensure that the second moments fit the object's support (S3 §V).

### 3.2 Information must reflect noise, units, and missing directions

For a local linear measurement model `w_t = Y_t π + ε_t`, with known independent Gaussian wrench covariance `R_t` and effectively known regressors,

\[
\mathcal I_\pi=\sum_tY_t^TR_t^{-1}Y_t.
\tag{N12}
\]

Actual pose derivatives, force estimates, and contact locations are noisy, making this an errors-in-variables problem. Unmodeled force offsets, friction, contact motion, torsion, and time offsets can mimic parameter changes. Equation (N12) is a conditional approximation, not automatically calibrated uncertainty. Correlated samples require a stacked covariance or an appropriate temporal model; a faster sample rate alone must not be counted as independent evidence.

**Recommendation based on S2/S4:** use fixed task-level characteristic scales, e.g. `D = diag(m₀, m₀ℓ₀, …, m₀ℓ₀², …)` with one mass, three first-moment, and six inertia entries, and define `π = D θ` for dimensionless `θ`. Then

\[
\widetilde Y_t=R_t^{-1/2}Y_tD,\qquad
\mathcal I_\theta=\sum_t\widetilde Y_t^T\widetilde Y_t.
\tag{N13}
\]

Do not use each episode's hidden true inertial parameters to normalize the policy's information reward.

- **A-optimal:** minimize `tr(I_θ⁻¹)`; matches ASID's criterion once a parameterization/scale is fixed.
- **D-optimal:** maximize `log det(I_θ)`; matches the determinant-based uncertainty principle in S4.
- **E-optimal diagnostic/objective:** improve the smallest eigenvalue, exposing the weakest identifiable direction.
- A regularized incremental objective such as `log det(Λ₀+I_{t+1}) − log det(Λ₀+I_t)` is a reasonable design proposal, **not ASID's exact reward**. Keep the unregularized data spectrum visible: a prior makes the matrix invertible without creating measurements.

Report singular values/rank of the whitened, scaled regressor, directional uncertainty, and held-out wrench prediction as well as parameter error. Trace alone can grow while a missing direction stays invisible; condition number alone ignores absolute information. If nuisance parameters are estimated jointly, inspect their coupling and marginalized inertial information, not just the inertial block as if nuisance quantities were known.

## 4. Recommendations for task redesign — not published LEAP results

1. **Separate acquisition, stabilization, and identification.** Start with a feasible pregrasp or an acquisition policy, then require four actual pulp contacts in bounded normal-force bands before starting an identification window. Random surface-point reaching/regrasping is a different objective and need not be part of the estimation baseline. Do not prescribe universal force thresholds from these papers; calibrate against the real pulps, object strength, force resolution, and actuators.
2. **Regulate support and preload separately from object motion.** Maintain enough internal force for friction margin while commanding an informative net wrench/pose trajectory. Penalize excess squeeze, shear utilization, saturation, and unobserved support contacts. Contact count and low drift are support metrics, not information metrics.
3. **Preserve contacts without demanding frozen geometry.** Coordinated finger motion, rolling, and compliant contact can preserve support while translating/rotating the object. Penalizing all object velocity or rigidly freezing every fingertip makes identification impossible. Contact persistence is compatible with excitation only within a feasible coordinated-motion subspace. Regrasp should be a recovery/repositioning option with measurement validity handled explicitly, not necessarily the normal action during a clean estimation window.
4. **Start with bounded, smooth, multi-direction experiments.** Low-amplitude periodic translations and rotations about more than one axis are a useful controlled baseline. Increase amplitude/bandwidth only while contact and wrench-model validity remain acceptable. S4 motivates periodic bandlimited design; it does not establish a safe frequency or amplitude for LEAP. Optimize the measured/estimated information under force, friction, joint, collision, workspace, and bandwidth constraints—not arbitrary joint action variance.
5. **Treat the fixed wrist as a feasibility constraint.** With no arm motion, all useful object motion must be produced inside the hand workspace. If fingers and object are frozen relative to a fixed palm, there is no excitation; stronger squeeze cannot fix it. If safe finger-only motion yields poor inertia sensitivity, either narrow the claimed identification target or add bounded wrist/arm motion. The arm-mounted platform in S1 is not evidence that a fixed-wrist LEAP achieves the same excitation.
6. **Build the sensing contract before the reward.** Needed quantities are time-aligned object pose/kinematics at a known origin, contact locations/normals, force vectors, and accounting for any contact moments/other supports. Distinguish simulator truth used for evaluation from channels realistically available to the estimator/policy. A normal-only sensor needs a justified extra model or additional sensing, not silent substitution for a full wrench.
7. **Use a held-out validity gate.** Compare mass, COM, all six body-frame inertia entries, physical consistency, identifiable subspace, and prediction of new motion/wrenches. Include force biases, pose/contact uncertainty, timing, and contact-model mismatch in evaluation. For nearly equal principal moments, principal-axis directions are intrinsically ill-conditioned; compare tensors in the agreed body frame rather than overinterpreting eigenvector errors.

### Key evidence tensions to retain

- **Firm contacts versus information:** S5 explains why preload helps slip margin; N5–N7 explain why preload is not full inertial excitation. These findings are complementary, not contradictory.
- **Tactile feasibility versus complete identification:** S1 demonstrates a sensing/inference construction but explicitly leaves active excitation unresolved. S2 motivates exploration design but does not validate S1's tactile sensing stack or full-tensor recovery. Combining them is a research proposal.
- **Physical consistency versus observability:** S3 removes impossible solutions; N5 shows that many possible solutions can still fit static data. A constrained estimate is not a uniqueness certificate.
- **Persistence versus contact-model validity:** S1 tolerates some contact changes but excludes significant soft-finger torsion. Requiring persistent soft contact may improve safety while increasing unmodeled torsional loading; this must be checked rather than assumed beneficial.

## Review limitations

The substantive citations above are primary papers, read through arXiv full text or author/institutional PDF copies. No survey was used as the authority for the conclusions. Supplemental code, raw experimental data, and source-paper implementations were not audited. No new experiments or measured LEAP results are claimed. PDF page anchors are deliberately not supplied; the stable locators are section/equation names. S4/S5 PDF extraction did not reliably preserve every mathematical glyph, so the newly written equations are explicitly labeled as standard mechanics/synthesis rather than verbatim transcriptions. The recommendations require reconciliation with the separate task-code and estimator audits and hardware sensing capabilities.
