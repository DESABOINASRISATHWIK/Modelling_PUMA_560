# PUMA-560: kinematics and MuJoCo physics

**Part 1 physics is implemented.** See [PART1.md](PART1.md) for the torque-driven
`PhysicsSimulation` API, scene/payload configuration, collision checking, gravity
derivation, model assumptions and validation. Run `python examples/physics_demo.py`
for a headless PD + gravity demonstration. The original viewer remains available.

Robotics course project, objectives 1-4: robot specification, DH model, forward/inverse kinematics (including
unreachable targets), and velocity kinematics with singularity analysis.
Kinematics algorithms are our own **NumPy** code; **MuJoCo** provides an independent
kinematics reference and the Part 1 physics/contact engine. It is never called by
FK / IK / Jacobian / singularity code.

```
src/robot_parameters.py     ONE place for all robot numbers (DH table, limits, masses, frames, conventions)
src/dh_model.py             generic DH transform + chain; prints the specification
src/forward_kinematics.py   FK (pose, all frames, wrist centre)
src/inverse_kinematics.py   closed-form IK, 8 branches, limits, reachability diagnostics   <- primary IK
src/numerical_ik.py         secondary damped-least-squares IK (cross-check only)
src/jacobian.py             geometric Jacobian (+ finite-difference validator, resolved-rate inverse)
src/velocity_kinematics.py  velocity ellipsoids, directional gain, singular directions, joint-speed blow-up
src/singularity_analysis.py SVD metrics, closed-form singularity conditions, singular-pose builders
src/mujoco_reference.py     MuJoCo model loader / reference FK (verification only)
src/simulation.py           interactive MuJoCo viewer + overlays + offscreen renders
src/visualization.py        matplotlib figures
src/verification.py         the numerical checks used by tests/ and examples/verification_report.py
tools/build_mjcf.py         generates models/puma560.xml        models/LICENSE_NOTICE.md  provenance + licences
tests/                      53 pytest tests                      examples/  demos + verification report
results/                    figures + verification_report.{md,json} (generated)
```

## Install and run

```bash
pip install -r requirements.txt
python -m pytest tests -q                      # 53 tests (about 10 s)
python examples/fk_demo.py                     # FK: tables, frames, MuJoCo cross-check, figures
python examples/ik_demo.py                     # IK: 8 branches, unreachable, joint-limit-infeasible
python examples/workspace_demo.py              # reachable workspace + analytic bound
python examples/singularity_demo.py            # SVD, sweeps, sigma_min map, gallery
python examples/velocity_demo.py               # velocity kinematics: J, ellipsoids, lost directions, joint-speed blow-up
python examples/verification_report.py         # all numbers quoted below -> results/
python -m src.simulation                       # INTERACTIVE MuJoCo demo (live evaluation)
python -m src.simulation --target 0.55 0.25 0.85 180 0 0     # start with a Cartesian target [m, deg]
```
Headless Linux (no display): prefix with `MUJOCO_GL=egl` (or `osmesa`); examples then still save MuJoCo renders.

### Interactive simulation
The viewer starts with opaque orange, silver and graphite links, with frames and diagnostic overlays hidden.
Diagnostic geometry is disabled in the renderer, including wrist markers, coordinate axes, velocity arrows and ellipsoids.
The XML and its generator share the same appearance settings. Optional overlays are drawn **from our own code**: DH frames {0}..{6}, the flange frame (large
triad), wrist centre (magenta), the target pose (translucent yellow triad), every IK branch as a coloured skeleton
(selected branch thick; branches violating joint limits are greyed), a singularity marker above the end effector
(green regular, orange sigma_min<0.05, red near-singular <0.01, magenta singular <1e-6) and optionally a workspace
cloud coloured by sigma_min. The left text panel shows q, EE pose, wrist centre, target, IK status, selected branch,
singularity status, sigma_min / condition / rank and the three singularity factors.

MuJoCo's own viewer binds almost every *letter* key, so project controls use other keys:

| input | action |
|---|---|
| Up / Down, Left / Right, PageUp / PageDown | move the Cartesian target +x/-x, +y/-y, +z/-z (2 cm) |
| Home / End | previous / next IK branch |
| Insert | move the robot to the selected IK branch |
| Delete | target := current end-effector pose |
| Enter | toggle workspace cloud |
| terminal `qd d1..d6` / `twist vx vy vz wx wy wz [dls]` | set joint velocity [deg/s], or a task twist resolved to joint velocity (pseudo-inverse or damped LS); the viewer then draws the end-effector linear (orange) and angular (purple) velocity arrows and the translucent linear velocity ellipsoid (`show velocity on/off`) |
| terminal (type `help`) | `q a1..a6`, `joint j deg`, `pose x y z roll pitch yaw`, `nudge ...`, `preset zero/ready/stretch/nominal/sing_shoulder/sing_elbow/sing_wrist/near_wrist`, `ik`, `next`, `prev`, `branch n`, `go`, `qd`, `twist`, `show frames/branches/workspace/singular/velocity`, `alpha a`, `info` |
| viewer "Joint" panel | joint sliders (the loop adopts changes made there) |

## 1-2. Robot specification and DH model

Parameter set: Corke & Armstrong-Helouvry (1994) / Armstrong, Khatib & Burdick (1986) as in Peter Corke's Robotics
Toolbox; flange offset and meshes from ARTE (see `models/LICENSE_NOTICE.md`). Units: m, rad, kg, kg m^2.

**Standard DH**: `A_{i-1,i} = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)`, `theta_i = q_i + offset_i` (all offsets 0).

| i | joint | theta | d [m] | a [m] | alpha | q range [deg] |
|---|---|---|---|---|---|---|
| 1 | waist | q1 | 0.67183 (26.45 in) | 0 | +90 | -160 .. 160 |
| 2 | shoulder | q2 | 0 | 0.4318 | 0 | -45 .. 225 |
| 3 | elbow | q3 | 0.15005 | 0.0203 | -90 | -225 .. 45 |
| 4 | forearm roll | q4 | 0.4318 | 0 | +90 | -110 .. 170 |
| 5 | wrist pitch | q5 | 0 | 0 | -90 | -100 .. 100 |
| 6 | wrist roll | q6 | 0 | 0 | 0 | -266 .. 266 |

Tool flange: `T_6E = Trans(z, 0.04 m)` (Corke uses 0; changing `D_FLANGE` is all that is needed, IK/Jacobian follow).

* **Base frame {0}** = MuJoCo world: origin on the waist axis at the mounting plane, z up, x "forward".
* **Joint axes**: joint i rotates about z_{i-1}, right-hand rule. At q = 0 ("L pose"): z0=+z, z1=z2=-y, z3=+z, z4=-y, z5=+z.
* **Wrist-centre frame**: origins of {4},{5},{6} coincide (a4=a5=a6=d5=d6=0): spherical wrist, wrist centre
  `p_w = p_3 + d4 z_3`. **End-effector frame {E}**: flange, z_E = approach axis.
* **Masses [kg]** 0 (m1 not published), 17.4, 4.8, 0.82, 0.34, 0.09 with COM and inertia tensors in `robot_parameters.py`.
* `python -c "from src.dh_model import describe_robot; print(describe_robot())"` prints the whole specification.

**MuJoCo <-> DH relationship** (built so it is *not* assumed): `models/puma560.xml` is written from absolute
zero-pose joint points/axes (world-aligned body frames, no DH matrices; `tools/build_mjcf.py` documents the table). It
was then *found* and is *tested* (`tests/test_model_consistency.py`, `test_fk.py`) that: world = {0}; `qpos_i = q_i`
(same sign and zero); joint-axis directions and anchor lines equal z_{i-1} and p_{i-1} of the DH chain at q=0;
`ee_site` = {6}·Trans(z,0.04) with the same orientation; `wrist_center_site` = origin of {4}.
Negative control: a 1 mm error in a3 or a flipped alpha4 is detected by the MuJoCo comparison.

## 3. Forward kinematics

`T_0E = A1(q1) A2(q2) A3(q3) A4(q4) A5(q5) A6(q6) T_6E`, returning `FKResult` (final 4x4 pose, position, rotation,
all T_0i, all relative A_i, wrist centre). Closed form of the wrist centre (used by IK):

`p_w = Rz(q1) [X, -d3, 0]^T + [0, 0, d1+Y]^T`, `[X, Y]^T = Rot(q2) [a2+u, v]^T`,
`u = a3 cos q3 - d4 sin q3`, `v = a3 sin q3 + d4 cos q3`.   Wrist: `R_36 = Rz(q4) Ry(-q5) Rz(q6)`.

## 4. Analytical inverse kinematics (`inverse_kinematics.py`)

1. **Wrist centre**: `T_06 = T_target T_6E^-1`, `p_w = (px,py,pz)` = its translation.
2. **Waist**: `r = hypot(px,py)`, `X = ±sqrt(r^2 - d3^2)`, `q1 = atan2(py,px) + atan2(d3, X)` (two branches).
3. **Elbow**: `rho^2 = X^2 + (pz-d1)^2 = a2^2 + L^2 + 2 a2 L cos(q3+phi)`, `L = sqrt(a3^2+d4^2)`, `phi = atan2(d4,a3)` ->
   `q3 = -phi ± acos((rho^2-a2^2-L^2)/(2 a2 L))` (two branches).
4. **Shoulder**: `q2 = atan2(Y, X) - atan2(v, a2+u)`.
5. **Wrist** (ZYZ Euler of `R_36 = R_03^T R_06`, middle angle `-q5`): `beta = atan2(sqrt(R02^2+R12^2), R22)`,
   `q4 = atan2(R12,R02)`, `q6 = atan2(R21,-R20)`, `q5 = -beta`; flipped: `q4+pi, q5=+beta, q6+pi`.
6. 2 x 2 x 2 = **8 branches**; every branch is re-verified with FK, 2π-equivalents are chosen inside the limits closest to
   `q_ref`, then joint limits are checked per branch. `result.closest_to(q_ref)` selects the nearest feasible branch.

**Reachability conditions**: (A) `r >= |d3|` (wrist centre outside the cylinder of radius |d3| about the waist axis);
(B) `|a2 - L| <= rho <= a2 + L` (a shell; outer limit 0.8641 m). Status codes: `ok`, `unreachable`
(`too_far` / `too_close` / `wrist_inside_shoulder_cylinder`, with margins in mm), `joint_limits` (geometrically reachable, every
branch violates a limit). Singular targets (X=0, sin(q3+phi)=0, sin q5=0) never crash: coinciding branches are merged and
flagged; at a wrist singularity one member of the one-parameter family is returned (q4 = reference).

## 5. Velocity kinematics (`jacobian.py`)

`J_v,i = z_{i-1} × (p_e - p_{i-1})`, `J_w,i = z_{i-1}`, `[v; w] = J(q) qdot`, 6x6, base frame, reference point = flange
(options: `frame="ee"`, `point="wrist"`). Derivation: rotating joint i alone turns the distal chain rigidly about z_{i-1};
velocities superpose. `jacobian_numerical` (central differences) exists only for validation.
`joint_velocity_from_twist` gives the resolved-rate inverse (SVD pseudo-inverse or damped least squares).

## 5b. Velocity kinematics in more depth (`velocity_kinematics.py`, `examples/velocity_demo.py`)

* **Forward velocity** `[v; w] = J(q) qdot`; **inverse velocity** `qdot = J^+ [v; w]` (SVD pseudo-inverse) or damped least
  squares `J^T (J J^T + lambda^2 I)^-1 [v; w]`. Frame choice (`base` / `ee`) and reference point (`flange` / `wrist`) are
  options; the wrist-centre Jacobian is block triangular and differs from the flange Jacobian by the unimodular map
  `[[I, -[r]x],[0, I]]` (tested, same determinant).
* **Velocity ellipsoids**: unit-norm joint velocities map under `J_v` (or `J_w`) onto an ellipsoid whose semi-axes are
  the singular values along the columns of U (`J_part = U S V^T`). **Directional gain** along a unit direction u is
  `|J_part^T u|` and always lies between the smallest and largest semi-axis.
* **Singular directions**: the smallest singular value gives `u_min`, the twist that is hard/impossible to produce, and
  `v_min`, the joint "self-motion" with (almost) no end-effector effect. At a singularity `J v_min = 0` exactly. Found
  and tested for each family:
  * wrist: self-motion `qd4 = -qd6` (the two collinear axes cancel); the lost angular direction is perpendicular to z3;
  * elbow: the lost linear direction is along the stretched arm (shoulder to wrist centre);
  * shoulder: the lost linear direction is horizontal and perpendicular to the arm plane, `Rz(q1) y`.
* **Joint-speed blow-up**: for the weakest twist `|qdot| = |twist| / sigma_min` with the pseudo-inverse (3.75e4 rad/s at
  q5 = 1e-4), whereas damped LS is bounded by `1/(2 lambda)` (10 rad/s for lambda = 0.05) at the cost of task error.
  `examples/velocity_demo.py` prints all of this and writes `results/velocity_ellipsoids.png`,
  `results/joint_speed_vs_singularity.png`, and `results/velocity_mujoco.png` is a MuJoCo render with the overlays.

## 6. Singularities (`singularity_analysis.py`)

With the wrist-centre reference point, J is block lower-triangular, hence (verified to 1e-16)

**`det J = a2 · X · v · sin(q5)`**, `v = a3 sin q3 + d4 cos q3 = L sin(q3+phi)`

| type | condition | meaning |
|---|---|---|
| shoulder | `X = a2 cos q2 + a3 cos(q2+q3) - d4 sin(q2+q3) = 0` | wrist centre on the cylinder radius \|d3\| about the waist axis: it cannot move horizontally perpendicular to the arm plane |
| elbow | `sin(q3+phi) = 0`, phi = 87.3° | arm stretched (q3 = -87.3°, workspace boundary) or folded (q3 = 92.7° ≡ -267.3°, outside the joint limits) |
| wrist | `sin q5 = 0` | axes of joints 4 and 6 collinear: only q4 ± q6 controllable |

Numerics use `J = U Σ Vᵀ`: all singular values, `sigma_min`, rank, condition number, manipulability. Status:
singular if `sigma_min < 1e-6` or rank < 6; near-singular if `sigma_min < 0.01` (threshold chosen from data: median
sigma_min over random valid poses 0.09, 8 % below 0.01 ≈ within 2° of a wrist singularity). Caveat: J mixes m/rad and rad/rad rows, so absolute sigma values depend on units; the zeros and relative trends are meaningful.

## Verification (actual results, `results/verification_report.md`, fixed seeds)

| check | result |
|---|---|
| FK vs MuJoCo `ee_site`, 2000 random valid q | position max 6.8e-16 m (RMS 2.2e-16); orientation max 7.4e-16 rad (RMS 2.6e-16) |
| IK round trip, 3000 random valid q | original q recovered 3000/3000; always 8 branches; every branch re-checked by FK: position max 9.6e-16 m, orientation max 2.8e-12 rad |
| branches inside joint limits | 1..8 per target (mean 4.67; 77 of 3000 targets have only one) |
| workspace boundary (arm stretched) | solved, 4 merged branches flagged elbow-singular, error 5e-16; the same pose pushed 1.01 mm outward -> `unreachable / too_far` |
| unreachable | target 2.2 m too far and wrist centre on the waist axis -> `unreachable` with reason and mm margin, no solutions |
| geometric vs joint-limit | 4000 random-angle poses: 2026 `joint_limits` (all 8 branches violate), 1974 `ok`; the two statuses are distinct |
| singular targets (50 each) | shoulder/elbow: 4 branches, wrist: 7 branches; all flagged, max FK error 7e-15 |
| near-singular (eps = 1e-3..1e-9) | all solved; max error 1e-9 (wrist) / 1e-14 (shoulder, elbow) |
| analytic vs secondary DLS IK | DLS converged 97/100 from perturbed starts (mean 19.5 iterations); 97/97 landed on an analytic branch |
| Jacobian vs central differences, 1000 q | max abs error 2.6e-10 (translational 2.6e-10, rotational 2.4e-10), RMS 4.8e-11 |
| `det J` vs closed form | max difference 1.2e-16 |
| velocity: ellipsoid axes, directional gain, twist round trip (500 q) | axes² vs eigenvalues of `J_part J_partᵀ` max diff 5.3e-15; no directional gain outside [smallest, largest] semi-axis; twist→qdot→twist error 2.2e-14 |
| velocity: singular directions (50 poses per type) | lost direction aligned with the predicted geometric direction (shoulder, elbow, wrist: ≥ 0.999999); `‖J·self-motion‖` ≤ 5.6e-15 |
| velocity: joint speed, weakest twist, wrist singularity | q5 = 0.1 / 0.01 / 0.001 / 0.0001: pseudo-inverse 38 / 375 / 3.75e3 / 3.75e4 rad/s, damped LS 8.2 / 1.07 / 0.11 / 0.011 rad/s (bound 10) |
| regular pose | sigma_min 0.0877, cond 21.6, rank 6 |
| near-singular pose (q5 = 0.02) | sigma_min 5.3e-3, cond 354, rank 6 |
| singular poses (shoulder / elbow / wrist) | sigma_min 1.6e-16 / 1.3e-18 / 4.4e-17, rank 5 each, only the matching analytic factor vanishes |

`results/*.png`: `fk_frames`, `fk_mujoco`, `ik_branches`, `ik_mujoco_branches`, `workspace` (wrist centre fills the
analytic sphere of radius `sqrt((a2+L)^2+d3^2) = 0.877 m` and respects r ≥ \|d3\|), `singular_value_sweeps`,
`sigma_min_map` (analytic curves X=0 and sin(q3+phi)=0 lie exactly on the numerical sigma_min valleys),
`singularity_gallery`, `singularity_mujoco`, `velocity_ellipsoids`, `joint_speed_vs_singularity`, `velocity_mujoco`.

## Known limitations and honest notes

* **Joint convention**: q = 0 is Corke's DH "L pose". Unimation's own joint-angle convention (and manuals) use different
  zeros; add `THETA_OFFSET` if you must match them. Sources disagree on some limits (ARTE lists ±110° for joint 2);
  we use Corke's, all in one array.
* **Flange offset 0.04 m** is a choice (ARTE 0.04, Corke 0, ROS URDF 0.0558) and is a single constant.
* **Dynamics assumptions:** Link 1 mass is a placeholder and link 3's published inertia was
  minimally corrected (see `models/LICENSE_NOTICE.md`).
* **IK** is valid for this PUMA DH structure only (raises `NotImplementedError` otherwise). At `|sin q5| < 1e-8` the returned
  wrist solution is one member of an infinite family (pose error ≤ ~3e-8 rad, trade-off documented in `inverse_kinematics.py`).
  Joint limits are checked per branch but IK does not search for non-closed-form workarounds.
* **Secondary DLS IK** may fail to converge (3/100 in the test) and is not used by anything else.
* **Interactive viewer**: the scripted self-test (`--selftest`) and all controls' *logic* were run and tested under a
  virtual display (Xvfb); I could not click GUI sliders or press keys on a real display, so those two input paths
  (viewer "Joint" sliders, keyboard) are untested by hand. On the virtual display the process ends with a segmentation
  fault at exit (Mesa/Xvfb teardown; also happens with a bare 3-line viewer script), not an error in the project.
* **Meshes**: licence/authorship should be checked upstream before public redistribution (`models/LICENSE_NOTICE.md`).
* Sigma values mix units (see above); the near-singular threshold is heuristic. The *linear* velocity ellipsoid alone
  need not collapse exactly at a singularity (the lost twist can have a small angular part); the exact statements are the
  singular directions of the full 6x6 Jacobian. Ellipsoids are drawn for unit-norm joint velocity, so their size depends on units.
* The velocity arrows/ellipsoid in the viewer are *drawn* from our Jacobian; no motion is simulated (no integration, control or trajectory).
* The original kinematics viewer does not execute dynamics. Use the Part 1 API for
  torque control, contacts, obstacles and payloads; trajectory planning and grasping remain separate work.

## Viva cheat-sheet

* Why 8 solutions? 2 (±X waist) × 2 (±acos elbow) × 2 (±beta wrist).
* Why is IK closed-form? Spherical wrist ⇒ wrist centre depends on q1..q3 only; R_36 is a ZYZ Euler rotation.
* Why `det J = a2 X v sin q5`? Reference the wrist centre: J is block triangular, `det = det(J_p)·det[z3 z4 z5]`,
  `det J_p = -X a2 v`, `det[z3 z4 z5] = -sin q5`.
* Which singularity is on the boundary? Stretched elbow (it is exactly ρ = a2 + L); folded elbow is outside the limits.
* What does a singularity mean at velocity level? `J` loses rank: some twist (`u_min`) cannot be produced and a joint self-motion (`v_min`) moves nothing; pseudo-inverse joint speed grows like `1/sigma_min`.
* How do you know the DH table is right? Independent MuJoCo tree agrees to 1e-16; wrong a3 / alpha4 is detected.
