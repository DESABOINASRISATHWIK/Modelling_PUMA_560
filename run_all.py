#!/usr/bin/env python
"""
run_all.py  -  ONE command: forward kinematics, inverse kinematics, velocity kinematics, and the
robot moving to the pose in the MuJoCo simulation.

Give ONE pose, either as joint angles or as a Cartesian target (default: a sample pose):

    python run_all.py                                        # default sample pose
    python run_all.py --q 30 50 -70 40 60 -20                # joint angles [deg]
    python run_all.py --target 0.55 0.25 0.85 180 0 0        # Cartesian target: x y z [m], roll pitch yaw [deg]
    python run_all.py --target 0.55 0.25 0.85 180 0 0 --branch 3   # choose another IK branch
    python run_all.py --qdot 10 -5 15 25 -20 30              # joint velocity [deg/s] used in the velocity part

Useful options
    --no-sim            print everything, do not open the viewer
    --png out.png       also save an offscreen MuJoCo snapshot at the final pose
                        (headless Linux: MUJOCO_GL=egl or osmesa)
    --move-time 3       seconds the robot takes to move in the viewer (smooth joint-space interpolation,
                        for visualisation only - it is not a trajectory planner or controller)
    --verify            also run a quick numerical verification (FK vs MuJoCo, IK round trip, Jacobian)
    --duration 20       close the viewer automatically after this many seconds

What happens, in order
    1. pose resolution   (--q  or  --target -> analytic IK picks the branch closest to the READY pose / --branch)
    2. forward kinematics: end-effector pose, every DH frame, wrist centre, MuJoCo cross-check
    3. inverse kinematics: all branches of that pose with joint-limit status
    4. velocity kinematics: Jacobian, qdot -> twist, twist -> qdot (pseudo-inverse and damped LS),
       velocity ellipsoids, singularity status and lost directions
    5. simulation: the arm moves to the pose; overlays (DH frames, target, IK branches, velocity
       ellipsoid and arrows, singularity marker) are drawn from our own code.  Viewer controls:
       see `python -m src.simulation` / README (arrows, Home/End, Insert, terminal commands).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import robot_parameters as rp  # noqa: E402
from src.dh_model import PUMA560, describe_robot  # noqa: E402
from src.forward_kinematics import forward_kinematics  # noqa: E402
from src.inverse_kinematics import inverse_kinematics  # noqa: E402
from src.jacobian import geometric_jacobian, jacobian_numerical, joint_velocity_from_twist  # noqa: E402
from src.mujoco_reference import mujoco_fk  # noqa: E402
from src.singularity_analysis import analyze_configuration  # noqa: E402
from src.utils import matrix_to_rpy, orientation_angle, rpy_to_matrix, wrap_to_pi  # noqa: E402
from src.velocity_kinematics import directional_gain, singular_directions, velocity_ellipsoid  # noqa: E402

DEFAULT_Q_DEG = [30, 50, -70, 40, 60, -20]
DEFAULT_QDOT_DEG = [10, -5, 15, 25, -20, 30]
START_Q = rp.Q_READY                      # where the robot starts before moving to the pose

np.set_printoptions(precision=4, suppress=True, linewidth=130)


def section(title: str):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


# ----------------------------------------------------------------------------
# 1. pose resolution
# ----------------------------------------------------------------------------
def resolve_pose(args):
    """Return (q_goal or None, T_goal, ik_result_or_None, chosen_solution_or_None, mode_text)."""
    if args.target is not None:
        x, y, z, r, p, yw = args.target
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(*np.radians([r, p, yw]))
        T[:3, 3] = [x, y, z]
        res = inverse_kinematics(T, q_ref=START_Q)
        section("1. POSE: Cartesian target  ->  analytic IK picks a joint configuration")
        print(f"target position [m]  = {T[:3, 3]}   rpy [deg] = {np.array([r, p, yw])}")
        print(res.summary())
        if not res.solutions:
            print("\n-> target is geometrically UNREACHABLE; the robot will not move.")
            for k in ("r", "radial_margin", "rho", "rho_min", "rho_max", "outer_margin"):
                print(f"   {k:<14} = {res.diagnostics[k]:.4f}")
            return None, T, res, None, "target"
        if args.branch is not None:
            if not 0 <= args.branch < len(res.solutions):
                sys.exit(f"--branch must be between 0 and {len(res.solutions) - 1}")
            sol = res.solutions[args.branch]
        else:
            sol = res.closest_to(START_Q, feasible_only=True)
            if sol is None:
                print("\n-> reachable, but EVERY branch violates a joint limit; the robot will not move.")
                return None, T, res, None, "target"
        if not sol.within_limits:
            print(f"\n[warning] selected branch {args.branch} violates a joint limit: {sol.limit_violation.round(3)} rad")
        idx = next(i for i, s in enumerate(res.solutions) if s is sol)
        print(f"\nselected branch [{idx}] {sol.label}   q [deg] = {np.degrees(sol.q).round(3)}")
        return sol.q.copy(), T, res, sol, "target"
    q = np.radians(args.q if args.q is not None else DEFAULT_Q_DEG)
    section("1. POSE: joint angles")
    print(f"q [deg] = {np.degrees(q)}")
    return q, forward_kinematics(q).T, None, None, "joint"


# ----------------------------------------------------------------------------
# 2. forward kinematics
# ----------------------------------------------------------------------------
def run_fk(q):
    section("2. FORWARD KINEMATICS")
    fk = forward_kinematics(q)
    print("DH table (standard):  A_i = Rz(q_i) Tz(d_i) Tx(a_i) Rx(alpha_i)")
    print(f"{'i':>2} {'theta_off':>10} {'d [m]':>9} {'a [m]':>9} {'alpha [deg]':>12} {'q_i [deg]':>10}")
    for i, (off, d, a, al) in enumerate(PUMA560.table()):
        print(f"{i + 1:>2} {np.degrees(off):10.1f} {d:9.5f} {a:9.5f} {np.degrees(al):12.1f} {np.degrees(q[i]):10.3f}")
    for i, T in enumerate(fk.frames):
        print(f"\nT_0{i}  (frame {{{i}}} in the base frame):\n{T}")
    print(f"\nflange / end-effector pose T_0E =\n{fk.T}")
    r, p, y = np.degrees(matrix_to_rpy(fk.rotation))
    print(f"position [m]       = {fk.position}")
    print(f"orientation rpy    = [{r:.3f} {p:.3f} {y:.3f}] deg")
    print(f"wrist centre [m]   = {fk.wrist_center}   (common origin of frames {{4}},{{5}},{{6}})")
    ref = mujoco_fk(q)
    print(f"MuJoCo cross-check : position error {np.linalg.norm(ref[:3, 3] - fk.position):.2e} m, "
          f"orientation error {orientation_angle(ref[:3, :3], fk.rotation):.2e} rad")
    lim = PUMA560.limits
    inside = np.all((q >= lim[:, 0] - 1e-9) & (q <= lim[:, 1] + 1e-9))
    print(f"joint limits       : {'all joints inside their limits' if inside else 'VIOLATED -> ' + str(np.degrees(q).round(1))}")
    return fk


# ----------------------------------------------------------------------------
# 3. inverse kinematics
# ----------------------------------------------------------------------------
def run_ik(q, T, res):
    section("3. INVERSE KINEMATICS  (closed form, all branches for the end-effector pose above)")
    if res is None:                        # joint-angle mode: solve for the pose reached by q
        res = inverse_kinematics(T, q_ref=q)
    print(res.summary())
    d = res.diagnostics
    print(f"\nreachability: r = {d['r']:.4f} m (needs >= |d3| = {abs(PUMA560.d[2]):.4f}),  "
          f"rho = {d['rho']:.4f} m in [{d['rho_min']:.4f}, {d['rho_max']:.4f}]  "
          f"(margins: radial {d['radial_margin'] * 1e3:.1f} mm, outer {d['outer_margin'] * 1e3:.1f} mm)")
    best = res.closest_to(q, feasible_only=False)
    if best is not None:
        err = np.degrees(np.linalg.norm(wrap_to_pi(best.q - q)))
        print(f"branch closest to the commanded joints: {best.label}   (joint-space distance {err:.2e} deg)")
    return res


# ----------------------------------------------------------------------------
# 4. velocity kinematics
# ----------------------------------------------------------------------------
def run_velocity(q, qdot_deg):
    section("4. VELOCITY KINEMATICS  ([v; w] = J(q) qdot, base frame, flange point)")
    J = geometric_jacobian(q)
    qd = np.radians(qdot_deg)
    tw = J @ qd
    print(f"J(q) =\n{J}")
    print(f"\nmax |J - J_finite_difference| = {np.abs(J - jacobian_numerical(q)).max():.2e}")
    print(f"\nqdot = {np.array(qdot_deg)} deg/s  =  {qd} rad/s")
    print(f"-> end-effector twist:  v = {tw[:3]} m/s  (|v| = {np.linalg.norm(tw[:3]):.4f})"
          f"   w = {tw[3:]} rad/s  (|w| = {np.linalg.norm(tw[3:]):.4f})")
    back = joint_velocity_from_twist(q, tw, method="pinv")
    dls = joint_velocity_from_twist(q, tw, method="dls", damping=0.05)
    print("\nback from the twist:")
    print(f"  pseudo-inverse qdot = {np.degrees(back).round(4)} deg/s   |J qdot - twist| = {np.abs(J @ back - tw).max():.1e}")
    print(f"  damped LS (0.05)    = {np.degrees(dls).round(4)} deg/s   |J qdot - twist| = {np.abs(J @ dls - tw).max():.1e}")
    rep = analyze_configuration(q)
    print(f"\nsingularity analysis: {rep.text()}")
    print(f"singular values of J : {rep.svd.singular_values}")
    f = rep.factors
    print(f"closed-form factors  : X = {f['shoulder']:+.4f} (shoulder),  v = {f['elbow']:+.4f} (elbow),  "
          f"sin q5 = {f['wrist']:+.4f} (wrist);   det J = a2*X*v*sin(q5)")
    for part in ("linear", "angular"):
        e = velocity_ellipsoid(q, part)
        unit = "m/s per rad/s" if part == "linear" else "rad/s per rad/s"
        print(f"{part:<7} velocity ellipsoid semi-axes = {e.semi_axes}  [{unit}]")
        print(f"        principal directions (columns) =\n{e.axes}")
    print(f"directional gain (linear): along x {directional_gain(q, [1, 0, 0]):.4f}, "
          f"y {directional_gain(q, [0, 1, 0]):.4f}, z {directional_gain(q, [0, 0, 1]):.4f}")
    sd = singular_directions(q)
    print(f"weakest task direction (sigma_min = {sd.sigma_min:.3e}): v = {sd.lost_twist[:3]}  w = {sd.lost_twist[3:]}")
    print(f"joint self-motion with least end-effector effect:   qdot = {sd.null_motion}")
    return qd


# ----------------------------------------------------------------------------
# optional verification
# ----------------------------------------------------------------------------
def run_verify():
    from src import verification as V
    section("VERIFICATION (quick; full numbers: python examples/verification_report.py)")
    f = V.fk_vs_mujoco(n=300, seed=0)
    print(f"FK vs MuJoCo ({f['n']} random q): position max {f['position_error_m']['max']:.1e} m, "
          f"orientation max {f['orientation_error_rad']['max']:.1e} rad")
    i = V.ik_roundtrip(n=300, seed=1)
    print(f"IK round trip ({i['n']}): original recovered {i['recovered_original']}/{i['n']}, "
          f"branches {i['branch_count_histogram']}, FK error of branches max {i['position_error_m']['max']:.1e} m")
    j = V.jacobian_vs_numerical(n=200, seed=3)
    print(f"Jacobian vs finite differences (200 q): max {j['all']['max_abs']:.1e};  det J vs closed form {j['determinant_vs_closed_form_max']:.1e}")
    v = V.velocity_checks(n=100, seed=6)
    print(f"velocity: ellipsoid-vs-eig {v['ellipsoid_axes_vs_eig_max_err']:.1e}, twist round trip {v['twist_roundtrip_max_err']:.1e}, "
          f"singular-direction alignment >= {min(v[f'singular_{k}']['lost_direction_alignment_min'] for k in ('shoulder', 'elbow', 'wrist')):.6f}")


# ----------------------------------------------------------------------------
# 5. simulation
# ----------------------------------------------------------------------------
def build_sim(q_goal, T, res, sol, qd_rad):
    from src.simulation import PumaSimulation
    sim = PumaSimulation(START_Q)
    sim.set_target(T)
    if sim.ik and sim.ik.solutions and q_goal is not None:
        best = min(sim.ik.solutions, key=lambda s: float(np.linalg.norm(wrap_to_pi(s.q - q_goal))))
        sim.branch = next(i for i, s in enumerate(sim.ik.solutions) if s is best)
    sim.q_goal, sim.qd_goal = q_goal, qd_rad
    return sim


def run_simulation(sim, q_goal, qd_rad, move_time, duration):
    import mujoco
    import mujoco.viewer
    from src.simulation import COMMAND_HELP

    stop = threading.Event()

    def console():
        print("\n" + COMMAND_HELP)
        while not stop.is_set():
            try:
                line = input("puma> ")
            except (EOFError, KeyboardInterrupt):
                return
            if line.strip().lower() in ("quit", "exit"):
                stop.set()
                return
            print(sim.handle_command(line))

    threading.Thread(target=console, daemon=True).start()
    q0 = sim.q.copy()
    arrived = False
    t0 = time.time()
    section("5. SIMULATION: the robot moves to the pose" if q_goal is not None else
            "5. SIMULATION: target shown (robot stays put)")
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=sim.key) as v:
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 135, -20, 3.2
        v.cam.lookat[:] = [0.2, -0.1, 0.8]
        while v.is_running() and not stop.is_set():
            t = time.time() - t0
            if duration is not None and t > duration:
                break
            with sim._lock:
                if q_goal is not None and not arrived:
                    s = min(t / max(move_time, 1e-6), 1.0)
                    s = s * s * (3 - 2 * s)                       # smoothstep
                    sim.q = q0 + (q_goal - q0) * s
                    if s >= 1.0:
                        arrived = True
                        sim.q = q_goal.copy()
                        sim.set_qdot(qd_rad)                      # show twist arrows + ellipsoid at the pose
                        print("arrived.")
                else:
                    sim.adopt_ui_changes()
                sim.apply()
                with v.lock():
                    v.user_scn.ngeom = 0
                    sim.draw_overlays(v.user_scn)
                left, right = sim.hud()
                v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))
            v.sync()
            time.sleep(1 / 60)
    stop.set()


def main(argv=None):
    ap = argparse.ArgumentParser(description="FK + IK + velocity kinematics + simulation for one PUMA-560 pose",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--q", nargs=6, type=float, metavar="DEG", help="joint angles [deg]")
    g.add_argument("--target", nargs=6, type=float, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                   help="Cartesian target: position [m] and roll/pitch/yaw [deg]")
    ap.add_argument("--branch", type=int, help="IK branch index to use with --target (default: closest feasible to READY)")
    ap.add_argument("--qdot", nargs=6, type=float, metavar="DEG_S", default=DEFAULT_QDOT_DEG,
                    help="joint velocity [deg/s] for the velocity-kinematics part")
    ap.add_argument("--no-sim", action="store_true", help="do not open the MuJoCo viewer")
    ap.add_argument("--png", type=str, help="save an offscreen MuJoCo snapshot at the final pose")
    ap.add_argument("--move-time", type=float, default=3.0, help="seconds for the move in the viewer")
    ap.add_argument("--duration", type=float, default=None, help="close the viewer after this many seconds")
    ap.add_argument("--verify", action="store_true", help="also run a quick numerical verification")
    ap.add_argument("--spec", action="store_true", help="also print the robot specification (DH, limits, masses)")
    a = ap.parse_args(argv)

    if a.spec:
        section("ROBOT SPECIFICATION")
        print(describe_robot())

    q, T, res, sol, mode = resolve_pose(a)
    if q is not None:
        run_fk(q)
        res = run_ik(q, T, res if mode == "target" else None)
        qd = run_velocity(q, a.qdot)
    else:
        qd = np.radians(a.qdot)
    if a.verify:
        run_verify()

    if a.png:
        try:
            from src.simulation import render_png
            sim = build_sim(q, T, res, sol, qd)
            if q is not None:
                sim.set_q(q)
                sim.set_qdot(qd)
            print("\nwrote", render_png(a.png, sim))
        except Exception as exc:
            print(f"\n[note] snapshot not written ({type(exc).__name__}: {exc}). On headless Linux set MUJOCO_GL=egl or osmesa.")
    if not a.no_sim:
        try:
            sim = build_sim(q, T, res, sol, qd)
            run_simulation(sim, q, qd, a.move_time, a.duration)
        except Exception as exc:
            print(f"\n[note] viewer could not be opened ({type(exc).__name__}: {exc}).\n"
                  "       Use --no-sim for text only, or --png out.png for a snapshot.")
    return 0 if q is not None else 2


if __name__ == "__main__":
    sys.exit(main())
