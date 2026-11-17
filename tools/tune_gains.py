#!/usr/bin/env python
"""Documented gain-tuning procedure for the joint-space PD(+G) controllers.

Procedure (the numbers in the report come from this script):

1. DESIGN.   Estimate the joint-space inertia range M_ii over the collision-free workspace
             (MuJoCo mass matrix at random valid poses -- analysis only, controllers never
             use M) and apply ``control.design_pd_gains``:  Kp = I_max wn^2,
             Kd = 2 zeta wn I_max - b, capped by Kd <= I_min / dt.
2. SWEEP.    Run the *slow* nominal trajectory (half speed) with PD + gravity compensation for
             every combination  Kp scale in {0.25 ... 64}  x  Kd scale in {0.5, 1, 2}
             and record tracking RMSE, maximum error, settling time, peak torque and
             saturation.  Pure PD is NOT used for tuning so the gravity sag does not hide the
             dynamic behaviour.
3. SELECT.   Findings recorded in sweep.md: tracking error falls as 1/Kp (the dominant term is the
             viscous-drag lag b*qd_des/Kp of the damping-dominated wrist), Kd has little effect
             because the simulator damping already gives zeta > 1 on the wrist and the arm Kd is
             set for critical damping, and the loop stays stable up to ~32x the design gains;
             at 64x the torque limits saturate and settling degrades.  The design values
             (x1) are therefore used: they meet the accuracy target (< 0.01 rad at half speed)
             with a >= 32x gain margin ("well behaved" = settles inside 0.01 rad, no sustained saturation) and keep headroom in the actuator limits for the payload
             case.  Caveat: the margin is a simulation margin (perfect state feedback, no delay or
             noise); on hardware the usable gains would be much lower.

Outputs: results/tuning/design_table.md, results/tuning/sweep.csv, results/tuning/sweep.md

Run:  python tools/tune_gains.py [--quick]
"""
import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import robot_parameters as rp  # noqa: E402
from src.collision import CollisionChecker  # noqa: E402
from src.control import (CONTROL_DT, DEFAULT_KD, DEFAULT_KP, INERTIA_MAX, INERTIA_MIN,  # noqa: E402
                         PDGravityController, design_pd_gains)
from src.execution import run_trajectory  # noqa: E402
from src.gravity import DHGravityModel  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.physics import PhysicsSimulation  # noqa: E402
from src.scene import SceneConfig  # noqa: E402
from src.tasks import nominal_task, prepare_task  # noqa: E402
from src.trajectory import QuinticTrajectory  # noqa: E402

OUT = ROOT / "results" / "tuning"


def inertia_range(n=600, seed=1):
    """Diagonal of the joint-space mass matrix over random collision-free poses (analysis only)."""
    sim = PhysicsSimulation()
    chk = CollisionChecker(sim.model)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        q = rng.uniform(rp.JOINT_LIMITS[:, 0], rp.JOINT_LIMITS[:, 1])
        if not chk.is_configuration_valid(q):
            continue
        sim.reset(q)
        M = np.zeros((6, 6))
        try:                                   # MuJoCo >= 4 signature
            mujoco.mj_fullM(sim.model, sim.data, M)
        except TypeError:                      # MuJoCo 3.x signature
            mujoco.mj_fullM(sim.model, M, sim.data.qM)
        out.append(np.diag(M).copy())
    out = np.array(out)
    return out.min(axis=0), out.max(axis=0)


def design_table():
    lo, hi = inertia_range()
    wn = np.array([12.0, 14.0, 25.0, 1.0, 1.0, 1.0])
    kp, kd = design_pd_gains(wn, 1.0, hi, lo)
    lines = ["| joint | I_min | I_max | wn [rad/s] | Kp design | Kd design | Kd cap I_min/dt | Kp used | Kd used |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i in range(6):
        wn_txt = f"{wn[i]:.0f}" if i < 3 else "n/a (damping-dominated)"
        kp_d = f"{kp[i]:.1f}" if i < 3 else "-"
        lines.append(f"| {i + 1} | {lo[i]:.3g} | {hi[i]:.3g} | {wn_txt} | {kp_d} | {kd[i]:.3g} | "
                     f"{lo[i] / CONTROL_DT:.3g} | {DEFAULT_KP[i]:g} | {DEFAULT_KD[i]:g} |")
    text = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "design_table.md").write_text(
        "# PD gain design (joints 1-3 from wn/zeta; wrist joints damping dominated)\n\n" + text + "\n\n"
        f"Simulator viscous damping b = 1 N m s/rad, control period {CONTROL_DT * 1e3:.0f} ms.\n"
        "Wrist Kp is set from the closed-loop time constant (b + Kd)/Kp ~ 50 ms.\n", encoding="utf-8")
    return text


def slow_trajectory():
    prepared = prepare_task(nominal_task())
    return QuinticTrajectory(prepared.waypoints, dt=0.002,
                             max_velocity=0.5 * np.asarray(prepared.task.v_max),
                             max_acceleration=0.5 * np.asarray(prepared.task.a_max)), prepared


def sweep(kp_scales, kd_scales):
    traj, prepared = slow_trajectory()
    sim = PhysicsSimulation(SceneConfig())
    grav = DHGravityModel()
    rows = []
    for ks in kp_scales:
        for ds in kd_scales:
            kp, kd = DEFAULT_KP * ks, DEFAULT_KD * ds
            ctrl = PDGravityController(grav, kp, kd)
            try:
                log = run_trajectory(sim, ctrl, traj, hold_time=2.0)
                m = compute_metrics(log, prepared.T_target)
                err = log.q_des - log.q
                # overshoot: largest error of the opposite sign of the last motion, in the hold phase
                hold = log.t > traj.duration
                overshoot = float(np.max(np.abs(err[hold]))) if hold.any() else 0.0
                # "well behaved": finite, settles inside 0.01 rad in the hold phase, no sustained saturation
                stable = bool(np.all(np.isfinite(log.q)) and np.all(np.abs(log.q) < 10)
                              and overshoot < 0.01 and m["saturation_time_s_any_joint"] < 0.5)
                rows.append(dict(kp_scale=ks, kd_scale=ds, stable=stable,
                                 rmse_rms_rad=m["rmse_rad_rms_over_joints"], max_err_rad=m["max_error_rad_overall"],
                                 settling_s=m["settling_time_s"], hold_err_rad=overshoot,
                                 peak_tau_ratio=max(np.array(m["torque_requested_peak_Nm"]) / np.array(log.meta["torque_limits"])),
                                 saturation_s=m["saturation_time_s_any_joint"],
                                 final_ee_err_mm=1e3 * m["final_ee_position_error_m"]))
            except Exception as exc:                         # divergence / NaN
                rows.append(dict(kp_scale=ks, kd_scale=ds, stable=False, rmse_rms_rad=np.nan, max_err_rad=np.nan,
                                 settling_s=np.nan, hold_err_rad=np.nan, peak_tau_ratio=np.nan,
                                 saturation_s=np.nan, final_ee_err_mm=np.nan, note=type(exc).__name__))
    return rows


def write(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = ["kp_scale", "kd_scale", "stable", "rmse_rms_rad", "max_err_rad", "settling_s", "hold_err_rad",
            "peak_tau_ratio", "saturation_s", "final_ee_err_mm"]
    with open(OUT / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    md = ["| Kp x | Kd x | well behaved | RMSE (rms over joints) [rad] | max err [rad] | settling [s] | "
          "hold-phase err [rad] | peak torque / limit | saturation [s] | final EE err [mm] |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['kp_scale']:g} | {r['kd_scale']:g} | {r['stable']} | {r['rmse_rms_rad']:.4f} | "
                  f"{r['max_err_rad']:.4f} | {r['settling_s']:.2f} | {r['hold_err_rad']:.4f} | "
                  f"{r['peak_tau_ratio']:.2f} | {r['saturation_s']:.3f} | {r['final_ee_err_mm']:.2f} |")
    (OUT / "sweep.md").write_text("# Gain sweep, slow nominal trajectory, PD + gravity compensation\n\n"
                                  + "\n".join(md) + "\n", encoding="utf-8")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer sweep points")
    a = ap.parse_args()
    print("== 1. DESIGN\n" + design_table())
    kps = [0.5, 1, 2] if a.quick else [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    kds = [1] if a.quick else [0.5, 1, 2]
    print("\n== 2. SWEEP (slow nominal trajectory, PD + gravity compensation)")
    rows = sweep(kps, kds)
    print(write(rows))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
