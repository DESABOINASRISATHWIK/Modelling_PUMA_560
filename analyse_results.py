#!/usr/bin/env python
"""Rebuild every table and figure from the saved raw logs (Part 4).

    python run_experiments.py && python analyse_results.py

Reads  results/experiments/**  (raw .npz logs, config.json, planner_study.json)
Writes results/figures/*.png, results/experiments/summary.md, results/experiments/comparison.csv
Metrics are RECOMPUTED here from the raw logs, not copied from metrics.json.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.experiments import CONTROLLER_LABELS, load_case  # noqa: E402
from src.metrics import ORIENTATION_TOL, POSITION_TOL, compute_metrics  # noqa: E402
from src.utils import rpy_to_matrix  # noqa: E402

COLORS = {"pd": "tab:red", "pd_gravity": "tab:blue", "pd_gravity_payload_aware": "tab:green"}
JN = [f"q{i}" for i in range(1, 7)]
EXPS = ["nominal", "obstacle", "payload"]


def target_pose(cfg):
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(*np.radians(cfg["target_rpy_deg"]))
    T[:3, 3] = cfg["target_xyz"]
    return T


def recompute(case):
    T = target_pose(case["config"])
    return {n: compute_metrics(log, T) for n, log in case["logs"].items()}


def fig_tracking(exp, case, out):
    fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for j, a in enumerate(ax.ravel()):
        first = next(iter(case["logs"].values()))
        a.plot(first.t, first.q_des[:, j], "k--", lw=1.2, label="reference")
        for n, log in case["logs"].items():
            a.plot(log.t, log.q[:, j], color=COLORS.get(n), lw=1.2, label=CONTROLLER_LABELS.get(n, n))
        a.set_ylabel(f"{JN[j]} [rad]"); a.grid(alpha=.3)
    ax[2, 0].set_xlabel("t [s]"); ax[2, 1].set_xlabel("t [s]")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"{exp}: joint tracking"); fig.tight_layout(); fig.savefig(out / f"tracking_{exp}.png", dpi=130); plt.close(fig)


def fig_error(exp, case, out):
    fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for j, a in enumerate(ax.ravel()):
        for n, log in case["logs"].items():
            a.plot(log.t, log.q_des[:, j] - log.q[:, j], color=COLORS.get(n), lw=1.1, label=CONTROLLER_LABELS.get(n, n))
        a.axvline(next(iter(case["logs"].values())).meta["trajectory_duration_s"], color="gray", ls=":")
        a.set_ylabel(f"e_{j + 1} [rad]"); a.grid(alpha=.3)
    ax[0, 0].legend(fontsize=7); ax[2, 0].set_xlabel("t [s]"); ax[2, 1].set_xlabel("t [s]")
    fig.suptitle(f"{exp}: tracking error (dotted line: end of the motion, then hold)")
    fig.tight_layout(); fig.savefig(out / f"error_{exp}.png", dpi=130); plt.close(fig)


def fig_torque(exp, case, out):
    fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for j, a in enumerate(ax.ravel()):
        lim = next(iter(case["logs"].values())).meta["torque_limits"][j]
        for n, log in case["logs"].items():
            a.plot(log.t[:-1], log.tau_req[:-1, j], color=COLORS.get(n), lw=1.0, label=CONTROLLER_LABELS.get(n, n))
        a.axhline(lim, color="k", ls="--", lw=.8); a.axhline(-lim, color="k", ls="--", lw=.8)
        a.set_ylabel(f"tau_{j + 1} [N m]"); a.grid(alpha=.3)
    ax[0, 0].legend(fontsize=7); ax[2, 0].set_xlabel("t [s]"); ax[2, 1].set_xlabel("t [s]")
    fig.suptitle(f"{exp}: requested torque (dashed: actuator limits)")
    fig.tight_layout(); fig.savefig(out / f"torque_{exp}.png", dpi=130); plt.close(fig)


def _box(ax, c, h, color="tab:red"):
    (cx, cy, cz), (hx, hy, hz) = c, h
    xs, ys, zs = [cx - hx, cx + hx], [cy - hy, cy + hy], [cz - hz, cz + hz]
    for x in xs:
        for y in ys:
            ax.plot([x, x], [y, y], zs, color=color, lw=1)
    for x in xs:
        for z in zs:
            ax.plot([x, x], ys, [z, z], color=color, lw=1)
    for y in ys:
        for z in zs:
            ax.plot(xs, [y, y], [z, z], color=color, lw=1)


def fig_ee_path(exp, case, out):
    cfg = case["config"]
    fig = plt.figure(figsize=(11, 4.8))
    a3 = fig.add_subplot(1, 2, 1, projection="3d")
    a2 = fig.add_subplot(1, 2, 2)
    for n, log in case["logs"].items():
        a3.plot(*log.ee_pos.T, color=COLORS.get(n), lw=1.4, label=CONTROLLER_LABELS.get(n, n))
        a2.plot(log.ee_pos[:, 0], log.ee_pos[:, 2], color=COLORS.get(n), lw=1.4)
    tx = cfg["target_xyz"]
    a3.scatter(*tx, color="k", marker="*", s=80, label="target"); a2.scatter(tx[0], tx[2], color="k", marker="*", s=80)
    for o in cfg["obstacles"]:
        _box(a3, o["position"], o["size"])
        p, s = o["position"], o["size"]
        a2.add_patch(plt.Rectangle((p[0] - s[0], p[2] - s[2]), 2 * s[0], 2 * s[2], color="tab:red", alpha=.3))
    a3.set_xlabel("x"); a3.set_ylabel("y"); a3.set_zlabel("z"); a3.legend(fontsize=7)
    a2.set_xlabel("x [m]"); a2.set_ylabel("z [m]"); a2.set_aspect("equal"); a2.grid(alpha=.3); a2.set_title("side view (x-z)")
    fig.suptitle(f"{exp}: end-effector path"); fig.tight_layout(); fig.savefig(out / f"ee_path_{exp}.png", dpi=130); plt.close(fig)


def fig_comparison(all_m, out):
    labels = [("rmse_rad_rms_over_joints", "tracking RMSE [rad]"),
              ("final_ee_position_error_m", "final EE position error [m]"),
              ("peak", "peak applied torque, joint 2 [N m]"),
              ("saturation_time_s_any_joint", "saturation time [s]")]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    names = ["pd", "pd_gravity", "pd_gravity_payload_aware"]
    for a, (key, title) in zip(ax, labels):
        w = 0.26
        for k, n in enumerate(names):
            vals = []
            for e in EXPS:
                m = all_m.get(e, {}).get(n)
                vals.append(np.nan if m is None else (m["torque_peak_Nm"][1] if key == "peak" else m[key]))
            a.bar(np.arange(len(EXPS)) + (k - 1) * w, vals, w, color=COLORS[n], label=CONTROLLER_LABELS[n])
        a.set_xticks(range(len(EXPS))); a.set_xticklabels(EXPS); a.set_title(title, fontsize=9); a.grid(alpha=.3, axis="y")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "controller_comparison.png", dpi=130); plt.close(fig)


def _boxplot(ax, data, labels):
    """matplotlib renamed boxplot(labels=) to tick_labels= in 3.9; support both."""
    try:
        ax.boxplot(data, tick_labels=labels)
    except TypeError:
        ax.boxplot(data, labels=labels)


def fig_planner(study, out):
    rows = study["rows"]
    methods = sorted({r["method"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ok_t = [[r["time_s"] for r in rows if r["method"] == m and r["success"]] for m in methods]
    ok_l = [[r["path_length_rad"] for r in rows if r["method"] == m and r["success"]] for m in methods]
    rate = [np.mean([r["success"] for r in rows if r["method"] == m]) * 100 for m in methods]
    ax[0].bar(methods, rate, color=["tab:blue", "tab:orange"][:len(methods)]); ax[0].set_ylabel("success rate [%]"); ax[0].set_ylim(0, 105)
    _boxplot(ax[1], ok_t, methods); ax[1].set_ylabel("planning time [s] (successful runs)"); ax[1].set_yscale("log")
    _boxplot(ax[2], ok_l, methods); ax[2].set_ylabel("path length [rad]")
    for a in ax:
        a.grid(alpha=.3)
    n = len([r for r in rows if r["method"] == methods[0]])
    fig.suptitle(f"Planner comparison on the obstacle task ({n} seeds each)")
    fig.tight_layout(); fig.savefig(out / "planner_comparison.png", dpi=130); plt.close(fig)


def fmt_metrics_table(m_by_ctrl):
    h = ("| controller | RMSE rms-over-joints [rad] | max joint err [rad] | final EE pos err [mm] | final EE ori err [rad] | "
         "holds target (<5 mm, <0.02 rad) | peak torque j1..j6 [N m] | RMS torque j1..j6 [N m] | saturation [s] | collision-free |\n"
         "|---|---|---|---|---|---|---|---|---|---|\n")
    for n, m in m_by_ctrl.items():
        h += (f"| {CONTROLLER_LABELS.get(n, n)} | {m['rmse_rad_rms_over_joints']:.4f} | {m['max_error_rad_overall']:.4f} | "
              f"{m['final_ee_position_error_m'] * 1e3:.2f} | {m['final_ee_orientation_error_rad']:.4f} | "
              f"{'yes' if m['holds_target_within_tolerance'] else 'NO'} | "
              f"{', '.join(f'{x:.1f}' for x in m['torque_peak_Nm'])} | {', '.join(f'{x:.1f}' for x in m['torque_rms_Nm'])} | "
              f"{m['saturation_time_s_any_joint']:.3f} | {'yes' if m['collision_free'] else 'NO ' + str(m['collision_pairs'])} |\n")
    return h


def planner_table(study):
    rows = study["rows"]
    t = "| planner | success | planning time median [s] (min-max) | path length median [rad] | waypoints median | iterations median |\n|---|---|---|---|---|---|\n"
    for m in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == m]
        ok = [r for r in sub if r["success"]]
        if ok:
            tt = [r["time_s"] for r in ok]
            t += (f"| {m} | {len(ok)}/{len(sub)} | {np.median(tt):.3f} ({min(tt):.3f}-{max(tt):.3f}) | "
                  f"{np.median([r['path_length_rad'] for r in ok]):.3f} | {np.median([r['waypoints'] for r in ok]):.0f} | "
                  f"{np.median([r['iterations'] for r in ok]):.0f} |\n")
        else:
            t += f"| {m} | 0/{len(sub)} | - | - | - | - |\n"
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "results" / "experiments"))
    ap.add_argument("--figs", default=str(ROOT / "results" / "figures"))
    a = ap.parse_args(argv)
    data, figs = Path(a.data), Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    md = ["# Experiment summary (generated by analyse_results.py from raw logs)\n"]
    all_m, csv_rows = {}, []
    for e in EXPS:
        if not (data / e).exists():
            continue
        case = load_case(data, e)
        m = recompute(case)
        all_m[e] = m
        cfg = case["config"]
        md.append(f"\n## Experiment: {e}\n\n{cfg['description']}.\n")
        md.append(f"- goal selection: {cfg['goal']['n_branches']} IK branches, {cfg['goal']['n_within_limits']} within limits, "
                  f"{cfg['goal']['n_collision_free']} collision-free; chosen `{cfg['goal']['chosen_label']}`")
        md.append(f"- direct joint path blocked: **{cfg['direct_joint_path_blocked']}**; plan: {cfg['plan']['summary']}")
        md.append(f"- trajectory: {cfg['trajectory']['duration_s']:.2f} s, collision-free against true geometry: "
                  f"{cfg['trajectory']['collision_free_true_scene']}, planned min clearance: {cfg['trajectory']['min_obstacle_clearance_m']}\n")
        md.append(fmt_metrics_table(m))
        for n, mm in m.items():
            csv_rows.append(dict(experiment=e, controller=n, rmse_rms_rad=mm["rmse_rad_rms_over_joints"],
                                 max_err_rad=mm["max_error_rad_overall"], final_ee_pos_err_m=mm["final_ee_position_error_m"],
                                 final_ee_ori_err_rad=mm["final_ee_orientation_error_rad"],
                                 peak_tau_max_Nm=max(mm["torque_peak_Nm"]), saturation_s=mm["saturation_time_s_any_joint"],
                                 collision_free=mm["collision_free"]))
        fig_tracking(e, case, figs); fig_error(e, case, figs); fig_torque(e, case, figs); fig_ee_path(e, case, figs)
    if all_m:
        fig_comparison(all_m, figs)
    sp = data / "planner_study.json"
    if sp.exists():
        study = json.loads(sp.read_text())
        md.append(f"\n## Planner comparison ({study['task']} task; direct path feasible: {study['direct_path_feasible']})\n")
        md.append(planner_table(study))
        fig_planner(study, figs)
    if csv_rows:
        with open(data / "comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    (data / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md)); print(f"\nfigures -> {figs}\nsummary -> {data / 'summary.md'}")


if __name__ == "__main__":
    main()
