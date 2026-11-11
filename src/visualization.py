"""
Matplotlib figures for FK, IK, workspace and singularity analysis (headless-safe, Agg backend by default).

All geometry shown comes from our own NumPy code (forward_kinematics, inverse_kinematics, jacobian,
singularity_analysis).  MuJoCo renders are produced by simulation.render_image().
"""

from __future__ import annotations

import os

import matplotlib

if not os.environ.get("DISPLAY") and matplotlib.get_backend().lower() not in ("agg",):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import robot_parameters as rp
from .dh_model import PUMA560
from .forward_kinematics import forward_kinematics, random_valid_configuration
from .inverse_kinematics import IKResult
from .jacobian import geometric_jacobian
from .singularity_analysis import (SIGMA_NEAR, SIGMA_SINGULAR, analyze_configuration, arm_plane_coordinates,
                                   singular_configuration, sigma_min_map_q2_q3, svd_metrics)

BRANCH_COLORS = ["#e6483d", "#338cf2", "#33bf59", "#f2bf26", "#b359e6", "#1abfcc", "#f280b3", "#8c8c33"]
AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")


# ----------------------------------------------------------------------------
# 3D helpers
# ----------------------------------------------------------------------------
def set_axes_equal(ax, center=None, radius=None):
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    c = lims.mean(axis=1) if center is None else np.asarray(center)
    r = 0.5 * np.max(lims[:, 1] - lims[:, 0]) if radius is None else radius
    ax.set_xlim3d(c[0] - r, c[0] + r)
    ax.set_ylim3d(c[1] - r, c[1] + r)
    ax.set_zlim3d(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


def draw_triad(ax, T, length=0.1, lw=1.5, label=None, alpha=1.0):
    o = T[:3, 3]
    for k in range(3):
        d = T[:3, k] * length
        ax.plot([o[0], o[0] + d[0]], [o[1], o[1] + d[1]], [o[2], o[2] + d[2]], color=AXIS_COLORS[k], lw=lw, alpha=alpha)
    if label:
        ax.text(*(o + 0.02), label, fontsize=7)


def skeleton(q):
    fk = forward_kinematics(q)
    pts = np.array([fk.frames[0][:3, 3], fk.frames[1][:3, 3], fk.frames[2][:3, 3], fk.frames[3][:3, 3],
                    fk.wrist_center, fk.position])
    return pts, fk


def draw_robot(ax, q, color="k", lw=3.0, frames=True, frame_len=0.08, alpha=1.0, label=None, ee_len=0.15):
    pts, fk = skeleton(q)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker="o", color=color, lw=lw, ms=4, alpha=alpha, label=label)
    ax.scatter(*fk.wrist_center, color="m", s=40, zorder=5)
    if frames:
        for i, T in enumerate(fk.frames):
            draw_triad(ax, T, frame_len, 1.2, label=f"{{{i}}}", alpha=alpha)
    draw_triad(ax, fk.T, ee_len, 2.5, label="E", alpha=alpha)
    return fk


def _floor(ax, half=0.9):
    g = np.array([[-half, -half], [half, -half], [half, half], [-half, half], [-half, -half]])
    ax.plot(g[:, 0], g[:, 1], 0 * g[:, 0], color="0.7", lw=0.8)


# ----------------------------------------------------------------------------
# FK figure
# ----------------------------------------------------------------------------
def plot_fk(q, path=None, title=None):
    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(projection="3d")
    fk = draw_robot(ax, q)
    _floor(ax)
    ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]"), ax.set_zlabel("z [m]")
    set_axes_equal(ax, center=(0.3, 0, 0.7), radius=0.8)
    ax.set_title(title or "PUMA-560 forward kinematics  (DH frames {0}..{6}, flange E)\n"
                 f"q = {np.round(np.degrees(q), 1).tolist()} deg", fontsize=9)
    ax.text2D(0.02, 0.02, f"p_E = [{fk.position[0]:.3f}, {fk.position[1]:.3f}, {fk.position[2]:.3f}] m\n"
              f"p_wrist = [{fk.wrist_center[0]:.3f}, {fk.wrist_center[1]:.3f}, {fk.wrist_center[2]:.3f}] m",
              transform=ax.transAxes, fontsize=8, family="monospace")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return fig


# ----------------------------------------------------------------------------
# IK figure
# ----------------------------------------------------------------------------
def plot_ik_branches(T_target, result: IKResult, path=None, q_highlight=None):
    fig = plt.figure(figsize=(13, 6.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    draw_triad(ax, T_target, 0.2, 3.0)
    ax.scatter(*T_target[:3, 3], color="gold", s=90, edgecolor="k", zorder=6, label="target")
    for k, s in enumerate(result.solutions):
        pts, _ = skeleton(s.q)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker="o", color=BRANCH_COLORS[k % 8], lw=2.2 if s.within_limits else 1.0,
                ms=3, ls="-" if s.within_limits else "--", alpha=0.9 if s.within_limits else 0.55,
                label=f"[{k}] {s.label}")
    _floor(ax)
    ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]"), ax.set_zlabel("z [m]")
    set_axes_equal(ax, center=(0.25, 0.1, 0.75), radius=0.7)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("analytic IK: all branches reaching the same pose\n(solid = inside joint limits, dashed = violates limits)", fontsize=9)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    rows = [[f"[{k}]", s.label, *[f"{np.degrees(v):7.1f}" for v in s.q], "yes" if s.within_limits else "NO",
             f"{s.position_error:.0e}"] for k, s in enumerate(result.solutions)]
    if rows:
        tb = ax2.table(cellText=rows, colLabels=["#", "branch", "q1", "q2", "q3", "q4", "q5", "q6", "limits ok", "e_pos"],
                       loc="center", cellLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(7)
        tb.auto_set_column_width(list(range(10)))
        tb.scale(1, 1.6)
        for (r, c), cell in tb.get_celld().items():
            if r > 0 and not result.solutions[r - 1].within_limits:
                cell.set_facecolor("#f3d9d9")
    ax2.set_title(f"{result.status}: {result.message}", fontsize=8, wrap=True)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return fig


# ----------------------------------------------------------------------------
# Workspace figure
# ----------------------------------------------------------------------------
def plot_workspace(n=20000, seed=0, path=None):
    rng = np.random.default_rng(seed)
    P, W, S = [], [], []
    for _ in range(n):
        q = random_valid_configuration(rng)
        fk = forward_kinematics(q)
        P.append(fk.position)
        W.append(fk.wrist_center)
        S.append(svd_metrics(geometric_jacobian(q)).sigma_min)
    P, W, S = map(np.array, (P, W, S))
    L = rp.L_FOREARM
    rho_max = rp.A2 + L
    R_max = np.hypot(rho_max, rp.D3)                 # wrist centre lies in a sphere around (0,0,d1) of this radius
    fig = plt.figure(figsize=(15, 5.5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=np.log10(np.maximum(S, 1e-6)), s=1.2, cmap="viridis_r", alpha=0.5)
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    set_axes_equal(ax, center=(0, 0, 0.7), radius=1.0)
    ax.set_title(f"end-effector positions of {n} random valid configs\n(colour = log10 sigma_min)", fontsize=9)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="log10 sigma_min")
    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(P[:, 0], P[:, 1], c=np.log10(np.maximum(S, 1e-6)), s=1, cmap="viridis_r", alpha=0.4)
    ax.set_aspect("equal"), ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]"), ax.set_title("top view", fontsize=9)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R_max * np.cos(th), R_max * np.sin(th), "r--", lw=1, label=f"wrist-centre outer bound  R={R_max:.3f} m")
    ax.plot(rp.D3 * np.cos(th), rp.D3 * np.sin(th), "k:", lw=1, label=f"shoulder cylinder |d3|={rp.D3:.3f} m")
    ax.legend(fontsize=7, loc="upper right")
    ax = fig.add_subplot(1, 3, 3)
    r = np.hypot(W[:, 0], W[:, 1])
    ax.scatter(r, W[:, 2], s=1, color="m", alpha=0.25, label="wrist centre")
    ax.plot(R_max * np.cos(np.linspace(0, np.pi, 200) - np.pi / 2), rp.D1 + R_max * np.sin(np.linspace(0, np.pi, 200) - np.pi / 2),
            "r--", lw=1, label="analytic outer bound (sphere about shoulder)")
    ax.axvline(abs(rp.D3), color="k", ls=":", lw=1, label="r = |d3|")
    ax.set_aspect("equal"), ax.set_xlabel("r = hypot(x,y) [m]"), ax.set_ylabel("z [m]"), ax.legend(fontsize=7)
    ax.set_title("wrist-centre workspace (r-z half plane)", fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return dict(positions=P, wrist=W, sigma_min=S, R_max=R_max, fig=fig)


# ----------------------------------------------------------------------------
# Singularity figures
# ----------------------------------------------------------------------------
def plot_singular_value_sweeps(path=None):
    q_reg = np.array([0.4, 0.8, -1.0, 0.5, 0.9, 0.3])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    specs = [("wrist", 4, "q5 [deg]  (wrist singularity at q5 = 0)", lambda: np.radians(np.linspace(-40, 40, 401))),
             ("elbow", 2, "q3 [deg]  (stretched-arm singularity at q3 = -phi)",
              lambda: -np.arctan2(rp.D4, rp.A3) + np.radians(np.linspace(-40, 40, 401))),
             ("shoulder", 1, "q2 [deg]  (shoulder singularity where X = 0)", None)]
    for ax, (kind, idx, xl, rng_fn) in zip(axes, specs):
        qs = singular_configuration(kind, q_reg)
        vals = rng_fn() if rng_fn else qs[1] + np.radians(np.linspace(-40, 40, 401))
        sv = np.empty((len(vals), 6))
        q = qs.copy()
        for k, v in enumerate(vals):
            q[idx] = v
            sv[k] = svd_metrics(geometric_jacobian(q)).singular_values
        x = np.degrees(vals)
        for i in range(6):
            ax.semilogy(x, np.maximum(sv[:, i], 1e-12), lw=1.4 if i < 5 else 2.4, color="C3" if i == 5 else f"C{i}",
                        label=f"sigma_{i + 1}" + (" (= sigma_min)" if i == 5 else ""))
        ax.axvline(np.degrees(qs[idx]), color="k", ls=":", lw=1)
        ax.axhline(SIGMA_NEAR, color="orange", ls="--", lw=0.8, label=f"near-singular  {SIGMA_NEAR}")
        ax.set_xlabel(xl), ax.set_ylabel("singular values of J"), ax.set_ylim(1e-6, 5)
        ax.set_title(f"{kind} singularity", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Singular values of the 6x6 geometric Jacobian across a singularity (other joints fixed)", fontsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return fig


def plot_sigma_map(q_base=(0.4, 0.0, 0.0, 0.5, 0.9, 0.3), n=161, path=None):
    lo, hi = PUMA560.limits[:, 0], PUMA560.limits[:, 1]
    q2 = np.linspace(PUMA560.limits[1, 0] - 0.3, PUMA560.limits[1, 1] + 0.3, n)
    q3 = np.linspace(PUMA560.limits[2, 0] - 0.3, PUMA560.limits[2, 1] + 0.3, n)
    S = sigma_min_map_q2_q3(q_base, q2, q3)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.pcolormesh(np.degrees(q2), np.degrees(q3), np.log10(np.maximum(S, 1e-8)), shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="log10 sigma_min(J)")
    # analytic singular curves
    Q2, Q3 = np.meshgrid(q2, q3)
    X = np.vectorize(lambda a, b: arm_plane_coordinates([0, a, b, 0, 0, 0])[0])(Q2, Q3)
    ax.contour(np.degrees(Q2), np.degrees(Q3), X, levels=[0], colors="r", linewidths=2)
    phi = np.arctan2(rp.D4, rp.A3)
    ax.axhline(np.degrees(-phi), color="w", lw=2, ls="--")
    ax.axhline(np.degrees(np.pi - phi), color="w", lw=2, ls="--")
    ax.axhline(np.degrees(-phi - 2 * np.pi), color="w", lw=2, ls="--")
    ax.plot([], [], "r", lw=2, label="shoulder: X(q2,q3) = 0")
    ax.plot([], [], "w--", lw=2, label="elbow: sin(q3+phi) = 0")
    ax.add_patch(plt.Rectangle((np.degrees(lo[1]), np.degrees(PUMA560.limits[2, 0])),
                               np.degrees(hi[1] - lo[1]), np.degrees(PUMA560.limits[2, 1] - PUMA560.limits[2, 0]),
                               fill=False, ec="cyan", lw=2, label="joint limits"))
    ax.set_xlim(np.degrees(q2[0]), np.degrees(q2[-1])), ax.set_ylim(np.degrees(q3[0]), np.degrees(q3[-1]))
    ax.set_xlabel("q2 [deg]"), ax.set_ylabel("q3 [deg]")
    ax.set_title(f"sigma_min over the (q2, q3) plane   (q1,q4,q5,q6 = {np.round(np.degrees(np.array(q_base)[[0,3,4,5]]),0).tolist()} deg)\n"
                 "dark = (near-)singular; analytic curves match exactly", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return fig


def plot_singularity_gallery(path=None):
    """Regular / near-singular / three singular configurations: 3D skeletons + status text."""
    q_reg = np.array([0.4, 0.8, -1.0, 0.5, 0.9, 0.3])
    q_near = singular_configuration("wrist", q_reg)
    q_near[4] = 0.02
    cases = [("regular", q_reg), ("near-singular (q5=0.02 rad)", q_near)] + \
            [(f"singular: {k}", singular_configuration(k, q_reg)) for k in ("shoulder", "elbow", "wrist")]
    fig = plt.figure(figsize=(17, 4.2))
    for i, (name, q) in enumerate(cases):
        ax = fig.add_subplot(1, 5, i + 1, projection="3d")
        draw_robot(ax, q, frames=False, ee_len=0.12)
        _floor(ax, 0.8)
        rep = analyze_configuration(q)
        col = {"regular": "green", "near-singular": "darkorange", "singular": "red"}[rep.status]
        ax.set_title(f"{name}\n{rep.status.upper()}  sigma_min={rep.svd.sigma_min:.1e}\ncond={rep.svd.condition_number:.1e}  rank={rep.svd.rank}",
                     fontsize=8, color=col)
        set_axes_equal(ax, center=(0.3, 0, 0.7), radius=0.8)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return fig


def image_grid(images, titles, path, ncols=3, figsize=None):
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (5.2 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Velocity-kinematics figures
# ----------------------------------------------------------------------------
def plot_velocity_ellipsoids(path=None):
    """Linear and angular velocity ellipsoids: regular pose vs poses 0.05 rad from each singularity."""
    from .velocity_kinematics import velocity_ellipsoid
    q_reg = np.array([0.4, 0.8, -1.0, 0.5, 0.9, 0.3])
    cases = [("regular", q_reg)]
    for kind, idx in (("wrist", 4), ("elbow", 2), ("shoulder", 1)):
        q = singular_configuration(kind, q_reg)
        q[idx] += 0.05
        cases.append((f"0.05 rad from {kind} singularity", q))
    fig = plt.figure(figsize=(16, 9.5))
    for col, (name, q) in enumerate(cases):
        for row, part in enumerate(("linear", "angular")):
            ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection="3d")
            e = velocity_ellipsoid(q, part)
            X, Y, Z = e.surface(n=28)
            ax.plot_surface(X, Y, Z, color=("tab:blue" if part == "linear" else "tab:orange"), alpha=0.35, linewidth=0)
            for k in range(3):
                d = e.axes[:, k] * e.semi_axes[k]
                ax.plot([-d[0], d[0]], [-d[1], d[1]], [-d[2], d[2]], color=AXIS_COLORS[k], lw=2.2)
            lim = 1.1 if part == "linear" else 1.9
            ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim), ax.set_zlim(-lim, lim)
            ax.set_box_aspect((1, 1, 1))
            unit = "m/s per rad/s" if part == "linear" else "rad/s per rad/s"
            ax.set_title(f"{name}\n{part}: semi-axes " + ", ".join(f"{s:.3f}" for s in e.semi_axes) + f"\n[{unit}]", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle("Velocity ellipsoids: image of unit-norm joint velocities under J_v (top) and J_w (bottom), base frame.\n"
                 "A flattening ellipsoid = the end effector loses mobility in that direction (axis scales differ per row)", fontsize=11)
    fig.subplots_adjust(top=0.86, bottom=0.03, left=0.02, right=0.98, hspace=0.32, wspace=0.05)
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def plot_joint_speed_vs_singularity(path=None, damping=0.05):
    """Joint speed needed to produce a unit twist while q5 approaches the wrist singularity."""
    from .jacobian import joint_velocity_from_twist
    from .velocity_kinematics import required_joint_speed, singular_directions, tracking_error
    eps = np.logspace(-4, np.log10(0.6), 120)
    q_base = singular_configuration("wrist", np.array([0.4, 0.8, -1.0, 0.5, 0.9, 0.3]))
    generic = np.array([0.05, 0.02, -0.04, 0.3, -0.2, 0.1])
    generic = generic / np.linalg.norm(generic)
    rows = {k: [] for k in ("sig", "pinv_w", "dls_w", "pinv_g", "dls_g", "err_w", "err_g")}
    for e in eps:
        q = q_base.copy()
        q[4] = e
        d = singular_directions(q)
        rows["sig"].append(d.sigma_min)
        rows["pinv_w"].append(required_joint_speed(q, d.lost_twist, "pinv"))
        rows["dls_w"].append(required_joint_speed(q, d.lost_twist, "dls", damping))
        rows["pinv_g"].append(required_joint_speed(q, generic, "pinv"))
        rows["dls_g"].append(required_joint_speed(q, generic, "dls", damping))
        rows["err_w"].append(tracking_error(q, d.lost_twist, joint_velocity_from_twist(q, d.lost_twist, method="dls", damping=damping)))
        rows["err_g"].append(tracking_error(q, generic, joint_velocity_from_twist(q, generic, method="dls", damping=damping)))
    R = {k: np.array(v) for k, v in rows.items()}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].loglog(eps, R["pinv_w"], "r-", lw=2, label="pseudo-inverse, weakest twist  (= 1/sigma_min)")
    ax[0].loglog(eps, R["dls_w"], "r--", lw=2, label=f"damped LS (lambda={damping}), weakest twist")
    ax[0].loglog(eps, R["pinv_g"], "b-", lw=1.5, label="pseudo-inverse, generic twist")
    ax[0].loglog(eps, R["dls_g"], "b--", lw=1.5, label="damped LS, generic twist")
    ax[0].axhline(1 / (2 * damping), color="k", ls=":", lw=1, label="DLS bound 1/(2 lambda)")
    ax[0].set_xlabel("q5 [rad]  (wrist singularity at 0)"), ax[0].set_ylabel("|qdot| [rad/s] for a unit-norm twist")
    ax[0].set_title("joint speed required for a unit task twist", fontsize=10)
    ax[0].legend(fontsize=7), ax[0].grid(alpha=0.3, which="both")
    ax[1].loglog(eps, R["err_w"], "r--", lw=2, label="DLS error, weakest twist")
    ax[1].loglog(eps, R["err_g"], "b--", lw=1.5, label="DLS error, generic twist")
    ax[1].loglog(eps, R["sig"], "k-", lw=1.2, label="sigma_min")
    ax[1].set_xlabel("q5 [rad]"), ax[1].set_ylabel("DLS task error |twist - J qdot|   and   sigma_min")
    ax[1].set_title("price of damping: task-space error", fontsize=10)
    ax[1].legend(fontsize=7), ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return R
