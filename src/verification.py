"""
Numerical verification routines shared by the tests and ``examples/verification_report.py``.

Every function takes an explicit random seed and returns plain dictionaries of
statistics, so the numbers in the report are exactly the numbers the tests assert on.
"""

from __future__ import annotations

import numpy as np

from . import robot_parameters as rp
from .dh_model import PUMA560, DHModel
from .forward_kinematics import forward_kinematics, random_valid_configuration
from .inverse_kinematics import inverse_kinematics
from .jacobian import geometric_jacobian, jacobian_numerical
from .mujoco_reference import mujoco_fk
from .numerical_ik import numerical_ik_dls
from .singularity_analysis import (analyze_configuration, determinant_closed_form,
                                   singular_configuration)
from .utils import orientation_angle, wrap_to_pi


def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    return dict(max=float(np.max(x)), rms=float(np.sqrt(np.mean(x ** 2))), mean=float(np.mean(x)))


# ---------------------------------------------------------------------------
# Forward kinematics vs MuJoCo
# ---------------------------------------------------------------------------
def fk_vs_mujoco(n: int = 1000, seed: int = 0, model: DHModel = PUMA560) -> dict:
    rng = np.random.default_rng(seed)
    pos, ori, wc = [], [], []
    for _ in range(n):
        q = random_valid_configuration(rng, model)
        mine = forward_kinematics(q, model)
        ref = mujoco_fk(q)
        pos.append(np.linalg.norm(mine.position - ref[:3, 3]))
        ori.append(orientation_angle(mine.rotation, ref[:3, :3]))
        wc.append(np.linalg.norm(mine.wrist_center - mujoco_fk(q, rp.MJCF_WRIST_SITE)[:3, 3]))
    return dict(n=n, seed=seed, position_error_m=_stats(pos), orientation_error_rad=_stats(ori),
                wrist_center_error_m=_stats(wc))


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------
def ik_roundtrip(n: int = 2000, seed: int = 1, model: DHModel = PUMA560) -> dict:
    """q -> FK -> IK -> every returned branch re-verified with FK."""
    rng = np.random.default_rng(seed)
    n_branch, recovered, pos, ori, lim_ok = [], 0, [], [], 0
    n_feas = []
    for _ in range(n):
        q = random_valid_configuration(rng, model)
        T = forward_kinematics(q, model).T
        res = inverse_kinematics(T, q_ref=q, model=model)
        n_branch.append(len(res.solutions))
        n_feas.append(len(res.feasible))
        for s in res.solutions:
            pos.append(s.position_error)
            ori.append(s.orientation_error)
            lim_ok += int(s.within_limits == bool(np.all(model.limits[:, 0] - 1e-9 <= s.q)
                                                   and np.all(s.q <= model.limits[:, 1] + 1e-9)))
        best = res.closest_to(q)
        if best is not None and np.linalg.norm(wrap_to_pi(best.q - q)) < 1e-6:
            recovered += 1
    hist = {int(k): int(np.sum(np.array(n_branch) == k)) for k in sorted(set(n_branch))}
    hist_f = {int(k): int(np.sum(np.array(n_feas) == k)) for k in sorted(set(n_feas))}
    return dict(n=n, seed=seed, recovered_original=recovered, branch_count_histogram=hist,
                feasible_count_histogram=hist_f, position_error_m=_stats(pos),
                orientation_error_rad=_stats(ori),
                limit_flag_consistent=lim_ok == len(pos))


def _sample_pose_status(q, model=PUMA560):
    return inverse_kinematics(forward_kinematics(q, model).T, q_ref=q, model=model)


def ik_edge_cases(seed: int = 2, model: DHModel = PUMA560) -> dict:
    """Boundary, unreachable, joint-limit-infeasible, singular and near-singular targets."""
    rng = np.random.default_rng(seed)
    out: dict = {}
    L, a2 = rp.L_FOREARM, model.a[1]

    # -- workspace boundary: arm fully stretched (rho = a2 + L), then 1 mm further out
    q = singular_configuration("elbow", q_base=random_valid_configuration(rng, model), model=model)
    T = forward_kinematics(q, model).T
    r = inverse_kinematics(T, q_ref=q, model=model)
    out["boundary_stretched"] = dict(status=r.status, reason=r.reason, n_solutions=len(r.solutions),
                                     max_err=max([max(s.position_error, s.orientation_error) for s in r.solutions] or [np.nan]),
                                     outer_margin_m=r.diagnostics["outer_margin"])
    S = np.array([0, 0, model.d[0]])
    direction = (r.wrist_center - S) / np.linalg.norm(r.wrist_center - S)
    T_out = T.copy()
    T_out[:3, 3] += 1e-3 * direction
    r2 = inverse_kinematics(T_out, q_ref=q, model=model)
    out["beyond_boundary_1mm"] = dict(status=r2.status, reason=r2.reason, message=r2.message)

    # -- clearly unreachable targets
    T_far = np.eye(4)
    T_far[:3, 3] = [3.0, 0.5, 1.0]
    r = inverse_kinematics(T_far, model=model)
    out["far_target"] = dict(status=r.status, reason=r.reason, message=r.message, n_solutions=len(r.solutions))
    T_axis = np.eye(4)
    T_axis[:3, 3] = [0.0, 0.0, 0.9]
    T_axis[:3, :3] = np.diag([1, -1, -1.0])           # tool pointing down, wrist centre on waist axis
    T_axis[:3, 3] = [0.0, 0.0, 0.9 + rp.D_FLANGE]
    r = inverse_kinematics(T_axis, model=model)
    out["on_waist_axis"] = dict(status=r.status, reason=r.reason, message=r.message)

    # -- reachable geometry but joint-limit infeasible: sample q outside the limits
    found, tried, n_ok, n_limit = None, 0, 0, 0
    for _ in range(4000):
        q = rng.uniform(-np.pi, np.pi, 6)
        tried += 1
        r = _sample_pose_status(q, model)
        if r.status == "joint_limits":
            n_limit += 1
            if found is None:
                found = (q, r)
        elif r.status == "ok":
            n_ok += 1
    q_f, r_f = found
    out["joint_limit_infeasible"] = dict(
        tried=tried, n_status_ok=n_ok, n_status_joint_limits=n_limit,
        example_q_deg=np.degrees(q_f).round(1).tolist(), n_geometric_branches=len(r_f.solutions),
        all_branches_violate=bool(all(not s.within_limits for s in r_f.solutions)), message=r_f.message)

    # -- exactly singular targets
    for kind in ("shoulder", "elbow", "wrist"):
        sing = {}
        errs, flagged, n_sol = [], 0, []
        for _ in range(50):
            qb = random_valid_configuration(rng, model, margin=0.05)
            qs = singular_configuration(kind, q_base=qb, model=model)
            r = inverse_kinematics(forward_kinematics(qs, model).T, q_ref=qs, model=model)
            errs += [max(s.position_error, s.orientation_error) for s in r.solutions]
            n_sol.append(len(r.solutions))
            flagged += int(any(kind in s.singular for s in r.solutions))
        sing = dict(n_cases=50, flagged_singular=flagged, max_err=float(np.max(errs)),
                    branches_min=int(np.min(n_sol)), branches_max=int(np.max(n_sol)))
        out[f"singular_{kind}"] = sing

    # -- near-singular targets (sin q5 = 1e-3 ... 1e-8, X, v similarly)
    near = {}
    for kind in ("shoulder", "elbow", "wrist"):
        for eps in (1e-3, 1e-5, 1e-7, 1e-9):
            errs, ok = [], 0
            for _ in range(30):
                qb = random_valid_configuration(rng, model, margin=0.05)
                qs = singular_configuration(kind, q_base=qb, model=model)
                idx = {"shoulder": 1, "elbow": 2, "wrist": 4}[kind]
                qs[idx] += eps
                r = inverse_kinematics(forward_kinematics(qs, model).T, q_ref=qs, model=model)
                errs += [max(s.position_error, s.orientation_error) for s in r.solutions] or [np.inf]
                ok += int(r.status in ("ok", "joint_limits") and len(r.solutions) > 0)
            near[f"{kind}_eps{eps:g}"] = dict(solved=ok, of=30, max_err=float(np.max(errs)))
    out["near_singular"] = near
    return out


def ik_analytic_vs_numerical(n: int = 100, seed: int = 4, model: DHModel = PUMA560) -> dict:
    """DLS (secondary) started near a random q must converge to one analytic branch."""
    rng = np.random.default_rng(seed)
    conv, matched, its, t_err = 0, 0, [], []
    for _ in range(n):
        q = random_valid_configuration(rng, model, margin=0.1)
        T = forward_kinematics(q, model).T
        q0 = q + rng.normal(scale=0.3, size=6)
        nr = numerical_ik_dls(T, q0, model)
        if nr.converged:
            conv += 1
            its.append(nr.iterations)
            res = inverse_kinematics(T, q_ref=nr.q, model=model)
            d = min(np.linalg.norm(wrap_to_pi(s.q - nr.q)) for s in res.solutions)
            matched += int(d < 1e-5)
            t_err.append(d)
    return dict(n=n, converged=conv, matched_an_analytic_branch=matched,
                mean_iterations=float(np.mean(its)) if its else None)


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------
def jacobian_vs_numerical(n: int = 500, seed: int = 3, model: DHModel = PUMA560) -> dict:
    rng = np.random.default_rng(seed)
    e_all, e_v, e_w, det_err = [], [], [], []
    for _ in range(n):
        q = random_valid_configuration(rng, model)
        Ja, Jn = geometric_jacobian(q, model), jacobian_numerical(q, model)
        D = Ja - Jn
        e_all.append(D), e_v.append(D[:3]), e_w.append(D[3:])
        det_err.append(abs(np.linalg.det(Ja) - determinant_closed_form(q, model)))
    cat = lambda L: np.concatenate([np.ravel(x) for x in L])
    f = lambda x: dict(max_abs=float(np.max(np.abs(x))), rms=float(np.sqrt(np.mean(x ** 2))))
    return dict(n=n, seed=seed, all=f(cat(e_all)), translational=f(cat(e_v)), rotational=f(cat(e_w)),
                determinant_vs_closed_form_max=float(np.max(det_err)))


# ---------------------------------------------------------------------------
# Singularities
# ---------------------------------------------------------------------------
def singularity_examples(model: DHModel = PUMA560) -> dict:
    q_reg = np.array([0.4, 0.8, -1.0, 0.5, 0.9, 0.3])
    out = {"regular": analyze_configuration(q_reg, model)}
    q_n = singular_configuration("wrist", q_reg, model)
    q_n[4] = 0.02
    out["near_singular(q5=0.02)"] = analyze_configuration(q_n, model)
    for k in ("shoulder", "elbow", "wrist"):
        out[f"singular_{k}"] = analyze_configuration(singular_configuration(k, q_reg, model), model)
    return out


# ---------------------------------------------------------------------------
# Velocity kinematics
# ---------------------------------------------------------------------------
def velocity_checks(n: int = 500, seed: int = 6, model: DHModel = PUMA560) -> dict:
    """Ellipsoids, directional gains, singular directions and joint-speed behaviour."""
    from .jacobian import joint_velocity_from_twist
    from .velocity_kinematics import (directional_gain, required_joint_speed, singular_directions,
                                      velocity_ellipsoid)
    rng = np.random.default_rng(seed)
    ell_err, gain_viol, twist_rt = [], 0.0, []
    for _ in range(n):
        q = random_valid_configuration(rng, model)
        J = geometric_jacobian(q, model)
        for part, blk in (("linear", J[:3]), ("angular", J[3:])):
            e = velocity_ellipsoid(q, part, model)
            ell_err.append(np.max(np.abs(e.semi_axes ** 2 - np.sort(np.linalg.eigvalsh(blk @ blk.T))[::-1])))
            u = rng.normal(size=3)
            g = directional_gain(q, u, part, model)
            gain_viol = max(gain_viol, g - e.semi_axes[0], e.semi_axes[2] - g, 0.0)
        qd = rng.normal(size=6)
        tw = J @ qd
        twist_rt.append(np.linalg.norm(J @ joint_velocity_from_twist(q, tw, model) - tw))
    out = dict(n=n, seed=seed, ellipsoid_axes_vs_eig_max_err=float(np.max(ell_err)),
               directional_gain_bound_violation=float(gain_viol),
               twist_roundtrip_max_err=float(np.max(twist_rt)))
    # singular directions
    for kind in ("shoulder", "elbow", "wrist"):
        cos_min, res_max = 1.0, 0.0
        for _ in range(50):
            qs = singular_configuration(kind, random_valid_configuration(rng, model, margin=0.05), model)
            d = singular_directions(qs, model)
            fk = forward_kinematics(qs, model)
            res_max = max(res_max, d.residual)
            if kind == "elbow":
                ref = fk.wrist_center - fk.frames[1][:3, 3]
                c = abs(d.lost_twist[:3] @ ref) / np.linalg.norm(ref) / np.linalg.norm(d.lost_twist[:3])
            elif kind == "shoulder":
                ref = np.array([-np.sin(qs[0]), np.cos(qs[0]), 0.0])
                c = abs(d.lost_twist[:3] @ ref) / np.linalg.norm(d.lost_twist[:3])
            else:   # wrist: self-motion qd4 = -qd6 and lost angular part perpendicular to z3
                z3 = fk.frames[3][:3, 2]
                c = 1.0 - max(abs(d.null_motion[3] + d.null_motion[5]), abs(d.lost_twist[3:] @ z3))
            cos_min = min(cos_min, c)
        out[f"singular_{kind}"] = dict(lost_direction_alignment_min=float(cos_min), null_motion_residual_max=float(res_max))
    # joint speed near the wrist singularity
    lam = 0.05
    rows = []
    for eps in (1e-1, 1e-2, 1e-3, 1e-4):
        q = singular_configuration("wrist", np.array([0.4, 0.8, -1.0, 0.5, 0.9, 0.3]), model)
        q[4] = eps
        d = singular_directions(q, model)
        rows.append(dict(q5=eps, sigma_min=d.sigma_min,
                         pinv_speed=required_joint_speed(q, d.lost_twist, "pinv", model=model),
                         dls_speed=required_joint_speed(q, d.lost_twist, "dls", lam, model)))
    out["joint_speed_weakest_twist"] = rows
    out["dls_bound_1_over_2lambda"] = 1 / (2 * lam)
    return out
