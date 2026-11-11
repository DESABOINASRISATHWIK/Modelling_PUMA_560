"""
Closed-form (analytical) inverse kinematics of the PUMA-560.   NumPy only.

Notation   c = cos, s = sin;  d1, a2, d3, a3, d4 are the DH constants of
robot_parameters.py;   L = sqrt(a3^2 + d4^2),  phi = atan2(d4, a3).

--------------------------------------------------------------------------
0. Remove the tool
--------------------------------------------------------------------------
    T_06 = T_base^-1  T_target  T_6E^-1

Because a4 = a5 = a6 = d5 = d6 = 0 the origins of frames {4},{5},{6} coincide:
the WRIST CENTRE  p_w = origin of T_06.  It depends on q1, q2, q3 only.

--------------------------------------------------------------------------
1. Position problem (joints 1-3)  -  wrist centre  p_w = (px, py, pz)
--------------------------------------------------------------------------
Chaining the DH transforms gives (see forward_kinematics.py)

    p_w = Rz(q1) [ X, -d3, 0 ]^T + [ 0, 0, d1 + Y ]^T                     (1)

    [X, Y]^T = Rot(q2) [ a2 + u , v ]^T                                   (2)
    u = a3 cos q3 - d4 sin q3 ,   v = a3 sin q3 + d4 cos q3

(X, Y) are the wrist-centre coordinates in the vertical "arm plane" that is
offset laterally by d3 from the waist axis (X horizontal, Y vertical).

* Waist.   From (1):  px^2 + py^2 = X^2 + d3^2   so   X = +/- sqrt(r^2 - d3^2),
           r = hypot(px, py), and  [px, py]^T = Rz(q1) [X, -d3]^T  gives

               q1 = atan2(py, px) + atan2(d3, X)               (two branches: X>0 "right", X<0 "left")

  Reachability condition A:   r >= |d3|   (wrist centre outside the cylinder of radius |d3|).

* Elbow.   |[X, Y]|^2 = rho^2 = a2^2 + L^2 + 2 a2 L cos(q3 + phi)  with  rho^2 = r^2 - d3^2 + (pz - d1)^2, so

               cos(q3 + phi) = (rho^2 - a2^2 - L^2) / (2 a2 L)         q3 = -phi +/- acos(.)   ("elbow up/down")

  Reachability condition B:   |a2 - L| <= rho <= a2 + L.

* Shoulder.  From (2):        q2 = atan2(Y, X) - atan2(v, a2 + u)

--------------------------------------------------------------------------
2. Orientation problem (joints 4-6)  -  spherical wrist
--------------------------------------------------------------------------
    R_36 = R_03^T R_06 ,   R_36 = Rz(q4) Rx(pi/2) Rz(q5) Rx(-pi/2) Rz(q6)
                                 = Rz(q4) Ry(-q5) Rz(q6)               (Rx(pi/2) Rz(q5) Rx(-pi/2) = Ry(-q5))
which is a Z-Y-Z Euler decomposition with middle angle beta = -q5:

    R_36[2,2] = cos beta,  R_36[0,2] = cos q4 sin beta,  R_36[1,2] = sin q4 sin beta,
    R_36[2,0] = -sin beta cos q6,  R_36[2,1] = sin beta sin q6.

   non-flipped:  beta = atan2(+sqrt(R02^2+R12^2), R22) in [0, pi],  q4 = atan2(R12, R02), q6 = atan2(R21, -R20),  q5 = -beta
   flipped:      q4 -> q4 + pi,  q6 -> q6 + pi,  q5 -> +beta

   Wrist singularity sin(q5) = 0: q4 and q6 axes are collinear, only q4 + q6 (q5=0) or
   q4 - q6 (q5=pi) is determined -> a one-parameter family.  We return one member
   (q4 = reference value) and flag it.

Total: 2 (waist) x 2 (elbow) x 2 (wrist) = 8 geometric branches.  Every returned
branch is verified with the forward kinematics.  Joint limits are applied afterwards
(together with the 2*pi equivalents of each angle), so geometric unreachability and
joint-limit infeasibility are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dh_model import PUMA560, DHModel
from .forward_kinematics import forward_kinematics
from .utils import invert_transform, is_rotation, pose_error, wrap_to_pi

GEOM_TOL = 1e-9          # tolerance of the reachability tests
SING_TOL = 1e-7          # |X| (shoulder) and |sin(q3+phi)| (elbow) below this -> singular, branches merged
WRIST_SING_TOL = 1e-8    # |sin q5| below this -> wrist singular.  Trade-off measured in
#                          examples/verification_report.py: the regular Euler formulas lose accuracy like
#                          1e-16/|sin q5|, the singular-member approximation has error ~|sin q5|;
#                          both are <= ~3e-8 rad at the crossover 1e-8.
VERIFY_TOL_POS = 1e-6    # [m]    an analytic branch must reproduce the target
VERIFY_TOL_ORI = 1e-6    # [rad]


# ----------------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------------
@dataclass
class IKSolution:
    q: np.ndarray                    # joint angles, 2*pi representative chosen inside the limits if possible
    shoulder: str                    # "right" (X >= 0) / "left" (X < 0)
    elbow: str                       # "up" / "down"
    wrist: str                       # "noflip" / "flip"
    within_limits: bool
    limit_violation: np.ndarray      # per joint [rad]
    position_error: float            # [m]   |p_FK(q) - p_target|
    orientation_error: float         # [rad] geodesic angle
    singular: list = field(default_factory=list)   # subset of {"shoulder","elbow","wrist"}

    @property
    def label(self) -> str:
        return f"{self.shoulder}/{self.elbow}/{self.wrist}"


@dataclass
class IKResult:
    status: str                      # "ok" | "unreachable" | "joint_limits"
    reason: str                      # machine-readable code
    message: str                     # human-readable explanation
    wrist_center: np.ndarray
    diagnostics: dict
    solutions: list                  # all geometric branches that passed FK verification

    @property
    def feasible(self) -> list:
        """Branches that also respect the joint limits."""
        return [s for s in self.solutions if s.within_limits]

    def closest_to(self, q_ref, feasible_only: bool = True):
        """Branch with the smallest joint-space distance to q_ref (None if none)."""
        pool = self.feasible if feasible_only else self.solutions
        if not pool:
            return None
        q_ref = np.asarray(q_ref, dtype=float)
        return min(pool, key=lambda s: float(np.linalg.norm(wrap_to_pi(s.q - q_ref))))

    def summary(self) -> str:
        lines = [f"IK status: {self.status} ({self.reason}) - {self.message}"]
        for k, s in enumerate(self.solutions):
            flag = "OK " if s.within_limits else "LIM"
            sing = f" singular:{','.join(s.singular)}" if s.singular else ""
            lines.append(f"  [{k}] {flag} {s.label:<18} q[deg]={np.round(np.degrees(s.q), 2).tolist()} "
                         f"e_pos={s.position_error:.1e} e_ori={s.orientation_error:.1e}{sing}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _check_puma_structure(model: DHModel) -> None:
    """The closed form is only valid for the PUMA DH structure (spherical wrist, a1=d2=0 ...)."""
    ok = (np.allclose(model.alpha, [np.pi / 2, 0, -np.pi / 2, np.pi / 2, -np.pi / 2, 0])
          and np.isclose(model.a[0], 0) and np.allclose(model.a[3:], 0)
          and np.isclose(model.d[1], 0) and np.allclose(model.d[4:], 0))
    if not ok:
        raise NotImplementedError("analytical IK requires the PUMA-560 DH structure")


def _validate_target(T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("target must be a finite 4x4 homogeneous transform")
    if not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError("target must have last row [0 0 0 1]")
    if not is_rotation(T[:3, :3], tol=1e-6):
        raise ValueError("target rotation block must be a proper rotation matrix")
    return T


def _fit_to_limits(q_raw: np.ndarray, limits: np.ndarray, q_ref: np.ndarray, tol: float = 1e-9):
    """Choose the 2*pi-equivalent of every angle that respects the limits, closest to q_ref."""
    q = np.empty(6)
    viol = np.zeros(6)
    for j in range(6):
        base = float(wrap_to_pi(q_raw[j]))
        cand = base + 2 * np.pi * np.arange(-2, 3)
        v = np.maximum(limits[j, 0] - cand, 0.0) + np.maximum(cand - limits[j, 1], 0.0)
        ok = v <= tol
        if ok.any():
            idx = np.flatnonzero(ok)
            k = idx[np.argmin(np.abs(cand[idx] - q_ref[j]))]
            q[j], viol[j] = cand[k], 0.0
        else:
            k = int(np.argmin(v + 1e-12 * np.abs(cand - q_ref[j])))
            q[j], viol[j] = cand[k], v[k]
    return q, viol


def reachability_diagnostics(T_target, model: DHModel = PUMA560) -> dict:
    """Geometric reachability quantities of a target (no joint limits involved)."""
    _check_puma_structure(model)
    T = _validate_target(T_target)
    T06 = invert_transform(model.T_base) @ T @ invert_transform(model.T_tool)
    pw = T06[:3, 3]
    d1, a2, d3, a3, d4 = model.d[0], model.a[1], model.d[2], model.a[2], model.d[3]
    L = float(np.hypot(a3, d4))
    r = float(np.hypot(pw[0], pw[1]))
    Y = float(pw[2] - d1)
    rho2 = max(r * r - d3 * d3, 0.0) + Y * Y
    rho = float(np.sqrt(rho2))
    return dict(wrist_center=pw.copy(), r=r, radial_margin=r - abs(d3), Y=Y, rho=rho,
                rho_min=abs(a2 - L), rho_max=a2 + L, outer_margin=(a2 + L) - rho,
                inner_margin=rho - abs(a2 - L), L=L, T06=T06)


# ----------------------------------------------------------------------------
# Main solver
# ----------------------------------------------------------------------------
def inverse_kinematics(T_target, q_ref=None, model: DHModel = PUMA560,
                       limits: np.ndarray | None = None) -> IKResult:
    """All analytic IK branches of a target flange pose.

    Parameters
    ----------
    T_target : 4x4 desired end-effector (flange) pose in the base frame.
    q_ref    : optional reference joint vector; used (i) to pick, among the 2*pi
               equivalents allowed by the limits, the one closest to q_ref and
               (ii) as the free value q4 at a wrist singularity.
    limits   : (6,2) joint limits, default = model limits.

    Returns an :class:`IKResult`.  ``status`` is
        "ok"           - at least one branch is inside the joint limits,
        "joint_limits" - target is geometrically reachable but every branch violates a limit,
        "unreachable"  - no real solution (see ``reason`` / ``diagnostics``).
    No exception is raised for unreachable or singular targets.
    """
    _check_puma_structure(model)
    T = _validate_target(T_target)
    limits = model.limits if limits is None else np.asarray(limits, dtype=float)
    q_ref = np.zeros(6) if q_ref is None else model.check_q(q_ref)

    dg = reachability_diagnostics(T, model)
    pw, T06 = dg["wrist_center"], dg["T06"]
    px, py, pz = pw
    d1, a2, d3, a3, d4 = model.d[0], model.a[1], model.d[2], model.a[2], model.d[3]
    L, phi = dg["L"], float(np.arctan2(d4, a3))

    # ---- geometric reachability ------------------------------------------------
    if dg["radial_margin"] < -GEOM_TOL:
        return IKResult("unreachable", "wrist_inside_shoulder_cylinder",
                        f"wrist centre is {abs(dg['radial_margin']) * 1e3:.2f} mm too close to the waist axis "
                        f"(r = {dg['r']:.4f} m < |d3| = {abs(d3):.4f} m)", pw, dg, [])
    if dg["outer_margin"] < -GEOM_TOL:
        return IKResult("unreachable", "too_far",
                        f"wrist centre is {abs(dg['outer_margin']) * 1e3:.2f} mm beyond the maximum reach "
                        f"(rho = {dg['rho']:.4f} m > a2 + L = {dg['rho_max']:.4f} m)", pw, dg, [])
    if dg["inner_margin"] < -GEOM_TOL:
        return IKResult("unreachable", "too_close",
                        f"wrist centre is {abs(dg['inner_margin']) * 1e3:.2f} mm inside the minimum reach "
                        f"(rho = {dg['rho']:.4f} m < |a2 - L| = {dg['rho_min']:.4f} m)", pw, dg, [])

    # ---- 1. joints 1-3 ------------------------------------------------------------
    rho2 = dg["rho"] ** 2
    c3 = float(np.clip((rho2 - a2 ** 2 - L ** 2) / (2 * a2 * L), -1.0, 1.0))
    acos3 = float(np.arccos(c3))
    X_abs = float(np.sqrt(max(dg["r"] ** 2 - d3 ** 2, 0.0)))
    Y = dg["Y"]

    arm_candidates = []                      # (theta1, theta2, theta3, shoulder, elbow, flags)
    shoulder_sing = X_abs < SING_TOL
    elbow_sing = abs(np.sin(acos3)) < SING_TOL
    for s_sign in ([+1.0] if shoulder_sing else [+1.0, -1.0]):
        X = s_sign * X_abs
        th1 = np.arctan2(py, px) + np.arctan2(d3, X)
        for e_sign in ([+1.0] if elbow_sing else [+1.0, -1.0]):
            th3 = -phi + e_sign * acos3
            u = a3 * np.cos(th3) - d4 * np.sin(th3)
            v = a3 * np.sin(th3) + d4 * np.cos(th3)
            th2 = np.arctan2(Y, X) - np.arctan2(v, a2 + u)
            # label the elbow geometrically: is the elbow joint above the shoulder->wrist line?
            ex, ey = a2 * np.cos(th2), a2 * np.sin(th2)
            side = (X * ey - Y * ex) * (1.0 if X >= 0 else -1.0)
            elbow_lbl = "up" if side > 0 else "down"
            flags = (["shoulder"] if shoulder_sing else []) + (["elbow"] if elbow_sing else [])
            arm_candidates.append((th1, th2, th3, "right" if X >= 0 else "left", elbow_lbl, flags))

    # ---- 2. joints 4-6 ------------------------------------------------------------
    solutions: list[IKSolution] = []
    off = model.theta_offset
    for th1, th2, th3, sh_lbl, el_lbl, flags in arm_candidates:
        q123 = np.array([th1, th2, th3]) - off[:3]
        R03 = model.frames(np.r_[q123, 0.0, 0.0, 0.0])[3][:3, :3]
        R36 = R03.T @ T06[:3, :3]
        sb = float(np.hypot(R36[0, 2], R36[1, 2]))           # sin(beta) >= 0
        wrist_sing = sb < WRIST_SING_TOL
        wrist_opts = []
        if wrist_sing:
            beta = 0.0 if R36[2, 2] > 0 else np.pi
            q4 = float(q_ref[3] + off[3])                    # free parameter
            if beta == 0.0:                                  # R36 = Rz(q4 + q6)
                q6 = np.arctan2(R36[1, 0], R36[0, 0]) - q4
            else:                                            # R36 = Rz(q4 - q6) Ry(pi)
                delta = np.arctan2(-R36[1, 0], -R36[0, 0])
                q6 = q4 - delta
            wrist_opts.append((q4, -beta, q6, "noflip", flags + ["wrist"]))
        else:
            beta = np.arctan2(sb, R36[2, 2])
            a_ = np.arctan2(R36[1, 2], R36[0, 2])
            g_ = np.arctan2(R36[2, 1], -R36[2, 0])
            wrist_opts.append((a_, -beta, g_, "noflip", flags))
            wrist_opts.append((a_ + np.pi, +beta, g_ + np.pi, "flip", flags))

        for th4, th5, th6, w_lbl, fl in wrist_opts:
            q_raw = np.array([th1, th2, th3, th4, th5, th6]) - off
            q_fit, viol = _fit_to_limits(q_raw, limits, q_ref)
            fk = forward_kinematics(q_fit, model)
            e_pos, e_ori = pose_error(T, fk.T)
            if e_pos > VERIFY_TOL_POS or e_ori > VERIFY_TOL_ORI:
                continue                                      # never return an unverified branch
            sol = IKSolution(q=q_fit, shoulder=sh_lbl, elbow=el_lbl, wrist=w_lbl,
                             within_limits=bool(np.all(viol <= 1e-9)), limit_violation=viol,
                             position_error=e_pos, orientation_error=e_ori, singular=list(fl))
            if not any(np.linalg.norm(wrap_to_pi(sol.q - o.q)) < 1e-6 for o in solutions):
                solutions.append(sol)

    if not solutions:                                         # extremely unlikely numerical failure
        return IKResult("unreachable", "verification_failed",
                        "analytic branches did not pass the FK check (numerically degenerate target)",
                        pw, dg, [])
    n_ok = sum(s.within_limits for s in solutions)
    if n_ok == 0:
        worst = min(float(np.max(s.limit_violation)) for s in solutions)
        return IKResult("joint_limits", "all_branches_violate_limits",
                        f"target is geometrically reachable ({len(solutions)} branch(es)) but every branch violates "
                        f"a joint limit (smallest violation {np.degrees(worst):.1f} deg)", pw, dg, solutions)
    sing = sorted({f for s in solutions for f in s.singular})
    msg = f"{len(solutions)} geometric branch(es), {n_ok} inside the joint limits"
    if sing:
        msg += f"; singular: {', '.join(sing)}"
    return IKResult("ok", "solved" if not sing else "solved_singular", msg, pw, dg, solutions)
