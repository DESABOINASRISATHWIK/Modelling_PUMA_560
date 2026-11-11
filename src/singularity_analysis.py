"""
Singularity analysis of the PUMA-560.   NumPy only.

Closed-form singularity conditions (derived from the DH model)
--------------------------------------------------------------
The Jacobian referred to the WRIST CENTRE p_w is block lower triangular
(the wrist axes pass through p_w, so their linear columns vanish):

        [ J_p(q1,q2,q3)   0              ]            det(J) is independent of the
  J_w = [ [z0 z1 z2]    [z3 z4 z5]       ] ,          reference point (J_flange = [[I, -[r]x],[0, I]] J_w),

    det J = det(J_p) * det[z3 z4 z5].

With the wrist-centre position  p_w = Rz(q1)(X, -d3, 0) + (0, 0, d1 + Y),  (X,Y) = Rot(q2)(a2+u, v)
(see inverse_kinematics.py) one finds

    det J_p        = - X * a2 * v ,      v = a3 sin q3 + d4 cos q3 = L sin(q3 + phi)
    det[z3 z4 z5]  = - sin q5   (z4 is orthogonal to z3 and z5; sign from the DH alphas)

    =>  det J = + a2 * X * v * sin(q5) = a2 * L * X * sin(q3 + phi) * sin(q5)   (verified numerically in tests)

so the arm is singular exactly when one of the three factors vanishes:

  SHOULDER   X = 0, i.e. the wrist centre lies on the cylinder of radius |d3| around the waist axis:
             a2 cos q2 + a3 cos(q2+q3) - d4 sin(q2+q3) = 0 .
             Then dp_w/dq1 = Rz(q1)(d3, 0, 0) and dp_w/dq2, dp_w/dq3 lie in the arm plane, so the wrist
             centre cannot move horizontally perpendicular to the arm plane (direction Rz(q1) y).
  ELBOW      sin(q3 + phi) = 0, phi = atan2(d4, a3): upper arm and forearm lines (shoulder->elbow->wrist
             centre) are collinear.  q3 = -phi is the fully STRETCHED arm (workspace boundary);
             q3 = pi - phi is the fully FOLDED arm.
  WRIST      sin q5 = 0: axes of joints 4 and 6 are collinear -> only q4 +/- q6 is controllable.

Numerical diagnostics use the SVD  J = U S V^T :  sigma_min, rank, condition number
sigma_max / sigma_min and Yoshikawa's manipulability prod(sigma_i) = |det J|.
(Units caveat: J mixes m/rad and rad/rad rows, so absolute sigma values depend on the choice of
units; the *zeros* and the relative behaviour are meaningful.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import robot_parameters as rp
from .dh_model import PUMA560, DHModel
from .jacobian import geometric_jacobian

SIGMA_SINGULAR = 1e-6       # sigma_min below this  -> singular
SIGMA_NEAR = 0.01           # sigma_min below this  -> near-singular.  Chosen from data: over 20000 random
#                             valid configurations the median sigma_min is 0.09 and 8% lie below 0.01
#                             (about 2 deg from a wrist singularity: sigma_min ~ 0.27 |sin q5| there).
ANALYTIC_TOL = 1e-6         # |factor| below this -> that analytic singularity is active


# ----------------------------------------------------------------------------
# SVD based metrics
# ----------------------------------------------------------------------------
@dataclass
class SVDMetrics:
    singular_values: np.ndarray
    sigma_min: float
    sigma_max: float
    rank: int
    condition_number: float
    manipulability: float


def svd_metrics(J: np.ndarray, rank_tol: float | None = None) -> SVDMetrics:
    s = np.linalg.svd(J, compute_uv=False)
    if rank_tol is None:
        rank_tol = max(J.shape) * np.finfo(float).eps * s[0]
    rank = int(np.sum(s > rank_tol))
    cond = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    return SVDMetrics(s, float(s[-1]), float(s[0]), rank, cond, float(np.prod(s)))


# ----------------------------------------------------------------------------
# Analytic singularity factors
# ----------------------------------------------------------------------------
def arm_plane_coordinates(q, model: DHModel = PUMA560) -> tuple[float, float]:
    """(X, Y): signed horizontal offset / height of the wrist centre in the arm plane."""
    q = model.check_q(q)
    th2, th3 = q[1] + model.theta_offset[1], q[2] + model.theta_offset[2]
    a2, a3, d4 = model.a[1], model.a[2], model.d[3]
    u = a3 * np.cos(th3) - d4 * np.sin(th3)
    v = a3 * np.sin(th3) + d4 * np.cos(th3)
    X = (a2 + u) * np.cos(th2) - v * np.sin(th2)
    Y = (a2 + u) * np.sin(th2) + v * np.cos(th2)
    return float(X), float(Y)


def singularity_factors(q, model: DHModel = PUMA560) -> dict:
    """The three factors of det J:  X (shoulder),  L sin(q3+phi) (elbow),  sin q5 (wrist)."""
    q = model.check_q(q)
    X, _ = arm_plane_coordinates(q, model)
    a3, d4 = model.a[2], model.d[3]
    th3, th5 = q[2] + model.theta_offset[2], q[4] + model.theta_offset[4]
    v = a3 * np.sin(th3) + d4 * np.cos(th3)
    return dict(shoulder=X, elbow=float(v), wrist=float(np.sin(th5)))


def determinant_closed_form(q, model: DHModel = PUMA560) -> float:
    """det J = a2 * X * v * sin(q5)  (v = L sin(q3 + phi))."""
    f = singularity_factors(q, model)
    return float(model.a[1] * f["shoulder"] * f["elbow"] * f["wrist"])


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------
@dataclass
class SingularityReport:
    q: np.ndarray
    svd: SVDMetrics
    factors: dict
    active: list              # analytic singularities with |factor| < ANALYTIC_TOL
    status: str               # "regular" | "near-singular" | "singular"

    def text(self) -> str:
        s = self.svd
        return (f"{self.status.upper():<13} sigma_min={s.sigma_min:.3e} cond={s.condition_number:.3e} "
                f"rank={s.rank}/6 w={s.manipulability:.3e} | X={self.factors['shoulder']:+.4f} "
                f"v={self.factors['elbow']:+.4f} sin(q5)={self.factors['wrist']:+.4f}"
                + (f" | active: {', '.join(self.active)}" if self.active else ""))


def analyze_configuration(q, model: DHModel = PUMA560) -> SingularityReport:
    q = model.check_q(q)
    J = geometric_jacobian(q, model)
    m = svd_metrics(J)
    fac = singularity_factors(q, model)
    active = [k for k, v in fac.items() if abs(v) < ANALYTIC_TOL]
    if m.sigma_min < SIGMA_SINGULAR or m.rank < model.n:
        status = "singular"
    elif m.sigma_min < SIGMA_NEAR:
        status = "near-singular"
    else:
        status = "regular"
    return SingularityReport(q.copy(), m, fac, active, status)


# ----------------------------------------------------------------------------
# Construction of exactly singular configurations
# ----------------------------------------------------------------------------
def singular_configuration(kind: str, q_base=None, model: DHModel = PUMA560) -> np.ndarray:
    """Return a configuration at which the requested singularity is exact.

    kind = "wrist"    : q5 = 0
           "elbow"    : q3 = -phi (arm stretched)
           "shoulder" : q2 chosen so that X(q2, q3) = 0  (tan q2 = (a2 + u)/v)
    All other joints are copied from ``q_base`` (default: a generic regular pose).
    """
    q = np.array([0.4, 0.8, -1.0, 0.5, 0.9, 0.3]) if q_base is None else model.check_q(q_base).copy()
    a2, a3, d4 = model.a[1], model.a[2], model.d[3]
    if kind == "wrist":
        q[4] = 0.0
    elif kind == "elbow":
        q[2] = -np.arctan2(d4, a3) - model.theta_offset[2]
    elif kind == "shoulder":
        th3 = q[2] + model.theta_offset[2]
        u = a3 * np.cos(th3) - d4 * np.sin(th3)
        v = a3 * np.sin(th3) + d4 * np.cos(th3)
        # X = (a2+u) cos th2 - v sin th2 = 0  ->  th2 = atan2(a2+u, v)   (or + pi)
        th2 = np.arctan2(a2 + u, v)
        cand = [th2, th2 - np.pi, th2 + np.pi]
        lo, hi = model.limits[1]
        ok = [c - model.theta_offset[1] for c in cand if lo <= c - model.theta_offset[1] <= hi]
        q[1] = ok[0] if ok else th2 - model.theta_offset[1]
    else:
        raise ValueError("kind must be 'shoulder', 'elbow' or 'wrist'")
    return q


def sweep_sigma_min(q_base, joint: int, values, model: DHModel = PUMA560) -> np.ndarray:
    """sigma_min(q) while joint index ``joint`` runs through ``values``."""
    out = np.empty(len(values))
    q = np.array(q_base, dtype=float)
    for k, val in enumerate(values):
        q[joint] = val
        out[k] = svd_metrics(geometric_jacobian(q, model)).sigma_min
    return out


def sigma_min_map_q2_q3(q_base, q2_vals, q3_vals, model: DHModel = PUMA560) -> np.ndarray:
    """sigma_min over a (q2, q3) grid; other joints from q_base.  Shape (len(q3), len(q2))."""
    q = np.array(q_base, dtype=float)
    out = np.empty((len(q3_vals), len(q2_vals)))
    for i, q3 in enumerate(q3_vals):
        for j, q2 in enumerate(q2_vals):
            q[1], q[2] = q2, q3
            out[i, j] = svd_metrics(geometric_jacobian(q, model)).sigma_min
    return out
