"""
SECONDARY inverse kinematics: damped-least-squares (Levenberg-Marquardt style) iteration.

This is NOT the primary IK of the project (that is the closed form in
inverse_kinematics.py).  It exists to (i) cross-check the analytic solver and
(ii) show how an iterative Jacobian method behaves near singularities.

    e   = [ p* - p(q) ;  log(R* R(q)^T) ]            (base-frame pose error)
    dq  = J^T (J J^T + lambda^2 I)^-1 e ,   q <- q + dq

It uses the analytic Jacobian of jacobian.py.  No SciPy / black-box optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dh_model import PUMA560, DHModel
from .forward_kinematics import forward_kinematics
from .jacobian import geometric_jacobian
from .utils import orientation_error_vector


@dataclass
class NumericalIKResult:
    q: np.ndarray
    converged: bool
    iterations: int
    position_error: float
    orientation_error: float


def numerical_ik_dls(T_target, q0, model: DHModel = PUMA560, max_iter: int = 300,
                     tol_pos: float = 1e-9, tol_ori: float = 1e-9, damping: float = 0.02,
                     clip_limits: bool = False) -> NumericalIKResult:
    T_target = np.asarray(T_target, dtype=float)
    q = model.check_q(q0).copy()
    for it in range(max_iter):
        fk = forward_kinematics(q, model)
        e = np.r_[T_target[:3, 3] - fk.position,
                  orientation_error_vector(T_target[:3, :3], fk.rotation)]
        pe, oe = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        if pe < tol_pos and oe < tol_ori:
            return NumericalIKResult(q, True, it, pe, oe)
        J = geometric_jacobian(q, model)
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
        q = q + dq
        if clip_limits:
            q = np.clip(q, model.limits[:, 0], model.limits[:, 1])
    fk = forward_kinematics(q, model)
    e = np.r_[T_target[:3, 3] - fk.position, orientation_error_vector(T_target[:3, :3], fk.rotation)]
    return NumericalIKResult(q, False, max_iter, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:])))
