"""
Velocity kinematics of the PUMA-560: geometric Jacobian.   NumPy only.

Derivation
----------
Let p_e be the end-effector (flange) position and, for joint i (revolute),
z_{i-1} the unit vector of its axis and p_{i-1} a point on that axis (origin of
DH frame {i-1}); both are read from the DH forward kinematics (FKResult.frames).

Rotating joint i alone with rate qd_i turns the whole distal chain rigidly about
the axis z_{i-1}:

    angular velocity of the end effector   w   = z_{i-1} qd_i
    linear  velocity of the end effector   v   = (w x (p_e - p_{i-1})) = z_{i-1} x (p_e - p_{i-1}) qd_i

Velocities of the individual joints superpose (the motion is a sum of rotations
about different axes), hence

    [v]            [J_v]
    [w] = J(q) qd = [J_w] qd ,   J_v,i = z_{i-1} x (p_e - p_{i-1}),   J_w,i = z_{i-1}.

J is 6x6 and, by default, expressed in the BASE frame {0}: v is the velocity of the
flange origin, w the angular velocity of the flange, both resolved in {0}.  Options:
``frame="ee"`` resolves both in the flange frame ({E}), J_E = blockdiag(R^T, R^T) J_0;
``point="wrist"`` uses the wrist centre as reference point (then the wrist columns of
J_v vanish and J is block lower triangular -> the singularity analysis decouples).

A finite-difference Jacobian is provided ONLY to validate the analytic one.
"""

from __future__ import annotations

import numpy as np

from .dh_model import PUMA560, DHModel
from .forward_kinematics import forward_kinematics
from .utils import rotation_log


def geometric_jacobian(q, model: DHModel = PUMA560, frame: str = "base", point: str = "flange") -> np.ndarray:
    """6x6 geometric Jacobian J(q) such that [v; w] = J(q) qdot."""
    fk = forward_kinematics(q, model)
    p_e = fk.position if point == "flange" else fk.wrist_center
    if point not in ("flange", "wrist"):
        raise ValueError("point must be 'flange' or 'wrist'")
    J = np.zeros((6, model.n))
    for i in range(model.n):
        z = fk.frames[i][:3, 2]            # axis of joint i+1  = z_i (0-based list: frame i-1 in the text)
        p = fk.frames[i][:3, 3]
        J[:3, i] = np.cross(z, p_e - p)    # translational part
        J[3:, i] = z                       # rotational part
    if frame == "base":
        return J
    if frame == "ee":
        R = fk.rotation.T
        out = np.zeros_like(J)
        out[:3], out[3:] = R @ J[:3], R @ J[3:]
        return out
    raise ValueError("frame must be 'base' or 'ee'")


def jacobian_numerical(q, model: DHModel = PUMA560, h: float = 1e-6) -> np.ndarray:
    """Central finite-difference Jacobian of the FK (base frame, flange point).

    Linear part: dp/dq_i.   Angular part: spatial angular velocity
    w_i = log(R(q+h e_i) R(q-h e_i)^T) / (2h)   (rotation vector of the incremental rotation).
    Used only for validation.
    """
    q = model.check_q(q)
    J = np.zeros((6, model.n))
    for i in range(model.n):
        dq = np.zeros(model.n)
        dq[i] = h
        fp, fm = forward_kinematics(q + dq, model), forward_kinematics(q - dq, model)
        J[:3, i] = (fp.position - fm.position) / (2 * h)
        J[3:, i] = rotation_log(fp.rotation @ fm.rotation.T) / (2 * h)
    return J


def end_effector_twist(q, qdot, model: DHModel = PUMA560) -> np.ndarray:
    """[v; w] = J(q) qdot (base frame)."""
    return geometric_jacobian(q, model) @ np.asarray(qdot, dtype=float)


def joint_velocity_from_twist(q, twist, model: DHModel = PUMA560, method: str = "pinv",
                              damping: float = 0.05, rcond: float = 1e-9) -> np.ndarray:
    """Resolved-rate inverse  qdot = J^+ [v; w].

    method = "pinv" : SVD pseudo-inverse (truncated at rcond * sigma_max)
             "dls"  : damped least squares  J^T (J J^T + lambda^2 I)^-1   (bounded near singularities)
    """
    J = geometric_jacobian(q, model)
    twist = np.asarray(twist, dtype=float)
    if method == "pinv":
        U, s, Vt = np.linalg.svd(J)
        keep = s > rcond * s[0]
        s_inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
        return Vt.T @ (s_inv * (U.T @ twist))
    if method == "dls":
        return J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), twist)
    raise ValueError("method must be 'pinv' or 'dls'")
