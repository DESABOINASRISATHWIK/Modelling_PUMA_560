"""
Velocity kinematics of the PUMA-560 built on the geometric Jacobian (NumPy only).

    [v]   [J_v]
    [w] = [J_w] qdot            v: flange-origin velocity, w: flange angular velocity (base frame)

What this module adds on top of jacobian.py (geometric interpretation, viva material)
------------------------------------------------------------------------------------
1. Velocity (manipulability) ellipsoids.  Joint velocities of unit norm |qdot| = 1 are mapped by
   J_v (or J_w) onto an ellipsoid in task space.  With the SVD  J_v = U S V^T  the semi-axes are the
   singular values s_i along the columns of U.  The ellipsoid collapses (an s_i -> 0) in the
   direction in which the end effector loses mobility.
2. Directional gain.  Largest end-effector speed along a unit direction u for |qdot| = 1:
   |J_v^T u| (linear) or |J_w^T u| (angular).
3. Singular directions.  For J = U S V^T the smallest singular value belongs to
   u_min (task twist that is hard / impossible to produce) and
   v_min (joint motion that produces almost / exactly no end-effector motion: the self-motion).
   At a singularity  J v_min = 0  exactly.
4. Joint speed needed for a task twist: |qdot| = |J^+ twist| ~ |twist| / sigma  -> blows up near a
   singularity for the pseudo-inverse and stays bounded for damped least squares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dh_model import PUMA560, DHModel
from .jacobian import geometric_jacobian, joint_velocity_from_twist


@dataclass
class VelocityEllipsoid:
    semi_axes: np.ndarray          # (3,) singular values (descending)
    axes: np.ndarray               # (3,3) columns = principal directions (task space, base frame)
    volume_measure: float          # product of semi axes (0 when the ellipsoid is flat)
    kind: str                      # "linear" [m/s per rad/s]  or  "angular" [rad/s per rad/s]

    def surface(self, center=np.zeros(3), scale: float = 1.0, n: int = 24):
        """Points of the ellipsoid surface (3 arrays of shape (n, n)) for plotting."""
        u, v = np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n)
        sphere = np.array([np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                           np.outer(np.ones_like(u), np.cos(v))])
        pts = np.tensordot(self.axes @ np.diag(self.semi_axes) * scale, sphere, axes=(1, 0))
        return tuple(pts[i] + center[i] for i in range(3))


def velocity_ellipsoid(q, part: str = "linear", model: DHModel = PUMA560) -> VelocityEllipsoid:
    """Ellipsoid reached by unit-norm joint velocities in linear or angular end-effector velocity."""
    J = geometric_jacobian(q, model)
    block = J[:3] if part == "linear" else J[3:]
    if part not in ("linear", "angular"):
        raise ValueError("part must be 'linear' or 'angular'")
    U, s, _ = np.linalg.svd(block)
    return VelocityEllipsoid(s, U, float(np.prod(s)), part)


def directional_gain(q, direction, part: str = "linear", model: DHModel = PUMA560) -> float:
    """max |u . (v or w)| over unit-norm qdot, i.e. |J_part^T u|."""
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    J = geometric_jacobian(q, model)
    block = J[:3] if part == "linear" else J[3:]
    return float(np.linalg.norm(block.T @ u))


@dataclass
class SingularDirections:
    sigma_min: float
    lost_twist: np.ndarray         # (6,) unit twist [v; w] hardest to produce  (left singular vector)
    null_motion: np.ndarray        # (6,) unit joint velocity producing the smallest end-effector motion
    residual: float                # |J null_motion| = sigma_min


def singular_directions(q, model: DHModel = PUMA560) -> SingularDirections:
    """Weakest task direction and the associated joint self-motion (exact null space at a singularity)."""
    J = geometric_jacobian(q, model)
    U, s, Vt = np.linalg.svd(J)
    return SingularDirections(float(s[-1]), U[:, -1].copy(), Vt[-1].copy(), float(np.linalg.norm(J @ Vt[-1])))


def required_joint_speed(q, twist, method: str = "pinv", damping: float = 0.05,
                         model: DHModel = PUMA560) -> float:
    """|qdot| needed for a task twist (pseudo-inverse truncated only at machine precision, or DLS)."""
    qd = joint_velocity_from_twist(q, twist, model, method=method, damping=damping, rcond=1e-14)
    return float(np.linalg.norm(qd))


def tracking_error(q, twist, qdot, model: DHModel = PUMA560) -> float:
    """|twist - J qdot|: what the chosen joint velocity fails to deliver (non-zero for DLS near singularities)."""
    return float(np.linalg.norm(np.asarray(twist, float) - geometric_jacobian(q, model) @ qdot))


def velocity_report(q, qdot, model: DHModel = PUMA560) -> dict:
    """Everything about one velocity mapping, for printing / the viewer HUD."""
    J = geometric_jacobian(q, model)
    tw = J @ np.asarray(qdot, dtype=float)
    return dict(linear_velocity=tw[:3], angular_velocity=tw[3:], linear_speed=float(np.linalg.norm(tw[:3])),
                angular_speed=float(np.linalg.norm(tw[3:])), jacobian=J)
