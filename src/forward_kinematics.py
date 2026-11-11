"""
Forward kinematics of the PUMA-560 from its DH table (NumPy only).

Derivation (standard DH, see dh_model.py)
-----------------------------------------
    T_0E(q) = T_base  A_1(q1) A_2(q2) A_3(q3) A_4(q4) A_5(q5) A_6(q6)  T_6E

with A_i = Rz(q_i + off_i) Tz(d_i) Tx(a_i) Rx(alpha_i).  Because the DH table of
this robot has a_4 = a_5 = a_6 = d_5 = d_6 = 0, the origins of frames {4},{5},{6}
coincide: the wrist centre

    p_w = p_3 + d_4 z_3 ,     p_3 = p_2 + d_3 z_2 + a_3 x_3 ,
    p_2 = p_1 + a_2 x_2 ,     p_1 = (0, 0, d_1)

depends on q1, q2, q3 only.  The last three joints only orient the tool
(R_36 = Rz(q4) Ry(-q5) Rz(q6)); this decoupling is what the closed-form IK uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dh_model import PUMA560, DHModel


@dataclass
class FKResult:
    T: np.ndarray                    # 4x4 end-effector (flange) pose in the base frame
    position: np.ndarray             # (3,)  end-effector position  [m]
    rotation: np.ndarray             # (3,3) end-effector orientation
    frames: list                     # [T_00, T_01, ..., T_06]  absolute frames
    link_transforms: list            # [A_1, ..., A_6]          relative DH transforms
    wrist_center: np.ndarray         # (3,)  common origin of frames {4},{5},{6}


def forward_kinematics(q, model: DHModel = PUMA560) -> FKResult:
    """Pose of the tool flange for the joint vector ``q`` (6 angles, radians).

    Works for any finite q (joint limits are *not* enforced here; use
    ``joint_limit_violation`` for that).
    """
    q = model.check_q(q)
    A = model.link_transforms(q)
    frames = [model.T_base.copy()]
    T = model.T_base.copy()
    for Ai in A:
        T = T @ Ai
        frames.append(T.copy())
    T_E = T @ model.T_tool
    return FKResult(T=T_E, position=T_E[:3, 3].copy(), rotation=T_E[:3, :3].copy(),
                    frames=frames, link_transforms=A, wrist_center=frames[4][:3, 3].copy())


def end_effector_pose(q, model: DHModel = PUMA560) -> np.ndarray:
    """Convenience: just the 4x4 flange pose."""
    return forward_kinematics(q, model).T


def wrist_center(q, model: DHModel = PUMA560) -> np.ndarray:
    return forward_kinematics(q, model).wrist_center


def joint_limit_violation(q, model: DHModel = PUMA560) -> np.ndarray:
    """Per-joint amount [rad] by which q is outside its limits (0 if inside)."""
    q = model.check_q(q)
    return np.maximum(model.limits[:, 0] - q, 0.0) + np.maximum(q - model.limits[:, 1], 0.0)


def within_limits(q, model: DHModel = PUMA560, tol: float = 1e-9) -> bool:
    return bool(np.all(joint_limit_violation(q, model) <= tol))


def random_valid_configuration(rng: np.random.Generator, model: DHModel = PUMA560,
                               margin: float = 0.0) -> np.ndarray:
    """Uniform random joint vector inside the joint limits (shrunk by ``margin``)."""
    lo, hi = model.limits[:, 0] + margin, model.limits[:, 1] - margin
    return rng.uniform(lo, hi)
