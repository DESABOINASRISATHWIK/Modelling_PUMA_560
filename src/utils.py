"""Small SO(3)/SE(3) helpers used by the whole project (NumPy only)."""

from __future__ import annotations

import numpy as np


def wrap_to_pi(angle):
    """Wrap angle(s) to the interval (-pi, pi]."""
    a = np.arctan2(np.sin(angle), np.cos(angle))
    a = np.where(a <= -np.pi + 1e-12, np.pi, a)
    return a if np.ndim(angle) else float(a)


def rot_x(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform (uses R^T, no general matrix inverse)."""
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float)


def is_rotation(R: np.ndarray, tol: float = 1e-8) -> bool:
    return (R.shape == (3, 3) and np.allclose(R.T @ R, np.eye(3), atol=tol)
            and abs(np.linalg.det(R) - 1.0) < tol)


def rotation_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, angle in [0, pi]) of a rotation matrix."""
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-7:                       # small angle: sin(a)/a ~ 1
        return 0.5 * v
    if np.pi - angle < 1e-5:               # near pi: use the symmetric part
        B = (R + np.eye(3)) / 2.0          # = n n^T
        k = int(np.argmax(np.diag(B)))
        n = B[:, k] / np.sqrt(B[k, k])
        # fix the sign with the (small) antisymmetric part when available
        if np.dot(n, v) < 0:
            n = -n
        return angle * n
    return angle / (2.0 * np.sin(angle)) * v


def orientation_error_vector(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector that rotates R_current into R_target."""
    return rotation_log(R_target @ R_current.T)


def orientation_angle(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle [rad] between two rotation matrices."""
    return float(np.linalg.norm(rotation_log(R_a.T @ R_b)))


def pose_error(T_target: np.ndarray, T_current: np.ndarray) -> tuple[float, float]:
    """(position error [m], orientation error [rad]) between two poses."""
    pos = float(np.linalg.norm(T_target[:3, 3] - T_current[:3, 3]))
    ori = orientation_angle(T_target[:3, :3], T_current[:3, :3])
    return pos, ori


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) Ry(pitch) Rx(roll)  (fixed-axis x-y-z / ZYX Euler)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`rpy_to_matrix` (pitch in [-pi/2, pi/2])."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-9:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:                                  # gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation (QR of a Gaussian matrix)."""
    Q, Rr = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(Rr)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q
