"""
Standard Denavit-Hartenberg model of the PUMA-560.

    A_{i-1,i}(theta, d, a, alpha) = Rot(z, theta) Trans(z, d) Trans(x, a) Rot(x, alpha)

                 | c_t  -s_t c_a   s_t s_a   a c_t |
               = | s_t   c_t c_a  -c_t s_a   a s_t |
                 |  0      s_a       c_a       d   |
                 |  0       0         0        1   |

The numbers come from :mod:`src.robot_parameters`; this module only turns the
DH table into matrices.  Nothing here is hard-coded for a particular joint
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import robot_parameters as rp


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Standard-DH homogeneous transform  Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


@dataclass(frozen=True)
class DHModel:
    """DH table + fixed base/tool transforms of a serial revolute arm."""

    alpha: np.ndarray = field(default_factory=lambda: rp.DH_ALPHA.copy())
    a: np.ndarray = field(default_factory=lambda: rp.DH_A.copy())
    d: np.ndarray = field(default_factory=lambda: rp.DH_D.copy())
    theta_offset: np.ndarray = field(default_factory=lambda: rp.THETA_OFFSET.copy())
    T_base: np.ndarray = field(default_factory=lambda: rp.T_BASE.copy())
    T_tool: np.ndarray = field(default_factory=lambda: rp.T_TOOL.copy())
    limits: np.ndarray = field(default_factory=lambda: rp.JOINT_LIMITS.copy())

    @property
    def n(self) -> int:
        return len(self.alpha)

    # ------------------------------------------------------------------ table
    def table(self) -> np.ndarray:
        """Rows [theta_offset, d, a, alpha] for each joint (q is added to theta)."""
        return np.column_stack([self.theta_offset, self.d, self.a, self.alpha])

    def check_q(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n,):
            raise ValueError(f"expected {self.n} joint angles, got shape {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError("joint angles must be finite")
        return q

    # ---------------------------------------------------------------- chains
    def link_transforms(self, q) -> list[np.ndarray]:
        """Relative transforms A_{i-1,i}, i = 1..n  (joint offsets included)."""
        q = self.check_q(q)
        return [dh_transform(q[i] + self.theta_offset[i], self.d[i], self.a[i], self.alpha[i])
                for i in range(self.n)]

    def frames(self, q) -> list[np.ndarray]:
        """Absolute transforms T_0i for i = 0..n  (index 0 is the base frame)."""
        T = self.T_base.copy()
        out = [T.copy()]
        for A in self.link_transforms(q):
            T = T @ A
            out.append(T.copy())
        return out

    def flange_pose(self, q) -> np.ndarray:
        return self.frames(q)[-1] @ self.T_tool


PUMA560 = DHModel()


def describe_robot(model: DHModel = PUMA560) -> str:
    """Human-readable robot specification (tables used in the README/report)."""
    lines = []
    lines.append("PUMA-560  -  standard DH  (A = Rz(theta) Tz(d) Tx(a) Rx(alpha),  theta = q + offset)")
    lines.append(f"{'joint':<15}{'type':<10}{'theta_off[deg]':>15}{'d [m]':>10}{'a [m]':>10}"
                 f"{'alpha[deg]':>12}{'q_min[deg]':>12}{'q_max[deg]':>12}")
    for i in range(model.n):
        lines.append(f"{str(i + 1) + ':' + rp.JOINT_NAMES[i]:<15}{rp.JOINT_TYPES[i]:<10}"
                     f"{np.degrees(model.theta_offset[i]):15.1f}{model.d[i]:10.5f}{model.a[i]:10.5f}"
                     f"{np.degrees(model.alpha[i]):12.1f}{np.degrees(model.limits[i, 0]):12.1f}"
                     f"{np.degrees(model.limits[i, 1]):12.1f}")
    lines.append(f"tool flange offset  T_6E = Trans(z, {rp.D_FLANGE} m)")
    lines.append("")
    lines.append("link   mass[kg]   COM in link frame [m]          diag(I_com) [kg m^2]")
    for i in range(model.n):
        com = rp.LINK_COM[i]
        Id = np.diag(rp.LINK_INERTIA[i])
        lines.append(f"{i + 1:<7}{rp.LINK_MASS[i]:<11.3f}[{com[0]:8.4f} {com[1]:8.4f} {com[2]:8.4f}]"
                     f"   [{Id[0]:.2e} {Id[1]:.2e} {Id[2]:.2e}]")
    lines.append("")
    lines.append("Joint axes and origins in the BASE frame at q = 0 (z_{i-1} through p_{i-1}):")
    F = model.frames(np.zeros(model.n))
    for i in range(model.n):
        z, p = F[i][:3, 2], F[i][:3, 3]
        lines.append(f"  joint {i + 1}: axis z_{i} = [{z[0]:+.0f} {z[1]:+.0f} {z[2]:+.0f}]   "
                     f"point p_{i} = [{p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}]")
    pw = F[4][:3, 3]
    lines.append(f"  wrist centre (origin of {{4}}={{5}}={{6}}) at q=0: [{pw[0]:+.4f} {pw[1]:+.4f} {pw[2]:+.4f}]")
    return "\n".join(lines)
