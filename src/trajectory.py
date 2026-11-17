"""Quintic joint-space trajectory generation for the PUMA 560.

A path is an (M, 6) array of joint-space waypoints in radians. Each segment
uses quintic time scaling and comes to rest at both ends. Consequently,
position, velocity and acceleration are continuous at every waypoint.

The coefficients and derivatives are implemented with NumPy; no trajectory
planning package is used.
"""
from __future__ import annotations

import numpy as np


class QuinticTrajectory:
    """Generate a time-parameterized, piecewise-quintic joint trajectory.

    Parameters
    ----------
    waypoints : array_like, shape (M, 6)
        Joint-space path in radians, including start and goal.
    dt : float
        Sampling interval [s].
    max_velocity : array_like, shape (6,), optional
        Per-joint velocity limits [rad/s]. Defaults to 1 rad/s per joint.
    max_acceleration : array_like, shape (6,), optional
        Per-joint acceleration limits [rad/s^2]. Defaults to 2 rad/s^2.
    segment_times : array_like, shape (M-1,), optional
        Explicit segment durations [s]. If omitted, durations are selected
        to satisfy the requested velocity and acceleration limits.
    """

    _MAX_S_DU = 1.875
    _MAX_S_DDU = 10.0 / np.sqrt(3.0)

    def __init__(self, waypoints, dt=0.002, max_velocity=None,
                 max_acceleration=None, segment_times=None):
        self.waypoints = np.asarray(waypoints, dtype=float)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 6:
            raise ValueError("waypoints must have shape (M, 6)")
        if self.waypoints.shape[0] < 2 or not np.all(np.isfinite(self.waypoints)):
            raise ValueError("at least two finite six-joint waypoints are required")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        self.dt = float(dt)
        self.max_velocity = self._limits(max_velocity, 1.0, "max_velocity")
        self.max_acceleration = self._limits(max_acceleration, 2.0, "max_acceleration")
        self.segment_times = self._durations(segment_times)
        self.segment_starts = np.concatenate(([0.0], np.cumsum(self.segment_times)[:-1]))
        self.duration = float(np.sum(self.segment_times))

    @staticmethod
    def _limits(values, default, name):
        arr = np.full(6, default, dtype=float) if values is None else np.asarray(values, dtype=float)
        if arr.shape != (6,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError(f"{name} must contain six positive finite values")
        return arr

    def _durations(self, segment_times):
        nseg = len(self.waypoints) - 1
        if segment_times is not None:
            times = np.asarray(segment_times, dtype=float)
            if times.shape != (nseg,) or not np.all(np.isfinite(times)) or np.any(times <= 0):
                raise ValueError(f"segment_times must contain {nseg} positive finite values")
            return times.copy()

        durations = []
        for qa, qb in zip(self.waypoints[:-1], self.waypoints[1:]):
            delta = np.abs(qb - qa)
            t_velocity = np.max(self._MAX_S_DU * delta / self.max_velocity)
            t_acceleration = np.max(np.sqrt(self._MAX_S_DDU * delta / self.max_acceleration))
            durations.append(max(float(t_velocity), float(t_acceleration), self.dt))
        return np.asarray(durations)

    @staticmethod
    def _scale(u):
        """Return quintic position, first derivative and second derivative w.r.t. u."""
        u = np.clip(float(u), 0.0, 1.0)
        s = 10*u**3 - 15*u**4 + 6*u**5
        ds = 30*u**2 - 60*u**3 + 30*u**4
        dds = 60*u - 180*u**2 + 120*u**3
        return s, ds, dds

    def evaluate(self, t):
        """Evaluate desired q, qdot, qddot at time t [s].

        Returns three independent arrays, each with shape (6,). Values before
        zero and after the trajectory duration are held at the endpoint with
        zero velocity and acceleration.
        """
        if not np.isfinite(t):
            raise ValueError("time must be finite")
        if t <= 0.0:
            return self.waypoints[0].copy(), np.zeros(6), np.zeros(6)
        if t >= self.duration:
            return self.waypoints[-1].copy(), np.zeros(6), np.zeros(6)

        idx = int(np.searchsorted(self.segment_starts, t, side="right") - 1)
        idx = min(max(idx, 0), len(self.segment_times) - 1)
        T = self.segment_times[idx]
        u = (t - self.segment_starts[idx]) / T
        s, ds, dds = self._scale(u)
        delta = self.waypoints[idx + 1] - self.waypoints[idx]
        q = self.waypoints[idx] + delta * s
        qd = delta * (ds / T)
        qdd = delta * (dds / (T*T))
        return q, qd, qdd

    def generate(self):
        """Return sampled arrays ``(time, q, qd, qdd)`` including final time."""
        times = np.arange(0.0, self.duration, self.dt)
        times = np.append(times, self.duration)
        samples = [self.evaluate(t) for t in times]
        q = np.vstack([x[0] for x in samples])
        qd = np.vstack([x[1] for x in samples])
        qdd = np.vstack([x[2] for x in samples])
        return times, q, qd, qdd


def check_trajectory_collision(checker, trajectory, resolution=0.02):
    """Check every consecutive sampled trajectory configuration/edge.

    ``trajectory`` is the (N, 6) q array returned by ``generate``. This is a
    discrete collision check whose conservatism depends on the sampling and
    edge resolution; it is not a continuous-collision guarantee.
    """
    q = np.asarray(trajectory, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6 or len(q) < 2:
        raise ValueError("trajectory must have shape (N, 6), N >= 2")
    for i in range(len(q) - 1):
        if not checker.is_edge_valid(q[i], q[i + 1], resolution=resolution):
            return False, i
    return True, None
