"""Collision-checked planning for a PUMA carrying a rigid cuboid payload.

RRT-Connect and quintic time scaling are the project's NumPy implementations;
MuJoCo is used by :class:`CollisionChecker` only to query geometric contacts.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .collision import CollisionChecker
from .path_planning import RRTConnectPlanner, path_length
from .trajectory import QuinticTrajectory


@dataclass
class CarryPlan:
    success: bool
    status: str
    waypoints: np.ndarray | None
    trajectory: QuinticTrajectory | None
    direct_blocked_segments: tuple[str, ...]
    path_length_rad: float
    planning_time_s: float
    min_clearance_m: float | None  # conservative lower bound from obstacle inflation
    first_collision_time_s: float | None
    segment_summaries: tuple[dict, ...]

    def summary(self):
        if not self.success:
            return f"carry plan failed: {self.status} ({self.planning_time_s:.2f} s)"
        direct = ", ".join(self.direct_blocked_segments) or "none"
        clearance = "n/a" if self.min_clearance_m is None else f"{self.min_clearance_m:.3f} m"
        return (f"carry route success: {len(self.waypoints)} waypoints, "
                f"{self.path_length_rad:.3f} rad, {self.planning_time_s:.3f} s, "
                f"min clearance {clearance}, direct-blocked segments: {direct}")


def _trajectory_collision(checker, trajectory, sample_dt=0.01, edge_resolution=0.008):
    times = np.append(np.arange(0.0, trajectory.duration, sample_dt), trajectory.duration)
    qs = np.vstack([trajectory.evaluate(t)[0] for t in times])
    for i in range(len(qs) - 1):
        if not checker.is_edge_valid(qs[i], qs[i + 1], resolution=edge_resolution):
            return False, float(times[i])
    return True, None


def plan_carry_route(plan_checker: CollisionChecker, true_checker: CollisionChecker,
                     q_start, q_targets, *, dt=0.002, max_velocity=None,
                     max_acceleration=None, seed=7, timeout=5.0,
                     clearance_lower_bound=None):
    """Plan through ordered joint-space targets and verify the timed route.

    Each adjacent target pair is solved independently. A valid direct edge is
    retained by RRT-Connect; blocked edges are routed around the inflated scene.
    The concatenated path is then timed and checked against the true geometry.
    ``clearance_lower_bound`` records the safety margin guaranteed by planning
    against the expanded obstacle boxes; it is not an exact closest distance.
    """
    t0 = time.perf_counter()
    q_start = np.asarray(q_start, dtype=float)
    targets = [q_start.copy()]
    targets.extend(np.asarray(q, dtype=float) for q in q_targets)
    if q_start.shape != (6,) or any(q.shape != (6,) or not np.all(np.isfinite(q)) for q in targets):
        raise ValueError("carry route start and targets must be finite six-joint vectors")
    if not plan_checker.is_configuration_valid(q_start):
        return CarryPlan(False, "invalid_start", None, None, (), np.nan,
                         time.perf_counter() - t0, None, 0.0, ())

    parts = []
    summaries = []
    blocked = []
    last = q_start
    for index, goal in enumerate(targets[1:]):
        planner = RRTConnectPlanner(plan_checker, edge_resolution=0.015,
                                   seed=int(seed) + index, max_iterations=5000,
                                   timeout=timeout, shortcut_attempts=100)
        result = planner.solve(last, goal)
        summaries.append({"segment": index, "status": result.status,
                          "planning_time_s": result.planning_time,
                          "path_length_rad": result.path_length,
                          "waypoints": result.n_waypoints, "seed": result.seed,
                          "iterations": result.iterations})
        if not result.success:
            return CarryPlan(False, f"segment_{index}_{result.status}", None, None,
                             tuple(blocked), np.nan, time.perf_counter() - t0,
                             None, None, tuple(summaries))
        if result.status != "success_direct":
            blocked.append(str(index))
        segment = result.path
        parts.append(segment if not parts else segment[1:])
        last = goal.copy()

    waypoints = np.vstack(parts)
    # Drop numerically duplicate waypoints from adjacent subplans.
    keep = np.r_[True, np.linalg.norm(np.diff(waypoints, axis=0), axis=1) > 1e-9]
    waypoints = waypoints[keep]
    if len(waypoints) < 2:
        return CarryPlan(False, "degenerate_route", None, None, tuple(blocked), 0.0,
                         time.perf_counter() - t0, None, None, tuple(summaries))
    trajectory = QuinticTrajectory(waypoints, dt=dt, max_velocity=max_velocity,
                                   max_acceleration=max_acceleration)
    # Check the complete time-scaled curve against the inflated scene too;
    # checking only the RRT waypoints would not preserve its safety margin
    # after quintic interpolation and timing.
    valid, collision_time = _trajectory_collision(plan_checker, trajectory)
    if not valid:
        return CarryPlan(False, "timed_trajectory_margin_violation", waypoints, trajectory,
                         tuple(blocked), path_length(waypoints), time.perf_counter() - t0,
                         None, collision_time, tuple(summaries))
    valid, collision_time = _trajectory_collision(true_checker, trajectory)
    if not valid:
        return CarryPlan(False, "timed_trajectory_collision", waypoints, trajectory,
                         tuple(blocked), path_length(waypoints), time.perf_counter() - t0,
                         None, collision_time, tuple(summaries))
    # MuJoCo's mj_geomDistance returns zero for some box/box pairs, including
    # separated boxes.  Report the conservative clearance guaranteed by the
    # inflated-obstacle check instead of presenting that zero as exact.
    clearance = (float(clearance_lower_bound)
                 if true_checker.obstacle_geoms and clearance_lower_bound is not None
                 else None)
    return CarryPlan(True, "success", waypoints, trajectory, tuple(blocked),
                     path_length(waypoints), time.perf_counter() - t0,
                     clearance, None, tuple(summaries))
