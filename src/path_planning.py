"""Collision-aware joint-space path planning for the PUMA 560 (Part 2).

Three planners, all written with NumPy only (MuJoCo is used *only* through
``CollisionChecker`` for contact queries, which is the permitted use):

* ``direct_joint_path``   straight line q(s) = q_start + s (q_goal - q_start)   (baseline)
* ``RRTPlanner``          single-tree RRT with goal bias                        (comparison)
* ``RRTConnectPlanner``   bidirectional RRT-Connect                             (main planner)

Both tree planners finish with collision-checked *shortcut smoothing*.

Every planner exposes ``solve(start, goal)`` which NEVER raises for an ordinary
planning failure; it returns a ``PlanResult`` with status, planning time, seed,
iteration count and path length.  ``plan(start, goal)`` is the older convenience
wrapper that returns the path or raises (``ValueError`` for an invalid endpoint,
``RuntimeError`` if the search fails).

Limitations (documented honestly)
---------------------------------
* Edges are checked at discrete samples (at most ``edge_resolution`` rad apart in any
  joint).  This is not continuous collision detection, so thin obstacles or fast
  sweeps can slip between samples.  The task layer therefore plans against obstacles
  *inflated by a safety margin* and every executed motion is re-checked.
* Distance is the plain Euclidean joint-space norm (all joints weighted equally).
* The planner is probabilistic and neither optimal nor complete; ``max_iterations``
  and ``timeout`` bound the effort.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .collision import CollisionChecker
from .scene import vector

# status codes of PlanResult
OK = "success"
OK_DIRECT = "success_direct"
INVALID_START = "invalid_start"
INVALID_GOAL = "invalid_goal"
TIMEOUT = "timeout"
MAX_ITER = "max_iterations"


@dataclass
class PlanResult:
    """Outcome of one planning query."""
    success: bool
    status: str
    path: np.ndarray | None          # (K, 6) waypoints [rad] including start and goal, None on failure
    planning_time: float             # wall-clock seconds
    iterations: int
    seed: int
    path_length: float               # joint-space Euclidean length of ``path`` [rad]; nan on failure
    raw_path_length: float           # length before smoothing; nan on failure
    n_nodes: int                     # total tree nodes created
    method: str
    settings: dict = field(default_factory=dict)

    @property
    def n_waypoints(self) -> int:
        return 0 if self.path is None else len(self.path)

    def summary(self) -> str:
        if not self.success:
            return (f"[{self.method}] FAILED ({self.status}) after {self.planning_time:.2f} s, "
                    f"{self.iterations} iterations, seed {self.seed}")
        return (f"[{self.method}] {self.status}: {self.n_waypoints} waypoints, length {self.path_length:.3f} rad "
                f"(raw {self.raw_path_length:.3f}), {self.planning_time * 1e3:.0f} ms, "
                f"{self.iterations} iterations, seed {self.seed}")


def path_length(path):
    """Euclidean joint-space path length [rad]."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 6 or len(path) < 2:
        raise ValueError("path must have shape (M, 6), M >= 2")
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


@dataclass
class _Tree:
    nodes: list
    parents: list

    @classmethod
    def rooted_at(cls, q):
        return cls([np.asarray(q, dtype=float).copy()], [-1])

    def nearest(self, q):
        distances = np.linalg.norm(np.asarray(self.nodes) - q, axis=1)
        return int(np.argmin(distances))

    def add(self, q, parent):
        self.nodes.append(np.asarray(q, dtype=float).copy())
        self.parents.append(int(parent))
        return len(self.nodes) - 1

    def path_to_root(self, index):
        path = []
        while index != -1:
            path.append(self.nodes[index].copy())
            index = self.parents[index]
        return path[::-1]


class _PlannerBase:
    """Shared sampling / steering / collision-checked extension / smoothing."""

    method = "base"

    def __init__(self, checker, step_size=0.20, edge_resolution=0.02,
                 goal_sample_rate=0.10, max_iterations=8000, seed=42,
                 shortcut_attempts=150, timeout=30.0):
        if not isinstance(checker, CollisionChecker):
            raise TypeError("checker must be a CollisionChecker")
        if not np.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be positive and finite")
        if not np.isfinite(edge_resolution) or edge_resolution <= 0:
            raise ValueError("edge_resolution must be positive and finite")
        if not 0 <= goal_sample_rate <= 1:
            raise ValueError("goal_sample_rate must be between 0 and 1")
        if max_iterations < 1 or shortcut_attempts < 0:
            raise ValueError("invalid iteration/shortcut count")
        if timeout is not None and (not np.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be positive and finite (or None)")

        self.checker = checker
        self.model = checker.model
        self.joints = checker.joints
        self.limits = self.model.jnt_range[self.joints].copy()
        self.lower = self.limits[:, 0]
        self.upper = self.limits[:, 1]
        self.step_size = float(step_size)
        self.edge_resolution = float(edge_resolution)
        self.goal_sample_rate = float(goal_sample_rate)
        self.max_iterations = int(max_iterations)
        self.shortcut_attempts = int(shortcut_attempts)
        self.timeout = None if timeout is None else float(timeout)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.last_iterations = 0
        self._n_nodes = 0

    # ----------------------------------------------------------------- recorded settings
    def settings(self):
        return dict(method=self.method, step_size=self.step_size, edge_resolution=self.edge_resolution,
                    goal_sample_rate=self.goal_sample_rate, max_iterations=self.max_iterations,
                    shortcut_attempts=self.shortcut_attempts, timeout=self.timeout, seed=self.seed)

    # ----------------------------------------------------------------- primitives
    def _sample(self, goal):
        if self.rng.random() < self.goal_sample_rate:
            return goal.copy()
        return self.rng.uniform(self.lower, self.upper)

    def _steer(self, source, target):
        delta = target - source
        distance = np.linalg.norm(delta)
        if distance <= self.step_size:
            return target.copy(), True
        return source + (self.step_size / distance) * delta, False

    def _extend(self, tree, target):
        """One collision-free step of ``tree`` toward ``target``: (status, index).

        status is 'reached', 'advanced', or 'trapped'.
        """
        nearest_index = tree.nearest(target)
        nearest = tree.nodes[nearest_index]
        new_q, reached = self._steer(nearest, target)
        if np.linalg.norm(new_q - nearest) < 1e-12:
            return ("reached" if reached else "trapped"), nearest_index
        if not self.checker.is_edge_valid(nearest, new_q, self.edge_resolution):
            return "trapped", None
        new_index = tree.add(new_q, nearest_index)
        self._n_nodes += 1
        return ("reached" if reached else "advanced"), new_index

    def _shortcut(self, path):
        """Random collision-checked shortcutting of non-adjacent waypoints."""
        path = [np.asarray(q).copy() for q in path]
        if len(path) <= 2:
            return np.asarray(path)
        for _ in range(self.shortcut_attempts):
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.choice(len(path), size=2, replace=False))
            if j <= i + 1:
                continue
            if self.checker.is_edge_valid(path[i], path[j], self.edge_resolution):
                path = path[:i + 1] + path[j:]
        return np.asarray(path)

    def _expired(self, t0):
        return self.timeout is not None and (time.perf_counter() - t0) > self.timeout

    # ----------------------------------------------------------------- public API
    def solve(self, start, goal, smooth=True):
        """Plan from ``start`` to ``goal``; always returns a ``PlanResult``."""
        t0 = time.perf_counter()
        self.rng = np.random.default_rng(self.seed)      # same seed -> same result
        self.last_iterations = 0
        self._n_nodes = 0

        def result(success, status, path=None, raw=np.nan):
            length = path_length(path) if path is not None else np.nan
            return PlanResult(success, status, path, time.perf_counter() - t0, self.last_iterations,
                              self.seed, length, raw, self._n_nodes, self.method, self.settings())

        start = vector(start, 6, "start")
        goal = vector(goal, 6, "goal")
        if not self.checker.is_configuration_valid(start):
            return result(False, INVALID_START)
        if not self.checker.is_configuration_valid(goal):
            return result(False, INVALID_GOAL)
        if self.checker.is_edge_valid(start, goal, self.edge_resolution):
            direct = np.vstack((start, goal))
            return result(True, OK_DIRECT, direct, path_length(direct))

        raw = self._search(start, goal, t0)
        if isinstance(raw, str):                          # failure status
            return result(False, raw)
        raw = np.asarray(raw)
        raw_len = path_length(raw)
        path = self._shortcut(raw) if smooth else raw
        # Defensive final validation: never return an invalid edge.
        if not self.checker.validate_path(path, self.edge_resolution):
            return result(False, "invalid_path")
        return result(True, OK, path, raw_len)

    def plan(self, start, goal, smooth=True):
        """Return the ``(M, 6)`` path or raise (ValueError: bad endpoint, RuntimeError: no path)."""
        res = self.solve(start, goal, smooth)
        if res.success:
            return res.path
        if res.status == INVALID_START:
            raise ValueError("start configuration is outside limits or in collision")
        if res.status == INVALID_GOAL:
            raise ValueError("goal configuration is outside limits or in collision")
        raise RuntimeError(f"{self.method} failed ({res.status}) after {res.iterations} iterations; "
                           "try a larger max_iterations/timeout or a different step_size/seed")

    def _search(self, start, goal, t0):                  # pragma: no cover - overridden
        raise NotImplementedError


class RRTPlanner(_PlannerBase):
    """Single-tree RRT rooted at the start with goal biasing (comparison baseline)."""

    method = "rrt"

    def _search(self, start, goal, t0):
        tree = _Tree.rooted_at(start)
        self._n_nodes = 1
        for iteration in range(1, self.max_iterations + 1):
            self.last_iterations = iteration
            if self._expired(t0):
                return TIMEOUT
            sample = self._sample(goal)
            status, index = self._extend(tree, sample)
            if status == "trapped":
                continue
            # try to close the gap to the goal from the newest node
            q_new = tree.nodes[index]
            if np.linalg.norm(goal - q_new) <= self.step_size and \
                    self.checker.is_edge_valid(q_new, goal, self.edge_resolution):
                path = tree.path_to_root(index)
                path.append(goal.copy())
                return path
        return MAX_ITER


class RRTConnectPlanner(_PlannerBase):
    """Bidirectional RRT-Connect (Kuffner & LaValle, 2000) in the 6-D joint space.

    Two trees (rooted at start and goal) take turns: one EXTENDS toward a random
    sample, the other CONNECTS greedily toward the new node until it reaches it
    (path found) or is blocked.  Trees swap roles every iteration.

    Parameters
    ----------
    checker : CollisionChecker for the same scene used for planning.
    step_size : max Euclidean joint-space extension per tree step [rad].
    edge_resolution : max change of any one joint between collision samples [rad].
    goal_sample_rate : probability of sampling the opposite tree's root.
    max_iterations, timeout : search budget (iterations, seconds).
    seed : random seed (recorded in the result; the same seed reproduces the same path).
    shortcut_attempts : random shortcut attempts after a path is found.
    """

    method = "rrt_connect"

    def _connect(self, tree, target):
        """Repeatedly extend ``tree`` toward ``target`` until trapped or reached."""
        while True:
            status, index = self._extend(tree, target)
            if status != "advanced":
                return status, index

    def _search(self, start, goal, t0):
        tree_start = _Tree.rooted_at(start)
        tree_goal = _Tree.rooted_at(goal)
        self._n_nodes = 2
        tree_a, tree_b = tree_start, tree_goal
        a_is_start_tree = True
        for iteration in range(1, self.max_iterations + 1):
            self.last_iterations = iteration
            if self._expired(t0):
                return TIMEOUT
            sample = self._sample(tree_b.nodes[0])
            status_a, index_a = self._extend(tree_a, sample)
            if status_a != "trapped":
                status_b, index_b = self._connect(tree_b, tree_a.nodes[index_a])
                if status_b == "reached":
                    path_a = tree_a.path_to_root(index_a)        # root(a) -> connection
                    path_b = tree_b.path_to_root(index_b)        # root(b) -> connection
                    if a_is_start_tree:
                        return path_a + path_b[-2::-1]
                    return path_b + path_a[-2::-1]
            tree_a, tree_b = tree_b, tree_a
            a_is_start_tree = not a_is_start_tree
        return MAX_ITER


def direct_joint_path(checker, start, goal, edge_resolution=0.02):
    """Straight-line joint interpolation (baseline): 2-waypoint path, or None if blocked."""
    start = vector(start, 6, "start")
    goal = vector(goal, 6, "goal")
    if checker.is_configuration_valid(start) and checker.is_configuration_valid(goal) \
            and checker.is_edge_valid(start, goal, edge_resolution):
        return np.vstack((start, goal))
    return None


def make_planner(method, checker, **kwargs):
    """Factory: 'rrt_connect' or 'rrt'."""
    classes = {"rrt_connect": RRTConnectPlanner, "rrt": RRTPlanner}
    if method not in classes:
        raise ValueError(f"unknown planner '{method}' (choose from {sorted(classes)})")
    return classes[method](checker, **kwargs)
