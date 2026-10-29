"""Joint-space path planning and IK branch selection for the PUMA 560."""

from dataclasses import dataclass

import numpy as np

from inverse_kinematics import DEFAULT_LIMITS, IKResult


@dataclass(frozen=True)
class JointPath:
    """Geometric joint path before timing is assigned."""

    position: np.ndarray
    distance: float


def _limits(joint_limits: np.ndarray | None) -> np.ndarray:
    if joint_limits is None:
        return DEFAULT_LIMITS.copy()
    limits = np.asarray(joint_limits, dtype=float)
    if limits.shape != (6, 2) or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("joint_limits must have shape (6, 2)")
    return limits


def _joint_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (6,):
        raise ValueError(f"{name} must contain six joint angles")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def wrapped_joint_delta(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Return the shortest signed angular displacement for each joint."""
    start = _joint_vector(start, "start")
    goal = _joint_vector(goal, "goal")
    return np.arctan2(np.sin(goal - start), np.cos(goal - start))


def select_ik_solution(
    solutions: list[IKResult],
    current: np.ndarray,
    *,
    singularity_threshold: float = 1e-4,
) -> IKResult:
    """Select the safest continuous IK branch from valid solutions.

    Solutions near a Jacobian singularity are rejected. Among the remaining
    branches, the score prioritizes small joint motion and then singularity
    margin. This is a local branch-selection rule, not collision checking.
    """
    current = _joint_vector(current, "current")
    if singularity_threshold <= 0:
        raise ValueError("singularity_threshold must be positive")
    candidates = [
        solution for solution in solutions
        if solution.converged
        and solution.smallest_singular_value >= singularity_threshold
    ]
    if not candidates:
        raise ValueError(
            "no valid IK solution remains after singularity filtering"
        )
    return min(
        candidates,
        key=lambda solution: (
            np.linalg.norm(wrapped_joint_delta(current, solution.q)),
            -solution.smallest_singular_value,
        ),
    )


def plan_joint_path(
    start: np.ndarray,
    goal: np.ndarray,
    *,
    num_samples: int = 101,
    joint_limits: np.ndarray | None = None,
) -> JointPath:
    """Create a shortest-angle straight-line path in joint space.

    This is a geometric path only; it has no time, velocity, or acceleration.
    Collision checking is intentionally left explicit for a future MuJoCo
    environment/obstacle layer.
    """
    limits = _limits(joint_limits)
    start = _joint_vector(start, "start")
    goal = _joint_vector(goal, "goal")
    if np.any(start < limits[:, 0]) or np.any(start > limits[:, 1]):
        raise ValueError("start exceeds the supplied joint limits")
    if np.any(goal < limits[:, 0]) or np.any(goal > limits[:, 1]):
        raise ValueError("goal exceeds the supplied joint limits")
    if num_samples < 2:
        raise ValueError("num_samples must be at least two")

    delta = wrapped_joint_delta(start, goal)
    position = start + np.linspace(0.0, 1.0, num_samples)[:, None] * delta
    if np.any(position < limits[:, 0]) or np.any(position > limits[:, 1]):
        raise ValueError("shortest-angle path exceeds the joint limits")
    return JointPath(
        position=position,
        distance=float(np.linalg.norm(delta)),
    )
