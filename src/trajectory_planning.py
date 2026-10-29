"""Smooth joint-space trajectory planning for the PUMA 560."""

import argparse
from dataclasses import dataclass

import numpy as np

from inverse_kinematics import DEFAULT_LIMITS
from path_planning import JointPath


@dataclass(frozen=True)
class JointTrajectory:
    """Sampled joint trajectory in radians, seconds, and SI units."""

    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


def _validate_waypoints(
    waypoints: np.ndarray,
    joint_limits: np.ndarray,
) -> np.ndarray:
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 6:
        raise ValueError("waypoints must have shape (number_of_points, 6)")
    if waypoints.shape[0] < 2:
        raise ValueError("at least two waypoints are required")
    if not np.all(np.isfinite(waypoints)):
        raise ValueError("waypoints must contain only finite values")
    if np.any(waypoints < joint_limits[:, 0]) or np.any(
        waypoints > joint_limits[:, 1]
    ):
        raise ValueError("waypoints exceed the supplied joint limits")
    return waypoints


def _validate_durations(
    durations: np.ndarray,
    segment_count: int,
) -> np.ndarray:
    durations = np.asarray(durations, dtype=float)
    if durations.shape != (segment_count,):
        raise ValueError(
            f"durations must contain {segment_count} segment values"
        )
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError("durations must be finite and greater than zero")
    return durations


def _validate_limits(joint_limits: np.ndarray | None) -> np.ndarray:
    if joint_limits is None:
        return DEFAULT_LIMITS.copy()
    limits = np.asarray(joint_limits, dtype=float)
    if limits.shape != (6, 2) or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("joint_limits must have shape (6, 2)")
    return limits


def _sample_times(total_time: float, num_samples: int) -> np.ndarray:
    if total_time <= 0 or not np.isfinite(total_time):
        raise ValueError("duration must be finite and greater than zero")
    if num_samples < 2:
        raise ValueError("num_samples must be at least two")
    return np.linspace(0.0, total_time, num_samples)


def _single_segment(
    start: np.ndarray,
    end: np.ndarray,
    time: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = time[-1]
    tau = time / duration
    delta = end - start

    if method == "cubic":
        blend = 3.0 * tau**2 - 2.0 * tau**3
        blend_velocity = (6.0 * tau - 6.0 * tau**2) / duration
        blend_acceleration = (6.0 - 12.0 * tau) / duration**2
    elif method == "quintic":
        blend = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        blend_velocity = (
            30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
        ) / duration
        blend_acceleration = (
            60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3
        ) / duration**2
    else:
        raise ValueError("method must be 'cubic' or 'quintic'")

    position = start + blend[:, None] * delta
    velocity = blend_velocity[:, None] * delta
    acceleration = blend_acceleration[:, None] * delta
    return position, velocity, acceleration


def plan_joint_trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    duration: float,
    *,
    num_samples: int = 101,
    method: str = "quintic",
    joint_limits: np.ndarray | None = None,
) -> JointTrajectory:
    """Plan a smooth point-to-point joint trajectory.

    Both endpoints have zero velocity and acceleration. ``quintic`` is the
    default because it also gives zero acceleration at both endpoints.
    """
    waypoints = np.vstack([start, goal])
    trajectory = plan_waypoint_trajectory(
        waypoints,
        [duration],
        num_samples=num_samples,
        method=method,
        joint_limits=joint_limits,
    )
    return trajectory


def plan_waypoint_trajectory(
    waypoints: np.ndarray,
    durations: np.ndarray | list[float],
    *,
    num_samples: int = 101,
    method: str = "quintic",
    joint_limits: np.ndarray | None = None,
) -> JointTrajectory:
    """Plan a smooth trajectory through joint-space waypoints.

    Each segment uses zero velocity and acceleration at its endpoints. This
    makes the path continuous and predictable, but it does not enforce a
    nonzero cruising velocity through intermediate waypoints.
    """
    limits = _validate_limits(joint_limits)
    waypoints = _validate_waypoints(waypoints, limits)
    durations = _validate_durations(durations, waypoints.shape[0] - 1)
    if num_samples < waypoints.shape[0]:
        raise ValueError("num_samples must cover all waypoint segments")

    segment_count = len(durations)
    samples_per_segment = [2] * segment_count
    extra_samples = num_samples - (segment_count + 1)
    if extra_samples > 0:
        proportions = durations / np.sum(durations)
        additions = np.floor(extra_samples * proportions).astype(int)
        samples_per_segment = [
            count + int(addition)
            for count, addition in zip(samples_per_segment, additions)
        ]
        remainder = extra_samples - int(np.sum(additions))
        samples_per_segment[-1] += remainder

    all_time: list[np.ndarray] = []
    all_position: list[np.ndarray] = []
    all_velocity: list[np.ndarray] = []
    all_acceleration: list[np.ndarray] = []
    time_offset = 0.0
    for index, duration in enumerate(durations):
        local_time = _sample_times(duration, samples_per_segment[index])
        position, velocity, acceleration = _single_segment(
            waypoints[index],
            waypoints[index + 1],
            local_time,
            method,
        )
        if index:
            local_time = local_time[1:]
            position = position[1:]
            velocity = velocity[1:]
            acceleration = acceleration[1:]
        all_time.append(local_time + time_offset)
        all_position.append(position)
        all_velocity.append(velocity)
        all_acceleration.append(acceleration)
        time_offset += duration

    return JointTrajectory(
        time=np.concatenate(all_time),
        position=np.vstack(all_position),
        velocity=np.vstack(all_velocity),
        acceleration=np.vstack(all_acceleration),
    )


def time_parameterize_path(
    path: JointPath,
    duration: float,
    *,
    method: str = "quintic",
    joint_limits: np.ndarray | None = None,
) -> JointTrajectory:
    """Assign time to a geometric path using the smooth waypoint planner."""
    return plan_waypoint_trajectory(
        path.position,
        np.full(path.position.shape[0] - 1, duration / (path.position.shape[0] - 1)),
        num_samples=path.position.shape[0],
        method=method,
        joint_limits=joint_limits,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a PUMA 560 joint-space trajectory."
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument(
        "--method",
        choices=("cubic", "quintic"),
        default="quintic",
    )
    args = parser.parse_args()

    start = np.zeros(6)
    goal = np.array([0.4, -0.5, 0.6, 0.3, -0.4, 0.2])
    trajectory = plan_joint_trajectory(
        start,
        goal,
        args.duration,
        num_samples=args.samples,
        method=args.method,
    )
    print("Samples:", len(trajectory.time))
    print("Duration:", trajectory.time[-1])
    print("Start position:", trajectory.position[0])
    print("End position:", trajectory.position[-1])
    print("Maximum joint speed:", np.max(np.abs(trajectory.velocity), axis=0))
