"""Numerical inverse kinematics for the PUMA 560 model."""

import argparse
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import least_squares

from forward_kinematics import forward_kinematics


DEFAULT_LIMITS = np.array(
    [[-np.pi, np.pi]] * 6,
    dtype=float,
)


@dataclass(frozen=True)
class IKResult:
    """One solution returned by :func:`inverse_kinematics`."""

    q: np.ndarray
    position_error: float
    orientation_error: float
    iterations: int
    converged: bool
    smallest_singular_value: float
    condition_number: float


def _validate_transform(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    if target.shape != (4, 4):
        raise ValueError("target must be a 4x4 homogeneous transformation")
    if not np.all(np.isfinite(target)):
        raise ValueError("target must contain only finite values")
    if not np.allclose(target[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("target must have a homogeneous last row")
    rotation = target[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("target rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("target rotation must have determinant 1")
    return target


def _rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return the shortest world-frame rotation vector target-current."""
    rotation = target @ current.T
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cosine)
    if angle < 1e-9:
        return 0.5 * np.array(
            [rotation[2, 1] - rotation[1, 2],
             rotation[0, 2] - rotation[2, 0],
             rotation[1, 0] - rotation[0, 1]]
        )
    if np.pi - angle < 1e-6:
        diagonal = np.diag(rotation)
        axis = np.sqrt(np.maximum((diagonal + 1.0) / 2.0, 0.0))
        index = int(np.argmax(axis))
        if axis[index] < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            for other in range(3):
                if other != index:
                    axis[other] = rotation[index, other] / (2.0 * axis[index])
            axis /= np.linalg.norm(axis)
        return angle * axis
    axis = np.array(
        [rotation[2, 1] - rotation[1, 2],
         rotation[0, 2] - rotation[2, 0],
         rotation[1, 0] - rotation[0, 1]]
    ) / (2.0 * np.sin(angle))
    return angle * axis


def _pose_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [target[:3, 3] - current[:3, 3],
         _rotation_error(target[:3, :3], current[:3, :3])]
    )


def _solve_inverse_kinematics(
    target: np.ndarray,
    initial_guess: np.ndarray | None = None,
    *,
    joint_limits: np.ndarray | None = None,
    position_weight: float = 1.0,
    orientation_weight: float = 0.25,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> IKResult:
    """Solve for six joint angles that reach a target pose.

    The solver uses bounded nonlinear least squares with a finite-difference
    Jacobian. ``target`` is a 4x4 pose in the same world frame as
    :func:`forward_kinematics`. Position is measured in metres and angles in
    radians.
    """
    target = _validate_transform(target)
    if initial_guess is None:
        q = np.zeros(6, dtype=float)
    else:
        q = np.asarray(initial_guess, dtype=float).copy()
        if q.shape != (6,):
            raise ValueError("initial_guess must contain six joint angles")
    if not np.all(np.isfinite(q)):
        raise ValueError("initial_guess must contain only finite values")
    if joint_limits is None:
        limits = DEFAULT_LIMITS.copy()
    else:
        limits = np.asarray(joint_limits, dtype=float)
        if limits.shape != (6, 2) or np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("joint_limits must have shape (6, 2)")
    if position_weight <= 0 or orientation_weight <= 0:
        raise ValueError("error weights must be positive")
    if max_iterations <= 0 or tolerance <= 0:
        raise ValueError("solver settings must be positive")

    q = np.clip(q, limits[:, 0], limits[:, 1])
    weights = np.array(
        [position_weight] * 3 + [orientation_weight] * 3,
        dtype=float,
    )
    def residual(angles: np.ndarray) -> np.ndarray:
        return weights * _pose_error(target, forward_kinematics(angles))

    result = least_squares(
        residual,
        q,
        bounds=(limits[:, 0], limits[:, 1]),
        max_nfev=max_iterations,
        xtol=tolerance,
        ftol=tolerance,
        gtol=tolerance,
        diff_step=1e-6,
        x_scale="jac",
    )
    q = result.x
    final_error = _pose_error(target, forward_kinematics(q))
    jacobian = np.empty((6, 6), dtype=float)
    for column in range(6):
        perturbed = q.copy()
        perturbed[column] += 1e-6
        jacobian[:, column] = (
            _pose_error(target, forward_kinematics(perturbed))
            - final_error
        ) / 1e-6
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    smallest_singular_value = float(singular_values[-1])
    condition_number = float(
        singular_values[0] / smallest_singular_value
        if smallest_singular_value > 0
        else np.inf
    )
    return IKResult(
        q=q,
        position_error=float(np.linalg.norm(final_error[:3])),
        orientation_error=float(np.linalg.norm(final_error[3:])),
        iterations=result.nfev,
        converged=bool(result.success),
        smallest_singular_value=smallest_singular_value,
        condition_number=condition_number,
    )


def inverse_kinematics(
    target: np.ndarray,
    initial_guess: np.ndarray | None = None,
    *,
    joint_limits: np.ndarray | None = None,
    position_weight: float = 1.0,
    orientation_weight: float = 0.25,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> list[IKResult]:
    """Return all distinct solutions found for a target pose.

    The bounded numerical solver is started from a deterministic 3-by-3-by-3
    grid for every joint, plus ``initial_guess`` when supplied. This finds
    the usual PUMA 560 branches without relying on a fragile closed-form
    derivation. Numerical multi-start cannot prove that no additional branch
    exists, especially at a singularity.
    """
    target = _validate_transform(target)
    if joint_limits is None:
        limits = DEFAULT_LIMITS.copy()
    else:
        limits = np.asarray(joint_limits, dtype=float)
        if limits.shape != (6, 2) or np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("joint_limits must have shape (6, 2)")

    seed_values = np.column_stack(
        [limits[:, 0], np.mean(limits, axis=1), limits[:, 1]]
    )
    guesses = [np.asarray(seed, dtype=float) for seed in product(*seed_values)]
    if initial_guess is not None:
        guess = np.asarray(initial_guess, dtype=float)
        if guess.shape != (6,):
            raise ValueError("initial_guess must contain six joint angles")
        if not np.all(np.isfinite(guess)):
            raise ValueError("initial_guess must contain only finite values")
        guesses.insert(0, np.clip(guess, limits[:, 0], limits[:, 1]))

    solutions: list[IKResult] = []
    for guess in guesses:
        result = _solve_inverse_kinematics(
            target,
            guess,
            joint_limits=limits,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        if not result.converged:
            continue
        if result.position_error > tolerance or result.orientation_error > tolerance:
            continue
        if not any(
            np.linalg.norm(np.arctan2(
                np.sin(result.q - existing.q),
                np.cos(result.q - existing.q),
            )) < 1e-5
            for existing in solutions
        ):
            solutions.append(result)
    return solutions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve PUMA 560 inverse kinematics."
    )
    parser.add_argument(
        "joint_angles",
        nargs=6,
        type=float,
        metavar="q",
        help="known joint angles used to create a demo target",
    )
    args = parser.parse_args()
    target = forward_kinematics(np.asarray(args.joint_angles))
    results = inverse_kinematics(target)
    print("Target pose:\n", target)
    print("Solutions found:", len(results))
    for index, result in enumerate(results, start=1):
        print(f"\nSolution {index}:")
        print("Joint angles:", result.q)
        print("Position error:", result.position_error)
        print("Orientation error:", result.orientation_error)
        print("Near-singularity sigma:", result.smallest_singular_value)
