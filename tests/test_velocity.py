"""Velocity kinematics: ellipsoids, directional gain, singular directions, joint-speed blow-up."""
import numpy as np
import pytest

from src.forward_kinematics import forward_kinematics, random_valid_configuration
from src.jacobian import end_effector_twist, geometric_jacobian, joint_velocity_from_twist
from src.singularity_analysis import singular_configuration
from src.velocity_kinematics import (directional_gain, required_joint_speed, singular_directions,
                                     tracking_error, velocity_ellipsoid, velocity_report)


def test_ellipsoid_semi_axes_are_sqrt_eigenvalues_of_JJt():
    rng = np.random.default_rng(0)
    for _ in range(30):
        q = random_valid_configuration(rng)
        J = geometric_jacobian(q)
        for part, block in (("linear", J[:3]), ("angular", J[3:])):
            e = velocity_ellipsoid(q, part)
            ev = np.sort(np.linalg.eigvalsh(block @ block.T))[::-1]
            assert np.allclose(e.semi_axes ** 2, ev, atol=1e-12)
            assert np.all(np.diff(e.semi_axes) <= 1e-15)
    assert np.isclose(velocity_ellipsoid(q, "linear").volume_measure, np.prod(np.linalg.svd(J[:3], compute_uv=False)))


def test_ellipsoid_maps_unit_joint_velocities_inside_its_surface():
    rng = np.random.default_rng(1)
    q = random_valid_configuration(rng)
    e = velocity_ellipsoid(q, "linear")
    J = geometric_jacobian(q)
    for _ in range(200):
        qd = rng.normal(size=6)
        qd /= np.linalg.norm(qd)
        v = J[:3] @ qd
        coords = e.axes.T @ v / e.semi_axes
        assert np.linalg.norm(coords) <= 1 + 1e-9          # inside (or on) the ellipsoid


def test_directional_gain_equals_semi_axes_along_principal_directions():
    rng = np.random.default_rng(2)
    q = random_valid_configuration(rng)
    for part in ("linear", "angular"):
        e = velocity_ellipsoid(q, part)
        for k in range(3):
            assert np.isclose(directional_gain(q, e.axes[:, k], part), e.semi_axes[k], rtol=1e-10)
        for _ in range(100):                                # no direction exceeds the largest semi-axis
            u = rng.normal(size=3)
            g = directional_gain(q, u, part)
            assert e.semi_axes[2] - 1e-12 <= g <= e.semi_axes[0] + 1e-12


def test_wrist_singularity_self_motion_and_lost_rotation():
    q = singular_configuration("wrist")
    d = singular_directions(q)
    assert d.residual < 1e-12
    # self-motion: q4 and q6 rotate in opposite senses with equal rate, everything else still
    assert np.allclose(d.null_motion[[0, 1, 2, 4]], 0, atol=1e-9)
    assert np.isclose(d.null_motion[3], -d.null_motion[5], atol=1e-9)
    # lost angular direction is perpendicular to the (now collinear) axes of joints 4 and 6
    z3 = forward_kinematics(q).frames[3][:3, 2]
    assert abs(d.lost_twist[3:] @ z3) < 1e-9


def test_elbow_singularity_lost_motion_is_along_the_arm():
    q = singular_configuration("elbow")
    d = singular_directions(q)
    fk = forward_kinematics(q)
    arm = fk.wrist_center - fk.frames[1][:3, 3]
    arm /= np.linalg.norm(arm)
    assert d.residual < 1e-12
    assert abs(d.lost_twist[:3] @ arm) / np.linalg.norm(d.lost_twist[:3]) > 0.999


def test_shoulder_singularity_lost_motion_is_horizontal_normal_to_arm_plane():
    q = singular_configuration("shoulder")
    d = singular_directions(q)
    n = np.array([-np.sin(q[0]), np.cos(q[0]), 0.0])         # Rz(q1) y
    assert d.residual < 1e-12
    assert abs(d.lost_twist[:3] @ n) / np.linalg.norm(d.lost_twist[:3]) > 0.999


def test_joint_speed_blow_up_for_pinv_and_bound_for_dls():
    lam = 0.05
    for eps in (1e-1, 1e-2, 1e-3, 1e-4):
        q = singular_configuration("wrist")
        q[4] = eps
        d = singular_directions(q)
        pinv = required_joint_speed(q, d.lost_twist, "pinv")
        dls = required_joint_speed(q, d.lost_twist, "dls", damping=lam)
        assert np.isclose(pinv, 1.0 / d.sigma_min, rtol=1e-6)      # |qdot| = |twist| / sigma_min
        assert dls <= 1.0 / (2 * lam) + 1e-9                       # max_s s/(s^2+lam^2) = 1/(2 lam)
        # DLS pays with a task error instead
        err = tracking_error(q, d.lost_twist, joint_velocity_from_twist(q, d.lost_twist, method="dls", damping=lam))
        assert 0 < err <= 1.0 + 1e-9
    assert pinv > 1e3 and dls < 10


def test_velocity_report_is_consistent_with_twist_and_validation():
    rng = np.random.default_rng(3)
    q, qd = random_valid_configuration(rng), rng.normal(size=6)
    r = velocity_report(q, qd)
    tw = end_effector_twist(q, qd)
    assert np.allclose(r["linear_velocity"], tw[:3]) and np.allclose(r["angular_velocity"], tw[3:])
    with pytest.raises(ValueError):
        velocity_ellipsoid(q, "bogus")
