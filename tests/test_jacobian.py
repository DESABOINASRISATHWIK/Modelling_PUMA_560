"""Jacobian tests: analytic vs finite differences, structure, frames, velocity relation."""
import numpy as np

from src.forward_kinematics import forward_kinematics, random_valid_configuration
from src.jacobian import (end_effector_twist, geometric_jacobian, jacobian_numerical,
                          joint_velocity_from_twist)
from src.singularity_analysis import singular_configuration
from src.verification import jacobian_vs_numerical


def test_analytic_jacobian_matches_finite_differences():
    st = jacobian_vs_numerical(n=500, seed=3)
    assert st["all"]["max_abs"] < 1e-7
    assert st["translational"]["max_abs"] < 1e-7 and st["rotational"]["max_abs"] < 1e-7
    assert st["determinant_vs_closed_form_max"] < 1e-12


def test_velocity_relation_matches_fk_motion():
    """x_dot = J qdot: move q a tiny step and compare with the FK displacement."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        q, qd = random_valid_configuration(rng), rng.normal(size=6)
        h = 1e-7
        f0, f1 = forward_kinematics(q), forward_kinematics(q + h * qd)
        v_num = (f1.position - f0.position) / h
        tw = end_effector_twist(q, qd)
        assert np.allclose(tw[:3], v_num, atol=1e-5)
        # angular: R1 R0^T = exp([w] h)  ->  w ~ axial vector of (R1 R0^T - I)/h
        S = (f1.rotation @ f0.rotation.T - np.eye(3)) / h
        w_num = np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]]) / 2
        assert np.allclose(tw[3:], w_num, atol=1e-5)


def test_wrist_point_jacobian_is_block_triangular():
    rng = np.random.default_rng(1)
    Jw = geometric_jacobian(random_valid_configuration(rng), point="wrist")
    assert np.allclose(Jw[:3, 3:], 0, atol=1e-12)


def test_changing_reference_point_is_a_unimodular_map():
    rng = np.random.default_rng(2)
    q = random_valid_configuration(rng)
    fk = forward_kinematics(q)
    r = fk.position - fk.wrist_center
    S = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    Jf, Jw = geometric_jacobian(q), geometric_jacobian(q, point="wrist")
    T = np.eye(6)
    T[:3, 3:] = -S
    assert np.allclose(T @ Jw, Jf, atol=1e-12)
    assert np.isclose(np.linalg.det(Jf), np.linalg.det(Jw))


def test_ee_frame_jacobian_is_rotated_base_jacobian():
    rng = np.random.default_rng(3)
    q = random_valid_configuration(rng)
    R = forward_kinematics(q).rotation
    Jb, Je = geometric_jacobian(q), geometric_jacobian(q, frame="ee")
    assert np.allclose(Je[:3], R.T @ Jb[:3]) and np.allclose(Je[3:], R.T @ Jb[3:])


def test_resolved_rate_inverse_and_damping_near_singularity():
    rng = np.random.default_rng(4)
    q = random_valid_configuration(rng)
    tw = np.array([0.1, -0.05, 0.02, 0.1, 0.0, -0.2])
    qd = joint_velocity_from_twist(q, tw, method="pinv")
    assert np.allclose(geometric_jacobian(q) @ qd, tw, atol=1e-9)
    qs = singular_configuration("wrist")
    qs[4] = 1e-4
    big = np.linalg.norm(joint_velocity_from_twist(qs, np.array([0, 0, 0, 0.1, 0.1, 0.1]), method="pinv", rcond=1e-12))
    damped = np.linalg.norm(joint_velocity_from_twist(qs, np.array([0, 0, 0, 0.1, 0.1, 0.1]), method="dls", damping=0.05))
    assert damped < big
