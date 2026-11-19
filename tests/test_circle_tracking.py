import numpy as np

from src.circle_tracking import CircleConfig, CircleReference


def test_inclined_circle_is_closed_and_has_zero_endpoint_velocity():
    reference = CircleReference(CircleConfig(sample_dt=0.05))
    p0, v0 = reference.cartesian(0.0)
    p1, v1 = reference.cartesian(reference.config.duration)
    np.testing.assert_allclose(p0, p1, atol=1e-12)
    np.testing.assert_allclose(v0, 0.0, atol=1e-12)
    np.testing.assert_allclose(v1, 0.0, atol=1e-12)


def test_circle_reference_stays_within_joint_limits_and_is_continuous():
    reference = CircleReference(CircleConfig(sample_dt=0.05))
    from src import robot_parameters as rp
    assert np.all(reference.q >= rp.JOINT_LIMITS[:, 0])
    assert np.all(reference.q <= rp.JOINT_LIMITS[:, 1])
    assert np.max(np.abs(np.diff(reference.q, axis=0))) < 0.1
    # The presentation task deliberately uses the upper-elbow branch rather
    # than the lower configuration that crowds the orange arm joint.
    assert reference.q[0, 1] > 2.0 and reference.q[0, 2] > -1.5
