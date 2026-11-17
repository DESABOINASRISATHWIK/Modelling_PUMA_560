"""Part 3: quintic trajectory generation."""
import numpy as np
import pytest

from src.trajectory import QuinticTrajectory

W = np.array([[0, 0, 0, 0, 0, 0], [0.5, -0.4, 0.3, 0.2, -0.1, 0.6], [0.2, 0.1, 0.5, 0.0, 0.3, 0.0]])


def test_boundary_conditions_and_shapes():
    tr = QuinticTrajectory(W, max_velocity=np.full(6, 1.0), max_acceleration=np.full(6, 2.0))
    q, qd, qdd = tr.evaluate(0.0)
    assert q.shape == qd.shape == qdd.shape == (6,)
    np.testing.assert_allclose(q, W[0]); np.testing.assert_allclose(qd, 0); np.testing.assert_allclose(qdd, 0)
    for k, t in enumerate(np.cumsum(tr.segment_times)):          # stops at every waypoint
        q, qd, qdd = tr.evaluate(t)
        np.testing.assert_allclose(q, W[k + 1], atol=1e-12); np.testing.assert_allclose(qd, 0, atol=1e-9)


def test_derivatives_match_finite_differences():
    tr = QuinticTrajectory(W)
    h = 1e-6
    for t in (0.3, 0.9, tr.segment_times[0] + 0.4):
        qp, qdp, _ = tr.evaluate(t + h); qm, qdm, _ = tr.evaluate(t - h); _, qd, qdd = tr.evaluate(t)
        np.testing.assert_allclose((qp - qm) / (2 * h), qd, atol=1e-5)
        np.testing.assert_allclose((qdp - qdm) / (2 * h), qdd, atol=1e-4)


def test_limits_respected():
    vmax = np.array([0.8, 0.8, 0.8, 1.5, 1.5, 2.0]); amax = np.array([1.5, 1.5, 1.5, 5, 5, 8.0])
    tr = QuinticTrajectory(W, max_velocity=vmax, max_acceleration=amax)
    _, q, qd, qdd = tr.generate()
    assert np.all(np.abs(qd) <= vmax * (1 + 1e-6)) and np.all(np.abs(qdd) <= amax * (1 + 1e-6))
    assert np.all(np.isfinite(q))


def test_validation_and_holding():
    with pytest.raises(ValueError):
        QuinticTrajectory(W[:1])
    with pytest.raises(ValueError):
        QuinticTrajectory(W, segment_times=[1.0])
    tr = QuinticTrajectory(W)
    np.testing.assert_allclose(tr.evaluate(tr.duration + 5)[0], W[-1])
