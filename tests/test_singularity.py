"""Singularity analysis tests."""
import numpy as np

from src.forward_kinematics import forward_kinematics
from src.inverse_kinematics import inverse_kinematics
from src.jacobian import geometric_jacobian
from src.singularity_analysis import (SIGMA_NEAR, analyze_configuration, singular_configuration,
                                      sweep_sigma_min, svd_metrics)
from src.verification import singularity_examples


def test_regular_near_and_singular_examples():
    ex = singularity_examples()
    assert ex["regular"].status == "regular" and ex["regular"].svd.rank == 6
    assert ex["near_singular(q5=0.02)"].status == "near-singular"
    for kind in ("shoulder", "elbow", "wrist"):
        r = ex[f"singular_{kind}"]
        assert r.status == "singular" and r.svd.rank == 5
        assert r.active == [kind]
        assert r.svd.sigma_min < 1e-12


def test_closed_form_determinant_vanishes_exactly_on_the_three_families():
    rng = np.random.default_rng(0)
    for kind in ("shoulder", "elbow", "wrist"):
        for _ in range(20):
            q = singular_configuration(kind, q_base=rng.uniform(-1, 1, 6))
            assert abs(np.linalg.det(geometric_jacobian(q))) < 1e-12


def test_sigma_min_decreases_towards_wrist_singularity():
    q = singular_configuration("wrist")
    eps = np.array([0.3, 0.1, 0.03, 0.01, 0.003, 0.001])
    s = sweep_sigma_min(q, 4, eps)
    assert np.all(np.diff(s) < 0)
    ratio = s / np.sin(eps)
    assert np.ptp(ratio) / ratio.mean() < 0.2             # sigma_min ~ c |sin q5|


def test_stretched_elbow_singularity_is_a_workspace_boundary():
    q = singular_configuration("elbow")
    res = inverse_kinematics(forward_kinematics(q).T, q_ref=q)
    assert abs(res.diagnostics["outer_margin"]) < 1e-9


def test_svd_metrics_values():
    m = svd_metrics(np.diag([3.0, 2, 1, 1, 1, 0.5]))
    assert np.isclose(m.condition_number, 6.0) and m.rank == 6 and np.isclose(m.manipulability, 3.0)
    m = svd_metrics(np.diag([1.0, 1, 1, 1, 1, 0.0]))
    assert m.rank == 5 and np.isinf(m.condition_number)
    assert SIGMA_NEAR > 0
