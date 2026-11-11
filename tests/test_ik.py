"""Analytical IK tests: round trip, branch structure, boundaries, unreachable, limits, singular."""
import re
from pathlib import Path

import numpy as np
import pytest

from src import robot_parameters as rp
from src.forward_kinematics import forward_kinematics, random_valid_configuration
from src.inverse_kinematics import inverse_kinematics, reachability_diagnostics
from src.utils import wrap_to_pi
from src.verification import ik_analytic_vs_numerical, ik_edge_cases, ik_roundtrip


@pytest.fixture(scope="module")
def roundtrip():
    return ik_roundtrip(n=1000, seed=1)


@pytest.fixture(scope="module")
def edge():
    return ik_edge_cases(seed=2)


def test_roundtrip_recovers_original_configuration(roundtrip):
    assert roundtrip["recovered_original"] == roundtrip["n"]


def test_generic_targets_have_eight_branches(roundtrip):
    assert roundtrip["branch_count_histogram"] == {8: roundtrip["n"]}


def test_every_branch_is_verified_by_fk(roundtrip):
    assert roundtrip["position_error_m"]["max"] < 1e-9
    assert roundtrip["orientation_error_rad"]["max"] < 1e-9
    assert roundtrip["limit_flag_consistent"]


def test_core_modules_do_not_use_scipy_or_mujoco_kinematics():
    """Kinematics stay independent; MuJoCo is allowed in reference/viewer/physics adapters."""
    src = Path(rp.PROJECT_ROOT, "src")
    for f in src.glob("*.py"):
        text = f.read_text()
        assert not re.search(r"^\s*(import|from)\s+scipy", text, re.M), f"{f.name} imports scipy"
        if f.name not in ("mujoco_reference.py", "simulation.py", "mujoco_visualization.py",
                          "scene.py", "physics.py", "gravity.py", "collision.py"):
            assert not re.search(r"^\s*(import|from)\s+mujoco", text, re.M), f"{f.name} imports mujoco"


def test_closest_branch_selection():
    rng = np.random.default_rng(10)
    q = random_valid_configuration(rng)
    res = inverse_kinematics(forward_kinematics(q).T)
    best = res.closest_to(q)
    assert best is not None and np.linalg.norm(wrap_to_pi(best.q - q)) < 1e-6
    # a far-away reference picks a *different* feasible branch
    far = res.closest_to(q + np.array([2.5, 0, 0, 0, 0, 0]))
    assert far is not None


def test_angle_representation_respects_limits_and_reference():
    """J6 allows +-266 deg: the 2*pi equivalent closest to the reference must be chosen."""
    q = np.array([0.2, 0.5, -0.8, 0.3, 0.6, np.radians(200.0)])
    res = inverse_kinematics(forward_kinematics(q).T, q_ref=q)
    sol = res.closest_to(q)
    assert abs(sol.q[5] - q[5]) < 1e-6
    res = inverse_kinematics(forward_kinematics(q).T, q_ref=q - np.array([0, 0, 0, 0, 0, 2 * np.pi]))
    sol = res.closest_to(q - np.array([0, 0, 0, 0, 0, 2 * np.pi]))
    assert abs(sol.q[5] - (q[5] - 2 * np.pi)) < 1e-6          # -160 deg is also inside [-266, 266]


def test_boundary_target_is_solved_and_flagged(edge):
    b = edge["boundary_stretched"]
    assert b["status"] == "ok" and b["n_solutions"] == 4
    assert b["max_err"] < 1e-9 and abs(b["outer_margin_m"]) < 1e-9


def test_just_outside_the_boundary_is_unreachable(edge):
    e = edge["beyond_boundary_1mm"]
    assert e["status"] == "unreachable" and e["reason"] == "too_far"


def test_far_and_axis_targets_are_unreachable_with_diagnostics(edge):
    assert edge["far_target"]["status"] == "unreachable" and edge["far_target"]["n_solutions"] == 0
    assert edge["on_waist_axis"]["reason"] == "wrist_inside_shoulder_cylinder"
    assert "mm" in edge["far_target"]["message"]


def test_joint_limit_infeasibility_is_distinguished_from_unreachability(edge):
    j = edge["joint_limit_infeasible"]
    assert j["n_status_joint_limits"] > 0 and j["all_branches_violate"]
    assert j["n_geometric_branches"] == 8


def test_exactly_singular_targets_do_not_crash_and_are_flagged(edge):
    for kind, n_min in (("shoulder", 4), ("elbow", 4), ("wrist", 7)):
        s = edge[f"singular_{kind}"]
        assert s["flagged_singular"] == s["n_cases"]
        assert s["max_err"] < 1e-9
        assert s["branches_min"] == n_min


def test_near_singular_targets_remain_accurate(edge):
    for key, v in edge["near_singular"].items():
        assert v["solved"] == v["of"], key
        assert v["max_err"] < 1e-7, key


def test_reachability_diagnostics_are_consistent():
    q = np.array([0.3, 1.0, -0.7, 0.2, 0.5, 0.1])
    d = reachability_diagnostics(forward_kinematics(q).T)
    assert d["rho_min"] <= d["rho"] <= d["rho_max"] and d["radial_margin"] > 0


@pytest.mark.parametrize("bad", [np.eye(3), np.full((4, 4), np.nan), np.diag([2.0, 1, 1, 1])])
def test_malformed_targets_raise_value_error(bad):
    with pytest.raises(ValueError):
        inverse_kinematics(bad)


def test_secondary_numerical_ik_agrees_with_analytic_branches():
    st = ik_analytic_vs_numerical(n=60, seed=4)
    assert st["converged"] >= 55 and st["matched_an_analytic_branch"] == st["converged"]
