"""Forward-kinematics tests (DH model vs closed forms vs MuJoCo)."""
from dataclasses import replace

import numpy as np
import pytest

from src import robot_parameters as rp
from src.dh_model import PUMA560, dh_transform
from src.forward_kinematics import (forward_kinematics, joint_limit_violation, random_valid_configuration,
                                    within_limits)
from src.utils import is_rotation, rot_x, rot_z, make_transform
from src.verification import fk_vs_mujoco


def test_dh_transform_equals_product_of_elementary_transforms():
    rng = np.random.default_rng(0)
    for _ in range(50):
        th, d, a, al = rng.uniform(-3, 3, 4)
        ref = (make_transform(rot_z(th), [0, 0, 0]) @ make_transform(np.eye(3), [0, 0, d])
               @ make_transform(np.eye(3), [a, 0, 0]) @ make_transform(rot_x(al), [0, 0, 0]))
        assert np.allclose(dh_transform(th, d, a, al), ref, atol=1e-14)


def test_zero_pose_geometry():
    """q = 0 is the 'L' pose: wrist centre at (a2+a3, -d3, d1+d4), tool frame = base orientation."""
    fk = forward_kinematics(np.zeros(6))
    assert np.allclose(fk.wrist_center, [rp.A2 + rp.A3, -rp.D3, rp.D1 + rp.D4], atol=1e-12)
    assert np.allclose(fk.position, fk.wrist_center + [0, 0, rp.D_FLANGE], atol=1e-12)
    assert np.allclose(fk.rotation, np.eye(3), atol=1e-12)


def test_fk_returns_all_intermediate_frames_and_proper_rotations():
    rng = np.random.default_rng(1)
    fk = forward_kinematics(random_valid_configuration(rng))
    assert len(fk.frames) == 7 and len(fk.link_transforms) == 6
    assert all(is_rotation(T[:3, :3]) for T in fk.frames)
    # wrist: origins of {4},{5},{6} coincide (spherical wrist)
    assert np.allclose(fk.frames[4][:3, 3], fk.frames[5][:3, 3]) and np.allclose(fk.frames[5][:3, 3], fk.frames[6][:3, 3])
    # the chain really is the product of the relative transforms
    T = np.eye(4)
    for A in fk.link_transforms:
        T = T @ A
    assert np.allclose(T @ rp.T_TOOL, fk.T)


def test_joint_offsets_are_applied():
    off = np.array([0.1, -0.2, 0.3, 0.0, 0.4, -0.5])
    shifted = replace(PUMA560, theta_offset=off)
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = rng.uniform(-2, 2, 6)
        assert np.allclose(forward_kinematics(q, shifted).T, forward_kinematics(q + off).T, atol=1e-13)


def test_input_validation():
    with pytest.raises(ValueError):
        forward_kinematics(np.zeros(5))
    with pytest.raises(ValueError):
        forward_kinematics([0, 0, 0, 0, 0, np.nan])


def test_limit_helpers():
    assert within_limits(np.zeros(6)) is True
    q = np.zeros(6)
    q[4] = np.radians(120)
    assert not within_limits(q)
    assert np.isclose(joint_limit_violation(q)[4], np.radians(20))


def test_fk_matches_mujoco_site_pose():
    st = fk_vs_mujoco(n=1000, seed=0)
    assert st["position_error_m"]["max"] < 1e-12
    assert st["orientation_error_rad"]["max"] < 1e-12
    assert st["wrist_center_error_m"]["max"] < 1e-12


def test_mujoco_comparison_is_sensitive_to_a_wrong_dh_table():
    """Negative control: a 1 mm error in a3 or a sign error in alpha4 must be detected."""
    bad_a = PUMA560.a.copy()
    bad_a[2] += 1e-3
    st = fk_vs_mujoco(n=50, seed=0, model=replace(PUMA560, a=bad_a))
    assert st["position_error_m"]["max"] > 5e-4
    bad_alpha = PUMA560.alpha.copy()
    bad_alpha[3] *= -1
    st = fk_vs_mujoco(n=50, seed=0, model=replace(PUMA560, alpha=bad_alpha))
    assert st["orientation_error_rad"]["max"] > 0.1
