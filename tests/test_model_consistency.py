"""The MuJoCo XML and robot_parameters.py describe the same robot (and how the frames relate)."""
import mujoco
import numpy as np

from src import robot_parameters as rp
from src.dh_model import PUMA560
from src.mujoco_reference import load_model, model_joint_ranges, new_data, set_joint_angles


def test_joint_limits_match():
    assert np.allclose(model_joint_ranges(), rp.JOINT_LIMITS, atol=1e-8)


def test_joint_axes_and_anchors_at_zero_equal_dh_frames():
    """XML joint i axis/anchor (absolute) == z_{i-1} and origin of DH frame {i-1} at q = 0.
    This is the documented relationship: world = {0}, qpos_i = q_i, same sign, same zero."""
    m = load_model()
    d = new_data(m)
    set_joint_angles(m, d, np.zeros(6))
    F = PUMA560.frames(np.zeros(6))
    for i, name in enumerate(rp.MJCF_JOINT_NAMES):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        axis, anchor = d.xaxis[jid], d.xanchor[jid]
        assert np.allclose(axis, F[i][:3, 2], atol=1e-12), name
        # anchor must lie on the DH axis (point may differ along the axis)
        diff = anchor - F[i][:3, 3]
        assert np.linalg.norm(np.cross(diff, F[i][:3, 2])) < 1e-12, name


def test_masses_and_centres_of_mass_match_parameters():
    m = load_model()
    d = new_data(m)
    set_joint_angles(m, d, np.zeros(6))
    F = PUMA560.frames(np.zeros(6))
    for i in range(1, 7):
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"link{i}")
        mass_ref = rp.LINK_MASS[i - 1] if i > 1 else rp.LINK1_PLACEHOLDER_MASS
        assert np.isclose(m.body_mass[body], mass_ref)
        com_ref = (F[i] @ np.r_[rp.LINK_COM[i - 1], 1.0])[:3]
        assert np.allclose(d.xipos[body], com_ref, atol=1e-9), i


def test_inertia_matches_except_documented_corrections():
    """Principal moments equal the published ones, except link 1 (placeholder) and link 3
    (smallest moment raised to satisfy the triangle inequality)."""
    m = load_model()
    for i in range(1, 7):
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"link{i}")
        got = np.sort(m.body_inertia[body])
        want = np.sort(np.diag(rp.LINK_INERTIA[i - 1]))
        if i == 1:
            assert np.isclose(got.max(), 0.35)
        elif i == 3:
            assert np.allclose(got[1:], want[1:]) and np.isclose(got[0], (want[2] - want[1]) * 1.005)
        else:
            assert np.allclose(got, want, rtol=1e-6)


def test_ee_site_is_flange_frame():
    m = load_model()
    d = new_data(m)
    set_joint_angles(m, d, np.zeros(6))
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, rp.MJCF_EE_SITE)
    F = PUMA560.frames(np.zeros(6))
    assert np.allclose(d.site_xpos[sid], (F[6] @ rp.T_TOOL)[:3, 3], atol=1e-12)
