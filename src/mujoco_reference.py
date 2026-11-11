"""
Thin wrapper around the MuJoCo model, used ONLY as an independent reference.

MuJoCo's own forward kinematics (``mj_kinematics``) is called here purely to
*verify* our NumPy DH implementation and to draw the robot.  None of the
kinematics / IK / Jacobian algorithms of this project call MuJoCo.

Relationship between the MuJoCo model and the DH model (established in
tools/build_mjcf.py and checked numerically by tests/test_fk.py):

* world frame  == DH base frame {0}
* qpos[i]      == q_{i+1}   (same sign, same zero: joint axes at q=0 coincide with z_{i-1})
* ``ee_site``  == flange frame {E} = {6} * Trans(z, D_FLANGE)   (same orientation as {6})
* ``wrist_center_site`` == origin of frames {4}={5}={6}
"""

from __future__ import annotations

from functools import lru_cache

import mujoco
import numpy as np

from . import robot_parameters as rp


@lru_cache(maxsize=1)
def _load():
    model = mujoco.MjModel.from_xml_path(str(rp.MJCF_PATH))
    return model


def load_model() -> mujoco.MjModel:
    """Return the (cached) compiled MuJoCo model of the PUMA-560."""
    return _load()


def new_data(model: mujoco.MjModel | None = None) -> mujoco.MjData:
    return mujoco.MjData(model or load_model())


def joint_qpos_addresses(model: mujoco.MjModel) -> list[int]:
    return [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in rp.MJCF_JOINT_NAMES]


def set_joint_angles(model: mujoco.MjModel, data: mujoco.MjData, q) -> None:
    """Write q into qpos and run MuJoCo's kinematics (no dynamics is stepped)."""
    adr = joint_qpos_addresses(model)
    for k, a in enumerate(adr):
        data.qpos[a] = q[k]
    mujoco.mj_kinematics(model, data)


def site_pose(model: mujoco.MjModel, data: mujoco.MjData, name: str = rp.MJCF_EE_SITE) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise KeyError(f"site '{name}' not found in model")
    T = np.eye(4)
    T[:3, :3] = data.site_xmat[sid].reshape(3, 3)
    T[:3, 3] = data.site_xpos[sid]
    return T


def mujoco_fk(q, site: str = rp.MJCF_EE_SITE) -> np.ndarray:
    """End-effector pose according to MuJoCo (reference only)."""
    model = load_model()
    data = new_data(model)
    set_joint_angles(model, data, q)
    return site_pose(model, data, site)


def model_joint_ranges(model: mujoco.MjModel | None = None) -> np.ndarray:
    model = model or load_model()
    out = np.zeros((rp.N_JOINTS, 2))
    for k, n in enumerate(rp.MJCF_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        out[k] = model.jnt_range[jid]
    return out
