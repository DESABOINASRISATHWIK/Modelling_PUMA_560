"""Optional MuJoCo visualization using the project's PUMA-560 MJCF asset."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import robot_parameters as rp

MJCF_PATH = Path(__file__).resolve().parents[1] / "models" / "puma560.xml"


def create_model():
    """Load the project's PUMA-560 MJCF model without opening a viewer."""
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required; install it with `pip install mujoco`.") from exc
    return mujoco.MjModel.from_xml_path(str(MJCF_PATH))


def set_joint_angles(model, data, q) -> None:
    """Set six joint angles in radians and forward the model state."""
    q = np.asarray(q, dtype=float)
    if q.shape != (rp.N_JOINTS,) or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite vector with six joint angles")
    if np.any(q < rp.JOINT_LIMITS[:, 0]) or np.any(q > rp.JOINT_LIMITS[:, 1]):
        raise ValueError("q contains an angle outside the PUMA-560 joint limits")
    data.qpos[:] = q
    import mujoco
    mujoco.mj_forward(model, data)


def show(q=rp.Q_ZERO) -> None:
    """Open an interactive MuJoCo viewer at configuration ``q``."""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required; install it with `pip install mujoco`.") from exc

    model = create_model()
    data = mujoco.MjData(model)
    set_joint_angles(model, data, q)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        viewer.opt.sitegroup[:] = 0
        viewer.cam.lookat[:] = [0.2, -0.1, 0.8]
        viewer.cam.distance = 2.5
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
