import mujoco
import numpy as np

from src.dh_model import PUMA560
from src.forward_kinematics import forward_kinematics
from src.mujoco_visualization import create_model


def test_mesh_model_matches_dh_frames_and_end_effector():
    model = create_model()
    data = mujoco.MjData(model)
    q = np.radians([30, 50, -70, 40, 60, -20])
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    fk = forward_kinematics(q)

    assert model.nq == 6 and model.njnt == 6 and model.nmesh == 7
    assert model.nu == 0  # Legacy visualization model; PhysicsSimulation adds torque motors.
    for i in range(6):
        assert np.allclose(data.xaxis[i], fk.frames[i][:3, 2], atol=1e-10)
    ee = model.site("ee_site").id
    assert np.allclose(data.site_xpos[ee], fk.position, atol=1e-10)
    assert np.allclose(data.site_xmat[ee].reshape(3, 3), fk.rotation, atol=1e-10)


def test_mesh_model_uses_the_same_joint_limits():
    model = create_model()
    assert np.allclose(model.jnt_range, PUMA560.limits, atol=1e-6)
