"""Student gravity calculation from COM positions and revolute joint axes."""
import mujoco
import numpy as np

from . import robot_parameters as rp
from .scene import vector


class GravityCompensator:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.joints = [model.joint(n).id for n in rp.MJCF_JOINT_NAMES]

    def torque(self, q):
        q = vector(q, 6, "q")
        m, d = self.model, self.data
        d.qpos[m.jnt_qposadr[self.joints]] = q
        mujoco.mj_kinematics(m, d)
        tau = np.zeros(6)
        # Only ancestors of a body's COM contribute. This includes fixed payload bodies.
        for body in range(1, m.nbody):
            ancestors = set()
            parent = body
            while parent:
                ancestors.add(parent)
                parent = int(m.body_parentid[parent])
            for i, joint in enumerate(self.joints):
                if int(m.jnt_bodyid[joint]) in ancestors:
                    jv = np.cross(d.xaxis[joint], d.xipos[body] - d.xanchor[joint])
                    tau[i] -= jv @ (m.body_mass[body] * m.opt.gravity)
        return tau
