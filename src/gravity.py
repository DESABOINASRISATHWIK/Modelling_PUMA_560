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


class DHGravityModel:
    """Gravity-compensation torque computed ONLY from our own DH forward kinematics.

    No MuJoCo call is made.  For link l (frame {l}, mass m_l, COM c_l in frame {l}):

        p_cl      = T_0l [c_l; 1]                     (COM in the base frame)
        Jv_cl[:,i] = z_{i-1} x (p_cl - p_{i-1}),  i <= l,  else 0
        g_hat(q)  = - sum_l  Jv_cl^T (m_l g_w)

    ``g_hat`` is the torque that CANCELS gravity, i.e. tau = g_hat(q) holds the arm
    still.  An optional rigid ``payload`` (mass, COM offset from the flange in the
    flange axes) is treated as one more body fixed to frame {6}.  Only mass and COM
    matter for gravity; the payload inertia is irrelevant here.

    Use ``DHGravityModel()`` (no payload) as the *nominal* controller model and
    ``DHGravityModel(payload=...)`` for a payload-aware model.
    """

    def __init__(self, payload=None, gravity=(0.0, 0.0, -9.81)):
        from .forward_kinematics import forward_kinematics  # local import: keeps module import light
        self._fk = forward_kinematics
        self.gravity = vector(gravity, 3, "gravity")
        self.masses = list(rp.LINK_MASS)
        self.coms = [np.asarray(c, dtype=float) for c in rp.LINK_COM]
        # The MJCF gives link 1 a 1 g placeholder mass at its frame origin; it has zero
        # lever arm about the vertical waist axis and a negligible effect elsewhere.
        self.masses[0] = rp.LINK1_PLACEHOLDER_MASS
        self.payload = payload
        if payload is not None:
            self.masses.append(float(payload.mass))
            self.coms.append(np.array([0.0, 0.0, rp.D_FLANGE]) + vector(payload.position, 3, "payload position"))
            self._owner = list(range(1, 7)) + [6]       # frame index each body is attached to
        else:
            self._owner = list(range(1, 7))

    def torque(self, q):
        q = vector(q, 6, "q")
        frames = self._fk(q).frames                      # T_00 ... T_06
        tau = np.zeros(6)
        for m, c, link in zip(self.masses, self.coms, self._owner):
            T = frames[link]
            p_c = T[:3, :3] @ c + T[:3, 3]
            force = m * self.gravity
            for i in range(link):                        # joints 1..link move this body
                z = frames[i][:3, 2]
                jv = np.cross(z, p_c - frames[i][:3, 3])
                tau[i] -= jv @ force
        return tau
