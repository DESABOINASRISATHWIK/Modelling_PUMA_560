"""Torque-driven execution; viewer positioning remains in simulation.PumaSimulation."""
import mujoco
import numpy as np

from . import robot_parameters as rp
from .mujoco_reference import site_pose
from .scene import SceneConfig, load_scene, vector


class PhysicsSimulation:
    def __init__(self, config=None, q0=None):
        self.config = config or SceneConfig()
        self.model = load_scene(self.config)
        self.data = mujoco.MjData(self.model)
        self.joints = [self.model.joint(n).id for n in rp.MJCF_JOINT_NAMES]
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        self.reset(rp.Q_ZERO if q0 is None else q0)

    def reset(self, q=None, qvel=None):
        q = vector(rp.Q_ZERO if q is None else q, 6, "q")
        v = vector(np.zeros(6) if qvel is None else qvel, 6, "qvel")
        limits = self.model.jnt_range[self.joints]
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            raise ValueError("reset configuration exceeds joint limits")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = q
        self.data.qvel[self.vadr] = v
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        """Return independent copies of q [rad], qvel [rad/s]."""
        return self.data.qpos[self.qadr].copy(), self.data.qvel[self.vadr].copy()

    def apply_torque(self, torque):
        """Clip joint torque [N m], convert tau=gear*ctrl; return accepted torque."""
        tau = vector(torque, 6, "torque")
        tau = np.clip(tau, -np.asarray(self.config.torque_limits), self.config.torque_limits)
        self.data.ctrl[:] = tau / np.asarray(self.config.gear)
        return tau.copy()

    def step(self, steps=1):
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("steps must be a positive integer")
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        # mj_step leaves some derived poses at the previous integration state.
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def end_effector_pose(self):
        return site_pose(self.model, self.data)
