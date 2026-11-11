"""Sampled configuration/edge checks using private MuJoCo data, never execution data."""
import mujoco
import numpy as np

from . import robot_parameters as rp
from .scene import vector


class CollisionChecker:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.joints = [model.joint(n).id for n in rp.MJCF_JOINT_NAMES]
        self.qadr = model.jnt_qposadr[self.joints]

    def contacts(self, q):
        """Return touching/penetrating geom-name pairs; ignores separated margin contacts."""
        q = vector(q, 6, "q")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        return [(self.model.geom(c.geom1).name, self.model.geom(c.geom2).name)
                for c in self.data.contact if c.dist <= 0]

    def is_configuration_valid(self, q):
        q = vector(q, 6, "q")
        limits = self.model.jnt_range[self.joints]
        return bool(np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1]) and not self.contacts(q))

    def is_edge_valid(self, start, end, resolution=0.02):
        """Linear joint-space samples, including both endpoints; max joint increment in rad."""
        start, end = vector(start, 6, "start"), vector(end, 6, "end")
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be positive and finite")
        count = max(1, int(np.ceil(np.max(np.abs(end-start)) / resolution)))
        return all(self.is_configuration_valid(start + t * (end-start))
                   for t in np.linspace(0, 1, count+1))
