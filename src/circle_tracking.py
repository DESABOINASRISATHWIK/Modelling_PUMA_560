"""Inclined-circle task-space tracking experiment (NumPy FK/IK/Jacobian).

MuJoCo supplies only the torque-driven plant and viewer.  The tilted circle,
quintic timing law, analytical IK branch selection, and DLS velocity mapping
are implemented here with the project's own NumPy routines.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from . import robot_parameters as rp
from .control import make_controller
from .gravity import DHGravityModel
from .inverse_kinematics import inverse_kinematics
from .jacobian import joint_velocity_from_twist
from .scene import SceneConfig, scene_xml
from .utils import rpy_to_matrix


@dataclass(frozen=True)
class CircleConfig:
    # Chosen from a small analytic-IK reachability check. This lower, front
    # workspace region stays away from the joint-3 torque limit in the
    # nominal MuJoCo model while remaining visibly three-dimensional.
    # Lower and farther forward than the first presentation layout.  This
    # keeps the full enlarged ring within the comfortable elbow/wrist region.
    center: tuple = (0.65, -0.35, 0.52)
    radius: float = 0.10
    tilt_deg: float = 45.0
    duration: float = 8.0
    sample_dt: float = 0.01
    damping: float = 0.04
    elbow_branch: str = "up"             # up: upper-arm/elbow-up IK branch

    def validate(self):
        if (not np.isfinite(self.radius) or self.radius <= 0 or
                not np.isfinite(self.duration) or self.duration <= 0 or
                not np.isfinite(self.sample_dt) or self.sample_dt <= 0):
            raise ValueError("radius, duration, and sample_dt must be positive and finite")
        c = np.asarray(self.center, dtype=float)
        if c.shape != (3,) or not np.all(np.isfinite(c)):
            raise ValueError("center must contain three finite values")
        if self.elbow_branch not in ("up", "down"):
            raise ValueError("elbow_branch must be 'up' or 'down'")
        return self


class CircleReference:
    """A closed inclined circle with zero start/end velocity and acceleration."""

    TOOL_R = rpy_to_matrix(np.pi, 0.0, np.pi)

    def __init__(self, config: CircleConfig | None = None):
        self.config = (config or CircleConfig()).validate()
        self.center = np.asarray(self.config.center, dtype=float)
        tilt = np.deg2rad(self.config.tilt_deg)
        self.u = np.array([1.0, 0.0, 0.0])
        self.v = np.array([0.0, np.cos(tilt), np.sin(tilt)])
        self.times = np.append(np.arange(0.0, self.config.duration, self.config.sample_dt),
                               self.config.duration)
        self.positions, self.velocities = zip(*(self.cartesian(t) for t in self.times))
        self.positions, self.velocities = np.asarray(self.positions), np.asarray(self.velocities)
        self.q, self.qd = self._build_joint_reference()

    def _phase(self, t):
        tau = float(np.clip(t / self.config.duration, 0.0, 1.0))
        f = 10*tau**3 - 15*tau**4 + 6*tau**5
        df = (30*tau**2 - 60*tau**3 + 30*tau**4) / self.config.duration
        return 2*np.pi*f, 2*np.pi*df

    def cartesian(self, t):
        s, sd = self._phase(t)
        p = self.center + self.config.radius * (self.u*np.cos(s) + self.v*np.sin(s))
        v = self.config.radius * (-self.u*np.sin(s) + self.v*np.cos(s)) * sd
        return p, v

    def pose(self, position):
        T = np.eye(4)
        T[:3, :3] = self.TOOL_R
        T[:3, 3] = position
        return T

    def _build_joint_reference(self):
        q_rows, qd_rows = [], []
        q_ref = None
        for index, (p, v) in enumerate(zip(self.positions, self.velocities)):
            result = inverse_kinematics(self.pose(p), q_ref=q_ref)
            options = [sol.q for sol in result.solutions if sol.within_limits]
            if not options:
                raise RuntimeError(f"circle sample unreachable or outside limits ({result.status})")
            if index == 0:
                # Select the desired physical elbow posture once, then use
                # nearest-neighbour continuation so the whole closed circle
                # stays on that branch.  Larger q3 is the elbow-up solution
                # at this task's initial pose.
                q_ref = (max(options, key=lambda q: q[2]) if self.config.elbow_branch == "up"
                         else min(options, key=lambda q: q[2])).copy()
            else:
                q_ref = min(options, key=lambda q: np.linalg.norm(q - q_ref)).copy()
            q_rows.append(q_ref)
            qd_rows.append(joint_velocity_from_twist(q_ref, np.r_[v, np.zeros(3)],
                                                      method="dls", damping=self.config.damping))
        return np.asarray(q_rows), np.asarray(qd_rows)

    def evaluate(self, t):
        t = float(np.clip(t, 0.0, self.config.duration))
        q = np.array([np.interp(t, self.times, self.q[:, j]) for j in range(6)])
        qd = np.array([np.interp(t, self.times, self.qd[:, j]) for j in range(6)])
        p, v = self.cartesian(t)
        return q, qd, p, v


def circle_scene_xml(reference: CircleReference, segments=64):
    """Return a standard no-obstacle scene with a visible non-colliding ring."""
    root = ET.fromstring(scene_xml(SceneConfig()))
    world = root.find("worldbody")
    points = [reference.cartesian(reference.config.duration * k / segments)[0]
              for k in range(segments + 1)]
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        ET.SubElement(world, "geom", name=f"circle_reference_{i}", type="capsule",
                      fromto=" ".join(map(str, np.r_[a, b])), size="0.0035",
                      rgba="0.15 0.95 0.72 0.70", contype="0", conaffinity="0", group="1")
    return ET.tostring(root, encoding="unicode")


class CircleTrackingSimulation:
    """Closed-loop MuJoCo execution plus task-space metrics for one controller."""

    TORQUE_LIMITS = np.array([150, 150, 100, 30, 20, 10], dtype=float)

    def __init__(self, reference: CircleReference, controller_name="pd_gravity"):
        self.reference = reference
        self.model = mujoco.MjModel.from_xml_string(circle_scene_xml(reference))
        self.data = mujoco.MjData(self.model)
        self.joints = np.array([self.model.joint(n).id for n in rp.MJCF_JOINT_NAMES])
        self.qadr, self.vadr = self.model.jnt_qposadr[self.joints], self.model.jnt_dofadr[self.joints]
        self.motor_ids = np.array([self.model.actuator(f"torque{i}").id for i in range(1, 7)])
        self.controller = make_controller(controller_name, gravity=DHGravityModel())
        self.controller_name = controller_name
        self.dt, self.time = float(self.model.opt.timestep), 0.0
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        q0, _, _, _ = self.reference.evaluate(0.0)
        self.data.qpos[self.qadr] = q0
        mujoco.mj_forward(self.model, self.data)
        self.time, self.log = 0.0, []

    def step(self):
        q, qd = self.data.qpos[self.qadr].copy(), self.data.qvel[self.vadr].copy()
        # The reference is periodic.  Retaining monotonically increasing
        # simulation time while wrapping only the phase makes an interactive
        # demonstration repeat forever without a discontinuity at 2*pi.
        phase_time = self.time % self.reference.config.duration
        q_ref, qd_ref, p_ref, _ = self.reference.evaluate(phase_time)
        tau_req = self.controller.torque(q, qd, q_ref, qd_ref)
        tau = np.clip(tau_req, -self.TORQUE_LIMITS, self.TORQUE_LIMITS)
        self.data.ctrl[self.motor_ids] = tau
        site = self.model.site("ee_site").id
        self.log.append(dict(time=self.time, q=q.tolist(), q_ref=q_ref.tolist(), tau=tau.tolist(),
                             tau_req=np.asarray(tau_req).tolist(), ee=self.data.site_xpos[site].tolist(),
                             reference=p_ref.tolist()))
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.time += self.dt

    def run(self, realtime=False, viewer=None, max_duration=None):
        import time
        wall0 = time.perf_counter()
        while max_duration is None or self.time < max_duration:
            self.step()
            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    break
            if realtime:
                delay = self.time - (time.perf_counter() - wall0)
                if delay > 0:
                    time.sleep(min(delay, 0.01))
        return self.metrics()

    def metrics(self):
        if not self.log:
            return {}
        q = np.asarray([x["q"] for x in self.log]); qref = np.asarray([x["q_ref"] for x in self.log])
        ee = np.asarray([x["ee"] for x in self.log]); pref = np.asarray([x["reference"] for x in self.log])
        tau = np.asarray([x["tau"] for x in self.log]); tau_req = np.asarray([x["tau_req"] for x in self.log])
        cart_error = np.linalg.norm(ee - pref, axis=1)
        return {"controller": self.controller.describe(), "duration_s": self.time,
                "cartesian_rmse_m": float(np.sqrt(np.mean(cart_error**2))),
                "cartesian_peak_error_m": float(np.max(cart_error)),
                "joint_rmse_rad": np.sqrt(np.mean((qref-q)**2, axis=0)).tolist(),
                "joint_peak_error_rad": np.max(np.abs(qref-q), axis=0).tolist(),
                "torque_rms_nm": np.sqrt(np.mean(tau**2, axis=0)).tolist(),
                "torque_peak_nm": np.max(np.abs(tau), axis=0).tolist(),
                "saturation_time_s": float(np.any(np.abs(tau_req) > self.TORQUE_LIMITS, axis=1).sum()*self.dt),
                "circle": {"center": self.reference.center.tolist(), "radius_m": self.reference.config.radius,
                           "tilt_deg": self.reference.config.tilt_deg,
                           "period_s": self.reference.config.duration,
                           "cycles_completed": int(self.time // self.reference.config.duration)}}

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"metrics": self.metrics(), "samples": self.log}, indent=2), encoding="utf-8")
