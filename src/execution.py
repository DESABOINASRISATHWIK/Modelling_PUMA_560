"""Closed-loop execution of a timed joint trajectory in MuJoCo (Part 3 / 4 glue).

The loop reads the TRUE simulator state, asks the controller for a torque, clips it to
the actuator limits, applies it and advances physics.  Joint positions are never
written during execution (only by the one-off ``reset`` at the start).

Logged at every control update:
    t, q, qd, q_des, qd_des, tau_requested, tau_applied, ee position, n_contacts
plus the set of colliding geom pairs.  Requested and applied torque differ only when
the actuator limit saturates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ARRAYS = ("t", "q", "qd", "q_des", "qd_des", "tau_req", "tau_app", "ee_pos", "n_contacts")


@dataclass
class ExecutionLog:
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    q_des: np.ndarray
    qd_des: np.ndarray
    tau_req: np.ndarray
    tau_app: np.ndarray
    ee_pos: np.ndarray
    n_contacts: np.ndarray
    contact_pairs: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        """Save raw numerical log (.npz) with the experiment settings embedded as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: getattr(self, k) for k in _ARRAYS},
                            contact_pairs=json.dumps(sorted({tuple(p) for p in self.contact_pairs})),
                            meta=json.dumps(self.meta, default=_json_default))
        return path

    @classmethod
    def load(cls, path):
        z = np.load(Path(path), allow_pickle=False)
        pairs = [tuple(p) for p in json.loads(str(z["contact_pairs"]))]
        return cls(**{k: z[k] for k in _ARRAYS}, contact_pairs=pairs, meta=json.loads(str(z["meta"])))


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def run_trajectory(sim, controller, trajectory, hold_time=2.0, control_decimation=1,
                   disturbance=None, meta=None, q_start=None):
    """Execute ``trajectory`` with ``controller`` on ``sim`` and return an ``ExecutionLog``.

    Parameters
    ----------
    sim : PhysicsSimulation          torque-driven MuJoCo environment (Part 1)
    controller : JointController     ``torque(q, qd, q_des, qd_des)``
    trajectory : object with ``duration`` and ``evaluate(t) -> (q_des, qd_des, qdd_des)``
    hold_time : seconds the final configuration is held after the trajectory ends
                (gives both controllers time to settle; same interval for every run)
    control_decimation : controller update every N physics steps (1 = 500 Hz at dt = 2 ms)
    disturbance : optional callable ``t -> (6,) external joint torque [N m]``
    q_start : robot start (default: trajectory.evaluate(0)); ``sim.reset`` is called ONCE here
    """
    if control_decimation < 1 or int(control_decimation) != control_decimation:
        raise ValueError("control_decimation must be a positive integer")
    if hold_time < 0:
        raise ValueError("hold_time must be non-negative")
    dt_sim = float(sim.model.opt.timestep)
    dt_ctrl = dt_sim * int(control_decimation)
    q0 = trajectory.evaluate(0.0)[0] if q_start is None else q_start
    sim.reset(q0)
    controller.reset()

    total = float(trajectory.duration) + float(hold_time)
    n = int(np.ceil(total / dt_ctrl - 1e-9))
    rec = {k: [] for k in _ARRAYS}
    pairs = set()

    for k in range(n):
        t = k * dt_ctrl
        q, qd = sim.state()
        q_des, qd_des, _ = trajectory.evaluate(t)
        tau_req = np.asarray(controller.torque(q, qd, q_des, qd_des), dtype=float)
        if not np.all(np.isfinite(tau_req)):
            raise FloatingPointError(f"controller returned a non-finite torque at t = {t:.3f} s")
        tau_app = sim.apply_torque(tau_req)
        if disturbance is not None:
            sim.apply_disturbance(disturbance(t))
        contacts = sim.contacts()
        rec["t"].append(t)
        rec["q"].append(q)
        rec["qd"].append(qd)
        rec["q_des"].append(q_des)
        rec["qd_des"].append(qd_des)
        rec["tau_req"].append(tau_req)
        rec["tau_app"].append(tau_app)
        rec["ee_pos"].append(sim.end_effector_pose()[:3, 3])
        rec["n_contacts"].append(len(contacts))
        pairs.update(tuple(sorted(p)) for p in contacts)
        sim.step(int(control_decimation))

    # final sample after the last step so the last row is the terminal state
    q, qd = sim.state()
    q_des, qd_des, _ = trajectory.evaluate(total)
    rec["t"].append(n * dt_ctrl)
    rec["q"].append(q)
    rec["qd"].append(qd)
    rec["q_des"].append(q_des)
    rec["qd_des"].append(qd_des)
    rec["tau_req"].append(rec["tau_req"][-1])
    rec["tau_app"].append(rec["tau_app"][-1])
    rec["ee_pos"].append(sim.end_effector_pose()[:3, 3])
    rec["n_contacts"].append(len(sim.contacts()))
    pairs.update(tuple(sorted(p)) for p in sim.contacts())

    info = dict(meta or {})
    info.update(controller=controller.describe(), control_period_s=dt_ctrl, physics_dt_s=dt_sim,
                trajectory_duration_s=float(trajectory.duration), hold_time_s=float(hold_time),
                torque_limits=list(map(float, sim.config.torque_limits)))
    return ExecutionLog(**{k: np.asarray(v) for k, v in rec.items()}, contact_pairs=sorted(pairs), meta=info)
