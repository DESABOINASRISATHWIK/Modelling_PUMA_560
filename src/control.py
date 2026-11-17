"""Joint-space feedback controllers for the PUMA 560 (Part 3).

Both controllers share one interface so they can be swapped by configuration:

    tau = controller.torque(q, qd, q_des, qd_des)          # (6,) N m, BEFORE actuator limits

* ``PDController``                tau = Kp e + Kd e_dot
* ``PDGravityController``         tau = Kp e + Kd e_dot + g_hat(q)

with  e = q_des - q,  e_dot = qd_des - qd  (all per-joint, element-wise gains).

Why these controllers
---------------------
The plant is  M(q) qdd + C(q, qd) qd + g(q) + B qd = tau.  A PD law needs no model, is
easy to tune per joint and is globally stable for set-point regulation *when gravity is
compensated* (Takegaki & Arimoto, 1981; the Lyapunov argument uses V = 1/2 qd^T M qd +
1/2 e^T Kp e, which only works if g(q) is cancelled).  Without the g_hat term the
equilibrium is displaced: for a slow motion  e_ss ~= Kp^-1 g(q), which is the effect the
comparison study quantifies.  ``g_hat`` is evaluated at the MEASURED q (feedback
gravity compensation), not at q_des.

Gains are per-joint vectors.  They are chosen by ``tools/tune_gains.py`` (see its
docstring for the procedure) and stored in ``DEFAULT_KP`` / ``DEFAULT_KD``.

Discrete-time limit on Kd
-------------------------
The controller runs at the physics rate (500 Hz).  The derivative term is an explicit
force, so a joint of inertia I is stable only if  Kd < 2 I / dt  (+ the simulator's own
viscous damping, which MuJoCo integrates implicitly; we do NOT rely on it).  The wrist
joints have tiny inertia (joint 6: about 4e-5 kg m^2) so their Kd must be very small;
this is why the wrist Kd below is far smaller than the arm Kd.  ``design_pd_gains``
enforces the conservative cap  Kd <= I_min / dt.
"""
from __future__ import annotations

import numpy as np

from .scene import vector

# --------------------------------------------------------------------------------------
# Gain design
# --------------------------------------------------------------------------------------
# Joint-space inertia M_ii over the collision-free workspace (MuJoCo mass matrix sampled at
# 600 random valid poses; analysis only, the controllers never read M):
INERTIA_MIN = np.array([1.572, 1.998, 0.3586, 1.640e-3, 6.42e-4, 4.0e-5])    # kg m^2
INERTIA_MAX = np.array([4.248, 2.882, 0.3618, 1.842e-3, 6.42e-4, 4.0e-5])
SIM_DAMPING = 1.0            # N m s/rad, the viscous coefficient assumed in the scene (Part 1)
CONTROL_DT = 0.002           # s


def design_pd_gains(omega_n, zeta=1.0, inertia_max=INERTIA_MAX, inertia_min=INERTIA_MIN,
                    damping=SIM_DAMPING, dt=CONTROL_DT, kd_floor=0.01):
    """Per-joint PD gains from a desired natural frequency and damping ratio.

    Joint model about the gravity-compensated equilibrium:  I qdd + b qd = Kp e - Kd qd, i.e.
        I s^2 + (b + Kd) s + Kp = 0   ->   wn = sqrt(Kp / I),  zeta = (b + Kd) / (2 sqrt(Kp I)).
    Design rule (worst case = largest inertia, so the response is at least critically
    damped everywhere in the workspace):
        Kp = I_max wn^2,      Kd = max( 2 zeta wn I_max - b, kd_floor ),
    then Kd is capped at I_min / dt for explicit-derivative stability at the control rate.
    For the three wrist joints the simulator damping b alone already gives zeta > 1, so
    only the floor remains: they behave as first-order lags with time constant (b+Kd)/Kp.
    """
    wn = np.broadcast_to(np.asarray(omega_n, dtype=float), (6,))
    kp = inertia_max * wn ** 2
    kd = np.maximum(2.0 * zeta * wn * inertia_max - damping, kd_floor)
    return kp, np.minimum(kd, inertia_min / dt)


# Final values (rounded), chosen with tools/tune_gains.py; see docs/METHODS.md.
#   joints 1-3  : wn = 12, 14, 25 rad/s, zeta = 1 -> design rule above
#   joints 4-6  : damping dominated; Kp sets the closed-loop time constant b/Kp ~ 50 ms
DEFAULT_KP = np.array([600.0, 550.0, 225.0, 20.0, 20.0, 20.0])
DEFAULT_KD = np.array([100.0, 80.0, 17.0, 0.5, 0.2, 0.01])


class JointController:
    """Base class: stores gains, defines the common ``torque`` interface."""

    name = "base"

    def __init__(self, kp=None, kd=None):
        self.kp = vector(DEFAULT_KP if kp is None else kp, 6, "kp")
        self.kd = vector(DEFAULT_KD if kd is None else kd, 6, "kd")
        if np.any(self.kp < 0) or np.any(self.kd < 0):
            raise ValueError("gains must be non-negative")

    def reset(self):
        """Clear internal state (none for PD); called at the start of every run."""

    def _pd(self, q, qd, q_des, qd_des):
        q, qd = vector(q, 6, "q"), vector(qd, 6, "qd")
        q_des, qd_des = vector(q_des, 6, "q_des"), vector(qd_des, 6, "qd_des")
        return self.kp * (q_des - q) + self.kd * (qd_des - qd)

    def torque(self, q, qd, q_des, qd_des):                # pragma: no cover - overridden
        raise NotImplementedError

    def describe(self):
        return dict(name=self.name, kp=self.kp.tolist(), kd=self.kd.tolist())


class PDController(JointController):
    """tau = Kp (q_des - q) + Kd (qd_des - qd)."""

    name = "pd"

    def torque(self, q, qd, q_des, qd_des):
        return self._pd(q, qd, q_des, qd_des)


class PDGravityController(JointController):
    """tau = Kp (q_des - q) + Kd (qd_des - qd) + g_hat(q).

    ``gravity`` is any object with ``torque(q) -> (6,)`` that returns the torque
    cancelling gravity (e.g. ``DHGravityModel``).  The model it carries (with or
    without payload) is what the study reports as "the model used by the compensation".
    """

    name = "pd_gravity"

    def __init__(self, gravity, kp=None, kd=None, label=None):
        super().__init__(kp, kd)
        if not hasattr(gravity, "torque"):
            raise TypeError("gravity must provide torque(q)")
        self.gravity = gravity
        if label:
            self.name = label

    def torque(self, q, qd, q_des, qd_des):
        return self._pd(q, qd, q_des, qd_des) + self.gravity.torque(q)

    def describe(self):
        d = super().describe()
        d["gravity_model"] = type(self.gravity).__name__
        payload = getattr(self.gravity, "payload", None)
        d["gravity_includes_payload_kg"] = float(payload.mass) if payload is not None else 0.0
        return d


def make_controller(name, gravity=None, kp=None, kd=None, label=None):
    """Select a controller by configuration string: 'pd' or 'pd_gravity'."""
    if name == "pd":
        c = PDController(kp, kd)
        if label:
            c.name = label
        return c
    if name == "pd_gravity":
        if gravity is None:
            raise ValueError("pd_gravity needs a gravity model")
        return PDGravityController(gravity, kp, kd, label)
    raise ValueError(f"unknown controller '{name}' (choose 'pd' or 'pd_gravity')")
