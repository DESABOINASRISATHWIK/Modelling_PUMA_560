"""Headless PD + gravity example; all execution motion comes from applied torque."""
import _common  # noqa: F401
import numpy as np

from src.simulation import PhysicsSimulation
from src.collision import CollisionChecker
from src.gravity import GravityCompensator
from src.scene import SceneConfig, Payload


def main():
    sim = PhysicsSimulation(SceneConfig(payload=Payload(mass=0.5)))
    collision = CollisionChecker(sim.model)
    gravity = GravityCompensator(sim.model)
    target = np.array([0.15, 0.15, -0.2, 0.1, 0.1, 0.1])
    q, _ = sim.state()
    if not collision.is_edge_valid(q, target):
        raise RuntimeError("demo joint-space path is blocked")
    kp = np.array([80, 100, 80, 15, 10, 5])
    # Wrist inertia is very small; use modest explicit velocity feedback at 500 Hz.
    # The assumed 1 N m s/rad joint damping is also integrated implicitly.
    kd = np.array([25, 25, 20, 0.2, 0.2, 0.2])
    for _ in range(1500):
        q, qvel = sim.state()
        sim.apply_torque(kp * (target-q) - kd * qvel + gravity.torque(q))
        sim.step()
    q, qvel = sim.state()
    if np.max(np.abs(q-target)) > 0.01 or np.max(np.abs(qvel)) > 0.01:
        raise RuntimeError("PD demonstration did not settle within tolerance")
    print("time [s]:", sim.data.time)
    print("joint error [rad]:", q-target)
    print("joint velocity [rad/s]:", qvel)
    print("end-effector pose:\n", sim.end_effector_pose())


if __name__ == "__main__":
    main()
