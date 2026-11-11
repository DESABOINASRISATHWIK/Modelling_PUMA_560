"""Core PUMA-560 kinematics walkthrough.

This example follows the four required outcomes without MuJoCo or plotting:
robot specification, DH/FK, analytical IK, and the Jacobian/singularities.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dh_model import describe_robot
from src.forward_kinematics import forward_kinematics
from src.inverse_kinematics import inverse_kinematics
from src.jacobian import geometric_jacobian
from src.singularity_analysis import analyze_configuration
from src.velocity_kinematics import twist_from_joint_rates


np.set_printoptions(precision=4, suppress=True)

q = np.radians([30, 50, -70, 40, 60, -20])

print("1. ROBOT SPECIFICATION AND DH MODEL")
print(describe_robot())

print("\n2. FORWARD KINEMATICS")
fk = forward_kinematics(q)
print("q [deg] =", np.degrees(q).round(2))
print("end-effector position [m] =", fk.position)
print("wrist centre [m] =", fk.wrist_center)
print("T_0E =\n", fk.T)

print("\n3. ANALYTICAL INVERSE KINEMATICS")
ik = inverse_kinematics(fk.T, q_ref=q)
print(ik.summary())
best = ik.closest_to(q)
if best is not None:
    print("closest feasible branch =", best.label)

print("\n4. JACOBIAN AND SINGULARITIES")
J = geometric_jacobian(q)
print("J(q) =\n", J)
print("[linear velocity; angular velocity] = J(q) qdot")
qdot = np.radians([10, -5, 8, 12, -6, 4])
print("example qdot [deg/s] =", np.degrees(qdot).round(2))
print("resulting twist [m/s; rad/s] =", twist_from_joint_rates(q, qdot))
report = analyze_configuration(q)
print(report.text())
print("singularity factors =", report.factors)
