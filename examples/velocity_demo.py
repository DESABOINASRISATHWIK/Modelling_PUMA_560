"""Velocity kinematics demo: J, qdot <-> twist, frames, ellipsoids, singular directions, joint-speed blow-up."""
import numpy as np

from _common import RESULTS
from src.forward_kinematics import forward_kinematics
from src.jacobian import (end_effector_twist, geometric_jacobian, jacobian_numerical,
                          joint_velocity_from_twist)
from src.singularity_analysis import analyze_configuration, singular_configuration
from src.velocity_kinematics import (directional_gain, singular_directions, velocity_ellipsoid,
                                     velocity_report)
from src.visualization import plot_joint_speed_vs_singularity, plot_velocity_ellipsoids

np.set_printoptions(precision=4, suppress=True, linewidth=120)
q = np.radians([30, 50, -70, 40, 60, -20])
qd = np.array([0.2, -0.1, 0.3, 0.5, -0.4, 0.6])                    # rad/s

print("=== 1. forward velocity:  [v; w] = J(q) qdot")
J = geometric_jacobian(q)
print("J(q) (base frame, flange point) =\n", J)
r = velocity_report(q, qd)
print("\nqdot [rad/s] =", qd)
print(f"v [m/s]     = {r['linear_velocity']}   |v| = {r['linear_speed']:.4f}")
print(f"w [rad/s]   = {r['angular_velocity']}   |w| = {r['angular_speed']:.4f}")
print(f"max |J - J_finite_difference| = {np.abs(J - jacobian_numerical(q)).max():.2e}")

print("\n=== 2. frames and reference point")
Je = geometric_jacobian(q, frame='ee')
R = forward_kinematics(q).rotation
tw_b, tw_e = J @ qd, Je @ qd
print("twist in base frame :", tw_b)
print("twist in flange frame:", tw_e, "  (= R^T applied to both blocks:", np.allclose(tw_e[:3], R.T @ tw_b[:3]), ")")
Jw = geometric_jacobian(q, point='wrist')
print("wrist-centre Jacobian has zero linear part for the wrist columns:", np.allclose(Jw[:3, 3:], 0))

print("\n=== 3. inverse velocity:  qdot = J^+ [v; w]")
tw = np.array([0.10, 0.00, -0.05, 0.0, 0.2, 0.0])
qd_pinv = joint_velocity_from_twist(q, tw, method="pinv")
qd_dls = joint_velocity_from_twist(q, tw, method="dls", damping=0.05)
print("desired twist       :", tw)
print("qdot (pseudo-inverse):", qd_pinv, "  J qdot - twist =", np.abs(J @ qd_pinv - tw).max())
print("qdot (damped LS)     :", qd_dls, "  J qdot - twist =", np.abs(J @ qd_dls - tw).max())

print("\n=== 4. velocity ellipsoids (unit-norm joint velocity) at this pose")
for part in ("linear", "angular"):
    e = velocity_ellipsoid(q, part)
    print(f"{part:<8} semi-axes {e.semi_axes}  volume measure {e.volume_measure:.4f}")
    print("         principal directions (columns):\n", e.axes)
print("gain along base-z (linear):", round(directional_gain(q, [0, 0, 1]), 4),
      " along base-x:", round(directional_gain(q, [1, 0, 0]), 4))

print("\n=== 5. what is lost at each singularity")
for kind in ("shoulder", "elbow", "wrist"):
    qs = singular_configuration(kind)
    d = singular_directions(qs)
    print(f"{kind:<9} {analyze_configuration(qs).text()}")
    print(f"{'':<9} lost twist  v={d.lost_twist[:3]}  w={d.lost_twist[3:]}")
    print(f"{'':<9} self-motion qdot={d.null_motion}   |J qdot| = {d.residual:.1e}")

plot_velocity_ellipsoids(RESULTS / "velocity_ellipsoids.png")
S = plot_joint_speed_vs_singularity(RESULTS / "joint_speed_vs_singularity.png")
print("\n=== 6. joint speed for the weakest twist near the wrist singularity (q5 = 1e-4):")
print(f"pseudo-inverse {S['pinv_w'][0]:.3e} rad/s   damped LS {S['dls_w'][0]:.3f} rad/s   (DLS bound 1/(2*0.05) = 10)")
print("figures written to", RESULTS)
