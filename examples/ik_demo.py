"""Inverse kinematics demo: all analytic branches, unreachable target, joint-limit infeasible target."""
import numpy as np

from _common import RESULTS, mujoco_images
from src.forward_kinematics import forward_kinematics
from src.inverse_kinematics import inverse_kinematics
from src.visualization import image_grid, plot_ik_branches

np.set_printoptions(precision=4, suppress=True)

# 1) a reachable pose generated from a known configuration
q_true = np.radians([30, 50, -70, 40, 60, -20])
T = forward_kinematics(q_true).T
res = inverse_kinematics(T, q_ref=q_true)
print(res.summary())
best = res.closest_to(q_true)
print("\nclosest to the initial configuration:", best.label, np.degrees(best.q).round(2), "deg")
plot_ik_branches(T, res, RESULTS / "ik_branches.png")

# 2) unreachable target (geometric)
T_far = np.eye(4)
T_far[:3, 3] = [1.4, 0.2, 0.9]
r = inverse_kinematics(T_far)
print("\nfar target       ->", r.status, "|", r.reason, "|", r.message)
T_axis = np.diag([1, -1, -1, 1.0])
T_axis[:3, 3] = [0.0, 0.0, 0.9]
r = inverse_kinematics(T_axis)
print("target on z-axis ->", r.status, "|", r.reason, "|", r.message)

# 3) reachable geometry but every branch violates a joint limit
rng = np.random.default_rng(2)
while True:
    q = rng.uniform(-np.pi, np.pi, 6)
    r = inverse_kinematics(forward_kinematics(q).T, q_ref=q)
    if r.status == "joint_limits":
        break
print("outside-limits pose ->", r.status, "|", r.reason, "|", r.message)

# 4) MuJoCo renders of three feasible branches (robot posed by OUR IK solutions)
feas = res.feasible
imgs = mujoco_images([(s.q, T) for s in feas[:3]]) if feas else None
if imgs:
    idx = {id(s): k for k, s in enumerate(res.solutions)}
    image_grid(imgs, [f"branch [{idx[id(s)]}]: {s.label}" for s in feas[:3]],
               RESULTS / "ik_mujoco_branches.png", ncols=3)
print("figures written to", RESULTS)
