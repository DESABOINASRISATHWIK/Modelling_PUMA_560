"""Forward kinematics demo: DH table, all intermediate frames, end-effector pose, MuJoCo cross-check."""
import numpy as np

from _common import RESULTS, mujoco_images
from src.dh_model import describe_robot
from src.forward_kinematics import forward_kinematics
from src.mujoco_reference import mujoco_fk
from src.utils import matrix_to_rpy, orientation_angle
from src.visualization import image_grid, plot_fk

np.set_printoptions(precision=4, suppress=True)
print(describe_robot(), "\n")

q = np.radians([30, 50, -70, 40, 60, -20])
fk = forward_kinematics(q)
print("q [deg] =", np.degrees(q))
for i, T in enumerate(fk.frames):
    print(f"T_0{i} =\n{T}")
print(f"\nflange pose T_0E =\n{fk.T}")
print("position [m]     =", fk.position)
print("rpy [deg]        =", np.degrees(matrix_to_rpy(fk.rotation)))
print("wrist centre [m] =", fk.wrist_center)

ref = mujoco_fk(q)
print(f"\nMuJoCo reference: position error = {np.linalg.norm(ref[:3, 3] - fk.position):.2e} m, "
      f"orientation error = {orientation_angle(ref[:3, :3], fk.rotation):.2e} rad")

plot_fk(q, RESULTS / "fk_frames.png")
imgs = mujoco_images([(q, None)])
if imgs:
    image_grid(imgs, ["MuJoCo render with our DH frames / wrist centre / EE frame"], RESULTS / "fk_mujoco.png", ncols=1)
print("figures written to", RESULTS)
