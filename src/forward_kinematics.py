
import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def translate(x=0.0, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def forward_kinematics(q):
    """
    Forward kinematics for the PUMA 560 mesh model.

    q: six joint angles in radians.
    Returns: 4x4 end-effector transformation matrix.
    """
    if len(q) != 6:
        raise ValueError("Expected exactly six joint angles")

    q1, q2, q3, q4, q5, q6 = q

    T = np.eye(4)
    T = (
        T
        @ translate(z=0.01)
        @ translate(z=0.6718)
        @ rot_x(np.pi / 2)
        @ rot_y(q1)
        @ rot_z(q2)
        @ translate(x=0.4318, y=-0.0203, z=0.1501)
        @ rot_z(-np.pi / 2)
        @ rot_z(q3)
        @ rot_x(np.pi / 2)
        @ rot_z(np.pi / 2)
        @ rot_z(q4)
        @ translate(z=0.4331)
        @ rot_x(np.pi / 2)
        @ rot_z(q5)
        @ translate(y=0.0558)
        @ rot_x(-np.pi / 2)
        @ rot_z(q6)
    )

    return T


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate PUMA 560 forward kinematics."
    )
    parser.add_argument(
        "joint_angles",
        nargs="*",
        type=float,
        metavar="q",
        help="six joint angles in radians",
    )
    args = parser.parse_args()
    if args.joint_angles:
        if len(args.joint_angles) != 6:
            parser.error("provide exactly six joint angles")
        q = np.asarray(args.joint_angles, dtype=float)
    else:
        try:
            q = np.asarray(
                [
                    float(input(f"Enter joint angle q{i} (radians): "))
                    for i in range(1, 7)
                ],
                dtype=float,
            )
        except ValueError as exc:
            parser.error(f"joint angles must be numeric: {exc}")

    T = forward_kinematics(q)

    print("Transformation matrix:")
    print(T)
    print("\nEnd-effector position (m):")
    print(T[:3, 3])
    print("\nEnd-effector orientation:")
    print(T[:3, :3])

    project_dir = Path(__file__).resolve().parent.parent
    model_path = project_dir / "models" / "puma560.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    for i in range(6):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}"
        )
        data.qpos[model.jnt_qposadr[joint_id]] = q[i]
        data.ctrl[i] = q[i]

    mujoco.mj_forward(model, data)
    print("\nMuJoCo viewer started. Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
