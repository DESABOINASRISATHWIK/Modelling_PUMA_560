
from pathlib import Path
import numpy as np
import mujoco

from forward_kinematics import forward_kinematics

root = Path(__file__).resolve().parent.parent
model = mujoco.MjModel.from_xml_path(
    str(root / "models" / "puma560.xml")
)
data = mujoco.MjData(model)

site_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
)

# Test different joint configurations (radians).
test_configs = [
    [0.2, -0.3, 0.4, 0.1, -0.2, 0.3],
    [-0.5, 0.4, -0.2, 0.6, 0.3, -0.4],
    [0.7, -0.6, 0.5, -0.3, 0.2, 0.1],
]

for k, q in enumerate(test_configs, start=1):
    q = np.asarray(q, dtype=float)

    # Set MuJoCo joint positions by joint name.
    for i in range(6):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT,
            f"joint{i + 1}"
        )
        qpos_id = model.jnt_qposadr[joint_id]
        data.qpos[qpos_id] = q[i]

    mujoco.mj_forward(model, data)

    T_numpy = forward_kinematics(q)
    p_numpy = T_numpy[:3, 3]
    R_numpy = T_numpy[:3, :3]

    p_mujoco = data.site_xpos[site_id].copy()
    R_mujoco = data.site_xmat[site_id].reshape(3, 3).copy()

    position_error = np.linalg.norm(p_numpy - p_mujoco)
    orientation_error = np.linalg.norm(R_numpy - R_mujoco)

    print(f"\nTest {k}")
    print("NumPy position: ", p_numpy)
    print("MuJoCo position:", p_mujoco)
    print(f"Position error: {position_error:.3e} m")
    print(f"Rotation matrix error: {orientation_error:.3e}")

    assert position_error < 1e-8, "Position mismatch!"
    assert orientation_error < 1e-8, "Orientation mismatch!"

print("\nAll FK validation tests passed!")
