
from pathlib import Path
import mujoco
import numpy as np

root = Path(__file__).resolve().parent.parent
model_path = root / "models" / "puma560.xml"

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

print("Joints:", model.njnt)
print("Actuators:", model.nu)
print("Degrees of freedom:", model.nv)

print("\nJoint names:")
for i in range(model.njnt):
    print("-", model.joint(i).name)

site_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
)

if site_id >= 0:
    print("\nEnd-effector position:",
          data.site_xpos[site_id].copy())
    print("End-effector orientation:\n",
          data.site_xmat[site_id].reshape(3, 3).copy())

assert model.njnt == 6, "Expected six joints"
assert model.nu == 6, "Expected six actuators"

print("\nBasic model checks passed.")
