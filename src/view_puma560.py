
from pathlib import Path

import mujoco
import mujoco.viewer

project_dir = Path(__file__).resolve().parent.parent
model_path = project_dir / "models" / "puma560.xml"

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Start at the default joint configuration.
mujoco.mj_resetData(model, data)
data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)

print("Model loaded successfully!")
print("Number of joints:", model.njnt)
print("Number of actuators:", model.nu)
print("Degrees of freedom:", model.nv)

for i in range(model.njnt):
    print(f"Joint {i + 1}: {model.joint(i).name}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
