
import mujoco
import mujoco.viewer

xml = """
<mujoco model="installation_test">
    <option gravity="0 0 -9.81"/>
    <worldbody>
        <geom name="floor"
              type="plane"
              size="2 2 0.1"
              rgba="0.7 0.7 0.7 1"/>

        <body name="ball" pos="0 0 1">
            <freejoint/>
            <geom type="sphere"
                  size="0.1"
                  mass="1"
                  rgba="0.1 0.4 0.9 1"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("MuJoCo model loaded successfully!")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
