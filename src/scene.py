"""Physics scenes built from the validated MJCF without changing its kinematics."""
from dataclasses import dataclass
from pathlib import Path
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from . import robot_parameters as rp


def vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} requires {size} finite numbers")
    return a


def fmt(value):
    return " ".join(str(float(x)) for x in value)


@dataclass(frozen=True)
class Obstacle:
    """Axis-aligned world box; size is HALF extent in metres."""
    position: tuple = (0.55, -0.15, 0.85)
    size: tuple = (0.10, 0.10, 0.15)


@dataclass(frozen=True)
class Payload:
    """Rigid box, centre relative to the flange; SI mass and half extents."""
    mass: float = 1.0
    position: tuple = (0.0, 0.0, 0.06)
    size: tuple = (0.04, 0.04, 0.04)


@dataclass(frozen=True)
class SceneConfig:
    timestep: float = 0.002
    gravity: tuple = (0.0, 0.0, -9.81)
    damping: tuple = (1.0,) * 6
    torque_limits: tuple = (150.0, 150.0, 100.0, 30.0, 20.0, 10.0)
    gear: tuple = (1.0,) * 6
    obstacles: tuple = ()
    payload: Payload | None = None

    @classmethod
    def obstacle_scene(cls, **kwargs):
        return cls(obstacles=(Obstacle(),), **kwargs)


def scene_xml(config=None):
    c = config or SceneConfig()
    if not np.isfinite(c.timestep) or c.timestep <= 0:
        raise ValueError("timestep must be positive and finite")
    gravity = vector(c.gravity, 3, "gravity")
    damping = vector(c.damping, 6, "damping")
    limits = vector(c.torque_limits, 6, "torque_limits")
    gear = vector(c.gear, 6, "gear")
    if np.any(damping < 0) or np.any(limits <= 0) or np.any(gear == 0):
        raise ValueError("require nonnegative damping, positive limits, nonzero gears")
    root = ET.parse(rp.MJCF_PATH).getroot()
    root.find("compiler").set("meshdir", str((rp.PROJECT_ROOT / "models/meshes").resolve()))
    root.find("option").attrib.update(timestep=str(c.timestep), gravity=fmt(gravity), integrator="implicitfast")
    world = root.find("worldbody")
    world.find("geom[@name='floor']").attrib.update(contype="1", conaffinity="1")
    # Simplified teaching proxies, independent of the render meshes and inertia.
    proxies = [
        dict(type="capsule", fromto="0 0 0.18 0 0 0.55", size="0.08"),
        dict(type="capsule", fromto="0.09 -0.15 0 0.37 -0.15 0", size="0.055"),
        dict(type="capsule", fromto="0 -0.06 0 0.02 -0.15 0", size="0.045"),
        dict(type="capsule", fromto="0 0 0.06 0 0 0.36", size="0.04"),
        dict(type="sphere", pos="0 0 0", size="0.025"),
        dict(type="sphere", pos="0 0 0.04", size="0.018"),
    ]
    contact = ET.SubElement(root, "contact")
    for i, proxy in enumerate(proxies, 1):
        body = world.find(f".//body[@name='link{i}']")
        body.find("joint").set("damping", str(damping[i-1]))
        ET.SubElement(body, "geom", name=f"collision_link{i}", contype="1", conaffinity="1",
                      rgba="0.2 0.8 0.3 0.15", group="3", **proxy)
        if i < 6:
            ET.SubElement(contact, "exclude", body1=f"link{i}", body2=f"link{i+1}")
    ET.SubElement(world, "geom", name="base_collision", type="cylinder", size="0.12 0.035",
                  pos="0 0 0.035", rgba="0.2 0.2 0.2 1")
    for i, obstacle in enumerate(c.obstacles):
        pos = vector(obstacle.position, 3, "obstacle position")
        size = vector(obstacle.size, 3, "obstacle size")
        if np.any(size <= 0):
            raise ValueError("obstacle half extents must be positive")
        ET.SubElement(world, "geom", name=f"obstacle_{i}", type="box", pos=fmt(pos), size=fmt(size),
                      rgba="0.8 0.2 0.2 1")
    if c.payload is not None:
        p = c.payload
        size = vector(p.size, 3, "payload size")
        pos = vector(p.position, 3, "payload position") + [0, 0, rp.D_FLANGE]
        if not np.isfinite(p.mass) or p.mass <= 0 or np.any(size <= 0):
            raise ValueError("payload mass and half extents must be positive")
        body = ET.SubElement(world.find(".//body[@name='link6']"), "body", name="payload", pos=fmt(pos))
        inertia = p.mass / 3 * np.array([size[1]**2 + size[2]**2, size[0]**2 + size[2]**2,
                                        size[0]**2 + size[1]**2])
        ET.SubElement(body, "inertial", pos="0 0 0", mass=str(p.mass), diaginertia=fmt(inertia))
        ET.SubElement(body, "geom", name="payload_collision", type="box", size=fmt(size), rgba="0.2 0.3 0.8 1")
        ET.SubElement(contact, "exclude", body1="link6", body2="payload")
    actuator = ET.SubElement(root, "actuator")
    for i in range(6):
        control_limit = limits[i] / abs(gear[i])
        ET.SubElement(actuator, "motor", name=f"torque{i+1}", joint=f"joint{i+1}", gear=str(gear[i]),
                      ctrllimited="true", ctrlrange=fmt([-control_limit, control_limit]),
                      forcelimited="true", forcerange=fmt([-control_limit, control_limit]))
    return ET.tostring(root, encoding="unicode")


def load_scene(config=None):
    return mujoco.MjModel.from_xml_string(scene_xml(config))


def write_scene(path, config=None):
    """Export a reviewable scene with mesh paths relative to the XML location."""
    path = Path(path)
    root = ET.fromstring(scene_xml(config))
    root.find("compiler").set("meshdir", os.path.relpath(rp.PROJECT_ROOT / "models/meshes", path.resolve().parent))
    ET.indent(root, space="  ")
    path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
