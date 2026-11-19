#!/usr/bin/env python
"""Static pillar detour: NumPy RRT / RRT-Connect planning plus torque execution."""
from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from src import robot_parameters as rp
from src.control import make_controller
from src.forward_kinematics import forward_kinematics
from src.gravity import DHGravityModel
from src.scene import scene_xml
from src.tasks import obstacle_task, prepare_task
from src.trajectory import QuinticTrajectory


def _path_model(prepared):
    """Pillar scene plus a visual-only end-effector route and endpoint markers."""
    root = ET.fromstring(scene_xml(prepared.task.exec_scene()))
    world = root.find("worldbody")
    times = np.linspace(0.0, prepared.trajectory.duration, 48)
    points = [forward_kinematics(prepared.trajectory.evaluate(t)[0]).position for t in times]
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        ET.SubElement(world, "geom", name=f"planned_path_{i}", type="capsule",
                      fromto=" ".join(map(str, np.r_[a, b])), size="0.004",
                      rgba="0.15 0.95 0.72 0.85", contype="0", conaffinity="0", group="1")
    for name, p, colour in (("path_start", points[0], "0.2 0.5 1 0.9"),
                             ("path_goal", points[-1], "1 0.85 0.1 0.9")):
        ET.SubElement(world, "geom", name=name, type="sphere", pos=" ".join(map(str, p)), size="0.018",
                      rgba=colour, contype="0", conaffinity="0", group="1")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def run(prepared, controller_name, headless=False):
    model = _path_model(prepared)
    data = mujoco.MjData(model)
    joints = np.array([model.joint(n).id for n in rp.MJCF_JOINT_NAMES])
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    motors = np.array([model.actuator(f"torque{i}").id for i in range(1, 7)])
    data.qpos[qadr] = prepared.q_start
    mujoco.mj_forward(model, data)
    controller = make_controller(controller_name, gravity=DHGravityModel())
    limits = np.asarray([150, 150, 100, 30, 20, 10], dtype=float)
    rows, dt, t = [], float(model.opt.timestep), 0.0
    forward = prepared.trajectory
    backward = QuinticTrajectory(
        forward.waypoints[::-1],
        dt=forward.dt,
        max_velocity=forward.max_velocity,
        max_acceleration=forward.max_acceleration,
    )
    cycle_duration = forward.duration + backward.duration

    def step():
        nonlocal t
        q, qd = data.qpos[qadr].copy(), data.qvel[vadr].copy()
        phase = t % cycle_duration
        if phase < forward.duration:
            qref, qdref, _ = forward.evaluate(phase)
        else:
            qref, qdref, _ = backward.evaluate(phase - forward.duration)
        requested = controller.torque(q, qd, qref, qdref)
        applied = np.clip(requested, -limits, limits)
        data.ctrl[motors] = applied
        rows.append({"time": t, "joint_error_norm_rad": float(np.linalg.norm(qref-q)),
                     "torque_nm": applied.tolist()})
        mujoco.mj_step(model, data); mujoco.mj_forward(model, data); t += dt

    if headless:
        while t < prepared.trajectory.duration:
            step()
    else:
        from mujoco import viewer as mujoco_viewer
        with mujoco_viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 135, -20, 2.6
            viewer.cam.lookat[:] = [0.55, 0.0, 0.65]
            wall0 = time.perf_counter()
        
            while viewer.is_running():
                step()
                viewer.sync()
        
                delay = t - (time.perf_counter() - wall0)
                if delay > 0:
                    time.sleep(min(delay, 0.01))
    return {"planner": prepared.plan.method, "direct_path_blocked": prepared.direct_blocked,
            "planning_time_s": prepared.plan.planning_time, "iterations": prepared.plan.iterations,
            "path_length_rad": prepared.plan.path_length, "trajectory_duration_s": prepared.trajectory.duration,
            "collision_free_reference": prepared.trajectory_collision_free,
            "execution_joint_error_rms_rad": float(np.sqrt(np.mean([x["joint_error_norm_rad"]**2 for x in rows]))),
            "execution_joint_error_peak_rad": float(max(x["joint_error_norm_rad"] for x in rows)),
            "controller": controller.describe()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planner", choices=("rrt", "rrt_connect"), default="rrt_connect")
    p.add_argument("--controller", choices=("pd", "pd_gravity"), default="pd_gravity")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--output", default="results/presentation/obstacle/latest.json")
    a = p.parse_args(argv)
    prepared = prepare_task(obstacle_task(), planner=a.planner, seed=7)
    metrics = run(prepared, a.controller, headless=a.headless)
    path = Path(a.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
