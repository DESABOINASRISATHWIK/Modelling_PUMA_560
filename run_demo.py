#!/usr/bin/env python
"""Live demonstration: select the case and the controller, watch the torque-driven motion.

    python run_demo.py --case nominal  --controller pd_gravity
    python run_demo.py --case obstacle --controller pd            # sags into the pillar
    python run_demo.py --case obstacle --controller pd_gravity
    python run_demo.py --case payload  --controller pd_gravity_payload_aware
    python run_demo.py --case obstacle --controller pd_gravity --gif demo.gif   # headless (MUJOCO_GL=egl)

Cases: nominal | obstacle | payload.   Controllers: pd | pd_gravity | pd_gravity_payload_aware
(the last only exists in the payload case).  The viewer runs the real closed loop in real time:
at every step the controller reads the state and sets joint torques; positions are never written.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.experiments import controller_variants  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.physics import PhysicsSimulation  # noqa: E402
from src.tasks import get_task, prepare_task  # noqa: E402


def setup(case, controller):
    task = get_task(case)
    prepared = prepare_task(task)
    variants = controller_variants(task)
    if controller not in variants:
        sys.exit(f"controller '{controller}' not available for case '{case}' (choose from {sorted(variants)})")
    print(prepared.plan.summary())
    print(f"trajectory {prepared.trajectory.duration:.2f} s + {task.hold_time:.1f} s hold; "
          f"direct path blocked: {prepared.direct_blocked}")
    return task, prepared, variants[controller]


def step_loop(sim, ctrl, traj, hold, on_step=None, realtime=False):
    """Generator-free closed loop; returns the recorded joint history (for a final report)."""
    sim.reset(traj.evaluate(0.0)[0])
    total, dt = traj.duration + hold, sim.model.opt.timestep
    t0, k, hist = time.time(), 0, []
    while k * dt < total:
        t = k * dt
        q, qd = sim.state()
        qd_des = traj.evaluate(t)
        sim.apply_torque(ctrl.torque(q, qd, qd_des[0], qd_des[1]))
        sim.step(1)
        hist.append(q)
        if on_step:
            if on_step(k, t) is False:
                break
        if realtime and k % 8 == 0:
            time.sleep(max(0.0, (k + 1) * dt - (time.time() - t0)))
        k += 1
    return np.array(hist)


def run_viewer(sim, ctrl, prepared, hold):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 135, -20, 3.0
        v.cam.lookat[:] = [0.4, 0.0, 0.6]
        step_loop(sim, ctrl, prepared.trajectory, hold,
                  on_step=lambda k, t: (v.sync() if k % 8 == 0 else None) or v.is_running(), realtime=True)
        print("motion finished; close the window to exit")
        while v.is_running():
            time.sleep(0.05)


def run_gif(sim, ctrl, prepared, hold, path, fps=25):
    import mujoco
    from PIL import Image
    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 135, -20, 3.0
    cam.lookat[:] = [0.4, 0.0, 0.6]
    frames, every = [], int(round(1.0 / (fps * sim.model.opt.timestep)))

    def grab(k, t):
        if k % every == 0:
            renderer.update_scene(sim.data, cam)
            frames.append(Image.fromarray(renderer.render()))

    step_loop(sim, ctrl, prepared.trajectory, hold, on_step=grab)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"wrote {path} ({len(frames)} frames)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="nominal", choices=["nominal", "obstacle", "payload"])
    ap.add_argument("--controller", default="pd_gravity")
    ap.add_argument("--gif", help="render headlessly to this GIF instead of opening a window")
    ap.add_argument("--no-viewer", action="store_true", help="just run and print the metrics")
    a = ap.parse_args(argv)
    task, prepared, ctrl = setup(a.case, a.controller)
    sim = PhysicsSimulation(task.exec_scene())
    if a.gif:
        run_gif(sim, ctrl, prepared, task.hold_time, a.gif)
    elif not a.no_viewer:
        try:
            run_viewer(sim, ctrl, prepared, task.hold_time)
        except Exception as exc:
            print(f"[note] viewer unavailable ({type(exc).__name__}: {exc}); use --gif out.gif or --no-viewer")
    from src.execution import run_trajectory
    m = compute_metrics(run_trajectory(sim, ctrl, prepared.trajectory, task.hold_time), prepared.T_target)
    print(f"RMSE {m['rmse_rad_rms_over_joints']:.4f} rad | final EE error {m['final_ee_position_error_m'] * 1e3:.2f} mm | "
          f"peak torque {max(m['torque_peak_Nm']):.1f} N m | collision-free: {m['collision_free']}")


if __name__ == "__main__":
    main()
