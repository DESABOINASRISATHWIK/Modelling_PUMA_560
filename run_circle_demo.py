#!/usr/bin/env python
"""Run the inclined-circle tracking experiment."""
import argparse
from pathlib import Path

from src.circle_tracking import CircleConfig, CircleReference, CircleTrackingSimulation


def _run(controller, config, headless, output, cycles):
    sim = CircleTrackingSimulation(CircleReference(config), controller)
    if headless:
        metrics = sim.run(max_duration=config.duration * cycles)
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 135, -20, 2.4
            viewer.cam.lookat[:] = [0.45, 0.0, 0.70]
            metrics = sim.run(realtime=True, viewer=viewer,
                              max_duration=None if cycles == 0 else config.duration * cycles)
            while viewer.is_running() and cycles != 0:
                viewer.sync()
    sim.save(output)
    print({k: metrics[k] for k in ("cartesian_rmse_m", "cartesian_peak_error_m", "saturation_time_s")})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=("nominal", "fast", "comparison"), default="nominal")
    p.add_argument("--controller", choices=("pd", "pd_gravity"), default="pd_gravity")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--cycles", type=int, default=0,
                   help="number of circle periods; 0 loops until the interactive viewer closes (headless: 1)")
    p.add_argument("--output", default="results/circle/circle_tracking.json")
    a = p.parse_args(argv)
    if a.cycles < 0:
        p.error("--cycles must be non-negative")
    cycles = 1 if a.headless and a.cycles == 0 else a.cycles
    duration = 8.0 if a.case != "fast" else 5.33
    config = CircleConfig(duration=duration)
    if a.case == "comparison":
        root = Path(a.output)
        for name in ("pd", "pd_gravity"):
            _run(name, config, True, root.with_name(f"{root.stem}_{name}{root.suffix}"), 1)
    else:
        _run(a.controller, config, a.headless, a.output, cycles)


if __name__ == "__main__":
    main()
