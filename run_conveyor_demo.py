#!/usr/bin/env python
"""Run the nominal stop-and-go PUMA conveyor pick-and-place simulation.

Interactive, until the viewer closes:
    python run_conveyor_demo.py

Headless bounded run:
    python run_conveyor_demo.py --headless --seconds 60
"""
import argparse
import time

from src.conveyor import ConveyorConfig, ConveyorSimulation


def concise_status(sim):
    status = sim.status
    return {key: status[key] for key in ("sim_time_s", "state", "spawned", "picked", "placed",
                                         "missed", "failed_grasps", "unexpected_obstacle_contacts", "fault")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without opening a viewer")
    parser.add_argument("--seconds", type=float, default=60.0, help="simulation duration; 0 runs until closed")
    parser.add_argument("--belt-speed", type=float, default=0.05, help="belt indexing speed [m/s]")
    parser.add_argument("--belt-ramp-time", type=float, default=0.5, help="smooth belt start/stop duration [s]")
    parser.add_argument("--spawn-interval", type=float, default=10.0, help="seconds between spawn attempts")
    parser.add_argument("--spawn-policy", choices=("ready", "paced"), default="ready",
                        help="ready feeds one item when the robot is home; paced uses a fixed interval")
    parser.add_argument("--max-items", type=int, default=5, help="fixed pool size")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="results/presentation/conveyor/latest.json")
    args = parser.parse_args(argv)
    if args.seconds < 0:
        parser.error("--seconds must be non-negative")
    config = ConveyorConfig(belt_speed=args.belt_speed, belt_ramp_time=args.belt_ramp_time,
                            spawn_interval=args.spawn_interval,
                            spawn_policy=args.spawn_policy, max_items=args.max_items, seed=args.seed,
                            obstacles=())
    sim = ConveyorSimulation(config)

    if args.headless:
        while args.seconds == 0 or sim.sim_time < args.seconds:
            sim.step()
            if int(sim.sim_time * 10) % 100 == 0 and abs(sim.sim_time * 10 - round(sim.sim_time * 10)) < 1e-9:
                print(concise_status(sim))
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            wall_start = time.perf_counter()
            while viewer.is_running() and (args.seconds == 0 or sim.sim_time < args.seconds):
                sim.step()
                viewer.sync()
                # Keep the viewer at the model timestep rather than running as fast
                # as possible; the conveyor/task clocks are simulation time.
                target = sim.sim_time
                delay = target - (time.perf_counter() - wall_start)
                if delay > 0:
                    time.sleep(min(delay, 0.01))
                if int(sim.sim_time) and abs(sim.sim_time - round(sim.sim_time)) < sim.dt / 2:
                    print(concise_status(sim))
    sim.save_log(args.output)
    print("Final:", concise_status(sim))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
