#!/usr/bin/env python
"""One entry point for the live project presentation.

Examples
--------
Open the current nominal conveyor demonstration:
    python run_presentation.py --show conveyor

Run it without a viewer and save a deterministic log:
    python run_presentation.py --show conveyor --headless --seconds 60

Open the existing torque-control/planning demonstration:
    python run_presentation.py --show control --case nominal --controller pd_gravity

The circle-tracking and dedicated planner demonstrations will be registered
here in their respective implementation phases.  Keeping this dispatcher
small avoids several presentation commands with incompatible arguments.
"""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _conveyor_args(args):
    duration = 60.0 if args.headless and args.seconds == 0.0 else args.seconds
    forwarded = ["--seconds", str(duration), "--belt-speed", str(args.belt_speed),
                 "--belt-ramp-time", str(args.belt_ramp_time), "--spawn-policy", args.spawn_policy,
                 "--max-items", str(args.max_items), "--seed", str(args.seed),
                 "--output", args.output or str(ROOT / "results" / "presentation" / "conveyor" / "latest.json")]
    if args.headless:
        forwarded.append("--headless")
    return forwarded


def _control_args(args):
    forwarded = ["--case", args.case, "--controller", args.controller]
    if args.headless:
        forwarded.append("--no-viewer")
    return forwarded


def _circle_args(args):
    forwarded = ["--case", args.case, "--controller", args.controller,
                 "--output", args.output or str(ROOT / "results" / "presentation" / "circle" / "latest.json")]
    if args.headless:
        forwarded.append("--headless")
    return forwarded


def _obstacle_args(args):
    forwarded = ["--planner", args.planner, "--controller", args.controller,
                 "--output", args.output or str(ROOT / "results" / "presentation" / "obstacle" / "latest.json")]
    if args.headless:
        forwarded.append("--headless")
    return forwarded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", required=True, choices=("conveyor", "control", "circle", "obstacle"),
                        help="presentation demonstration to run")
    parser.add_argument("--headless", action="store_true", help="run without opening a MuJoCo viewer")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="conveyor duration in simulation seconds; 0 runs until the viewer closes")
    parser.add_argument("--belt-speed", type=float, default=0.05, help="conveyor indexing speed [m/s]")
    parser.add_argument("--belt-ramp-time", type=float, default=0.5, help="conveyor start/stop ramp [s]")
    parser.add_argument("--spawn-policy", choices=("ready", "paced"), default="ready")
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", help="conveyor JSON output path")
    parser.add_argument("--planner", choices=("rrt", "rrt_connect"), default="rrt_connect",
                        help="planner for --show obstacle")
    parser.add_argument("--case", choices=("nominal", "obstacle", "payload", "fast", "comparison"), default="nominal",
                        help="case for --show control or --show circle")
    parser.add_argument("--controller", default="pd_gravity", choices=("pd", "pd_gravity", "pd_gravity_payload_aware"),
                        help="controller for --show control or --show circle")
    args = parser.parse_args(argv)

    if args.seconds < 0:
        parser.error("--seconds must be non-negative")
    if args.show == "conveyor":
        from run_conveyor_demo import main as run_conveyor
        return run_conveyor(_conveyor_args(args))
    if args.show == "circle":
        from run_circle_demo import main as run_circle
        # Circle accepts only its three task cases and two controller labels.
        if args.case in ("obstacle", "payload"):
            parser.error("--show circle accepts --case nominal, fast, or comparison")
        if args.controller == "pd_gravity_payload_aware":
            parser.error("--show circle accepts --controller pd or pd_gravity")
        return run_circle(_circle_args(args))
    if args.show == "obstacle":
        from run_obstacle_demo import main as run_obstacle
        if args.controller == "pd_gravity_payload_aware":
            parser.error("--show obstacle accepts --controller pd or pd_gravity")
        return run_obstacle(_obstacle_args(args))

    from run_demo import main as run_control
    if args.case in ("fast", "comparison"):
        parser.error("--show control accepts --case nominal, obstacle, or payload")
    return run_control(_control_args(args))


if __name__ == "__main__":
    main()
