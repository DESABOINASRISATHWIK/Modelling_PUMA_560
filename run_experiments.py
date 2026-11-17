#!/usr/bin/env python
"""Run the project's simulation experiments and save raw logs (Part 4).

    python run_experiments.py                    # everything (about 1-3 minutes)
    python run_experiments.py --exp nominal obstacle payload
    python run_experiments.py --exp planner --seeds 30
    python run_experiments.py --out results/experiments

Experiments
  nominal   free space, nominal model                       PD vs PD + gravity
  obstacle  pillar blocks the direct joint path; RRT-Connect  PD vs PD + gravity (executed contacts checked)
  payload   2 kg payload, controller model NOT updated      PD vs PD+G (nominal) vs PD+G (payload-aware, extra)
  planner   RRT-Connect vs RRT over many seeds on the obstacle task (planning only)

Raw data: <out>/<experiment>/*.npz (one per controller), config.json, metrics.json,
          <out>/planner_study.json.  Then run  python analyse_results.py
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.experiments import planner_study, run_case  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", default=["all"], choices=["all", "nominal", "obstacle", "payload", "planner"])
    ap.add_argument("--out", default=str(ROOT / "results" / "experiments"))
    ap.add_argument("--seeds", type=int, default=20, help="number of planner seeds for the planner study")
    a = ap.parse_args(argv)
    chosen = ["nominal", "obstacle", "payload", "planner"] if "all" in a.exp else a.exp
    for name in chosen:
        print(f"== {name}")
        if name == "planner":
            planner_study(seeds=range(a.seeds), out_dir=a.out)
        else:
            run_case(name, out_dir=a.out)
    print(f"\nraw logs written to {a.out}\nnext: python analyse_results.py")


if __name__ == "__main__":
    main()
