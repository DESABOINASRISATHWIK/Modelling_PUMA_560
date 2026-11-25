# PUMA 560: Kinematics, Planning, and Torque Control

Simulation project for a six-axis PUMA 560 manipulator. The core algorithms are implemented with Python and NumPy; MuJoCo supplies rigid-body simulation, contact queries, visualization, and independent kinematics checks.

## Included work

- Standard-DH forward kinematics, geometric Jacobian, singularity analysis, and closed-form eight-branch inverse kinematics.
- Damped-least-squares resolved-rate inverse kinematics.
- Joint-space planning with direct interpolation, RRT, RRT-Connect, collision checking, shortcut smoothing, and quintic trajectories.
- Torque control with PD and PD plus DH-model gravity compensation.
- Nominal, obstacle, payload, circle-tracking, and conveyor pick-and-place simulations.

## Setup

Requires Python 3.10 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Reproduce the reported experiments

```powershell
python -m pytest tests -q
python run_experiments.py
python analyse_results.py
```

The experiment runner writes raw logs to `results/experiments/`; the analysis script regenerates the figures in `results/figures/` and the comparison table.

## Run demonstrations

```powershell
# Interactive torque-controlled simulation
python run_demo.py --case obstacle --controller pd_gravity

# Headless metrics-only run
python run_demo.py --case payload --controller pd_gravity_payload_aware --no-viewer

# Kinematics and numerical verification
python examples/verification_report.py

# Presentation demonstrations
python run_presentation.py --show obstacle --planner rrt_connect --controller pd_gravity
python run_presentation.py --show circle --case comparison --headless
python run_presentation.py --show conveyor --headless --seconds 60
```

For a headless Linux renderer, set `MUJOCO_GL=egl` before running a command that renders images or video.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | Project implementation: kinematics, planning, trajectories, control, simulation, and evaluation. |
| `models/` | MuJoCo model and required visual meshes. |
| `tools/` | Model-generation and controller-tuning utilities. |
| `tests/` | Automated verification tests. |
| `examples/` | Small focused demonstrations and verification scripts. |
| `results/` | Reproducible raw logs and figures used by the technical report. |

The model provenance and mesh licence information are retained in `models/LICENSE_NOTICE.md`.
