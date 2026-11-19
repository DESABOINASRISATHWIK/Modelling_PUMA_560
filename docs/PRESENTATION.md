# Presentation runner and phased implementation

Use the course environment from the project root:

```powershell
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' run_presentation.py --show conveyor
```

The command opens the nominal conveyor scene. It contains no obstacle. Close
the MuJoCo viewer to stop it. The run log is written to
`results/presentation/conveyor/latest.json`.

For a deterministic check that does not open a viewer:

```powershell
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' run_presentation.py --show conveyor --headless --seconds 60
```

Without `--seconds`, a headless conveyor run defaults to 60 simulation seconds;
an interactive run continues until the viewer is closed.

The existing torque-control demonstration remains accessible through the same
entry point:

```powershell
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' run_presentation.py --show control --case nominal --controller pd_gravity
```

For the inclined-circle experiment:

```powershell
# Interactive viewer with the visible 20 cm-diameter green reference ring.
# It repeats until you close the viewer.
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' run_presentation.py --show circle --case nominal --controller pd_gravity

# Saved PD versus gravity-compensated-PD comparison
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' run_presentation.py --show circle --case comparison --headless
```

## Phases

1. **Shared presentation interface and nominal conveyor cleanup** — complete.
   The nominal conveyor invocation has no obstacle flags or obstacle geometry.
2. **Inclined-circle tracking** — complete. `--show circle` renders a
   non-colliding tilted reference ring and executes NumPy FK/IK/Jacobian-DLS
   references with PD or gravity-compensated PD.
3. **Robot-centred conveyor rebuild** — complete. The belt is in front of the
   rotated base at `y=-0.65 m`; heavy/light bins are at negative/positive `x`; every
   transfer releases above a rim, verifies the gravity drop, retracts
   vertically, and then returns to a work-side standby pose.
4. **Dedicated obstacle-planning demonstration** — add a separate blocked
   transfer scene and selectable RRT/RRT-Connect planners — complete. Use
   `run_presentation.py --show obstacle --planner rrt_connect`; the red pillar
   blocks the direct edge and the green line is the NumPy-planned detour.
5. **Final presentation bundle** — register all three demonstrations with
   `run_presentation.py` and collect their result artifacts in one run folder.
