# PUMA 560 MuJoCo Model

This project models a PUMA 560 robot arm in MuJoCo using actual link meshes,
validated forward kinematics, and numerical inverse kinematics that searches
for multiple joint configurations.

## Current capabilities

- Seven real PUMA 560 STL link meshes in `models/meshes/`.
- MuJoCo model with six revolute joints and six position actuators.
- PUMA link-frame transforms and dimensions adapted from the mesh-based URDF
  description.
- Forward kinematics returning a 4x4 homogeneous end-effector pose.
- Forward-kinematics regression checks against the MuJoCo `ee_site`.
- Bounded numerical inverse kinematics for position and orientation targets.
- Multiple IK solutions found by deterministic multi-start optimization.
- Duplicate solution filtering using wrapped joint-angle distance.
- Jacobian singular-value and condition-number diagnostics for every IK result.
- Smooth cubic and quintic joint-space trajectory generation.

## Project layout

```text
.
├── models/
│   ├── puma560.xml
│   └── meshes/
│       ├── puma_link1.stl ... puma_link7.stl
│       └── LICENSE
├── src/
│   ├── check_model.py
│   ├── forward_kinematics.py
│   ├── inverse_kinematics.py
│   ├── trajectory_planning.py
│   ├── validate_fk.py
│   └── view_puma560.py
└── mujoco_test.py
```

## Requirements

Python 3.10 or newer is recommended.

Install the numerical and simulation dependencies:

```powershell
python -m pip install numpy scipy mujoco
```

## Run the checks

Load the XML and verify the six-joint model:

```powershell
python src\check_model.py
```

Compare analytical FK against MuJoCo for several joint configurations:

```powershell
python src\validate_fk.py
```

Both checks should finish with a passed message.

## View the robot

Launch the passive MuJoCo viewer:

```powershell
python src\view_puma560.py
```

The model uses the `ee_site` site as the end-effector reference.

## Forward kinematics

Joint angles are expressed in radians. The returned transform is in metres:

```python
import numpy as np

from src.forward_kinematics import forward_kinematics

q = np.array([0.2, -0.3, 0.4, 0.1, -0.2, 0.3])
T = forward_kinematics(q)

position = T[:3, 3]
orientation = T[:3, :3]
```

## Multiple inverse-kinematics solutions

Create a target pose and search for distinct joint configurations:

```python
import numpy as np

from src.forward_kinematics import forward_kinematics
from src.inverse_kinematics import inverse_kinematics

target_joints = np.array([0.2, -0.3, 0.4, 0.1, -0.2, 0.3])
target_pose = forward_kinematics(target_joints)

solutions = inverse_kinematics(
    target_pose,
    initial_guess=target_joints,
)

for solution in solutions:
    print("q:", solution.q)
    print("position error:", solution.position_error)
    print("orientation error:", solution.orientation_error)
    print("smallest Jacobian singular value:",
          solution.smallest_singular_value)
```

The function returns a `list[IKResult]`. Each result contains:

- `q`: six joint angles in radians.
- `position_error`: end-effector position error in metres.
- `orientation_error`: rotation-vector error in radians.
- `iterations`: optimizer function evaluations.
- `converged`: optimizer convergence status.
- `smallest_singular_value`: smallest singular value of the numerical Jacobian.
- `condition_number`: Jacobian condition number.

The command-line demo creates a target pose from supplied joint angles and
prints every solution found:

```powershell
python src\inverse_kinematics.py 0.2 -0.3 0.4 0.1 -0.2 0.3
```

## IK method and limitations

The IK implementation uses bounded SciPy nonlinear least squares. It starts
from a deterministic 3-value seed set for each joint, giving up to
`3^6 = 729` initial guesses, and removes duplicate converged solutions.
Joint limits default to `[-pi, pi]` for all six joints and can be overridden
with the `joint_limits` argument.

This approach is useful for discovering the usual PUMA 560 branches, but
numerical multi-start is not a mathematical proof that every solution exists
or has been found. At singularities, the smallest Jacobian singular value
approaches zero and the condition number becomes large. Solutions should
therefore be checked for both pose error and singularity metrics before being
used for motion planning.

## Joint-space trajectory planning

The first trajectory-planning layer generates smooth point-to-point or
multi-waypoint joint trajectories. It returns sampled position, velocity, and
acceleration arrays:

```python
import numpy as np

from src.trajectory_planning import plan_joint_trajectory

start = np.zeros(6)
goal = np.array([0.4, -0.5, 0.6, 0.3, -0.4, 0.2])

trajectory = plan_joint_trajectory(
    start,
    goal,
    duration=5.0,
    num_samples=201,
    method="quintic",
)

print(trajectory.time)
print(trajectory.position)
print(trajectory.velocity)
print(trajectory.acceleration)
```

Use `method="cubic"` for zero endpoint velocity, or the default
`method="quintic"` for zero endpoint velocity and acceleration. Multiple
segments can be generated with `plan_waypoint_trajectory()`:

```python
from src.trajectory_planning import plan_waypoint_trajectory

waypoints = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.3, -0.2, 0.4, 0.1, 0.0, 0.2],
    [0.5, -0.4, 0.6, 0.2, -0.2, 0.3],
])
trajectory = plan_waypoint_trajectory(
    waypoints,
    durations=[2.0, 3.0],
)
```

The planner validates that every waypoint is inside the configured joint
limits. The current waypoint planner stops at each waypoint, so future work
can add velocity-continuous blending and Cartesian trajectory generation.

## Mesh attribution

The PUMA 560 STL meshes are distributed under the MIT license. The license
notice is retained in [models/meshes/LICENSE](models/meshes/LICENSE).
