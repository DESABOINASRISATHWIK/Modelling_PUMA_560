# Part 1: physics, collision and gravity interface

Use `from src.simulation import PhysicsSimulation` for dynamics. The existing
`PumaSimulation` class and `python -m src.simulation` remain the kinematics viewer.
The validated `models/puma560.xml` supplies the same robot tree, joints, inertial
transforms, and end-effector site to both. `src/scene.py` adds physics settings,
collision proxies, torque motors, optional world boxes and a fixed payload body.

## Shared conventions and API

Joint order is waist, shoulder, elbow, forearm roll, wrist pitch, wrist roll
(`joint1` through `joint6`). All API angles are radians; lengths are metres,
mass kilograms, time seconds, torque N m, inertia kg m². World equals DH base,
z points up. Positive angles and zeros follow `robot_parameters.py`.
`ee_site` is the flange: `T_6E = Trans(z, 0.04 m)`.
Limits come from the validated model, matching `JOINT_LIMITS`.

```python
from src.simulation import PhysicsSimulation
from src.scene import SceneConfig, Obstacle, Payload
from src.collision import CollisionChecker
from src.gravity import GravityCompensator

sim = PhysicsSimulation(SceneConfig(
    obstacles=(Obstacle(position=(0.8, 0.3, 0.8), size=(0.1, 0.1, 0.2)),),
    payload=Payload(mass=0.5)))
planner = CollisionChecker(sim.model)
gravity = GravityCompensator(sim.model)
q, qvel = sim.state()                 # copies, rad and rad/s
assert planner.is_configuration_valid(q)
accepted_tau = sim.apply_torque(gravity.torque(q))
sim.step(10)                         # integrates 10 fixed timesteps
T_world_flange = sim.end_effector_pose()  # 4x4 copy
sim.reset(q, qvel)                   # deliberate state assignment, time/control reset
```

`reset()` defaults to zero position and velocity and rejects joint-limit violations.
Execution only changes positions through `mj_step`; calling `reset` is reserved for
initialization or a deliberate restart. `apply_torque` holds its command until the
next call or reset. `step` refreshes derived poses after integration. Model and data
are exposed for MuJoCo viewer integration; do not mutate a shared model while a
planner or gravity query is running. Query objects own separate data, and each
query object should be used by one thread at a time.

Default timestep is 0.002 s with `implicitfast` integration. Each actuator is a
direct hinge motor: actuator force = control and joint torque = gear × control.
Defaults are unit gears and joint torque limits `[150,150,100,30,20,10]` N m.
`apply_torque` clips at these joint torque limits then divides by signed gear.
Control and actuator-force limits are ±(joint torque limit / |gear|), so changing
gear preserves joint torque bounds. No motor electrical model is implied.

## Model assumptions

Published parameter provenance and the original corrections are recorded in
`models/LICENSE_NOTICE.md`. Tests check compiled masses, COMs and inertia moments.
These are a literature parameter set, not measurements of a particular robot.

* Link 1 uses the existing 0.001 kg placeholder and principal inertias
  `[0.18, 0.18, 0.35]` kg m² in world-aligned zero-pose axes.
* Link 3's smallest principal moment is increased from 0.0125 to 0.0201 kg m²
  to satisfy the inertia triangle inequality. Other link inertias are preserved.
* Gravity defaults to `[0,0,-9.81]` m/s² and is configurable.
* Joint damping defaults to 1 N m s/rad on each joint. It is an assumed viscous
  coefficient. Coulomb friction, rotor inertia, elasticity and motor dynamics
  are omitted. Torque limits and timestep are teaching choices, not PUMA ratings.
* Collision capsules/spheres and base cylinder are assumed simplified envelopes,
  not measured physical surfaces. Visual meshes stay noncolliding. Their transparent
  proxies use geom group 3 and do not change explicitly specified inertias.
* Payload is an axis-aligned box in flange coordinates, rigidly fixed with no extra
  DOFs. Position specifies its COM relative to the flange; size is half extent.
  Its uniform-box inertia is calculated explicitly. Default payload is absent.
* Obstacles are static world boxes with half extents. `SceneConfig()` is nominal;
  `SceneConfig.obstacle_scene()` supplies one example obstacle. Collision-enabled
  geoms use MuJoCo default contact friction/solver settings; these are assumptions.

Run `python tools/build_scenes.py` to regenerate nominal/obstacle XMLs. Runtime
configurations use `load_scene` and do not rewrite the source robot XML.

## Collision checking

`CollisionChecker(model).is_configuration_valid(q)` checks limits and contacts.
`contacts(q)` returns touching or penetrating geom-name pairs (it does not check
joint limits). Floor/base/obstacle/payload and nonadjacent robot contacts are included.
Adjacent robot bodies are explicitly excluded as intended articulation contacts;
payload/last-link mounting contact is excluded. The environment floor and base
are both static, so their mutual contact is not relevant.
The filtering follows [MuJoCo's collision model](https://mujoco.readthedocs.io/en/stable/computation/#collision-detection).

`is_edge_valid(start, end, resolution=0.02)` linearly interpolates joint angles,
checking endpoints and intermediate samples. The largest increment in any joint
is at most resolution radians. This is sampled checking, not continuous collision
detection: thin obstacles can require smaller resolution. It does not wrap angles,
since physical joint limits apply. Invalid shape/nonfinite input raises ValueError;
out-of-limit or colliding configurations return False.

These geometric checks do not certify a controller's actual tracked trajectory.
Planner and gravity queries do not touch live qpos, qvel, controls, time or poses.

## Dynamics and student gravity calculation

The nominal rigid-body equation is

`M(q) qdd + C(q,qvel) qvel + g(q) = tau`.

Here M is joint-space inertia, C qvel is the velocity-dependent bias and g is
the compensating gravity torque. Actual simulation also includes viscous damping,
joint-limit constraints and contact forces. MuJoCo evaluates dynamics and integrates
the state; no student inverse-dynamics solver is claimed.

For a body COM c and ancestor revolute joint i with world axis z_i through o_i,
`Jv_c[:,i] = cross(z_i, c-o_i)`; nonancestor columns are zero. A gravitational
force `m*g_world` produces generalized force `Jv_c.T @ (m*g_world)`. Therefore
the command that cancels gravity is its negative, summed over all moving bodies:

`g_hat(q) = -sum(Jv_c.T @ (m*g_world))`.

`GravityCompensator.torque(q)` implements this cross-product sum itself, using
MuJoCo only for world COM/axis/anchor transforms and compiled masses. It includes
the fixed payload and the documented placeholder. It does not call inverse dynamics,
MuJoCo's Jacobian routines or qfrc_bias to compute its answer.

For Part 3, use `tau = Kp*(q_des-q) + Kd*(qvel_des-qvel) + g_hat(q)` and pass
the result to `apply_torque`. With z-up and downward gravity this sign holds the
arm rather than doubling gravity. At zero velocity `qfrc_bias` equals the gravity
term and is the validation reference; at nonzero velocity it also contains Coriolis
and centrifugal effects and should not be interpreted as gravity alone.

## Verification

Run `python -m pytest tests -q` and `python examples/physics_demo.py`.
`tests/test_physics.py` checks torque clipping and signed nonunit gears, integrated
motion, FK after stepping, gravity against simulator bias at random poses (with
and without payload and tilted gravity), static gravity hold, payload gravity
changes, floor/self/obstacle collisions, a blocked edge with clear endpoints,
limit/input rejection, scene export and preservation of live simulation state.
The demo executes PD plus gravity for three seconds without writing joint positions.

Validated on 2026-11-10 using Python 3.12 and MuJoCo 4.0.0: **63 tests passed**,
including eight Part 1 acceptance cases. The payload PD demo completed 3 s with
maximum final joint error 9.95e-7 rad and maximum speed 4.73e-6 rad/s. Both generated
scene XMLs compile with six torque actuators. Collision results apply to the simplified
proxies; they do not certify clearance of every visual mesh surface or real hardware.
