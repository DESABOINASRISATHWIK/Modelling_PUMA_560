# Stop-and-go conveyor pick-and-place

This is the first reliable conveyor task baseline for the PUMA simulation. It indexes one block to a marked station, smoothly stops it, waits for the arm to settle, and then picks and routes it by type to one of two bins.

## Run

From the project root, use the course venv:

```powershell
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' run_presentation.py --show conveyor
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' run_presentation.py --show conveyor --headless --seconds 60
```

The viewer runs until closed. Headless runs save a JSON event log and time series under `results/presentation/conveyor/latest.json`. `--belt-speed` defaults to 0.05 m/s and `--belt-ramp-time` to 0.5 s. The fixed-rate `paced` spawn policy remains available for stress tests; `ready` is the default.

## Task and control sequence

1. Spawn a free-joint cuboid at the infeed with its belt-direction velocity initially zero.
2. Ramp the scripted belt speed up and cruise along the x-axis toward the pickup mark at `pickup_x`.
3. Start a quintic S-curve deceleration early enough to stop at the mark. The arm does not descend until the belt reports stopped and the item is at the mark.
4. Move to pregrasp, descend, then hold the tool at the grasp pose until its measured linear speed is below 0.015 m/s and position error below 5 mm for 80 ms.
5. Close the two fingers. A grasp is accepted only after both finger geoms contact the item and both measured contact-normal forces exceed the mass-based threshold.
6. Apply payload-aware gravity-compensated PD control to lift, carry, release into the item's configured bin, and return.
7. Reuse item slots after the retention delay. The item pool remains fixed at `max_items` bodies.

The two compiled blocks are 0.12 kg (blue, bin 0) and 0.28 kg (orange, bin 1). The nominal required normal force per finger is

`N_required = 1.5 * m * (9.81 + 1.0) / (2 * mu)`

using the modeled effective friction coefficient `mu = 1.0` and a 1.5 safety factor. This gives 0.973 N per finger for the light block and 2.270 N for the heavy block. Grasp events log the actual MuJoCo contact-normal forces, relative position and velocity, and belt-axis velocity error. The JSON also records applied motor torques, belt ramp state, miss reasons, contact penetration, and placement error.

The type is **known from the simulated spawn metadata**: `_spawn` selects a compiled `ItemSpec`, and that slot's `mass`, dimensions, color, and `bin_id` drive the gripper-force threshold, payload model, and destination. The robot does not identify the type from measured torque or infer mass from the grasp. A real-robot version still needs a sensor-based classifier.

The robot installation is yawed by 180° about world `z`; its forward direction is therefore world `-y` and its right direction is world `-x`. The belt centreline is at `y=-0.65 m` and runs from `x=-0.40 m` to the pickup mark `P=(0, -0.65) m`. Bin 0 (blue/light, robot right) is at `( -0.45, -0.05 ) m`; bin 1 (orange/heavy, robot left) is at `( +0.45, -0.05 ) m`. Their 24 cm outer footprints do not overlap, and neither occupies the conveyor. The table top is `z=0.34 m`; the bin floors are raised to `z=0.35 m`, so their bases do not overlap the table. Each bin has a physical floor and rim for released blocks.

Transfer motion rises above a rim, moves horizontally to the bin centre, and releases the block with its bottom 40 mm above the rim. It never inserts the fingers into a bin. The state machine then waits for a floor contact and a low block velocity before counting a placement, rises vertically by 100 mm over the bin centre, and only then returns to standby. The conveyor uses one regular standby for each elbow branch: the light/right bin starts from `Q_CONVEYOR_STANDBY_RIGHT = [1.911, 1.211, -3.625, 0, -0.728, 1.911]`; the heavy/left bin uses `Q_CONVEYOR_STANDBY_LEFT = [-1.911, 1.931, 0.577, 0, 0.634, -1.911]`. They have the same work-side flange pose, but avoid switching elbow branches while holding an item. The original library `Q_READY` is preserved for kinematics examples; it is exactly wrist-singular because q5=0.

The finger target uses the actual 116 mm open inner gap and block width along the closing axis. It commands 2 mm total nominal pad compression instead of the old 4 mm geometric overclosure. Pad/item contact uses a stiffer solver setting; the measured maximum finger-item penetration in the current 40-second run was about 2.1 mm.

## Current verification and limits

The no-obstacle stop-go regression at 0.05 m/s checks the physical grasp, gravity drop, bin-floor contact, placement error, vertical retract, grasp forces, and contact penetration. The task starts and returns to `Q_CONVEYOR_STANDBY`.

The belt motion is still prescribed kinematically in Python; it is not a friction-driven moving surface. Thus the smooth ramps prevent scripted velocity jumps but do not yet validate real belt friction or toppling under physical acceleration. The contact model and friction coefficient are idealized, and the robot torque caps remain generic rather than verified PUMA motor limits.

The conveyor task excludes the link-2/link-4 collision-proxy pair because those simplified capsules overlap at ready/pick configurations and create a false blocking contact. The dedicated obstacle-planning experiment is separate from this nominal pick-and-place scene.

## Next project extensions

1. Add placement poses for a stack or pattern, and measure placement repeatability and stack stability.
3. Compare stop-and-go with continuous moving-belt interception at 0.05 and 0.10 m/s after velocity and contact behavior are validated.
4. Repeat nominal and challenging conditions across seeds and payloads; report cycle time, tracking error, joint effort/saturation, grasp success, placement accuracy, and failures.
