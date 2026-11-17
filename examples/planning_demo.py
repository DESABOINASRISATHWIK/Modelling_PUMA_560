"""Part 2 demo: pose target -> goal selection -> direct path / RRT-Connect around an obstacle.

Run:  python examples/planning_demo.py
(The previous version crashed: it did not use _common for the import path, and its start
pose was in self-collision, so the planner correctly rejected it.)
"""
import _common  # noqa: F401  (adds the project root to sys.path)
import numpy as np

from src.collision import CollisionChecker
from src.path_planning import direct_joint_path
from src.scene import load_scene
from src.tasks import get_task, prepare_task

np.set_printoptions(precision=3, suppress=True, linewidth=120)
for name in ("nominal", "obstacle"):
    task = get_task(name)
    p = prepare_task(task)
    true_checker = CollisionChecker(load_scene(task.exec_scene()))
    print(f"\n=== {name}: {task.description}")
    print("start q [rad]       :", p.q_start)
    print("target xyz [m]      :", task.target_xyz, " rpy [deg]:", task.target_rpy_deg)
    print("IK goal selection   :", p.goal_info)
    print("goal q [rad]        :", p.q_goal)
    print("direct path blocked :", p.direct_blocked,
          "| direct path valid in TRUE scene:", direct_joint_path(true_checker, p.q_start, p.q_goal) is not None)
    print("plan                :", p.plan.summary())
    print("waypoints           :\n", p.waypoints)
    print(f"trajectory          : {p.trajectory.duration:.2f} s, collision-free vs true geometry: "
          f"{p.trajectory_collision_free}, min clearance: {p.min_obstacle_clearance_m}")
