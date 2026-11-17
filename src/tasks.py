"""Task definition and the planning pipeline (Part 2 -> Part 3 hand-over).

A task is: a start joint configuration, a Cartesian TARGET POSE for the end effector,
an optional obstacle / payload, and the planner / trajectory settings.  ``prepare_task``
turns it into an executable, collision-checked timed trajectory:

    target pose --IK--> candidate branches --joint limits + collision--> best goal q_goal
    (q_start, q_goal) --planner (direct, else RRT-Connect + shortcut)--> waypoints (K, 6)
    waypoints --quintic segments--> q_des(t), qd_des(t), qdd_des(t)
    trajectory --dense collision re-check against the TRUE obstacle geometry--> verdict

Safety margin: the planner works on a scene whose obstacles are inflated by
``obstacle_margin`` [m] on every side, so that small tracking errors and the discrete
edge sampling cannot turn a planned path into a real contact.  Execution and the final
re-check use the true (un-inflated) obstacles.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .collision import CollisionChecker
from .forward_kinematics import forward_kinematics
from .inverse_kinematics import inverse_kinematics
from .path_planning import PlanResult, make_planner, path_length
from .scene import Obstacle, Payload, SceneConfig, load_scene
from .trajectory import QuinticTrajectory
from .utils import rpy_to_matrix


class TaskError(RuntimeError):
    """The task cannot be executed (unreachable, joint-limit infeasible, blocked, ...)."""


# ---------------------------------------------------------------------------------------
# task description
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskConfig:
    name: str
    description: str
    q_start: tuple
    target_xyz: tuple                      # [m]  flange position in the base frame
    target_rpy_deg: tuple                  # [deg] roll, pitch, yaw (R = Rz Ry Rx)
    obstacles: tuple = ()
    payload: Payload | None = None
    planner: str = "rrt_connect"           # used only when the direct joint path is blocked
    planner_seed: int = 0
    planner_kwargs: dict = field(default_factory=dict)
    obstacle_margin: float = 0.03          # [m] planning inflation
    v_max: tuple = (0.8, 0.8, 0.8, 1.5, 1.5, 2.0)        # [rad/s] trajectory velocity limits
    a_max: tuple = (1.5, 1.5, 1.5, 5.0, 5.0, 8.0)        # [rad/s^2] trajectory acceleration limits
    hold_time: float = 2.0                 # [s] final hold used for settling
    scene_overrides: dict = field(default_factory=dict)   # e.g. {"damping": (...,)} for robot-parameter changes

    @property
    def T_target(self):
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(*np.radians(self.target_rpy_deg))
        T[:3, 3] = self.target_xyz
        return T

    def exec_scene(self) -> SceneConfig:
        """The REAL scene used for physics execution and final collision verification."""
        return SceneConfig(obstacles=tuple(self.obstacles), payload=self.payload, **self.scene_overrides)

    def plan_scene(self) -> SceneConfig:
        """Same scene with every obstacle inflated by ``obstacle_margin`` (planning only)."""
        m = float(self.obstacle_margin)
        inflated = tuple(Obstacle(o.position, tuple(np.asarray(o.size) + m)) for o in self.obstacles)
        return replace(self.exec_scene(), obstacles=inflated)


@dataclass
class PreparedTask:
    task: TaskConfig
    q_start: np.ndarray
    q_goal: np.ndarray
    goal_info: dict
    direct_blocked: bool
    plan: PlanResult
    trajectory: QuinticTrajectory
    trajectory_collision_free: bool
    trajectory_first_collision_time: float | None
    min_obstacle_clearance_m: float | None
    T_target: np.ndarray
    waypoints: np.ndarray = field(init=False)

    def __post_init__(self):
        self.waypoints = self.plan.path


# ---------------------------------------------------------------------------------------
# goal selection
# ---------------------------------------------------------------------------------------
def select_goal(T_target, q_start, checker):
    """Pick the IK branch that is inside joint limits, collision-free and closest to ``q_start``.

    Returns ``(q_goal, info)``; raises ``TaskError`` with the reason if no branch qualifies.
    """
    res = inverse_kinematics(T_target, q_ref=q_start)
    info = dict(ik_status=res.status, n_branches=len(res.solutions), n_within_limits=0,
                n_collision_free=0, chosen_branch=None, chosen_label=None)
    if not res.solutions:
        raise TaskError(f"target pose is unreachable ({res.status}); no IK solution")
    best, best_d = None, np.inf
    for i, s in enumerate(res.solutions):
        if not s.within_limits:
            continue
        info["n_within_limits"] += 1
        if not checker.is_configuration_valid(s.q):
            continue
        info["n_collision_free"] += 1
        d = float(np.linalg.norm(s.q - q_start))
        if d < best_d:
            best, best_d = i, d
    if best is None:
        why = ("every IK branch violates a joint limit" if info["n_within_limits"] == 0
               else "every in-limit IK branch is in collision")
        raise TaskError(f"no feasible goal configuration: {why}")
    s = res.solutions[best]
    info.update(chosen_branch=best, chosen_label=s.label, joint_distance_rad=best_d)
    return s.q.copy(), info


# ---------------------------------------------------------------------------------------
# trajectory re-check against the true geometry
# ---------------------------------------------------------------------------------------
def validate_trajectory(checker, trajectory, sample_dt=0.02, resolution=0.01):
    """Dense collision check of the SAMPLED, TIMED trajectory (not just its waypoints).

    Returns ``(ok, first_collision_time_or_None)``.  Samples are ``sample_dt`` apart and
    consecutive samples are additionally interpolated at ``resolution`` rad.
    """
    times = np.append(np.arange(0.0, trajectory.duration, sample_dt), trajectory.duration)
    q = np.vstack([trajectory.evaluate(t)[0] for t in times])
    for i in range(len(q) - 1):
        if not checker.is_edge_valid(q[i], q[i + 1], resolution):
            return False, float(times[i])
    return True, None


def min_clearance(checker, trajectory, n=60):
    """Smallest robot-obstacle distance [m] sampled along the trajectory (None without obstacles)."""
    if not checker.obstacle_geoms:
        return None
    times = np.linspace(0.0, trajectory.duration, n)
    return float(min(checker.obstacle_clearance(trajectory.evaluate(t)[0]) for t in times))


# ---------------------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------------------
def prepare_task(task, planner=None, seed=None, **planner_overrides):
    """Goal selection -> planning -> time scaling -> collision verification."""
    planner_name = planner or task.planner
    seed = task.planner_seed if seed is None else int(seed)
    plan_checker = CollisionChecker(load_scene(task.plan_scene()))
    true_checker = CollisionChecker(load_scene(task.exec_scene()))

    q_start = np.asarray(task.q_start, dtype=float)
    if not plan_checker.is_configuration_valid(q_start):
        raise TaskError("start configuration is outside the joint limits or in collision")
    T_target = task.T_target
    q_goal, goal_info = select_goal(T_target, q_start, plan_checker)

    direct_blocked = not plan_checker.is_edge_valid(q_start, q_goal, 0.02)
    kwargs = dict(task.planner_kwargs)
    kwargs.update(planner_overrides)
    plan = make_planner(planner_name, plan_checker, seed=seed, **kwargs).solve(q_start, q_goal)
    if not plan.success:
        raise TaskError(f"planning failed: {plan.status} ({plan.summary()})")

    traj = QuinticTrajectory(plan.path, dt=0.002, max_velocity=np.asarray(task.v_max),
                             max_acceleration=np.asarray(task.a_max))
    ok, t_hit = validate_trajectory(true_checker, traj)
    clearance = min_clearance(true_checker, traj)
    return PreparedTask(task, q_start, q_goal, goal_info, direct_blocked, plan, traj, ok, t_hit, clearance, T_target)


# ---------------------------------------------------------------------------------------
# the project's experiment tasks
# ---------------------------------------------------------------------------------------
# Pick-and-place sweep: tool pointing down, from the left of the workspace to the right.
Q_START = (-0.6, 0.4, -2.5, 0.0, -1.0, 0.0)
# Target = FK of the configuration (0.6, 0.6, -2.3, 0.3, -0.9, 0.5), rounded; IK recovers it.
TARGET_XYZ = (0.7420, 0.3371, 0.8070)
TARGET_RPY_DEG = (149.149, 16.722, 174.094)

# A floor-standing pillar (half extents) in the middle of the sweep: blocks the direct path.
PILLAR = Obstacle(position=(0.76, -0.15, 0.35), size=(0.10, 0.10, 0.35))
# 2 kg box 6 cm beyond the flange (teaching choice; PUMA 560 literature quotes a ~2.5 kg rating).
PAYLOAD_2KG = Payload(mass=2.0, position=(0.0, 0.0, 0.06), size=(0.04, 0.04, 0.04))


def nominal_task():
    return TaskConfig("nominal", "Free-space motion, nominal model, moderate speed",
                      Q_START, TARGET_XYZ, TARGET_RPY_DEG)


def obstacle_task():
    return TaskConfig("obstacle", "Pillar blocks the direct joint path; RRT-Connect finds a detour",
                      Q_START, TARGET_XYZ, TARGET_RPY_DEG, obstacles=(PILLAR,))


def payload_task():
    return TaskConfig("payload", "Nominal motion carrying a 2 kg payload (model mismatch for the controller)",
                      Q_START, TARGET_XYZ, TARGET_RPY_DEG, payload=PAYLOAD_2KG)


TASKS = {"nominal": nominal_task, "obstacle": obstacle_task, "payload": payload_task}


def get_task(name):
    if name not in TASKS:
        raise ValueError(f"unknown task '{name}' (choose from {sorted(TASKS)})")
    return TASKS[name]()
