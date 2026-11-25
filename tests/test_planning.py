"""Part 2: planners, result objects, reproducibility, failure handling, task pipeline."""
import numpy as np
import pytest

from src.collision import CollisionChecker
from src.path_planning import (INVALID_GOAL, INVALID_START, OK_DIRECT, RRTConnectPlanner, RRTPlanner,
                               direct_joint_path, make_planner, path_length)
from src.scene import load_scene
from src.tasks import TaskError, get_task, obstacle_task, prepare_task, select_goal

QA = np.array(get_task("nominal").q_start)


@pytest.fixture(scope="module")
def obstacle_prepared():
    return prepare_task(obstacle_task())


def test_direct_path_in_free_space():
    p = prepare_task(get_task("nominal"))
    assert not p.direct_blocked and p.plan.status == OK_DIRECT and p.waypoints.shape == (2, 6)


def test_obstacle_blocks_direct_but_rrt_connect_succeeds(obstacle_prepared):
    p = obstacle_prepared
    assert p.direct_blocked and p.plan.success and p.waypoints.shape[1] == 6 and len(p.waypoints) >= 3
    checker = CollisionChecker(load_scene(p.task.exec_scene()))           # TRUE geometry
    assert checker.validate_path(p.waypoints, 0.01)
    assert p.trajectory_collision_free and p.min_obstacle_clearance_m > 0.0
    assert direct_joint_path(checker, p.q_start, p.q_goal) is None


def test_path_endpoints_limits_and_length(obstacle_prepared):
    p = obstacle_prepared
    np.testing.assert_allclose(p.waypoints[0], p.q_start)
    np.testing.assert_allclose(p.waypoints[-1], p.q_goal)
    lim = CollisionChecker(load_scene()).model.jnt_range[:6]
    assert np.all(p.waypoints >= lim[:, 0]) and np.all(p.waypoints <= lim[:, 1])
    assert p.plan.path_length == pytest.approx(path_length(p.waypoints))


@pytest.mark.parametrize("cls", [RRTConnectPlanner, RRTPlanner])
def test_reproducible_with_seed_and_result_fields(cls, obstacle_prepared):
    p = obstacle_prepared
    checker = CollisionChecker(load_scene(p.task.plan_scene()))
    # Limit the deliberately unsuccessful single-tree search: this test
    # exercises result reproducibility, not the full planner benchmark.
    kwargs = {"seed": 7}
    if cls is RRTPlanner:
        kwargs["max_iterations"] = 100
    a = cls(checker, **kwargs).solve(p.q_start, p.q_goal)
    b = cls(checker, **kwargs).solve(p.q_start, p.q_goal)
    # A tall obstacle makes single-tree RRT legitimately fail for this fixed
    # seed.  Success is not an invariant; reproducibility and a complete,
    # explicit result object are.
    assert a.success == b.success and a.status == b.status and a.seed == 7
    if a.success:
        assert np.array_equal(a.path, b.path)
    else:
        assert a.path is None and b.path is None
    assert a.planning_time > 0 and a.iterations > 0 and a.settings["seed"] == 7


def test_failures_are_explicit_not_exceptions(obstacle_prepared):
    p = obstacle_prepared
    checker = CollisionChecker(load_scene(p.task.plan_scene()))
    planner = RRTConnectPlanner(checker, seed=0)
    assert planner.solve(p.q_start, [9, 0, 0, 0, 0, 0]).status == INVALID_GOAL
    assert planner.solve([9, 0, 0, 0, 0, 0], p.q_goal).status == INVALID_START
    tiny = RRTConnectPlanner(checker, seed=0, max_iterations=1).solve(p.q_start, p.q_goal)
    assert not tiny.success and tiny.path is None and np.isnan(tiny.path_length)
    with pytest.raises(ValueError):
        planner.plan(p.q_start, [9, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        make_planner("nope", checker)


def test_timeout_status(obstacle_prepared):
    p = obstacle_prepared
    checker = CollisionChecker(load_scene(p.task.plan_scene()))
    r = RRTPlanner(checker, seed=2, timeout=1e-6).solve(p.q_start, p.q_goal)
    assert not r.success and r.status in ("timeout", "max_iterations")


def test_goal_selection_rejects_unreachable_and_picks_feasible():
    checker = CollisionChecker(load_scene())
    T = np.eye(4)
    T[:3, 3] = [3.0, 0, 1.0]
    with pytest.raises(TaskError):
        select_goal(T, QA, checker)
    q, info = select_goal(get_task("nominal").T_target, QA, checker)
    assert info["n_collision_free"] >= 1 and checker.is_configuration_valid(q)
