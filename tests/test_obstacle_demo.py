from src.tasks import obstacle_task, prepare_task


def test_pillar_blocks_direct_path_and_rrt_connect_returns_safe_timed_route():
    prepared = prepare_task(obstacle_task(), planner="rrt_connect", seed=7)
    assert prepared.direct_blocked
    assert prepared.plan.success
    assert prepared.trajectory_collision_free
