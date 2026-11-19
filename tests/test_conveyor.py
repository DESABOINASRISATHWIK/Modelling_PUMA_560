"""Regression checks for the bounded conveyor pick-and-place demo."""
import numpy as np

from src.conveyor import ConveyorConfig, ConveyorSimulation
from src.trajectory import QuinticBoundaryTrajectory


def test_quintic_boundary_trajectory_hits_all_endpoint_conditions():
    q0 = np.zeros(6)
    q1 = np.arange(1.0, 7.0)
    qd0, qd1 = np.full(6, 0.1), np.full(6, -0.2)
    qdd0, qdd1 = np.full(6, 0.3), np.full(6, -0.4)
    tr = QuinticBoundaryTrajectory(q0, q1, 1.7, qd0, qd1, qdd0, qdd1)
    for t, expected in ((0.0, (q0, qd0, qdd0)), (1.7, (q1, qd1, qdd1))):
        actual = tr.evaluate(t)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, atol=1e-9)


def test_conveyor_pool_is_bounded_and_completes_a_pick_and_place():
    sim = ConveyorSimulation(ConveyorConfig(spawn_interval=10.0, max_items=3))
    # The first physical pick/place includes the deliberate 0.4 m indexing
    # segment, settled grasp, free drop, and vertical retract.
    while sim.sim_time < 22.0:
        sim.step()
        assert sim._live_count() <= 3
    assert sim.spawned >= 1
    assert sim.picked >= 1
    assert sim.placed >= 1
    assert sim.status["max_live_items"] <= 3
    assert sim.status["tracking_error_max_rad"] < 0.5


def test_conveyor_config_rejects_invalid_capacity():
    try:
        ConveyorConfig(max_items=0).validate()
    except ValueError as exc:
        assert "max_items" in str(exc)
    else:
        raise AssertionError("invalid pool capacity was accepted")


def test_conveyor_bins_are_raised_above_the_work_surface():
    config = ConveyorConfig()
    assert np.isclose(config.bin_floor, config.work_surface_top + 0.01)


def test_gripper_target_matches_box_width_with_only_small_pad_compression():
    sim = ConveyorSimulation(ConveyorConfig(max_items=2))
    for slot in sim.item_slots:
        sim._slot = slot
        travel = sim._grip_target()[0]
        target_inner_gap = sim.FINGER_OPEN_INNER_GAP - 2.0 * travel
        box_width = 2.0 * slot.spec.half_size[0]
        assert target_inner_gap == box_width - sim.GRIP_COMPRESSION
        assert 0.0 < sim.GRIP_COMPRESSION <= 0.003


def test_stop_go_cycle_physically_sorts_the_first_payload_class():
    sim = ConveyorSimulation(ConveyorConfig(belt_speed=0.05, spawn_policy="ready"))
    while sim.sim_time < 22.0:
        sim.step()
    assert sim.picked >= 1
    assert sim.placed >= 1
    assert sim.missed == 0
    assert sim.status["max_finger_item_penetration_m"] < 0.004
    assert max(sim.status["joint_torque_peak_nm"]) < 100.0
    grasps = [event for event in sim._events if event["event"] == "grasped"]
    assert {event["item"] for event in grasps} == {"blue_light"}
    for event in grasps:
        forces = event["finger_contact_normal_forces_n"]
        assert min(forces.values()) >= event["required_jaw_normal_force_n"]
    assert sum(event["event"] == "belt_stopped" for event in sim._events) >= 1


def test_ready_feed_never_queues_multiple_unpicked_items_and_logs_sync():
    sim = ConveyorSimulation(ConveyorConfig(spawn_policy="ready", max_items=5))
    while sim.sim_time < 12.0:
        sim.step()
        assert sum(slot.state == "belt" for slot in sim.item_slots) <= 1
    assert sim.picked >= 1
    assert sim.status["grasp_sync"]
    sync = sim.status["grasp_sync"][0]
    assert sync["relative_position_error_m"] >= 0
    assert sync["relative_velocity_error_m_s"] >= 0
    # This is the component that determines whether the robot has matched the
    # conveyor's translational motion at contact; lateral pad disturbance is
    # still reported separately in the full 3D velocity error.
    assert sync["belt_axis_velocity_error_m_s"] < 0.01


def test_miss_reason_is_counted_and_written_as_an_event():
    sim = ConveyorSimulation(ConveyorConfig(max_items=1))
    slot = sim.item_slots[0]
    sim._mark_missed(slot, "belt_exit_unpicked")
    assert sim.status["miss_reasons"] == {"belt_exit_unpicked": 1}
    assert sim._events[-1]["reason"] == "belt_exit_unpicked"
