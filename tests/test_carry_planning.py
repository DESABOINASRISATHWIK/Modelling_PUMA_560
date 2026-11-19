"""Collision checks for payload-aware conveyor transfer paths."""
import numpy as np
import pytest

from src import robot_parameters as rp
from src.carry_planning import plan_carry_route
from src.conveyor import ConveyorConfig, ConveyorSimulation
from src.scene import Obstacle
from src.singularity_analysis import analyze_configuration


# This fixture is deliberately absent from the nominal conveyor presentation.
# Dedicated blocked-route cases are introduced with the obstacle experiment.
PILLAR = Obstacle(position=(0.45, -0.30, 0.56), size=(0.07, 0.07, 0.22))


@pytest.mark.parametrize("slot_index", [0, 1])
def test_payload_carry_path_is_collision_checked_for_both_payloads(slot_index):
    sim = ConveyorSimulation(ConveyorConfig(obstacles=(PILLAR,), obstacle_margin=0.03))
    slot = sim.item_slots[slot_index]
    plan_checker, true_checker = sim._carry_collision_checkers(slot)
    position = np.array([sim.config.pickup_x, sim.config.belt_y,
                         sim.config.belt_top + slot.spec.half_size[2]])
    grasp = position - sim.TOOL_R @ np.array([0.0, 0.0, sim.GRASP_OFFSET])
    lift = grasp.copy(); lift[2] += 0.20
    over = lift.copy(); over[:2] = sim.config.bin_centers[slot.spec.bin_id]
    drop = over.copy()
    drop[2] = (sim.config.bin_floor + slot.spec.half_size[2] + 0.015
               - (sim.TOOL_R @ np.array([0.0, 0.0, sim.GRASP_OFFSET]))[2])
    targets = []
    q_ref = rp.Q_TASK_HOME.copy()
    for target in (lift, over, drop):
        q_ref = sim._target_joints(target, q_ref=q_ref, checker=plan_checker)
        targets.append(q_ref)

    result = plan_carry_route(
        plan_checker, true_checker, rp.Q_TASK_HOME, targets, dt=sim.dt,
        max_velocity=[0.6, 0.6, 0.6, 1.0, 1.0, 1.2],
        max_acceleration=[1.0, 1.0, 1.0, 2.0, 2.0, 2.5],
        seed=sim.config.seed + slot.index * 101, timeout=5.0,
        clearance_lower_bound=sim.config.obstacle_margin)

    assert result.success, result.summary()
    assert result.min_clearance_m == pytest.approx(0.03)
    assert plan_checker.validate_path(result.waypoints, resolution=0.008)
    assert result.trajectory is not None


def test_empty_obstacle_set_has_no_clearance_claim():
    sim = ConveyorSimulation(ConveyorConfig(obstacles=()))
    plan_checker, true_checker = sim._robot_collision_checkers()
    assert plan_checker.validate_path(np.vstack([rp.Q_TASK_HOME, rp.Q_TASK_HOME + [0.02, 0, 0, 0, 0, 0]]))
    result = plan_carry_route(plan_checker, true_checker, rp.Q_TASK_HOME,
                              [rp.Q_TASK_HOME + [0.02, 0, 0, 0, 0, 0]],
                              dt=sim.dt, clearance_lower_bound=sim.config.obstacle_margin)
    assert result.success
    assert result.min_clearance_m is None


def test_bins_are_separated_and_overlap_is_rejected():
    config = ConveyorConfig()
    assert abs(config.bin_centers[0][0] - config.bin_centers[1][0]) >= 0.256
    assert config.bin_centers[0][0] < 0.0 and config.bin_centers[1][0] > 0.0
    assert config.belt_y < min(y for _, y in config.bin_centers)
    with pytest.raises(ValueError, match="overlap"):
        ConveyorConfig(bin_centers=((0.55, -0.48), (0.65, -0.30))).validate()


def test_conveyor_standby_is_regular_and_faces_the_front_work_area():
    assert analyze_configuration(rp.Q_READY).status == "singular"
    standby = analyze_configuration(rp.Q_CONVEYOR_STANDBY_RIGHT)
    assert standby.status == "regular"
    assert standby.svd.sigma_min > 0.1
    sim = ConveyorSimulation(ConveyorConfig())
    np.testing.assert_allclose(sim.state()[0], rp.Q_CONVEYOR_STANDBY_RIGHT)
    from src.forward_kinematics import forward_kinematics
    assert sim._base_to_world_position(forward_kinematics(sim.state()[0]).position)[1] < 0.0
