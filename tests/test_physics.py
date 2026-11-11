"""Part 1 acceptance checks: dynamics, contacts, payloads, gravity, isolation."""
import mujoco
import numpy as np
import pytest

from src import robot_parameters as rp
from src.collision import CollisionChecker
from src.forward_kinematics import forward_kinematics
from src.gravity import GravityCompensator
from src.simulation import PhysicsSimulation
from src.scene import Obstacle, Payload, SceneConfig, load_scene, write_scene


def test_torque_mapping_clipping_and_dynamic_motion():
    s = PhysicsSimulation(SceneConfig(gear=(2, -3, 1, 1, 1, 1)))
    q0, _ = s.state()
    accepted = s.apply_torque([1e6, 12, 0, 0, 0, 0])
    mujoco.mj_forward(s.model, s.data)
    np.testing.assert_allclose(s.data.qfrc_actuator, accepted)
    assert accepted[0] == 150 and s.data.ctrl[1] == -4
    s.step(5)
    q, v = s.state()
    assert np.linalg.norm(q-q0) > 1e-6 and np.linalg.norm(v) > 1e-4
    assert s.data.time == pytest.approx(0.01)
    np.testing.assert_allclose(s.end_effector_pose(), forward_kinematics(q).T, atol=1e-9)
    q[:] = 999
    assert np.max(s.state()[0]) < 999
    s.reset()
    assert s.data.time == 0 and not np.any(s.data.ctrl)


@pytest.mark.parametrize("payload", [None, Payload(mass=2.0, position=(0.03, 0.01, 0.08))])
def test_gravity_against_bias_and_static_hold(payload):
    s = PhysicsSimulation(SceneConfig(payload=payload, gravity=(0.2, -0.3, -9.81)))
    gravity = GravityCompensator(s.model)
    rng = np.random.default_rng(42)
    for _ in range(40):
        q = rng.uniform(rp.JOINT_LIMITS[:, 0]+0.01, rp.JOINT_LIMITS[:, 1]-0.01)
        s.reset(q)
        tau = gravity.torque(q)
        np.testing.assert_allclose(tau, s.data.qfrc_bias, atol=1e-9)
    s.reset()
    q, _ = s.state()
    s.apply_torque(gravity.torque(q))
    s.step(100)
    np.testing.assert_allclose(s.state()[0], q, atol=1e-9)
    np.testing.assert_allclose(s.state()[1], 0, atol=1e-9)


def test_payload_changes_gravity():
    bare, loaded = load_scene(), load_scene(SceneConfig(payload=Payload()))
    a = GravityCompensator(bare).torque(np.zeros(6))
    b = GravityCompensator(loaded).torque(np.zeros(6))
    assert np.linalg.norm(a-b) > 1


def test_collision_and_live_state_isolation():
    s = PhysicsSimulation()
    checker = CollisionChecker(s.model)
    assert checker.is_configuration_valid(np.zeros(6))
    # A box centred on the forearm proxy blocks the zero configuration.
    blocked = CollisionChecker(load_scene(SceneConfig(obstacles=(
        Obstacle(position=(rp.A2+rp.A3, -rp.D3, rp.D1+0.2), size=(0.06,)*3),))))
    assert not blocked.is_configuration_valid(np.zeros(6))
    assert any("obstacle_0" in pair for pair in blocked.contacts(np.zeros(6)))
    assert not blocked.is_edge_valid(np.zeros(6), np.array([0.1, 0, 0, 0, 0, 0]))
    assert checker.is_edge_valid(np.zeros(6), np.array([0.1, 0, 0, 0, 0, 0]))
    assert not checker.is_configuration_valid([10, 0, 0, 0, 0, 0])
    s.apply_torque([2, 3, 4, 0, 0, 0])
    s.step(3)
    before = [s.data.qpos.copy(), s.data.qvel.copy(), s.data.ctrl.copy(), s.data.time,
              s.data.site_xpos.copy()]
    checker.is_edge_valid(np.zeros(6), rp.Q_READY)
    GravityCompensator(s.model).torque(rp.Q_READY)
    for a, b in zip(before, [s.data.qpos, s.data.qvel, s.data.ctrl, s.data.time, s.data.site_xpos]):
        np.testing.assert_array_equal(a, b)


def test_floor_and_self_collision_detected():
    checker = CollisionChecker(load_scene())
    rng = np.random.default_rng(17)
    found_floor = found_self = False
    observed = set()
    for _ in range(2000):
        q = rng.uniform(rp.JOINT_LIMITS[:, 0], rp.JOINT_LIMITS[:, 1])
        pairs = checker.contacts(q)
        observed.update(pairs)
        found_floor |= any("floor" in pair for pair in pairs)
        found_self |= any(all(name.startswith("collision_link") for name in pair) for pair in pairs)
        if found_floor and found_self:
            break
    assert found_floor and found_self, observed


def test_edge_checks_interior_when_endpoints_clear():
    # Obstacle intersects the forearm at the midpoint of a waist sweep.
    checker = CollisionChecker(load_scene(SceneConfig(obstacles=(
        Obstacle(position=(rp.A2+rp.A3, -rp.D3, rp.D1+0.2), size=(0.04,)*3),))))
    a, b = np.zeros(6), np.zeros(6)
    a[0], b[0] = -0.8, 0.8
    assert checker.is_configuration_valid(a) and checker.is_configuration_valid(b)
    assert not checker.is_edge_valid(a, b)


def test_scene_export_and_bad_inputs(tmp_path):
    path = tmp_path / "scene.xml"
    write_scene(path, SceneConfig.obstacle_scene(payload=Payload()))
    m = mujoco.MjModel.from_xml_path(str(path))
    assert m.nu == 6 and m.body("payload").mass == 1
    for c in (SceneConfig(gear=(0,)*6), SceneConfig(timestep=-1),
              SceneConfig(payload=Payload(mass=-1)), SceneConfig(damping=(-1,)*6)):
        with pytest.raises(ValueError):
            load_scene(c)
    s = PhysicsSimulation()
    with pytest.raises(ValueError):
        s.reset([99]*6)
    with pytest.raises(ValueError):
        s.apply_torque([np.nan]*6)
    with pytest.raises(ValueError):
        s.step(1.5)
