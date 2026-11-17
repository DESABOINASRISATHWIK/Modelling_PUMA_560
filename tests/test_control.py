"""Part 3/4: controllers, gravity model, closed-loop execution, metrics."""
import numpy as np
import pytest

from src import robot_parameters as rp
from src.control import DEFAULT_KD, DEFAULT_KP, PDController, PDGravityController, design_pd_gains, make_controller
from src.execution import ExecutionLog, run_trajectory
from src.experiments import run_case
from src.gravity import DHGravityModel, GravityCompensator
from src.metrics import compute_metrics
from src.physics import PhysicsSimulation
from src.scene import Payload, SceneConfig
from src.trajectory import QuinticTrajectory


@pytest.mark.parametrize("payload", [None, Payload(mass=2.0, position=(0.03, 0.01, 0.08))])
def test_dh_gravity_matches_mujoco_bias_and_mujoco_based_model(payload):
    s = PhysicsSimulation(SceneConfig(payload=payload))
    dh, ref = DHGravityModel(payload), GravityCompensator(s.model)
    rng = np.random.default_rng(5)
    for _ in range(40):
        q = rng.uniform(rp.JOINT_LIMITS[:, 0] + .01, rp.JOINT_LIMITS[:, 1] - .01)
        s.reset(q)
        np.testing.assert_allclose(dh.torque(q), s.data.qfrc_bias, atol=1e-9)
        np.testing.assert_allclose(dh.torque(q), ref.torque(q), atol=1e-9)


def test_control_laws_exact():
    q, qd = np.arange(6) * .1, np.ones(6) * .2
    qdes, qddes = q + .3, np.ones(6) * .5
    kp, kd = np.arange(1, 7) * 10., np.arange(1, 7) * 1.
    pd = PDController(kp, kd)
    np.testing.assert_allclose(pd.torque(q, qd, qdes, qddes), kp * .3 + kd * .3)
    g = DHGravityModel()
    pdg = PDGravityController(g, kp, kd)
    np.testing.assert_allclose(pdg.torque(q, qd, qdes, qddes), pd.torque(q, qd, qdes, qddes) + g.torque(q))
    assert make_controller("pd").name == "pd" and make_controller("pd_gravity", gravity=g).name == "pd_gravity"
    with pytest.raises(ValueError):
        make_controller("pd_gravity")
    with pytest.raises(ValueError):
        PDController(kp=-np.ones(6))


def test_default_gains_follow_design_rule():
    kp, kd = design_pd_gains([12, 14, 25, 1, 1, 1])
    np.testing.assert_allclose(DEFAULT_KP[:3], kp[:3], rtol=0.05)
    np.testing.assert_allclose(DEFAULT_KD[:3], kd[:3], rtol=0.05)
    assert np.all(DEFAULT_KD[3:] <= 0.0 + np.array([0.82, 0.321, 0.02]) + 1e-12)   # explicit-derivative cap I_min/dt


def test_pd_gravity_beats_pd_and_log_roundtrip(tmp_path):
    r = run_case("nominal", out_dir=tmp_path, verbose=False)
    pd, pdg = r["metrics"]["pd"], r["metrics"]["pd_gravity"]
    assert pdg["final_ee_position_error_m"] < 5e-3 < pd["final_ee_position_error_m"]
    assert pdg["rmse_rad_rms_over_joints"] < pd["rmse_rad_rms_over_joints"]
    assert pdg["collision_free"] and pdg["saturation_time_s_any_joint"] == 0
    log = ExecutionLog.load(tmp_path / "nominal" / "pd_gravity.npz")
    np.testing.assert_allclose(log.q, r["logs"]["pd_gravity"].q)
    again = compute_metrics(log, r["prepared"].T_target)
    assert again["rmse_rad_rms_over_joints"] == pytest.approx(pdg["rmse_rad_rms_over_joints"])


def test_executed_motion_never_overwrites_state_and_records_saturation():
    sim = PhysicsSimulation()
    q0, q1 = np.zeros(6), np.array([1.0, 0.5, -0.5, 0, 0, 0])
    tr = QuinticTrajectory(np.vstack([q0, q1]), segment_times=[0.15])     # absurdly fast -> saturates
    log = run_trajectory(sim, PDController(), tr, hold_time=0.2)
    assert log.tau_req.shape == log.tau_app.shape == (len(log.t), 6)
    assert np.all(np.abs(log.tau_app) <= np.array(sim.config.torque_limits) + 1e-9)
    m = compute_metrics(log)
    assert m["saturation_time_s_any_joint"] > 0 and np.max(np.abs(log.tau_req)) > np.max(np.abs(log.tau_app))


def test_obstacle_execution_checks_real_contacts():
    r = run_case("obstacle", verbose=False)
    assert r["prepared"].direct_blocked and r["metrics"]["pd_gravity"]["collision_free"]
    assert r["metrics"]["pd_gravity"]["holds_target_within_tolerance"]


def test_payload_mismatch_is_visible_and_fixed_by_aware_model():
    m = run_case("payload", verbose=False)["metrics"]
    assert m["pd_gravity"]["final_ee_position_error_m"] > 5e-3          # model mismatch
    assert m["pd_gravity_payload_aware"]["final_ee_position_error_m"] < 5e-3
