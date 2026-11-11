"""Simulation state/logic tests (no OpenGL, no viewer window needed)."""
import numpy as np

from src.forward_kinematics import forward_kinematics
from src.simulation import KEY_END, KEY_HOME, KEY_INSERT, KEY_UP, PumaSimulation


def test_command_interface_and_state_integrity():
    s = PumaSimulation()
    assert "branch" in s.handle_command("pose 0.55 0.25 0.85 180 0 0")
    n = len(s.ik.solutions)
    assert n == 8
    s.handle_command("go")
    best = s.ik.solutions[s.branch]
    assert np.allclose(forward_kinematics(s.q).T[:3, 3], [0.55, 0.25, 0.85], atol=1e-9)
    q_before = s.q.copy()
    for bad in ("q 1 2", "joint 9 5", "preset nope", "pose a b c", "bogus"):
        s.handle_command(bad)
    assert np.allclose(s.q, q_before) and s.q.shape == (6,)        # bad input never corrupts the state
    s.preset("sing_wrist")
    assert abs(np.sin(s.q[4])) < 1e-12


def test_keys_move_target_and_never_raise():
    s = PumaSimulation()
    t0 = s.target[:3, 3].copy()
    s.key(KEY_UP)
    assert np.isclose(s.target[0, 3] - t0[0], 0.02)
    for _ in range(300):                   # push the target out of reach
        s.key(KEY_UP)
    assert s.ik.status == "unreachable" and s.ik.solutions == []
    s.key(KEY_HOME), s.key(KEY_END), s.key(KEY_INSERT)             # no solutions: must not crash
    assert s.q.shape == (6,)


def test_unreachable_then_recover():
    s = PumaSimulation()
    s.set_target_xyzrpy(3, 0, 0)
    assert s.ik.status == "unreachable"
    s.target_from_current()
    assert s.ik.status == "ok" and len(s.ik.solutions) == 8


def test_velocity_commands_and_overlay_drawing():
    import mujoco
    from src.jacobian import geometric_jacobian
    s = PumaSimulation()
    out = s.handle_command("twist 0.1 0 -0.05 0 0.2 0")
    assert "pseudo-inverse" in out and s.show["velocity"]
    assert np.allclose(geometric_jacobian(s.q) @ s.qdot, [0.1, 0, -0.05, 0, 0.2, 0], atol=1e-9)
    assert "DLS" in s.handle_command("twist 0 0 0 0 0.2 0 dls")
    s.handle_command("qd 10 -20 30 20 -10 40")
    assert np.allclose(np.degrees(s.qdot), [10, -20, 30, 20, -10, 40])
    before = s.qdot.copy()
    for bad in ("qd 1 2", "twist 1 2 3", "twist a b c d e f"):
        s.handle_command(bad)
    assert np.allclose(s.qdot, before)                         # malformed input leaves the state untouched
    scn = mujoco.MjvScene(s.model, 3000)
    s.show["workspace"] = False
    s.draw_overlays(scn)
    assert scn.ngeom == 0 and "ellipsoid" in s.hud()[0]
    for kind in ("sing_wrist", "sing_elbow", "sing_shoulder"):  # drawing at singular poses must not raise
        s.preset(kind)
        scn.ngeom = 0
        s.draw_overlays(scn)
