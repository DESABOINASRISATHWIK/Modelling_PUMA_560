"""
Interactive MuJoCo demonstration of FK, IK and the Jacobian/singularities of the PUMA-560.

MuJoCo is used to DRAW the arm (meshes) and to hold the joint state.  Everything
displayed on top of it -- DH frames, end-effector frame, wrist centre, the target
pose, the IK branches (as skeletons), the singularity marker and the workspace cloud --
is computed by OUR NumPy code (forward_kinematics / inverse_kinematics / jacobian /
singularity_analysis) and added to the scene as extra geoms.

Run
    python -m src.simulation                       # interactive viewer
    python -m src.simulation --target 0.6 0.2 0.9 180 0 0     # start with a Cartesian target [m, deg]
    python -m src.simulation --selftest            # scripted run (no user input), for CI/demo checks
    python -m src.simulation --png out.png         # offscreen snapshot (needs MUJOCO_GL=egl|osmesa)

Controls inside the viewer window (letters are used by MuJoCo's own shortcuts, so
we only use non-letter keys):
    Up / Down       target  +x / -x          (0.02 m)
    Left / Right    target  +y / -y
    PageUp / PageDn target  +z / -z
    Home / End      previous / next IK branch
    Insert          move the robot to the selected IK branch
    Delete          set target := current end-effector pose
    Enter           toggle workspace point cloud
    Joint angles    left panel "Joint" sliders (if offered by your viewer) or the terminal command  q / joint

Terminal commands (type in the console that launched the viewer):  help
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from . import robot_parameters as rp
from .physics import PhysicsSimulation  # Public Part 1 dynamics API; PumaSimulation is the legacy viewer.
from .dh_model import PUMA560
from .forward_kinematics import forward_kinematics, random_valid_configuration
from .inverse_kinematics import IKResult, inverse_kinematics
from .jacobian import geometric_jacobian, joint_velocity_from_twist
from .mujoco_reference import joint_qpos_addresses, new_data, set_joint_angles
from .singularity_analysis import (SIGMA_NEAR, SIGMA_SINGULAR, analyze_configuration,
                                   singular_configuration, svd_metrics)
from .utils import matrix_to_rpy, rpy_to_matrix
from .velocity_kinematics import velocity_ellipsoid

# GLFW key codes (non-letter keys only)
KEY_ENTER, KEY_INSERT, KEY_DELETE = 257, 260, 261
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_PAGE_UP, KEY_PAGE_DOWN, KEY_HOME, KEY_END = 266, 267, 268, 269

BRANCH_COLORS = [(0.95, 0.30, 0.25), (0.20, 0.55, 0.95), (0.20, 0.75, 0.35), (0.95, 0.75, 0.15),
                 (0.70, 0.35, 0.90), (0.10, 0.75, 0.80), (0.95, 0.50, 0.70), (0.55, 0.55, 0.20)]

PRESETS = {
    "zero": rp.Q_ZERO, "ready": rp.Q_READY, "stretch": rp.Q_STRETCH, "nominal": rp.Q_NOMINAL,
}
DEFAULT_Q = np.array([0.35, 0.9, -1.3, 0.5, 0.8, 0.4])


# ----------------------------------------------------------------------------
# scene primitives (work for viewer.user_scn and Renderer.scene alike)
# ----------------------------------------------------------------------------
def _next_geom(scn):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    scn.ngeom += 1
    return g


def add_sphere(scn, p, radius, rgba):
    g = _next_geom(scn)
    if g is None:
        return
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(p, float),
                        np.eye(3).reshape(9), np.asarray(rgba, np.float32))


def add_connector(scn, kind, p0, p1, width, rgba):
    g = _next_geom(scn)
    if g is None:
        return
    mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), np.eye(3).reshape(9), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, kind, width, np.asarray(p0, float), np.asarray(p1, float))


def add_ellipsoid(scn, center, semi_axes, axes, rgba):
    """Ellipsoid with semi-axes along the columns of ``axes`` (made a proper rotation)."""
    g = _next_geom(scn)
    if g is None:
        return
    R = np.array(axes, dtype=float)
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ELLIPSOID, np.maximum(np.asarray(semi_axes, float), 1e-4),
                        np.asarray(center, float), R.reshape(9), np.asarray(rgba, np.float32))


def add_capsule(scn, p0, p1, width, rgba):
    add_connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, p0, p1, width, rgba)


def add_arrow(scn, p0, p1, width, rgba):
    add_connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, p0, p1, width, rgba)


def add_frame(scn, T, length=0.08, width=0.004, alpha=1.0):
    """RGB triad (x red, y green, z blue) of a 4x4 pose."""
    o = T[:3, 3]
    for k, col in enumerate(((1, 0.1, 0.1), (0.1, 0.8, 0.1), (0.15, 0.3, 1.0))):
        add_arrow(scn, o, o + length * T[:3, k], width, (*col, alpha))


def sigma_color(sigma: float, alpha: float = 1.0):
    """green regular / orange near-singular / red singular."""
    if sigma < SIGMA_SINGULAR:
        return (0.9, 0.0, 0.9, alpha)
    if sigma < SIGMA_NEAR:
        return (1.0, 0.1, 0.1, alpha)
    if sigma < 5 * SIGMA_NEAR:
        return (1.0, 0.6, 0.0, alpha)
    return (0.1, 0.8, 0.2, alpha)


def skeleton_points(q) -> np.ndarray:
    """Origins of {0},{1},{2},{3}, wrist centre, flange  (from OUR forward kinematics)."""
    fk = forward_kinematics(q)
    return np.array([fk.frames[0][:3, 3], fk.frames[1][:3, 3], fk.frames[2][:3, 3], fk.frames[3][:3, 3],
                     fk.wrist_center, fk.position])


# ----------------------------------------------------------------------------
# State / logic (no GUI dependency -> unit-testable)
# ----------------------------------------------------------------------------
class PumaSimulation:
    def __init__(self, q0=None):
        # private copy: we change mesh transparency, which must not leak into the shared cached model
        self.model = mujoco.MjModel.from_xml_path(str(rp.MJCF_PATH))
        self.data = new_data(self.model)
        self._mesh_geoms = [g for g in range(self.model.ngeom)
                            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        self.set_mesh_alpha(1.0)
        self.adr = joint_qpos_addresses(self.model)
        self.q = np.array(DEFAULT_Q if q0 is None else q0, dtype=float)
        self.target = forward_kinematics(self.q).T.copy()
        self.ik: IKResult | None = None
        self.branch = 0
        self.show = dict(frames=False, branches=False, workspace=False, singular=False, velocity=False)
        self.qdot = np.zeros(6)                      # joint velocity [rad/s] whose end-effector twist is drawn
        self.ellipsoid_scale = 0.25                  # metres per (m/s per rad/s)
        self.message = ""
        self._workspace = None
        self._lock = threading.RLock()
        self.solve_ik()
        self.apply()

    def set_mesh_alpha(self, alpha: float):
        """Transparency of the robot meshes (so that frames / skeletons inside stay visible)."""
        alpha = float(np.clip(alpha, 0.05, 1.0))
        for g in self._mesh_geoms:
            self.model.geom_rgba[g, 3] = alpha
        self.mesh_alpha = alpha

    # ---------------------------------------------------------------- state
    def apply(self):
        """Push q to MuJoCo (for drawing only)."""
        set_joint_angles(self.model, self.data, self.q)

    def adopt_ui_changes(self):
        """If the viewer's joint sliders changed qpos behind our back, take it over."""
        qp = np.array([self.data.qpos[a] for a in self.adr])
        if np.max(np.abs(qp - self.q)) > 1e-9:
            self.q = qp
            self.message = "joint angles changed from the viewer panel"

    def set_q(self, q):
        q = PUMA560.check_q(q).copy()          # validate BEFORE touching the state
        with self._lock:
            self.q = q
            self.apply()

    def move_joint(self, j: int, delta_rad: float):
        q = self.q.copy()
        q[j] += delta_rad
        self.set_q(q)

    def preset(self, name: str):
        if name in PRESETS:
            q = PRESETS[name]
        elif name.startswith("sing_"):
            q = singular_configuration(name[5:], q_base=DEFAULT_Q)
        elif name == "near_wrist":
            q = singular_configuration("wrist", q_base=DEFAULT_Q)
            q[4] = 0.02
        else:
            raise KeyError(name)
        self.set_q(q)
        self.message = f"preset '{name}'"

    # ------------------------------------------------------------- velocity
    def set_qdot(self, qdot):
        qdot = PUMA560.check_q(qdot).copy()
        with self._lock:
            self.qdot = qdot
            self.show["velocity"] = True

    def set_twist(self, twist, damping: float | None = None):
        """Resolved-rate: qdot = J^+ twist (pseudo-inverse, or damped LS when damping is given)."""
        tw = np.asarray(twist, dtype=float)
        if tw.shape != (6,) or not np.all(np.isfinite(tw)):
            raise ValueError("twist needs 6 finite numbers [vx vy vz wx wy wz]")
        method = "pinv" if damping is None else "dls"
        qd = joint_velocity_from_twist(self.q, tw, method=method, damping=damping or 0.05, rcond=1e-9)
        self.set_qdot(qd)
        return qd

    # ----------------------------------------------------------------- IK
    def set_target(self, T):
        with self._lock:
            self.target = np.asarray(T, dtype=float).copy()
            self.solve_ik()

    def set_target_xyzrpy(self, x, y, z, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(*np.radians([roll_deg, pitch_deg, yaw_deg]))
        T[:3, 3] = [x, y, z]
        self.set_target(T)

    def nudge_target(self, d_pos=(0, 0, 0), d_rpy_deg=(0, 0, 0)):
        T = self.target.copy()
        T[:3, 3] += d_pos
        if np.any(d_rpy_deg):
            r, p, y = matrix_to_rpy(T[:3, :3])
            T[:3, :3] = rpy_to_matrix(r + np.radians(d_rpy_deg[0]), p + np.radians(d_rpy_deg[1]),
                                      y + np.radians(d_rpy_deg[2]))
        self.set_target(T)

    def solve_ik(self):
        self.ik = inverse_kinematics(self.target, q_ref=self.q)
        self.branch = 0
        best = self.ik.closest_to(self.q, feasible_only=False)
        if best is not None:
            self.branch = next(i for i, sol in enumerate(self.ik.solutions) if sol is best)
        self.message = self.ik.message

    def select_branch(self, step: int):
        if self.ik and self.ik.solutions:
            self.branch = (self.branch + step) % len(self.ik.solutions)

    def go_to_branch(self):
        if self.ik and self.ik.solutions:
            sol = self.ik.solutions[self.branch]
            self.set_q(sol.q)
            self.message = f"moved to branch {self.branch}: {sol.label}" + \
                ("" if sol.within_limits else "  (WARNING: violates joint limits)")
        else:
            self.message = "no IK solution to move to"

    def target_from_current(self):
        self.set_target(forward_kinematics(self.q).T)

    # --------------------------------------------------------------- workspace
    def workspace_cloud(self, n=1500, seed=0):
        if self._workspace is None:
            rng = np.random.default_rng(seed)
            pts, sig = [], []
            for _ in range(n):
                q = random_valid_configuration(rng)
                pts.append(forward_kinematics(q).position)
                sig.append(svd_metrics(geometric_jacobian(q)).sigma_min)
            self._workspace = (np.array(pts), np.array(sig))
        return self._workspace

    # ------------------------------------------------------------------- HUD
    def hud(self):
        fk = forward_kinematics(self.q)
        r, p, y = np.degrees(matrix_to_rpy(fk.rotation))
        rep = analyze_configuration(self.q)
        lim = "" if np.all((self.q >= PUMA560.limits[:, 0] - 1e-9) & (self.q <= PUMA560.limits[:, 1] + 1e-9)) \
            else "  [JOINT LIMIT VIOLATED]"
        left = ["q [deg]", "end-effector xyz [m]", "end-effector rpy [deg]", "wrist centre [m]", "target xyz [m]",
                "target rpy [deg]", "IK", "branch", "singularity", "sigma_min / cond / rank", "X, v, sin(q5)",
                "qdot [deg/s]", "v [m/s]  w [rad/s]", "linear ellipsoid axes"]
        tr, tp, ty = np.degrees(matrix_to_rpy(self.target[:3, :3]))
        sol_txt = "-"
        if self.ik and self.ik.solutions:
            s = self.ik.solutions[self.branch]
            sol_txt = (f"{self.branch + 1}/{len(self.ik.solutions)}  {s.label}  "
                       f"{'in limits' if s.within_limits else 'VIOLATES LIMITS'}"
                       + (f"  singular:{','.join(s.singular)}" if s.singular else ""))
        f = rep.factors
        tw_ = geometric_jacobian(self.q) @ self.qdot
        right = [np.array2string(np.degrees(self.q), precision=1, suppress_small=True) + lim,
                 f"{fk.position[0]:+.4f} {fk.position[1]:+.4f} {fk.position[2]:+.4f}",
                 f"{r:+.1f} {p:+.1f} {y:+.1f}",
                 f"{fk.wrist_center[0]:+.4f} {fk.wrist_center[1]:+.4f} {fk.wrist_center[2]:+.4f}",
                 f"{self.target[0, 3]:+.4f} {self.target[1, 3]:+.4f} {self.target[2, 3]:+.4f}",
                 f"{tr:+.1f} {tp:+.1f} {ty:+.1f}",
                 f"{self.ik.status}: {self.ik.message}" if self.ik else "-",
                 sol_txt,
                 rep.status.upper() + (f" ({', '.join(rep.active)})" if rep.active else ""),
                 f"{rep.svd.sigma_min:.2e} / {rep.svd.condition_number:.1e} / {rep.svd.rank}",
                 f"{f['shoulder']:+.3f}, {f['elbow']:+.3f}, {f['wrist']:+.3f}",
                 np.array2string(np.degrees(self.qdot), precision=1, suppress_small=True),
                 f"|v|={np.linalg.norm(tw_[:3]):.3f}  |w|={np.linalg.norm(tw_[3:]):.3f}",
                 np.array2string(velocity_ellipsoid(self.q, "linear").semi_axes, precision=3, suppress_small=True)]
        return "\n".join(left), "\n".join(right)

    # --------------------------------------------------------------- overlays
    def draw_overlays(self, scn):
        """Keep the scene free of diagnostic markers, frames and arrows."""
        return

    # ----------------------------------------------------------- command line
    def handle_command(self, line: str) -> str:
        parts = line.strip().split()
        if not parts:
            return ""
        cmd, args = parts[0].lower(), parts[1:]
        f = lambda xs: [float(x) for x in xs]
        try:
            with self._lock:
                if cmd == "help":
                    return COMMAND_HELP
                if cmd == "q":
                    self.set_q(np.radians(f(args)))
                    return "q set"
                if cmd == "joint":
                    self.move_joint(int(args[0]) - 1, np.radians(float(args[1])))
                    return "joint moved"
                if cmd == "pose":
                    self.set_target_xyzrpy(*f(args))
                    return self.ik.message
                if cmd == "nudge":
                    v = f(args) + [0.0] * (6 - len(args))
                    self.nudge_target(v[:3], v[3:6])
                    return self.ik.message
                if cmd == "here":
                    self.target_from_current()
                    return "target := current end-effector pose"
                if cmd == "ik":
                    self.solve_ik()
                    return self.ik.summary()
                if cmd in ("branch", "b"):
                    self.branch = int(args[0]) % max(len(self.ik.solutions), 1)
                    return f"branch {self.branch}"
                if cmd in ("next", "prev"):
                    self.select_branch(1 if cmd == "next" else -1)
                    return f"branch {self.branch}"
                if cmd == "go":
                    self.go_to_branch()
                    return self.message
                if cmd == "preset":
                    self.preset(args[0])
                    return self.message
                if cmd == "show":
                    self.show[args[0]] = (args[1].lower() in ("on", "1", "true")) if len(args) > 1 \
                        else not self.show[args[0]]
                    return f"show {args[0]} = {self.show[args[0]]}"
                if cmd == "qd":
                    self.set_qdot(np.radians(f(args)))
                    return "joint velocity set (shown as EE twist arrows)"
                if cmd == "twist":
                    lam = None
                    vals = [a for a in args if a.lower() != "dls"]
                    if "dls" in [a.lower() for a in args]:
                        lam = 0.05
                    qd = self.set_twist(f(vals), lam)
                    return f"qdot [deg/s] = {np.round(np.degrees(qd), 2).tolist()}  ({'DLS' if lam else 'pseudo-inverse'})"
                if cmd == "alpha":
                    self.set_mesh_alpha(float(args[0]))
                    return f"mesh alpha = {self.mesh_alpha}"
                if cmd == "info":
                    return self.hud()[0] + "\n" + self.hud()[1]
                return f"unknown command '{cmd}' (type help)"
        except (IndexError, ValueError, KeyError, NotImplementedError) as exc:
            return f"bad command: {exc}"

    # ----------------------------------------------------------------- keys
    def key(self, keycode: int):
        try:
            self._key(keycode)
        except Exception as exc:          # never let an exception escape into the viewer thread
            self.message = f"key error: {exc}"

    def _key(self, keycode: int):
        with self._lock:
            step = 0.02
            moves = {KEY_UP: (step, 0, 0), KEY_DOWN: (-step, 0, 0), KEY_LEFT: (0, step, 0),
                     KEY_RIGHT: (0, -step, 0), KEY_PAGE_UP: (0, 0, step), KEY_PAGE_DOWN: (0, 0, -step)}
            if keycode in moves:
                self.nudge_target(moves[keycode])
            elif keycode == KEY_HOME:
                self.select_branch(-1)
            elif keycode == KEY_END:
                self.select_branch(+1)
            elif keycode == KEY_INSERT:
                self.go_to_branch()
            elif keycode == KEY_DELETE:
                self.target_from_current()
            elif keycode == KEY_ENTER:
                self.show["workspace"] = not self.show["workspace"]


COMMAND_HELP = """commands:
  q a1 .. a6            set joint angles [deg]            joint j delta_deg   rotate joint j (1-6)
  pose x y z r p y      set Cartesian target [m, deg]      nudge dx dy dz [dr dp dy]
  here                  target := current pose             ik                  re-solve and list all branches
  next | prev | branch n   choose IK branch                go                  move robot to selected branch
  preset zero|ready|stretch|nominal|sing_shoulder|sing_elbow|sing_wrist|near_wrist
  qd d1..d6             joint velocity [deg/s] -> draws EE v / w     twist vx vy vz wx wy wz [dls]   qdot = J^+ twist
  show frames|branches|workspace|singular|velocity [on|off]   (velocity = ellipsoid + twist arrows)
  alpha a (robot transparency)   info | help | quit
keys: Up/Down/Left/Right/PageUp/PageDown move target; Home/End branch; Insert go; Delete target:=current; Enter workspace"""


# ----------------------------------------------------------------------------
# Viewer loop
# ----------------------------------------------------------------------------
def run_viewer(sim: PumaSimulation, duration: float | None = None, script: list | None = None):
    import mujoco.viewer

    stop = threading.Event()

    def console():
        print(COMMAND_HELP)
        while not stop.is_set():
            try:
                line = input("puma> ")
            except (EOFError, KeyboardInterrupt):
                return
            if line.strip().lower() in ("quit", "exit"):
                stop.set()
                return
            print(sim.handle_command(line))

    if script is None:
        threading.Thread(target=console, daemon=True).start()

    t0, k_script = time.time(), 0
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=sim.key,
                                      show_left_ui=True, show_right_ui=True) as v:
        v.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        v.opt.sitegroup[:] = 0
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 135, -20, 2.5
        v.cam.lookat[:] = [0.2, -0.1, 0.8]
        while v.is_running() and not stop.is_set():
            if duration is not None and time.time() - t0 > duration:
                break
            if script is not None and k_script < len(script) and time.time() - t0 > 0.3 * (k_script + 1):
                print(f"[script] {script[k_script]} -> {sim.handle_command(script[k_script])}")
                k_script += 1
            with sim._lock:
                sim.adopt_ui_changes()
                sim.apply()
                with v.lock():
                    v.user_scn.ngeom = 0
                    sim.draw_overlays(v.user_scn)
                left, right = sim.hud()
                v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))
            v.sync()
            time.sleep(1 / 60)
    stop.set()


def render_image(sim: PumaSimulation | None = None, size=(720, 960), azimuth=135, elevation=-20,
                 distance=2.3, lookat=(0.3, -0.05, 0.8)) -> np.ndarray:
    """Offscreen RGB snapshot with all overlays (set MUJOCO_GL=egl or osmesa on a headless machine)."""
    sim = sim or PumaSimulation()
    sim.apply()
    renderer = mujoco.Renderer(sim.model, height=size[0], width=size[1], max_geom=20000)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    cam.lookat[:] = lookat
    renderer.update_scene(sim.data, cam)
    sim.draw_overlays(renderer.scene)
    img = renderer.render().copy()
    renderer.close()
    return img


def render_png(path, sim: PumaSimulation | None = None, **kw):
    from PIL import Image
    Image.fromarray(render_image(sim, **kw)).save(path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="PUMA-560 FK/IK/Jacobian MuJoCo demo")
    ap.add_argument("--q", nargs=6, type=float, metavar="DEG", help="initial joint angles [deg]")
    ap.add_argument("--target", nargs=6, type=float, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                    help="Cartesian target [m, deg]")
    ap.add_argument("--selftest", action="store_true", help="scripted run without user input")
    ap.add_argument("--png", type=str, help="write an offscreen snapshot and exit")
    ap.add_argument("--duration", type=float, default=None, help="close the viewer after this many seconds")
    a = ap.parse_args(argv)
    sim = PumaSimulation(np.radians(a.q) if a.q else None)
    if a.target:
        sim.set_target_xyzrpy(*a.target)
    if a.png:
        print("wrote", render_png(a.png, sim))
        return
    script = None
    if a.selftest:
        script = ["preset ready", "pose 0.55 0.25 0.85 180 0 0", "ik", "next", "go", "show workspace on",
                  "qd 10 -20 30 20 -10 40", "twist 0.1 0 -0.05 0 0.2 0", "twist 0 0 0 0 0.2 0 dls",
                  "preset sing_wrist", "preset near_wrist", "preset nominal"]
    run_viewer(sim, duration=a.duration if not a.selftest else 6.0, script=script)


if __name__ == "__main__":
    main()
