"""Experiment definitions shared by run_experiments.py, the live demo and the tests (Part 4)."""
from __future__ import annotations

import json
import platform
from importlib import metadata
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from .control import make_controller
from .execution import ExecutionLog, run_trajectory
from .gravity import DHGravityModel
from .metrics import compute_metrics
from .path_planning import make_planner
from .physics import PhysicsSimulation
from .tasks import TaskConfig, get_task, prepare_task, select_goal  # noqa: F401  (re-exported)

CONTROLLER_LABELS = {
    "pd": "PD",
    "pd_gravity": "PD + gravity (nominal model)",
    "pd_gravity_payload_aware": "PD + gravity (payload-aware model)",
}


def controller_variants(task: TaskConfig):
    """Controllers compared on a task, all with identical gains.

    The gravity term of ``pd_gravity`` ALWAYS uses the nominal (payload-free) model, so in the
    payload experiment it deliberately does not know about the payload (robustness test).
    ``pd_gravity_payload_aware`` is the clearly-labelled extra case that does.
    """
    nominal = DHGravityModel()
    out = {"pd": make_controller("pd"),
           "pd_gravity": make_controller("pd_gravity", gravity=nominal, label="pd_gravity")}
    if task.payload is not None:
        out["pd_gravity_payload_aware"] = make_controller(
            "pd_gravity", gravity=DHGravityModel(payload=task.payload), label="pd_gravity_payload_aware")
    return out


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if is_dataclass(o):
        return asdict(o)
    return str(o)


def task_record(prepared):
    """Everything needed to reproduce the planning/trajectory of a run (saved as JSON)."""
    t = prepared.task
    return dict(
        name=t.name, description=t.description, q_start=list(t.q_start), target_xyz=list(t.target_xyz),
        target_rpy_deg=list(t.target_rpy_deg), obstacles=[asdict(o) for o in t.obstacles],
        payload=asdict(t.payload) if t.payload else None, obstacle_margin_m=t.obstacle_margin,
        v_max=list(t.v_max), a_max=list(t.a_max), hold_time_s=t.hold_time,
        goal=dict(q_goal=prepared.q_goal.tolist(), **prepared.goal_info),
        direct_joint_path_blocked=bool(prepared.direct_blocked),
        plan=dict(summary=prepared.plan.summary(), status=prepared.plan.status, seed=prepared.plan.seed,
                  method=prepared.plan.method, settings=prepared.plan.settings,
                  waypoints=prepared.waypoints.tolist(), path_length_rad=prepared.plan.path_length,
                  planning_time_s=prepared.plan.planning_time, iterations=prepared.plan.iterations),
        trajectory=dict(duration_s=prepared.trajectory.duration,
                        segment_times_s=prepared.trajectory.segment_times.tolist(),
                        collision_free_true_scene=bool(prepared.trajectory_collision_free),
                        first_collision_time_s=prepared.trajectory_first_collision_time,
                        min_obstacle_clearance_m=prepared.min_obstacle_clearance_m),
        environment=dict(python=platform.python_version(), numpy=np.__version__, mujoco=metadata.version("mujoco")))


def run_case(task_name_or_config, out_dir=None, variants=None, planner=None, seed=None, verbose=True):
    """Prepare a task and execute it with every controller variant.

    Returns ``dict(prepared=..., logs={name: ExecutionLog}, metrics={name: dict})``.  When
    ``out_dir`` is given, raw logs (.npz), ``config.json`` and ``metrics.json`` are written there.
    """
    task = get_task(task_name_or_config) if isinstance(task_name_or_config, str) else task_name_or_config
    prepared = prepare_task(task, planner=planner, seed=seed)
    variants = variants or controller_variants(task)
    sim = PhysicsSimulation(task.exec_scene())
    logs, metrics = {}, {}
    for name, ctrl in variants.items():
        log = run_trajectory(sim, ctrl, prepared.trajectory, hold_time=task.hold_time,
                             meta=dict(experiment=task.name, controller_name=name, planner_seed=prepared.plan.seed))
        logs[name] = log
        metrics[name] = compute_metrics(log, prepared.T_target, plan=prepared.plan)
        if verbose:
            m = metrics[name]
            print(f"  [{task.name:<8} | {name:<26}] RMSE {m['rmse_rad_rms_over_joints']:.4f} rad | "
                  f"final EE err {m['final_ee_position_error_m'] * 1e3:7.2f} mm | "
                  f"peak |tau| {max(m['torque_peak_Nm']):6.1f} N m | sat {m['saturation_time_s_any_joint']:.3f} s | "
                  f"collision-free {m['collision_free']}")
    if out_dir is not None:
        out = Path(out_dir) / task.name
        out.mkdir(parents=True, exist_ok=True)
        for name, log in logs.items():
            log.save(out / f"{name}.npz")
        (out / "config.json").write_text(json.dumps(task_record(prepared), indent=2, default=_jsonable))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_jsonable))
    return dict(prepared=prepared, logs=logs, metrics=metrics)


def planner_study(task_name="obstacle", seeds=range(20), methods=("rrt_connect", "rrt"), timeout=20.0,
                  out_dir=None, verbose=True):
    """Compare planners on the obstacle task over many random seeds (planning only)."""
    task = get_task(task_name)
    from .collision import CollisionChecker
    from .scene import load_scene
    checker = CollisionChecker(load_scene(task.plan_scene()))
    q_start = np.asarray(task.q_start, dtype=float)
    q_goal, _ = select_goal(task.T_target, q_start, checker)
    from .path_planning import direct_joint_path
    direct_ok = direct_joint_path(checker, q_start, q_goal) is not None
    rows = []
    for method in methods:
        for s in seeds:
            r = make_planner(method, checker, seed=int(s), timeout=timeout).solve(q_start, q_goal)
            rows.append(dict(method=method, seed=int(s), success=bool(r.success), status=r.status,
                             time_s=float(r.planning_time), iterations=int(r.iterations), nodes=int(r.n_nodes),
                             path_length_rad=float(r.path_length), waypoints=int(r.n_waypoints)))
        if verbose:
            sub = [x for x in rows if x["method"] == method]
            ok = [x for x in sub if x["success"]]
            print(f"  [{method:<12}] success {len(ok)}/{len(sub)} | median time "
                  f"{np.median([x['time_s'] for x in ok]) if ok else float('nan'):.3f} s")
    result = dict(task=task_name, direct_path_feasible=bool(direct_ok), timeout_s=timeout,
                  settings={m: make_planner(m, checker).settings() for m in methods}, rows=rows)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "planner_study.json").write_text(json.dumps(result, indent=2, default=_jsonable))
    return result


def load_case(out_dir, task_name):
    """Reload raw logs, config and metrics of a saved experiment."""
    d = Path(out_dir) / task_name
    logs = {p.stem: ExecutionLog.load(p) for p in sorted(d.glob("*.npz"))}
    return dict(logs=logs, config=json.loads((d / "config.json").read_text()),
                metrics=json.loads((d / "metrics.json").read_text()))
