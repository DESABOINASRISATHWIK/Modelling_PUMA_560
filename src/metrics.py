"""Quantitative performance measures for executed motions (Part 4).

All inputs come from an ``ExecutionLog`` (raw simulator data), so a metric can always
be recomputed from the saved ``.npz`` files.

Definitions (N samples, joint i):
    RMSE_i      = sqrt( 1/N sum_k (q_des,i[k] - q_i[k])^2 )          [rad]
    max error_i = max_k |q_des,i[k] - q_i[k]|                        [rad]
    final EE position error    = || p_target - p_ee(t_end) ||        [m]
    final EE orientation error = geodesic angle(R_target, R_ee)      [rad]
    torque RMS / peak          = per joint, over the APPLIED torque  [N m]
    saturation duration        = time during which |requested| > limit [s]
RMS torque is a measure of effort, not of energy consumption.
"""
from __future__ import annotations

import numpy as np

from .utils import orientation_angle

POSITION_TOL = 5e-3      # m    declared tolerance for "holds the target"
ORIENTATION_TOL = 0.02   # rad
JOINT_SETTLE_TOL = 0.01  # rad  per-joint band used for the settling time


def _settling_time(t, err_norm, tol):
    """First time after which the error norm stays inside ``tol`` (nan if never)."""
    outside = np.flatnonzero(err_norm > tol)
    if len(outside) == 0:
        return float(t[0])
    if outside[-1] == len(t) - 1:
        return float("nan")
    return float(t[outside[-1] + 1])


def compute_metrics(log, T_target=None, torque_limits=None, traj_duration=None, plan=None):
    """Return a flat dict of scalar metrics (plus per-joint lists) for one run.

    Parameters
    ----------
    log : ExecutionLog
    T_target : 4x4 desired end-effector pose (goal of the task); enables final-pose errors
    torque_limits : (6,) actuator limits; defaults to the ones stored in the log
    traj_duration : seconds of the moving part (default from the log meta)
    plan : optional PlanResult, adds planning time / path length / status
    """
    t = log.t
    dt = float(log.meta.get("control_period_s", t[1] - t[0]))
    limits = np.asarray(torque_limits if torque_limits is not None else log.meta["torque_limits"], dtype=float)
    T_move = float(traj_duration if traj_duration is not None else log.meta.get("trajectory_duration_s", t[-1]))

    err = log.q_des - log.q
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    max_err = np.max(np.abs(err), axis=0)
    moving = t <= T_move
    rmse_move = np.sqrt(np.mean(err[moving] ** 2, axis=0))

    tau_req = log.tau_req[:-1]            # last row only repeats the previous command
    tau_app = log.tau_app[:-1]
    sat = np.abs(tau_req) > limits + 1e-9
    m = {
        "rmse_rad": rmse.tolist(),
        "rmse_rad_moving_phase": rmse_move.tolist(),
        "rmse_rad_rms_over_joints": float(np.sqrt(np.mean(rmse ** 2))),
        "max_error_rad": max_err.tolist(),
        "max_error_rad_overall": float(max_err.max()),
        "final_joint_error_rad": err[-1].tolist(),
        "final_joint_error_norm_rad": float(np.linalg.norm(err[-1])),
        "final_speed_norm_rad_s": float(np.linalg.norm(log.qd[-1])),
        "settling_time_s": _settling_time(t, np.max(np.abs(err), axis=1), JOINT_SETTLE_TOL),
        "torque_rms_Nm": np.sqrt(np.mean(tau_app ** 2, axis=0)).tolist(),
        "torque_peak_Nm": np.max(np.abs(tau_app), axis=0).tolist(),
        "torque_requested_peak_Nm": np.max(np.abs(tau_req), axis=0).tolist(),
        "saturation_time_s_per_joint": (sat.sum(axis=0) * dt).tolist(),
        "saturation_time_s_any_joint": float(np.any(sat, axis=1).sum() * dt),
        "collision_free": bool(np.all(log.n_contacts == 0)),
        "collision_samples": int(np.count_nonzero(log.n_contacts)),
        "collision_pairs": [list(p) for p in log.contact_pairs],
        "duration_s": float(t[-1]),
    }
    if T_target is not None:
        p_end = log.ee_pos[-1]
        pos_err = float(np.linalg.norm(np.asarray(T_target)[:3, 3] - p_end))
        m["final_ee_position_error_m"] = pos_err
        # end-effector orientation: reconstructed from the final joint state with our own FK
        from .forward_kinematics import forward_kinematics
        R_end = forward_kinematics(log.q[-1]).rotation
        ori_err = float(orientation_angle(np.asarray(T_target)[:3, :3], R_end))
        m["final_ee_orientation_error_rad"] = ori_err
        m["holds_target_within_tolerance"] = bool(pos_err <= POSITION_TOL and ori_err <= ORIENTATION_TOL)
    if plan is not None:
        m["planning_success"] = bool(plan.success)
        m["planning_status"] = plan.status
        m["planning_time_s"] = float(plan.planning_time)
        m["planning_path_length_rad"] = float(plan.path_length)
        m["planning_seed"] = int(plan.seed)
        m["planning_method"] = plan.method
    return m


def ee_path_error(log, ee_ref):
    """RMS and max distance [m] between the logged EE path and a reference EE path (N,3)."""
    d = np.linalg.norm(log.ee_pos[:len(ee_ref)] - np.asarray(ee_ref), axis=1)
    return float(np.sqrt(np.mean(d ** 2))), float(d.max())
