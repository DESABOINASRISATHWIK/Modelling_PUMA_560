#!/usr/bin/env python
"""Conveyor demo with the robot facing the pickup and the bins on its LEFT and RIGHT.

Why: in run_conveyor_demo.py the robot base is yawed pi, which puts the pickup 90 deg to the
robot's left, bin 0 straight ahead (6 deg) and bin 1 directly BEHIND it (174 deg).  Only two IK
branches exist at that pickup, each compact for one bin only, so the arm flips ~206 deg
between every item.  Here the base is yawed -pi/2 so, in the robot's own frame, the pickup is
straight ahead and the bins are at +/-90 deg.  One IK branch (right/up/noflip) serves the
pickup AND both bins, so there is no flip and no idle swing.

    python run_conveyor_leftright_demo.py                      # interactive viewer
    python run_conveyor_leftright_demo.py --headless --seconds 60
(all options of run_conveyor_demo.py are accepted)
"""
import numpy as np

import run_conveyor_demo
from src.conveyor import ConveyorSimulation
from src.inverse_kinematics import inverse_kinematics


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class LeftRightConveyor(ConveyorSimulation):
    BASE_YAW = -np.pi / 2
    R_WORLD_FROM_BASE = _Rz(-np.pi / 2)
    R_BASE_FROM_WORLD = _Rz(-np.pi / 2).T
    TOOL_R = np.diag([1.0, -1.0, -1.0])        # tool points down; fingers close along the robot's x axis

    def _standby_for_spec(self, spec):
        """ONE standby pose above the pickup, on the common branch, for every item type."""
        if not hasattr(self, "_common_standby"):
            pick = np.array([self.config.pickup_x, self.config.belt_y, self.config.belt_top + 0.034])
            flange = pick - self.TOOL_R @ np.array([0.0, 0.0, self.GRASP_OFFSET]) + [0.0, 0.0, 0.30]
            T = np.eye(4)
            T[:3, :3] = self.TOOL_R
            T[:3, 3] = self.R_BASE_FROM_WORLD @ flange
            sols = [s for s in inverse_kinematics(T, q_ref=np.zeros(6)).solutions
                    if s.within_limits and s.label == "right/up/noflip"]
            if not sols:
                raise RuntimeError("no standby pose on the common IK branch")
            self._common_standby = sols[0].q.copy()
        return self._common_standby.copy()

    def _grip_target(self):
        """Fingers now close along the item's y extent (not x)."""
        if self._slot is None:
            return 0.0
        gap = 2.0 * self._slot.spec.half_size[1] - self.GRIP_COMPRESSION
        travel = np.clip((self.FINGER_OPEN_INNER_GAP - gap) / 2.0, 0.0, 0.05)
        return np.array([travel, travel])


if __name__ == "__main__":
    run_conveyor_demo.ConveyorSimulation = LeftRightConveyor
    run_conveyor_demo.main()
