"""Continuous, bounded-object conveyor pick-and-place demonstration.

The conveyor drive is a scripted planar belt: ungrasped free bodies advance at
the configured belt speed while retaining their orientation.  Once the two
finger pads contact an item, its free-body dynamics take over and the gripper
must carry it using simulated contacts and friction.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from . import robot_parameters as rp
from .collision import CollisionChecker
from .control import PDGravityController
from .forward_kinematics import forward_kinematics
from .gravity import DHGravityModel
from .inverse_kinematics import inverse_kinematics
from .carry_planning import CarryPlan, plan_carry_route
from .scene import Obstacle, Payload, SceneConfig, scene_xml
from .trajectory import QuinticTrajectory
from .utils import rpy_to_matrix


@dataclass(frozen=True)
class ItemSpec:
    name: str
    mass: float = 0.12
    half_size: tuple = (0.03, 0.03, 0.03)
    bin_id: int = 0
    rgba: tuple = (0.18, 0.62, 0.92, 1.0)
    max_grip_force_n: float = 25.0

    def validate(self):
        size = np.asarray(self.half_size, dtype=float)
        if not self.name or not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("item name and positive finite mass are required")
        if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise ValueError("half_size must contain three positive finite values")
        if self.bin_id not in (0, 1):
            raise ValueError("bin_id must be 0 or 1")
        if self.max_grip_force_n <= 0 or not np.isfinite(self.max_grip_force_n):
            raise ValueError("max_grip_force_n must be positive and finite")


@dataclass(frozen=True)
class ConveyorConfig:
    # The installed robot is yawed pi radians. World -y is therefore its
    # forward direction and world -x is its right direction.
    # The belt indexes blocks along +x toward P.
    belt_speed: float = 0.05              # m/s indexing speed, along +world x
    belt_ramp_time: float = 0.5           # smooth acceleration/deceleration duration [s]
    spawn_interval: float = 10.0          # simulation seconds; below cycle rate creates intentional misses
    spawn_policy: str = "ready"            # ready: one item at a time; paced: fixed interval
    max_items: int = 5
    pickup_x: float = 0.00
    spawn_x: float = -0.40
    belt_y: float = -0.65
    belt_half_width: float = 0.10
    belt_top: float = 0.342
    belt_half_length: float = 0.52
    # Bins sit on opposite sides of the arm, not beside the conveyor.
    # After installation yaw, bin 0 (robot right) is at negative world x;
    # bin 1 (robot left) is at positive world x.
    bin_centers: tuple = ((-0.45, -0.05), (0.45, -0.05))
    # The table top is z=0.34 m.  Raised bin floors sit 10 mm above it rather
    # than intersecting the work surface.
    work_surface_top: float = 0.34
    bin_floor: float = 0.35
    bin_inner_half_size: float = 0.11
    bin_wall_thickness: float = 0.01
    bin_wall_height: float = 0.07
    recycle_delay: float = 3.0
    seed: int = 7
    obstacles: tuple = ()
    obstacle_margin: float = 0.03
    items: tuple = (
        ItemSpec("blue_light", 0.12, (0.03, 0.03, 0.03), 0, (0.12, 0.48, 0.92, 1), 25.0),
        ItemSpec("orange_heavy", 0.28, (0.035, 0.03, 0.03), 1, (0.95, 0.34, 0.08, 1), 35.0),
    )

    def validate(self):
        for name in ("belt_speed", "belt_ramp_time", "spawn_interval", "recycle_delay", "obstacle_margin",
                     "belt_half_width", "belt_half_length", "bin_inner_half_size",
                     "bin_wall_thickness", "bin_wall_height"):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(self.work_surface_top) or not np.isfinite(self.bin_floor):
            raise ValueError("work_surface_top and bin_floor must be finite")
        if self.bin_floor < self.work_surface_top:
            raise ValueError("bin_floor must not be below the work surface")
        if self.spawn_policy not in ("ready", "paced"):
            raise ValueError("spawn_policy must be 'ready' or 'paced'")
        if isinstance(self.max_items, bool) or self.max_items < 1 or self.max_items > 12:
            raise ValueError("max_items must be in 1..12")
        if len(self.items) < 1 or len(self.bin_centers) != 2:
            raise ValueError("configure at least one item type and exactly two bins")
        bin_delta = np.abs(np.asarray(self.bin_centers[0], dtype=float)
                           - np.asarray(self.bin_centers[1], dtype=float))
        if bin_delta.shape != (2,) or not np.all(np.isfinite(bin_delta)):
            raise ValueError("bin_centers must contain two finite planar coordinates")
        bin_outer_half = self.bin_inner_half_size + self.bin_wall_thickness
        if np.max(bin_delta) < 2.0 * bin_outer_half:
            raise ValueError("bin tray footprints overlap; separate their centres")
        for item in self.items:
            item.validate()
        for item in self.items:
            if item.bin_id >= len(self.bin_centers):
                raise ValueError("item references a missing bin")
        return self


@dataclass
class ItemSlot:
    index: int
    spec: ItemSpec
    joint_id: int
    qpos_adr: int
    dof_adr: int
    geom_id: int
    state: str = "inactive"     # inactive, belt, grasped, placed, missed
    spawned_at: float = 0.0
    recycle_at: float = 0.0
    miss_reason: str = ""


class ConveyorSimulation:
    """MuJoCo scene, bounded item pool, robot controller and task state machine."""

    TOOL_R = rpy_to_matrix(np.pi, 0.0, np.pi)   # tool approach points downward
    BASE_YAW = np.pi
    R_WORLD_FROM_BASE = np.diag([-1.0, -1.0, 1.0])
    R_BASE_FROM_WORLD = R_WORLD_FROM_BASE
    GRASP_OFFSET = 0.105                        # flange/site to box centre [m]
    FINGER_OPEN_INNER_GAP = 0.116               # 2*(0.070 m center offset - 0.012 m pad half-width)
    GRIP_COMPRESSION = 0.0020                   # total nominal pad compression [m]

    def __init__(self, config: ConveyorConfig | None = None, obstacles=None):
        self.config = (config or ConveyorConfig()).validate()
        self.obstacles = tuple(self.config.obstacles if obstacles is None else obstacles)
        self.model = mujoco.MjModel.from_xml_string(self._xml(self.obstacles))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.joints = np.array([self.model.joint(n).id for n in rp.MJCF_JOINT_NAMES])
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        self.motor_ids = np.array([self.model.actuator(f"torque{i}").id for i in range(1, 7)])
        self.finger_act = [self.model.actuator("grip_left").id,
                           self.model.actuator("grip_right").id]
        self.finger_joints = [self.model.joint("grip_left_joint").id,
                              self.model.joint("grip_right_joint").id]
        self.finger_qadr = self.model.jnt_qposadr[self.finger_joints]
        self.finger_dadr = self.model.jnt_dofadr[self.finger_joints]
        self.item_slots: list[ItemSlot] = []
        for i in range(self.config.max_items):
            spec = self.config.items[i % len(self.config.items)]
            j = self.model.joint(f"item{i}_free").id
            g = self.model.geom(f"item{i}_geom").id
            slot = ItemSlot(i, spec, j, int(self.model.jnt_qposadr[j]),
                             int(self.model.jnt_dofadr[j]), g)
            self.item_slots.append(slot)
            self.model.geom_contype[g] = 0
            self.model.geom_conaffinity[g] = 0
        self.sim_time = 0.0
        self.rng = np.random.default_rng(self.config.seed)
        self.spawned = self.picked = self.placed = self.missed = 0
        self.failed_grasps = self.delayed_spawns = self.replans = 0
        self.max_live = 0
        self._next_spawn = 0.5
        self._spawn_type = 0
        self._task_state = "HOME"
        self._slot: ItemSlot | None = None
        self._phase_time = 0.0
        self._motion: QuinticTrajectory | None = None
        self._motion_elapsed = 0.0
        self._standby_q = self._standby_for_spec(self.config.items[0])
        self._q_command = self._standby_q.copy()
        self._qd_command = np.zeros(6)
        self._last_ik = 0.0
        self._pickup_ik_ref = None
        self._grip_open = True
        self._grip_deadline = 0.0
        self._settle_stable_time = 0.0
        self._settle_deadline = 0.0
        self._drop_stable_time = 0.0
        self._drop_deadline = 0.0
        self._belt_state = "stopped"
        self._belt_elapsed = 0.0
        self._belt_speed_now = 0.0
        self._carry_controller = None
        self._carry_plan: CarryPlan | None = None
        self._carry_plan_history = []
        self._unexpected_obstacle_contacts = 0
        self._fault_q = None
        self._carry_checkers = {}
        self._robot_checkers = None
        self._fault = ""
        self._last_contacts = []
        self._last_grasp_force = 0.0
        self._last_grasp_forces = {"left": 0.0, "right": 0.0}
        self._min_finger_item_contact_dist = float("inf")
        self._max_finger_item_penetration = 0.0
        self._peak_joint_torque = np.zeros(6)
        self._track_error_max = 0.0
        self._log = []
        self._events = []
        self._miss_reasons = {}
        self._grasp_sync = []
        self._controller_key = object()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self._standby_q
        self.data.ctrl[self.finger_act] = 0.0
        for slot in self.item_slots:
            self._park_slot(slot)
        mujoco.mj_forward(self.model, self.data)
        self.checker = CollisionChecker(mujoco.MjModel.from_xml_string(
            self._xml(self.obstacles, include_items=False, include_gripper=False)))
        self.controller = PDGravityController(DHGravityModel())
        self._sync_commands()

    def _xml(self, obstacles=(), include_items=True, include_gripper=True, payload=None):
        root = ET.fromstring(scene_xml(SceneConfig(obstacles=tuple(obstacles), payload=payload,
                                                   base_yaw=self.BASE_YAW)))
        world = root.find("worldbody")
        link6 = root.find(".//body[@name='link6']")
        if link6 is not None:
            claw_proxy = link6.find("./geom[@name='collision_link6']")
        if claw_proxy is not None:
            link6.remove(claw_proxy)
        contact = root.find("contact")
        actuator = root.find("actuator")
        # The simplified collision capsules for these two nonadjacent links
        # overlap at the ready and vertical pick poses even when the render
        # meshes are clear. Treat that proxy overlap as a known model artifact;
        # obstacle and all other self-collision checks remain active.
        ET.SubElement(contact, "exclude", body1="link2", body2="link4")
        # The belt is in front of the arm.  The broad table supports both bins;
        # only the narrow central strip indexes loose blocks.
        ET.SubElement(world, "geom", name="work_surface", type="box",
                      pos=f"0 {self.config.belt_y / 2.0} {self.config.work_surface_top-0.022}",
                      size=f"0.78 {abs(self.config.belt_y) / 2.0 + 0.30} 0.022",
                      rgba="0.10 0.12 0.15 1", contype="0", conaffinity="0")
        ET.SubElement(world, "geom", name="conveyor_belt", type="box",
                      pos=f"0 {self.config.belt_y} {self.config.belt_top-0.025}",
                      size=f"{self.config.belt_half_length} {self.config.belt_half_width} 0.025",
                      rgba="0.12 0.16 0.20 1", contype="2", conaffinity="2", friction="0.8 0.08 0.02")
        for i, (x, y) in enumerate(self.config.bin_centers):
            color = ("0.10 0.38 0.78 1" if i == 0 else "0.82 0.32 0.10 1")
            rim_color = ("0.10 0.38 0.78 0.55" if i == 0 else "0.82 0.32 0.10 0.55")
            inner = self.config.bin_inner_half_size
            wall = self.config.bin_wall_thickness
            outer = inner + wall
            wall_z = self.config.bin_floor + self.config.bin_wall_height / 2.0
            # A 10 mm floor rests on top of the table rather than overlapping
            # it.  ``bin_floor`` remains the top/support height used by drop
            # metrics and wall geometry.
            ET.SubElement(world, "geom", name=f"bin{i}_floor", type="box",
                          pos=f"{x} {y} {self.config.bin_floor-0.005}", size=f"{outer} {outer} 0.005",
                          rgba=color, contype="2", conaffinity="2", friction="0.8 0.08 0.02")
            # Walls collide with released items. Robot geoms use contact class
            # 1 while bins use class 2, so the planned vertical gripper entry
            # remains kinematic; the block itself (affinity 3) is contained.
            for dx, dy, sx, sy in ((0, outer, outer, wall), (0, -outer, outer, wall),
                                   (outer, 0, wall, outer), (-outer, 0, wall, outer)):
                ET.SubElement(world, "geom", name=f"bin{i}_rim_{dx}_{dy}", type="box",
                              pos=f"{x+dx} {y+dy} {wall_z}",
                              size=f"{sx} {sy} {self.config.bin_wall_height / 2.0}", rgba=rim_color,
                              contype="2", conaffinity="2", friction="0.8 0.08 0.02")
        if include_gripper:
            link6 = world.find(".//body[@name='link6']")
            body = ET.SubElement(link6, "body", name="gripper_base", pos="0 0 0.04")
            ET.SubElement(body, "geom", name="gripper_palm", type="box", pos="0 0 0.015",
                          size="0.045 0.035 0.018", rgba="0.12 0.16 0.2 1", contype="1", conaffinity="1")
            for side, x, axis in (("left", -0.07, "1 0 0"), ("right", 0.07, "-1 0 0")):
                finger = ET.SubElement(body, "body", name=f"grip_{side}_body", pos=f"{x} 0 0")
                ET.SubElement(finger, "joint", name=f"grip_{side}_joint", type="slide", axis=axis,
                              range="0 0.055", limited="true", damping="0.2")
                ET.SubElement(finger, "inertial", pos="0 0 0.09", mass="0.08",
                              diaginertia="0.00025 0.00025 0.00004")
                ET.SubElement(finger, "geom", name=f"finger_{side}_geom", type="box", pos="0 0 0.09",
                              size="0.012 0.026 0.065", rgba="0.25 0.30 0.34 1",
                              contype="1", conaffinity="1", friction="1.0 0.01 0.001",
                              solref="0.008 1", solimp="0.95 0.99 0.001")
                ET.SubElement(actuator, "position", name=f"grip_{side}", joint=f"grip_{side}_joint",
                              kp="5000", kv="45", ctrllimited="true", ctrlrange="0 0.055",
                              forcelimited="true", forcerange="-35 35")
            ET.SubElement(contact, "exclude", body1="grip_left_body", body2="grip_right_body")
            ET.SubElement(contact, "exclude", body1="link6", body2="gripper_base")
        if payload is not None:
            payload_geom = root.find(".//geom[@name='payload_collision']")
            payload_geom.set("contype", "1")
            payload_geom.set("conaffinity", "3")
            # These contacts are intentional in the real grasp; the carry model
            # treats the box as rigidly attached and checks its outside envelope.
            for body_name in ("gripper_base", "grip_left_body", "grip_right_body"):
                ET.SubElement(contact, "exclude", body1="payload", body2=body_name)
        if include_items:
            for i in range(self.config.max_items):
                spec = self.config.items[i % len(self.config.items)]
                hx, hy, hz = map(float, spec.half_size)
                mass = float(spec.mass)
                inertia = mass / 3.0 * np.array([hy*hy + hz*hz, hx*hx + hz*hz, hx*hx + hy*hy])
                b = ET.SubElement(world, "body", name=f"item{i}_body", pos=f"4 {i*0.2} 2.0")
                ET.SubElement(b, "freejoint", name=f"item{i}_free")
                ET.SubElement(b, "inertial", pos="0 0 0", mass=str(mass), diaginertia=" ".join(map(str, inertia)))
                ET.SubElement(b, "geom", name=f"item{i}_geom", type="box", size=f"{hx} {hy} {hz}",
                              rgba=" ".join(map(str, spec.rgba)), contype="1", conaffinity="3",
                              friction="1.0 0.02 0.002", solref="0.008 1", solimp="0.95 0.99 0.001")
        return ET.tostring(root, encoding="unicode")

    def _park_slot(self, slot):
        self.data.qpos[slot.qpos_adr:slot.qpos_adr+3] = [4.0, slot.index * 0.2, 2.0]
        self.data.qpos[slot.qpos_adr+3:slot.qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[slot.dof_adr:slot.dof_adr+6] = 0.0
        self.model.geom_contype[slot.geom_id] = 0
        self.model.geom_conaffinity[slot.geom_id] = 0
        slot.state = "inactive"

    def _sync_commands(self):
        self.data.ctrl[self.finger_act] = 0.0 if self._grip_open else self._grip_target()

    def _grip_target(self):
        if self._slot is None:
            return 0.0
        # Fingers close along x, so use the item's x width.  The former target
        # used a nominal 120 mm gap although the actual open pad gap is 116 mm,
        # causing about 2 mm of geometric overlap on each side.
        width = 2.0 * self._slot.spec.half_size[0]
        target_gap = width - self.GRIP_COMPRESSION
        travel = np.clip((self.FINGER_OPEN_INNER_GAP - target_gap) / 2.0, 0.0, 0.05)
        return np.array([travel, travel])

    @staticmethod
    def _smoothstep5(u):
        u = float(np.clip(u, 0.0, 1.0))
        return 10*u**3 - 15*u**4 + 6*u**5

    def _start_belt(self):
        if self._belt_state == "stopped":
            self._belt_state = "accelerating"
            self._belt_elapsed = 0.0
            self._events.append({"time": self.sim_time, "event": "belt_started",
                                 "speed_m_s": self.config.belt_speed})

    def _update_belt(self):
        """Prescribe a smooth shared belt velocity and stop at the pickup mark.

        The boxes remain free bodies for grasp physics, but their belt travel is
        scripted.  This S-curve has zero velocity and acceleration at the ends
        of each ramp to avoid a velocity step at start/stop.
        """
        items = [slot for slot in self.item_slots if slot.state == "belt"]
        if not items:
            self._belt_state = "stopped"
            self._belt_speed_now = 0.0
            self._belt_elapsed = 0.0
            return
        T = self.config.belt_ramp_time
        speed = self.config.belt_speed
        if self._belt_state == "stopped":
            for slot in items:
                self.data.qvel[slot.dof_adr:slot.dof_adr+6] = 0.0
            return
        if self._belt_state == "cruising":
            leading = max(items, key=lambda slot: self._slot_position(slot)[0])
            stop_start_x = self.config.pickup_x - speed * (T / 2.0 + self.dt)
            if self._slot_position(leading)[0] >= stop_start_x:
                self._belt_state = "decelerating"
                self._belt_elapsed = 0.0
        if self._belt_state == "accelerating":
            u = self._belt_elapsed / T
            self._belt_speed_now = speed * self._smoothstep5(u)
        elif self._belt_state == "cruising":
            self._belt_speed_now = speed
        elif self._belt_state == "decelerating":
            u = self._belt_elapsed / T
            self._belt_speed_now = speed * (1.0 - self._smoothstep5(u))
        else:
            self._belt_state = "stopped"
            self._belt_speed_now = 0.0
        for slot in items:
            self.data.qpos[slot.qpos_adr] += self._belt_speed_now * self.dt
            self.data.qvel[slot.dof_adr:slot.dof_adr+6] = [self._belt_speed_now, 0.0, 0.0, 0.0, 0.0, 0.0]
            if self._slot is not slot and self._slot_position(slot)[0] > self.config.belt_half_length:
                self._mark_missed(slot, "belt_exit_unpicked")
        if self._belt_state in ("accelerating", "decelerating"):
            self._belt_elapsed += self.dt
            if self._belt_elapsed >= T:
                if self._belt_state == "accelerating":
                    self._belt_state = "cruising"
                    self._belt_speed_now = speed
                else:
                    self._belt_state = "stopped"
                    self._belt_speed_now = 0.0
                    # Snap only the tiny integration remainder to the mark.
                    leading = max(items, key=lambda slot: self._slot_position(slot)[0])
                    self.data.qpos[leading.qpos_adr] = self.config.pickup_x
                    self.data.qvel[leading.dof_adr:leading.dof_adr+6] = 0.0
                    self._events.append({"time": self.sim_time, "event": "belt_stopped",
                                         "item": leading.spec.name, "slot": leading.index,
                                         "position_x_m": float(self.config.pickup_x)})

    def _spawn(self):
        free = next((s for s in self.item_slots if s.state == "inactive"), None)
        if free is None:
            self.delayed_spawns += 1
            return False
        # Pool slots have a fixed compiled item type (mass/inertia/size).
        desired = self.config.items[self._spawn_type % len(self.config.items)]
        candidates = [s for s in self.item_slots if s.state == "inactive" and s.spec.name == desired.name]
        if candidates:
            free = candidates[0]
        else:
            # Reuse any idle slot if the next category's dedicated slots are
            # occupied.  The geometry/mass stays the slot's compiled type.
            candidates = [s for s in self.item_slots if s.state == "inactive"]
            if not candidates:
                self.delayed_spawns += 1
                return False
            free = candidates[0]
        self._spawn_type = (self._spawn_type + 1) % len(self.config.items)
        z = self.config.belt_top + free.spec.half_size[2] + 0.004
        self.data.qpos[free.qpos_adr:free.qpos_adr+3] = [self.config.spawn_x, self.config.belt_y, z]
        self.data.qpos[free.qpos_adr+3:free.qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[free.dof_adr:free.dof_adr+6] = [self._belt_speed_now, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.model.geom_contype[free.geom_id] = 1
        self.model.geom_conaffinity[free.geom_id] = 3
        free.state = "belt"
        free.spawned_at = self.sim_time
        self.spawned += 1
        self._start_belt()
        return True

    def _recycle(self, slot):
        self._park_slot(slot)
        mujoco.mj_forward(self.model, self.data)

    def _mark_missed(self, slot, reason, recycle_delay=0.1):
        slot.state = "missed"
        slot.miss_reason = reason
        slot.recycle_at = self.sim_time + recycle_delay
        self.missed += 1
        self._miss_reasons[reason] = self._miss_reasons.get(reason, 0) + 1
        self._events.append({"time": self.sim_time, "event": "missed", "item": slot.spec.name,
                             "slot": slot.index, "reason": reason,
                             "position_m": self._slot_position(slot).tolist()})

    def _record_grasp_sync(self, slot):
        from .jacobian import end_effector_twist
        q, qd = self.state()
        twist = end_effector_twist(q, qd)
        twist = np.r_[self.R_WORLD_FROM_BASE @ twist[:3],
                      self.R_WORLD_FROM_BASE @ twist[3:]]
        site = self.model.site("ee_site").id
        R = self.data.site_xmat[site].reshape(3, 3)
        offset = R @ np.array([0.0, 0.0, self.GRASP_OFFSET])
        tool_velocity_at_box = twist[:3] + np.cross(twist[3:], offset)
        item_velocity = self.data.qvel[slot.dof_adr:slot.dof_adr+3].copy()
        relative_velocity = item_velocity - tool_velocity_at_box
        belt_axis = np.array([1.0, 0.0, 0.0])
        sample = {"time": self.sim_time, "item": slot.spec.name, "slot": slot.index,
                  "relative_position_error_m": self._grasp_relative_error(slot),
                  "relative_velocity_error_m_s": float(np.linalg.norm(relative_velocity)),
                  "relative_velocity_error_vector_m_s": relative_velocity.tolist(),
                  "belt_axis_velocity_error_m_s": float(abs(np.dot(relative_velocity, belt_axis))),
                  "finger_contact_normal_forces_n": dict(self._last_grasp_forces),
                  "required_jaw_normal_force_n": float(self._required_jaw_force(slot)),
                  "belt_velocity_m_s": self.config.belt_speed}
        self._grasp_sync.append(sample)
        self._events.append({"time": self.sim_time, "event": "grasped", **sample})

    def _slot_position(self, slot):
        return self.data.qpos[slot.qpos_adr:slot.qpos_adr+3].copy()

    def _live_count(self):
        return sum(s.state != "inactive" for s in self.item_slots)

    def _check_pool_invariant(self):
        live = self._live_count()
        if live > self.config.max_items:
            raise RuntimeError(f"item pool invariant violated: {live} live > {self.config.max_items}")
        return live

    def _item_contacts(self, slot):
        found = {"left": False, "right": False}
        forces = {"left": 0.0, "right": 0.0}
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            names = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)),
                     mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)))
            for side in ("left", "right"):
                if f"finger_{side}_geom" in names and f"item{slot.index}_geom" in names:
                    found[side] = True
                    mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    # MuJoCo reports the contact wrench in the contact frame;
                    # component 0 is the compressive normal force in newtons.
                    forces[side] += max(0.0, float(wrench[0]))
        return found, forces

    def _required_jaw_force(self, slot):
        # Symmetric two-pad grasp: 2*mu*N must support m*(g+a_z).
        # Use a conservative effective friction estimate and 1.5 safety factor.
        effective_mu = 1.0  # modeled pad and item sliding-friction coefficients are both 1.0
        return 1.5 * slot.spec.mass * (9.81 + 1.0) / (2.0 * effective_mu)

    def _grasp_relative_error(self, slot):
        site = self.model.site("ee_site").id
        p_ee = self.data.site_xpos[site]
        R_ee = self.data.site_xmat[site].reshape(3, 3)
        expected = p_ee + R_ee @ np.array([0.0, 0.0, self.GRASP_OFFSET])
        return float(np.linalg.norm(self._slot_position(slot) - expected))

    def _world_to_base_position(self, position):
        return self.R_BASE_FROM_WORLD @ np.asarray(position, dtype=float)

    def _base_to_world_position(self, position):
        return self.R_WORLD_FROM_BASE @ np.asarray(position, dtype=float)

    def _target_pose(self, position):
        T = np.eye(4)
        # The physical installation is yawed, but the gripper is symmetric
        # about its downward approach axis. Preserve the original downward
        # tool orientation *relative to the installed base*; transforming the
        # same orientation a second time selected a wrist-roll branch roughly
        # pi radians away at pickup.
        T[:3, :3] = self.TOOL_R
        T[:3, 3] = self._world_to_base_position(position)
        return T

    def _target_joints(self, position, q_ref, checker=None):
        result = inverse_kinematics(self._target_pose(position), q_ref=q_ref)
        options = [s for s in result.solutions if s.within_limits
                   and (checker is None or checker.is_configuration_valid(s.q))]
        if not options:
            qualifier = "collision-free " if checker is not None else ""
            raise RuntimeError(f"no in-limit {qualifier}IK solution for target ({result.status})")
        return min(options, key=lambda s: np.linalg.norm(s.q - q_ref)).q.copy()

    @staticmethod
    def _standby_for_spec(spec):
        return (rp.Q_CONVEYOR_STANDBY_RIGHT if spec.bin_id == 0
                else rp.Q_CONVEYOR_STANDBY_LEFT).copy()

    def _select_pickup_branch(self, slot, flange, q_ref):
        """Choose the grasp IK branch that is compact for this item's bin.

        A PUMA has two valid elbow branches at the central pickup station.
        The light/right bin is compact from one branch and the heavy/left bin
        from the other.  Selecting solely by closeness to standby made blue
        switch elbows after grasping, producing the visible acrobatics.
        """
        result = inverse_kinematics(self._target_pose(flange), q_ref=q_ref)
        grasp_options = []
        limits = rp.JOINT_LIMITS
        for sol in result.solutions:
            if not sol.within_limits:
                continue
            # The analytic wrist solution is periodic.  Keep equivalent q6
            # representations that are still inside the real joint range;
            # the branch best for blue is reported as q6-2pi by the solver,
            # although q6 itself is legal and continuous for the carry.
            for turns in (-1, 0, 1):
                candidate = sol.q.copy()
                candidate[5] += turns * 2.0 * np.pi
                if np.all(candidate >= limits[:, 0]) and np.all(candidate <= limits[:, 1]):
                    if not any(np.linalg.norm(candidate - prior) < 1e-8 for prior in grasp_options):
                        grasp_options.append(candidate)
        if not grasp_options:
            raise RuntimeError(f"no in-limit pickup IK solution ({result.status})")
        tool_to_item = self.TOOL_R @ np.array([0.0, 0.0, self.GRASP_OFFSET])
        bx, by = self.config.bin_centers[slot.spec.bin_id]
        release_center_z = (self.config.bin_floor + self.config.bin_wall_height
                            + slot.spec.half_size[2] + 0.040)
        release = np.array([bx, by, release_center_z]) - tool_to_item
        ranked = []
        for q_grasp in grasp_options:
            try:
                q_lift = self._target_joints(flange + [0.0, 0.0, 0.20], q_grasp)
                q_release = self._target_joints(release, q_lift)
            except RuntimeError:
                continue
            # Carry compactness is the dominant criterion. The small standby
            # term only breaks a tie between equally good carrying branches.
            # cost = np.linalg.norm(q_release - q_lift) + 0.05 * np.linalg.norm(q_grasp - q_ref)
            weights = np.array([1.0, 1.0, 1.0, 0.75, 0.75, 0.5])

            cost = (
            np.linalg.norm(weights * (q_grasp - q_ref))
                + np.linalg.norm(weights * (q_lift - q_grasp))
                + np.linalg.norm(weights * (q_release - q_lift))
            )
            ranked.append((cost, q_grasp))
        if not ranked:
            raise RuntimeError("no pickup branch can reach the configured bin")
        return min(ranked, key=lambda pair: pair[0])[1]

    def _track_item(self, slot, z_clearance, belt_velocity=True, q_ik_ref=None):
        """IK tracking update at 50 Hz; feed-forward velocity matches the belt."""
        q, qd = self.data.qpos[self.qadr].copy(), self.data.qvel[self.vadr].copy()
        pos = self._slot_position(slot)
        # With z_E down, the item centre lies 105 mm along the tool z axis.
        flange = pos - self.TOOL_R @ np.array([0.0, 0.0, self.GRASP_OFFSET])
        flange[2] += z_clearance
        q_new = self._target_joints(flange, q_ref=(self._q_command if q_ik_ref is None else q_ik_ref))
        update_dt = max(self.dt, min(self.sim_time - self._last_ik, 0.04))
        max_joint_step = np.array([0.8, 0.8, 0.8, 1.2, 1.2, 1.5]) * update_dt
        delta = np.clip(q_new - self._q_command, -max_joint_step, max_joint_step)
        old_command = self._q_command.copy()
        self._q_command = old_command + delta
        world_linear = np.array([self.config.belt_speed if belt_velocity else 0.0, 0.0, 0.0])
        twist = np.r_[self.R_BASE_FROM_WORLD @ world_linear, np.zeros(3)]
        from .jacobian import joint_velocity_from_twist
        qd_new = joint_velocity_from_twist(q_new, twist, method="dls", damping=0.04)
        qd_from_path = delta / update_dt
        self._qd_command = np.clip(0.5 * qd_new + 0.5 * qd_from_path, -0.8, 0.8)

    def _move_trajectory(self, waypoints):
        traj = QuinticTrajectory(np.asarray(waypoints, dtype=float), dt=self.dt,
                                 max_velocity=np.array([0.6, 0.6, 0.6, 1.0, 1.0, 1.2]),
                                 max_acceleration=np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.5]))
        self._motion = traj
        self._motion_elapsed = 0.0
        self._task_state = "CARRY"

    def _carry_collision_checkers(self, slot):
        key = (slot.spec.name, tuple(slot.spec.half_size), tuple(self.obstacles), self.config.obstacle_margin)
        if key in self._carry_checkers:
            return self._carry_checkers[key]
        payload = Payload(slot.spec.mass, (0.0, 0.0, self.GRASP_OFFSET), slot.spec.half_size)
        inflated = tuple(Obstacle(o.position, tuple(np.asarray(o.size, dtype=float)
                                                    + self.config.obstacle_margin))
                         for o in self.obstacles)
        plan_model = self._compile_checker_model(inflated, payload=payload)
        true_model = self._compile_checker_model(self.obstacles, payload=payload)
        # Carry geometry includes closed fingers. The attached-payload model
        # excludes intended pad/block contacts while retaining obstacle tests.
        width = 2.0 * slot.spec.half_size[0]
        travel = float(np.clip((self.FINGER_OPEN_INNER_GAP
                                 - (width - self.GRIP_COMPRESSION)) / 2.0, 0.0, 0.05))
        overrides = {"grip_left_joint": travel, "grip_right_joint": travel}
        checkers = (CollisionChecker(plan_model, qpos_overrides=overrides),
                    CollisionChecker(true_model, qpos_overrides=overrides))
        self._carry_checkers[key] = checkers
        return checkers

    def _compile_checker_model(self, obstacles, payload=None):
        root = ET.fromstring(self._xml(obstacles, include_items=False,
                                       include_gripper=True, payload=payload))
        # The grasp begins with the cuboid resting on the conveyor. That support
        # contact is intentional; it must not make the route's start invalid.
        belt = root.find(".//geom[@name='conveyor_belt']")
        if belt is not None:
            belt.set("contype", "0")
            belt.set("conaffinity", "0")
        # The final release deliberately places the attached planning payload
        # inside the open bin. Bin collisions are enabled in the live physics
        # model for a released item, but must not reject that pre-release pose.
        for geom in root.findall(".//geom"):
            if (geom.get("name") or "").startswith("bin"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
        return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    def _robot_collision_checkers(self):
        """Obstacle checkers for motions with no attached payload and open jaws."""
        if self._robot_checkers is None:
            inflated = tuple(Obstacle(o.position, tuple(np.asarray(o.size, dtype=float)
                                                        + self.config.obstacle_margin))
                             for o in self.obstacles)
            plan_model = mujoco.MjModel.from_xml_string(
                self._xml(inflated, include_items=False, include_gripper=True))
            true_model = mujoco.MjModel.from_xml_string(
                self._xml(self.obstacles, include_items=False, include_gripper=True))
            overrides = {"grip_left_joint": 0.0, "grip_right_joint": 0.0}
            self._robot_checkers = (CollisionChecker(plan_model, qpos_overrides=overrides),
                                    CollisionChecker(true_model, qpos_overrides=overrides))
        return self._robot_checkers

    def _plan_return_motion(self, q):
        plan_checker, true_checker = self._robot_collision_checkers()
        route = plan_carry_route(
            plan_checker, true_checker, q, [self._standby_q], dt=self.dt,
            max_velocity=np.array([0.6, 0.6, 0.6, 1.0, 1.0, 1.2]),
            max_acceleration=np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.5]),
            seed=self.config.seed + 9001, timeout=5.0,
            clearance_lower_bound=self.config.obstacle_margin)
        self._carry_plan = route
        self._carry_plan_history.append({"time": self.sim_time, "task": "return",
                                         "result": route.summary(), "success": route.success,
                                         "status": route.status,
                                         "path_length_rad": route.path_length_rad,
                                         "planning_time_s": route.planning_time_s,
                                         "min_clearance_m": route.min_clearance_m,
                                         "direct_blocked_segments": list(route.direct_blocked_segments),
                                         "segments": list(route.segment_summaries)})
        self._events.append({"time": self.sim_time, "event": "return_plan",
                             "success": route.success, "status": route.status,
                             "path_length_rad": route.path_length_rad,
                             "planning_time_s": route.planning_time_s,
                             "min_clearance_m": route.min_clearance_m,
                             "direct_blocked_segments": list(route.direct_blocked_segments)})
        if not route.success or route.trajectory is None:
            self._enter_task_fault(f"safe return path unavailable: {route.summary()}")
            return False
        self._motion = route.trajectory
        self._motion_elapsed = 0.0
        self._task_state = "RETURN"
        return True

    def _carry_targets(self, slot):
        q, _ = self.state()
        pos = self._slot_position(slot)
        bin_x, bin_y = self.config.bin_centers[slot.spec.bin_id]
        tool_to_item = self.TOOL_R @ np.array([0.0, 0.0, self.GRASP_OFFSET])
        # Nominal task: release from the air, rather than inserting fingers
        # into a tray.  The object's bottom clears the rim by 40 mm at the
        # stationary release pose, then MuJoCo gravity/contact settles it.
        release_center_z = (self.config.bin_floor + self.config.bin_wall_height
                            + slot.spec.half_size[2] + 0.040)
        grasp = pos - self.TOOL_R @ np.array([0.0, 0.0, self.GRASP_OFFSET])
        lift = grasp.copy(); lift[2] += 0.20
        release = np.array([bin_x, bin_y, release_center_z], dtype=float) - tool_to_item
        plan_checker, true_checker = self._carry_collision_checkers(slot)
        targets = []
        q_ref = q
        for p in (lift, release):
            q_ref = self._target_joints(p, q_ref=q_ref, checker=plan_checker)
            targets.append(q_ref)
        route = plan_carry_route(
            plan_checker, true_checker, q, targets, dt=self.dt,
            max_velocity=np.array([0.6, 0.6, 0.6, 1.0, 1.0, 1.2]),
            max_acceleration=np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.5]),
            seed=self.config.seed + slot.index * 101, timeout=5.0,
            clearance_lower_bound=self.config.obstacle_margin)
        self._carry_plan = route
        self._carry_plan_history.append({"time": self.sim_time, "item": slot.spec.name,
                                         "bin": slot.spec.bin_id, "result": route.summary(),
                                         "success": route.success, "status": route.status,
                                         "path_length_rad": route.path_length_rad,
                                         "planning_time_s": route.planning_time_s,
                                         "min_clearance_m": route.min_clearance_m,
                                         "direct_blocked_segments": list(route.direct_blocked_segments),
                                         "segments": list(route.segment_summaries)})
        self._events.append({"time": self.sim_time, "event": "carry_plan",
                             "item": slot.spec.name, "bin": slot.spec.bin_id,
                             "success": route.success, "status": route.status,
                             "path_length_rad": route.path_length_rad,
                             "planning_time_s": route.planning_time_s,
                             "min_clearance_m": route.min_clearance_m,
                             "direct_blocked_segments": list(route.direct_blocked_segments),
                             "segments": list(route.segment_summaries)})
        if not route.success or route.trajectory is None:
            raise RuntimeError(route.summary())
        self._motion = route.trajectory
        self._motion_elapsed = 0.0
        self._task_state = "CARRY"

    def _plan_retract_motion(self, q, slot):
        """Rise vertically above the release pose before moving home."""
        bin_x, bin_y = self.config.bin_centers[slot.spec.bin_id]
        tool_to_item = self.TOOL_R @ np.array([0.0, 0.0, self.GRASP_OFFSET])
        release_center_z = (self.config.bin_floor + self.config.bin_wall_height
                            + slot.spec.half_size[2] + 0.040)
        retract = np.array([bin_x, bin_y, release_center_z + 0.10]) - tool_to_item
        plan_checker, true_checker = self._robot_collision_checkers()
        q_retract = self._target_joints(retract, q_ref=q, checker=plan_checker)
        route = plan_carry_route(
            plan_checker, true_checker, q, [q_retract], dt=self.dt,
            max_velocity=np.array([0.6, 0.6, 0.6, 1.0, 1.0, 1.2]),
            max_acceleration=np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.5]),
            seed=self.config.seed + 7000 + slot.index, timeout=5.0,
            clearance_lower_bound=self.config.obstacle_margin)
        if not route.success or route.trajectory is None:
            self._enter_task_fault(f"vertical retract unavailable: {route.summary()}")
            return False
        self._motion = route.trajectory
        self._motion_elapsed = 0.0
        self._task_state = "RETRACT"
        self._events.append({"time": self.sim_time, "event": "retract_plan",
                             "item": slot.spec.name, "bin": slot.spec.bin_id,
                             "success": True, "vertical_clearance_m": 0.10})
        return True

    def _enter_task_fault(self, reason, contacts=()):
        q, _ = self.state()
        self._fault = reason
        self._fault_q = q.copy()
        self._q_command = q.copy()
        self._qd_command = np.zeros(6)
        self._motion = None
        self._task_state = "FAULT"
        self._events.append({"time": self.sim_time, "event": "task_fault",
                             "reason": reason, "contacts": [list(p) for p in contacts]})

    def state(self):
        return self.data.qpos[self.qadr].copy(), self.data.qvel[self.vadr].copy()

    def _set_controller(self):
        key = None if self._slot is None or self._slot.state != "grasped" else (self._slot.index, self._slot.spec.mass)
        if key == self._controller_key:
            return
        if key is None:
            self.controller = PDGravityController(DHGravityModel())
        else:
            self.controller = PDGravityController(
                DHGravityModel(payload=Payload(self._slot.spec.mass,
                                               (0.0, 0.0, self.GRASP_OFFSET),
                                               self._slot.spec.half_size)))
        self._controller_key = key

    def _advance_task(self):
        q, qd = self.state()
        if self._task_state == "HOME":
            self._q_command, self._qd_command = self._standby_q.copy(), np.zeros(6)
            eligible = [s for s in self.item_slots if s.state == "belt"
                        and self._belt_state == "stopped"
                        and abs(self._slot_position(s)[0] - self.config.pickup_x) <= 0.01]
            if eligible:
                self._slot = min(eligible, key=lambda s: abs(self._slot_position(s)[0] - self.config.pickup_x))
                self._phase_time = 0.0
                self._task_state = "PREGRASP"
                flange = self._slot_position(self._slot) - self.TOOL_R @ np.array([0, 0, self.GRASP_OFFSET])
                self._pickup_ik_ref = self._select_pickup_branch(self._slot, flange, q)
                self._track_item(self._slot, 0.14, belt_velocity=False, q_ik_ref=self._pickup_ik_ref)
            elif self._belt_state == "stopped" and any(s.state == "belt" for s in self.item_slots):
                # In paced mode, a queued item waits at the mark until the
                # robot returns home; then the belt indexes the next one.
                self._start_belt()
        elif self._task_state == "PREGRASP":
            if self.sim_time - self._last_ik >= 0.02:
                self._track_item(self._slot, 0.14, belt_velocity=False, q_ik_ref=self._pickup_ik_ref)
                self._last_ik = self.sim_time
            p = self._slot_position(self._slot) - self.TOOL_R @ np.array([0, 0, self.GRASP_OFFSET])
            p[2] += 0.14
            ee = self._base_to_world_position(forward_kinematics(q).position)
            if np.linalg.norm(ee - p) < 0.025:
                self._task_state = "DESCEND"
                self._phase_time = 0.0
        elif self._task_state == "DESCEND":
            T = 0.65
            u = np.clip(self._phase_time / T, 0.0, 1.0)
            blend = 10*u**3 - 15*u**4 + 6*u**5
            clearance = 0.14 * (1.0 - blend)
            if self.sim_time - self._last_ik >= 0.02:
                self._track_item(self._slot, clearance, belt_velocity=False, q_ik_ref=self._pickup_ik_ref)
                self._last_ik = self.sim_time
            if u >= 1.0:
                self._task_state = "SETTLE"
                self._phase_time = 0.0
                self._settle_stable_time = 0.0
                self._settle_deadline = self.sim_time + 0.8
        elif self._task_state == "SETTLE":
            if self.sim_time - self._last_ik >= 0.02:
                self._track_item(self._slot, 0.0, belt_velocity=False, q_ik_ref=self._pickup_ik_ref)
                self._last_ik = self.sim_time
            from .jacobian import end_effector_twist
            ee_speed = float(np.linalg.norm(end_effector_twist(q, qd)[:3]))
            position_error = self._grasp_relative_error(self._slot)
            if ee_speed < 0.015 and position_error < 0.005:
                self._settle_stable_time += self.dt
            else:
                self._settle_stable_time = 0.0
            if self._settle_stable_time >= 0.08:
                self._grip_open = False
                self._grip_deadline = self.sim_time + 1.3
                self._task_state = "CLOSE"
                self._phase_time = 0.0
            elif self.sim_time >= self._settle_deadline:
                self.failed_grasps += 1
                self._fault = f"arm did not settle before pickup (speed={ee_speed:.3f} m/s, error={position_error:.3f} m)"
                self._events.append({"time": self.sim_time, "event": "settle_retry",
                                     "speed_m_s": ee_speed, "position_error_m": position_error})
                self._task_state = "PREGRASP"
                self._phase_time = 0.0
        elif self._task_state == "CLOSE":
            if self.sim_time - self._last_ik >= 0.02:
                self._track_item(self._slot, 0.0, belt_velocity=False, q_ik_ref=self._pickup_ik_ref)
                self._last_ik = self.sim_time
            contacts, forces = self._item_contacts(self._slot)
            self._last_grasp_forces = dict(forces)
            self._last_grasp_force = min(forces.values())
            required = self._required_jaw_force(self._slot)
            self.model.actuator_forcerange[self.finger_act, 0] = -self._slot.spec.max_grip_force_n
            self.model.actuator_forcerange[self.finger_act, 1] = self._slot.spec.max_grip_force_n
            if all(contacts.values()) and min(forces.values()) >= required:
                self._record_grasp_sync(self._slot)
                self._slot.state = "grasped"
                self.picked += 1
                self._pickup_ik_ref = None
                self._phase_time = 0.0
                self._set_controller()
                try:
                    self._carry_targets(self._slot)
                except RuntimeError as exc:
                    # Keep the block closed in the jaws and hold position. A
                    # failed route must not silently drop the item.
                    self.failed_grasps += 1
                    self._enter_task_fault(f"carry path unavailable: {exc}")
            elif self.sim_time >= self._grip_deadline:
                self.failed_grasps += 1
                self._grip_open = True
                if self._slot_position(self._slot)[0] > self.config.pickup_x + 0.12:
                    self._mark_missed(self._slot, "pickup_passed_after_grasp_timeout")
                    self._slot = None
                    self._task_state = "HOME"
                else:
                    self._task_state = "PREGRASP"
        elif self._task_state == "CARRY":
            if self._slot is not None:
                slip = self._grasp_relative_error(self._slot)
                if slip > 0.06:
                    self._fault = f"item slipped from jaw grasp ({slip:.3f} m)"
                    self.failed_grasps += 1
                    self._grip_open = True
                    self._mark_missed(self._slot, "grasp_slip", recycle_delay=0.2)
                    self._slot = None
                    self._plan_return_motion(q)
                    return
            # Evaluate returns q, qd, qdd; retain the existing torque controller
            # interface while advancing the task trajectory.
            q_des, qd_des, _ = self._motion.evaluate(self._motion_elapsed)
            self._q_command, self._qd_command = q_des, qd_des
            self._motion_elapsed += self.dt
            if self._motion_elapsed >= self._motion.duration:
                self._q_command, self._qd_command = self._motion.evaluate(self._motion.duration)[:2]
                self._grip_open = True
                self._phase_time = 0.0
                self._task_state = "RELEASE"
        elif self._task_state == "RELEASE":
            # Keep the jaws open and motionless long enough to clear the pads,
            # then judge the *physical* fall rather than treating release as a
            # successful placement.
            if self._phase_time >= 0.20:
                self._task_state = "DROP_SETTLE"
                self._phase_time = 0.0
                self._drop_stable_time = 0.0
                self._drop_deadline = self.sim_time + 2.0
        elif self._task_state == "DROP_SETTLE":
            pos = self._slot_position(self._slot)
            linear_speed = float(np.linalg.norm(self.data.qvel[self._slot.dof_adr:self._slot.dof_adr+3]))
            bx, by = self.config.bin_centers[self._slot.spec.bin_id]
            on_bin_floor = any(f"item{self._slot.index}_geom" in pair and
                               f"bin{self._slot.spec.bin_id}_floor" in pair
                               for pair in self._last_contacts)
            inside = (abs(pos[0] - bx) <= self.config.bin_inner_half_size
                      and abs(pos[1] - by) <= self.config.bin_inner_half_size)
            if on_bin_floor and inside and linear_speed < 0.03:
                self._drop_stable_time += self.dt
            else:
                self._drop_stable_time = 0.0
            if self._drop_stable_time >= 0.15:
                self._slot.state = "placed"
                self.placed += 1
                self._slot.recycle_at = self.sim_time + self.config.recycle_delay
                self._events.append({"time": self.sim_time, "event": "placed", "item": self._slot.spec.name,
                                     "slot": self._slot.index, "position_m": pos.tolist(),
                                     "bin": self._slot.spec.bin_id,
                                     "placement_error_m": float(np.linalg.norm(pos[:2] - [bx, by])),
                                     "drop_settle_speed_m_s": linear_speed})
            elif self.sim_time < self._drop_deadline:
                return
            else:
                self._mark_missed(self._slot, "drop_did_not_settle_in_bin",
                                  recycle_delay=self.config.recycle_delay)
            slot = self._slot
            self._slot = None
            # Return on the same elbow branch used for the completed carry.
            # Switching to the *next* item's branch here made every alternating
            # blue/orange cycle perform an unnecessary visible flip at idle.
            self._standby_q = self._standby_for_spec(slot.spec)
            self._plan_retract_motion(q, slot)
        elif self._task_state == "RETRACT":
            self._q_command, self._qd_command, _ = self._motion.evaluate(self._motion_elapsed)
            self._motion_elapsed += self.dt
            if self._motion_elapsed >= self._motion.duration:
                self._q_command, self._qd_command = self._motion.evaluate(self._motion.duration)[:2]
                self._plan_return_motion(q)
        elif self._task_state == "RETURN":
            self._q_command, self._qd_command, _ = self._motion.evaluate(self._motion_elapsed)
            self._motion_elapsed += self.dt
            if self._motion_elapsed >= self._motion.duration:
                self._task_state = "HOME"
                self._q_command, self._qd_command = self._standby_q.copy(), np.zeros(6)
        elif self._task_state == "FAULT":
            if self._fault_q is not None:
                self._q_command = self._fault_q.copy()
            self._qd_command = np.zeros(6)

    def step(self):
        self._advance_task()
        self._set_controller()
        q, qd = self.state()
        error = float(np.max(np.abs(self._q_command - q)))
        self._track_error_max = max(self._track_error_max, error)
        tau = self.controller.torque(q, qd, self._q_command, self._qd_command)
        limits = np.asarray([150, 150, 100, 30, 20, 10], dtype=float)
        applied_tau = np.clip(tau, -limits, limits)
        self.data.ctrl[self.motor_ids] = applied_tau
        self._peak_joint_torque = np.maximum(self._peak_joint_torque, np.abs(applied_tau))
        self.data.ctrl[self.finger_act] = 0.0 if self._grip_open else self._grip_target()
        if self._slot is not None:
            force_cap = float(self._slot.spec.max_grip_force_n)
            self.model.actuator_forcerange[self.finger_act, 0] = -force_cap
            self.model.actuator_forcerange[self.finger_act, 1] = force_cap
        # Parked slots have contacts disabled. Belt motion uses a smooth shared
        # S-curve until the items stop at the pickup mark.
        for slot in self.item_slots:
            if slot.state in ("inactive", "placed", "missed"):
                if slot.state == "inactive":
                    self._park_slot(slot)
                elif self.sim_time >= slot.recycle_at:
                    self._recycle(slot)
            # A grasped body's gravity/contact dynamics are integrated normally.
        self._update_belt()
        mujoco.mj_forward(self.model, self.data)
        self._last_contacts = self.contacts()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            names = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)) or "",
                     mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)) or "")
            if any(name.startswith("finger_") for name in names) and any(name.endswith("_geom") and name.startswith("item") for name in names):
                self._min_finger_item_contact_dist = min(self._min_finger_item_contact_dist, float(c.dist))
                self._max_finger_item_penetration = max(self._max_finger_item_penetration, max(0.0, -float(c.dist)))
        mujoco.mj_step(self.model, self.data)
        self._last_contacts = self.contacts()
        obstacle_contacts = [pair for pair in self._last_contacts
                             if any(name.startswith("obstacle_") for name in pair)]
        if obstacle_contacts and self._task_state != "FAULT":
            self._unexpected_obstacle_contacts += len(obstacle_contacts)
            self._enter_task_fault("unexpected obstacle contact during execution", obstacle_contacts)
        self.sim_time += self.dt
        self._phase_time += self.dt
        self.max_live = max(self.max_live, self._check_pool_invariant())
        if len(self._log) == 0 or self.sim_time - self._log[-1]["time"] >= 0.1:
            self._log.append({"time": self.sim_time, "state": self._task_state,
                              "active": self._live_count(), "spawned": self.spawned,
                              "picked": self.picked, "placed": self.placed,
                              "missed": self.missed, "contacts": len(self._last_contacts),
                              "belt_state": self._belt_state,
                              "belt_speed_m_s": self._belt_speed_now,
                              "grip_force_n": self._last_grasp_force,
                              "grip_contact_normals_n": dict(self._last_grasp_forces),
                              "applied_joint_torque_nm": applied_tau.tolist(),
                              "tracking_error_rad": error})
        if self.config.spawn_policy == "paced":
            if self.sim_time >= self._next_spawn:
                self._spawn()
                self._next_spawn += self.config.spawn_interval
        elif (self.sim_time >= self._next_spawn and self._task_state == "HOME" and self._slot is None
              and not any(s.state == "belt" for s in self.item_slots)):
            # Feed only when the robot is home and no unpicked item is already
            # on the belt. The robot and belt therefore share a simple
            # readiness handshake rather than allowing arrivals to queue past it.
            if self._spawn():
                self._next_spawn = self.sim_time + min(self.config.spawn_interval, 1.0)
        return self.sim_time

    def contacts(self):
        names = []
        for c in self.data.contact[:self.data.ncon]:
            if c.dist > 0:
                continue
            a = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)) or str(c.geom1)
            b = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)) or str(c.geom2)
            names.append((a, b))
        return names

    @property
    def status(self):
        # Collapse contact manifold points to distinct geom pairs for a usable
        # terminal summary; the raw samples remain available in the log.
        unique_contacts = sorted({tuple(sorted(pair)) for pair in self._last_contacts})
        return {"sim_time_s": self.sim_time, "state": self._task_state,
                "spawned": self.spawned, "picked": self.picked, "placed": self.placed,
                "missed": self.missed, "failed_grasps": self.failed_grasps,
                "delayed_spawns": self.delayed_spawns, "max_live_items": self.max_live,
                "tracking_error_max_rad": self._track_error_max,
                "last_grip_force_n": self._last_grasp_force,
                "last_grip_contact_normals_n": dict(self._last_grasp_forces),
                "joint_torque_peak_nm": self._peak_joint_torque.tolist(),
                "min_finger_item_contact_dist_m": (None if not np.isfinite(self._min_finger_item_contact_dist)
                                                     else self._min_finger_item_contact_dist),
                "max_finger_item_penetration_m": self._max_finger_item_penetration,
                "belt_state": self._belt_state, "belt_speed_now_m_s": self._belt_speed_now,
                "last_contacts": unique_contacts, "miss_reasons": dict(self._miss_reasons),
                "grasp_sync": self._grasp_sync, "fault": self._fault,
                "carry_plan_history": self._carry_plan_history,
                "unexpected_obstacle_contacts": self._unexpected_obstacle_contacts,
                "configured_obstacles": self.obstacles}

    def save_log(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": self.config, "status": self.status, "samples": self._log,
                   "events": self._events}
        path.write_text(json.dumps(payload, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o), indent=2),
                        encoding="utf-8")
        return path


def run_headless(config=None, duration=30.0, obstacles=None):
    sim = ConveyorSimulation(config, obstacles=obstacles)
    while sim.sim_time < duration:
        sim.step()
    return sim
