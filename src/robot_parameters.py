"""
PUMA-560 robot specification  (single source of truth for ALL robot numbers)
============================================================================

Every other module (FK, IK, Jacobian, MJCF generator, tests, plots) imports its
robot data from here.  Nothing about the robot geometry is duplicated elsewhere.

Reference parameter set
-----------------------
Kinematic and dynamic data follow the standard-DH PUMA-560 model of

    P. Corke and B. Armstrong-Helouvry, "A search for consensus among model
    parameters reported for the PUMA 560 robot", Proc. IEEE ICRA, 1994,

as implemented in Peter Corke's Robotics Toolbox (``DH/Puma560.py``, MIT
licence).  Dynamic data originate from Armstrong, Khatib & Burdick (1986).
Joint limits are the ones listed in that model (Wang & Chen, 1991).
The tool-flange offset (0.04 m) and the link meshes come from ARTE
(A. Gil, Univ. Miguel Hernandez, LGPL-3.0), which uses the same a/d/alpha table.

Units
-----
metres, radians, kilograms, kg*m^2.  (Degrees only appear in comments.)

DH convention  (STANDARD / distal DH)
-------------------------------------
    A_{i-1,i} = Rot(z, theta_i) Trans(z, d_i) Trans(x, a_i) Rot(x, alpha_i)
    theta_i   = q_i + THETA_OFFSET_i

* Joint i is revolute and rotates about z_{i-1} (right-hand rule, +q_i is
  counter-clockwise looking down the +z_{i-1} axis).
* Frame {0} is the base frame: origin on the waist axis at the level of the
  mounting surface (floor / pedestal base), z_0 vertical up, x_0 pointing
  "forward".  Frame {0} coincides with the MuJoCo world frame.
* d_1 is the pedestal height (shoulder axis is 0.6718 m = 26.45 in above the
  mounting surface).

Frames of interest
------------------
{0}  base frame
{1}  waist frame        (z_1 = shoulder axis, joint 2)
{2}  upper-arm frame    (z_2 = elbow axis, joint 3), origin on the elbow axis
{3}  forearm frame      (z_3 = forearm roll axis, joint 4)
{4},{5},{6}  wrist frames.  Because a_4=a_5=a_6=d_5=d_6=0 all three origins
     coincide at the WRIST CENTRE (the point where the three wrist axes
     meet -> spherical wrist).
{E}  end-effector (tool flange) frame:  T_6E = Trans(z, D_FLANGE).
     z_E = approach direction, x_E, y_E the plate axes.

Sign / offset conventions
-------------------------
* THETA_OFFSET = 0 for all joints: the zero configuration q = 0 is Corke's
  "qz" 'L-shaped' pose (upper arm horizontal along +x_0, forearm vertical up).
* Other named poses: READY (qr), STRETCHED (qs), NOMINAL (qn) below.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Units / constants
# ----------------------------------------------------------------------------
DEG = np.pi / 180.0
INCH = 0.0254
N_JOINTS = 6

# ----------------------------------------------------------------------------
# Link dimensions  [m]
# ----------------------------------------------------------------------------
D1 = 26.45 * INCH      # pedestal height: base plane -> shoulder (joint 2) axis
A2 = 0.4318            # upper-arm length   (shoulder axis -> elbow axis)
D3 = 0.15005           # lateral shoulder offset along the elbow axis
A3 = 0.0203            # elbow offset (elbow axis -> forearm axis)
D4 = 0.4318            # forearm length     (elbow -> wrist centre)
D_FLANGE = 0.04        # wrist centre -> tool-flange plate along z_6  [ARTE: d6 = 0.04]
#                        (Corke's model uses 0, i.e. the tool at the wrist centre; the link-6
#                         mesh starts at z = 0.043 m, consistent with 0.04.  Changing this one
#                         number moves the end-effector frame; IK/Jacobian follow automatically.)

# ----------------------------------------------------------------------------
# Standard-DH table  (one entry per joint 1..6)
# ----------------------------------------------------------------------------
#                        joint:   1          2     3          4         5          6
DH_ALPHA = np.array([np.pi / 2,   0.0,  -np.pi / 2,  np.pi / 2, -np.pi / 2,  0.0])
DH_A     = np.array([0.0,         A2,    A3,          0.0,       0.0,        0.0])
DH_D     = np.array([D1,          0.0,   D3,          D4,        0.0,        0.0])
THETA_OFFSET = np.zeros(N_JOINTS)   # theta_i = q_i + THETA_OFFSET_i

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_pitch", "wrist_roll"]
JOINT_TYPES = ["revolute"] * N_JOINTS

# ----------------------------------------------------------------------------
# Fixed transforms
# ----------------------------------------------------------------------------
T_BASE = np.eye(4)                       # world -> frame {0}
T_TOOL = np.eye(4)                       # frame {6} -> end-effector flange {E}
T_TOOL[2, 3] = D_FLANGE

# ----------------------------------------------------------------------------
# Joint limits  [rad]   (rows: joints, columns: lower, upper)
# ----------------------------------------------------------------------------
JOINT_LIMITS = np.array([
    [-160 * DEG,  160 * DEG],
    [ -45 * DEG,  225 * DEG],
    [-225 * DEG,   45 * DEG],
    [-110 * DEG,  170 * DEG],
    [-100 * DEG,  100 * DEG],
    [-266 * DEG,  266 * DEG],
])

# ----------------------------------------------------------------------------
# Dynamic parameters (Armstrong et al. 1986 via Corke & Armstrong-Helouvry 1994)
# expressed in the standard-DH link frame {i} of link i
# ----------------------------------------------------------------------------
LINK_MASS = np.array([0.0, 17.4, 4.8, 0.82, 0.34, 0.09])          # [kg]

# centre of mass position in link frame {i}  [m]
LINK_COM = np.array([
    [0.0,     0.0,    0.0],
    [-0.3638, 0.006,  0.2275],
    [-0.0203, -0.0141, 0.070],
    [0.0,     0.019,  0.0],
    [0.0,     0.0,    0.0],
    [0.0,     0.0,    0.032],
])

# inertia about the COM, link-frame axes: [Ixx, Iyy, Izz, Ixy, Iyz, Ixz]  [kg m^2]
_LINK_INERTIA_VEC = np.array([
    [0.0,     0.35,    0.0,     0, 0, 0],
    [0.13,    0.524,   0.539,   0, 0, 0],
    [0.066,   0.086,   0.0125,  0, 0, 0],
    [1.8e-3,  1.3e-3,  1.8e-3,  0, 0, 0],
    [0.3e-3,  0.4e-3,  0.3e-3,  0, 0, 0],
    [0.15e-3, 0.15e-3, 0.04e-3, 0, 0, 0],
])


def inertia_tensors() -> np.ndarray:
    """Return the (6, 3, 3) array of link inertia tensors about the COM."""
    out = np.zeros((N_JOINTS, 3, 3))
    for i, (xx, yy, zz, xy, yz, xz) in enumerate(_LINK_INERTIA_VEC):
        out[i] = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    return out


LINK_INERTIA = inertia_tensors()

# Armstrong et al. give no mass for link 1 (m1 = 0 in the reference model); only
# its inertia about the waist axis (Iyy = 0.35 in frame {1}) matters.  MuJoCo
# requires strictly positive mass, so the MJCF generator uses this placeholder.
LINK1_PLACEHOLDER_MASS = 1e-3

# ----------------------------------------------------------------------------
# Named configurations (Corke's definitions, same DH zero)
# ----------------------------------------------------------------------------
Q_ZERO = np.zeros(N_JOINTS)                                   # 'L' shape
Q_READY = np.array([0.0, np.pi / 2, -np.pi / 2, 0.0, 0.0, 0.0])  # arm straight up
Q_STRETCH = np.array([0.0, 0.0, -np.pi / 2, 0.0, 0.0, 0.0])      # arm out along +x
Q_NOMINAL = np.array([0.0, np.pi / 4, -np.pi, 0.0, np.pi / 4, 0.0])

# ----------------------------------------------------------------------------
# MuJoCo model files
# ----------------------------------------------------------------------------
import pathlib as _pl

PROJECT_ROOT = _pl.Path(__file__).resolve().parents[1]
MJCF_PATH = PROJECT_ROOT / "models" / "puma560.xml"
MJCF_JOINT_NAMES = [f"joint{i}" for i in range(1, N_JOINTS + 1)]
MJCF_EE_SITE = "ee_site"
MJCF_WRIST_SITE = "wrist_center_site"

# ----------------------------------------------------------------------------
# Derived reach quantities (used by IK diagnostics)
# ----------------------------------------------------------------------------
L_FOREARM = float(np.hypot(A3, D4))        # elbow axis -> wrist centre distance
PHI_ELBOW = float(np.arctan2(D4, A3))      # angle used in the elbow equations
