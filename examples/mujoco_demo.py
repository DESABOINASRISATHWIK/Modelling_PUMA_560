"""Open the optional MuJoCo visualization of the PUMA-560."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mujoco_visualization import show


show(np.radians([30, 50, -70, 40, 60, -20]))
