"""Reachable workspace (sampling) + analytic bound of the wrist centre."""
import numpy as np

from _common import RESULTS
from src import robot_parameters as rp
from src.visualization import plot_workspace

out = plot_workspace(n=20000, seed=0, path=RESULTS / "workspace.png")
P, W, S, R_max = out["positions"], out["wrist"], out["sigma_min"], out["R_max"]
S0 = np.array([0, 0, rp.D1])
dist = np.linalg.norm(W - S0, axis=1)
r = np.hypot(W[:, 0], W[:, 1])
print(f"samples: {len(P)}")
print(f"end-effector reach: x in [{P[:,0].min():.3f}, {P[:,0].max():.3f}]  y in [{P[:,1].min():.3f}, {P[:,1].max():.3f}]"
      f"  z in [{P[:,2].min():.3f}, {P[:,2].max():.3f}] m")
print(f"wrist centre distance from shoulder point: max {dist.max():.4f} m <= analytic bound sqrt((a2+L)^2+d3^2) = {R_max:.4f} m")
print(f"wrist centre radial distance from waist axis: min {r.min():.5f} m >= |d3| = {abs(rp.D3):.5f} m")
print(f"fraction of samples near-singular (sigma_min < 0.01): {(S < 0.01).mean():.3f}")
print("figure written to", RESULTS / "workspace.png")
