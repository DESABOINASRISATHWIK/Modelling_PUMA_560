"""Singularity analysis: SVD table, sweeps through each singularity family, sigma_min map, gallery."""
import numpy as np

from _common import RESULTS, mujoco_images
from src.singularity_analysis import analyze_configuration, singular_configuration
from src.verification import singularity_examples
from src.visualization import (image_grid, plot_sigma_map, plot_singular_value_sweeps,
                               plot_singularity_gallery)

print("Regular / near-singular / singular configurations\n")
ex = singularity_examples()
for name, rep in ex.items():
    print(f"{name:<26}q[deg]={np.round(np.degrees(rep.q), 1).tolist()}")
    print(f"{'':<26}{rep.text()}")
    print(f"{'':<26}singular values = {np.array2string(rep.svd.singular_values, precision=3)}\n")

plot_singular_value_sweeps(RESULTS / "singular_value_sweeps.png")
plot_sigma_map(path=RESULTS / "sigma_min_map.png")
plot_singularity_gallery(RESULTS / "singularity_gallery.png")

cfgs = [(rep.q, None) for rep in ex.values()]
imgs = mujoco_images(cfgs, size=(480, 640))
if imgs:
    image_grid(imgs, [f"{n}\n{r.status}  sigma_min={r.svd.sigma_min:.1e}" for n, r in ex.items()],
               RESULTS / "singularity_mujoco.png", ncols=3)
print("figures written to", RESULTS)
