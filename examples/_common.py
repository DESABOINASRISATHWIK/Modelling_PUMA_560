"""Shared bootstrap for the example scripts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def mujoco_images(configs, **kw):
    """Offscreen MuJoCo renders (None if no OpenGL backend is available -> figures are skipped)."""
    try:
        from src.simulation import PumaSimulation, render_image
        out = []
        for q, target in configs:
            sim = PumaSimulation(q)
            if target is not None:
                sim.set_target(target)
            out.append(render_image(sim, **kw))
        return out
    except Exception as exc:                                  # pragma: no cover
        print(f"[note] MuJoCo offscreen rendering unavailable ({type(exc).__name__}); skipping MuJoCo images.\n"
              "       On a headless Linux box try:  MUJOCO_GL=egl  (or osmesa)  python examples/...")
        return None
