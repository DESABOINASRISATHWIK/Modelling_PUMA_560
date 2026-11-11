"""Generate portable nominal and obstacle physics XMLs from the validated robot."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import robot_parameters as rp
from src.scene import SceneConfig, write_scene

if __name__ == "__main__":
    for name, config in [("nominal", SceneConfig()), ("obstacles", SceneConfig.obstacle_scene())]:
        path = rp.PROJECT_ROOT / "models" / f"puma560_{name}.xml"
        write_scene(path, config)
        print(path)
