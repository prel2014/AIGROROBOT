"""Repository paths with an override for installations outside this checkout."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("DEXMAN_ROOT", Path(__file__).resolve().parents[2]))
ROBOT_USD = ROOT / "assets/robots/ur10e_shadow/robot.usda"
JOINT_MANIFEST = ROOT / "assets/robots/ur10e_shadow/joint_manifest.json"
