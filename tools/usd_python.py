"""Load bundled OpenUSD without starting Kit, a renderer, or a physics engine."""
import os
import sys
from pathlib import Path

_DLL_HANDLES = []


def configure_usd():
    sim = Path(os.environ.get("ISAAC_SIM_PATH", Path(__file__).resolve().parents[2] / "isaac-sim-6.2.0"))
    libs = next((sim / "extscache").glob("omni.usd.libs-*"))
    sys.path.insert(0, str(libs))
    if os.name == "nt":
        _DLL_HANDLES.append(os.add_dll_directory(str(libs / "bin")))
