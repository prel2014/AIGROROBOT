"""Check installed packages without importing them or launching Isaac Sim."""
import argparse
import importlib.util
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True)
    parser.add_argument("--live-plot", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if sys.version_info[:2] != (3, 12):
        print(f"Python incorrecto: {sys.executable}. Se requiere el Python 3.12 de Isaac Sim.", file=sys.stderr)
        return 2
    offline = args.command in {"train_vla", "infer_vla", "dataset", "doctor"}
    packages = ["numpy", "torch", "gymnasium"]
    if not offline:
        packages += ["isaaclab", "isaaclab_rl", "isaaclab_physx", "isaaclab_ov", "isaaclab_newton"]
    if args.command in {"agriculture_preview", "rover_eyes", "scene_gallery", "train_vla", "infer_vla", "play_vla", "collect_demos", "collect_corrections", "dataset"}:
        packages += ["h5py", "PIL"]
    if args.command in {"train_vla", "infer_vla", "play_vla", "memory_rl", "collect_corrections"}:
        packages += ["transformers"]
    if args.command == "memory_rl":
        packages += ["rsl_rl", "tensordict", "PIL"]
    if args.command in {"agriculture_teacher", "agriculture_collect"}:
        packages += ["rsl_rl", "tensordict"]
    if args.command == "agriculture_teacher":
        packages += ["tensorboard"]
    if args.command == "agriculture_collect":
        packages += ["h5py"]
    if args.live_plot:
        packages += ["matplotlib"]
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if not missing:
        return 0
    print("Entorno incompleto; no se ha iniciado la simulacion.", file=sys.stderr)
    print(f"Python: {sys.executable}", file=sys.stderr)
    print("Paquetes no encontrados: " + ", ".join(missing), file=sys.stderr)
    print("Desde PowerShell ejecuta primero:", file=sys.stderr)
    print(f'  & "{root / "scripts/setup.ps1"}"', file=sys.stderr)
    if args.command in {"train_vla", "infer_vla", "play_vla", "memory_rl"} or args.live_plot:
        sim = Path(sys.executable).resolve().parents[2]
        requirements = "vla-ui-requirements.txt" if args.live_plot else "vla-requirements.txt"
        print(f'  & "{sim / "python.bat"}" -m pip install -r "{root / "configs" / requirements}"',
              file=sys.stderr)
    print("Despues vuelve a ejecutar el escenario.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
