"""Guarda una muestra RGB-D de cada cámara lateral del rover."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("La carpeta de salida ya existe. Elige una carpeta nueva.")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "multi_gpu": False})
    exit_code = 1
    try:
        import numpy as np
        import omni.timeline
        from PIL import Image
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import open_stage
        from dexman.paths import ROOT
        from dexman.sensors.mobile import RoverSensors

        enable_extension("isaacsim.sensors.experimental.physics")
        enable_extension("isaacsim.sensors.experimental.rtx")
        scene = ROOT / "assets/agriculture/robot/rover/surco.usdc"
        if not open_stage(str(scene)):
            raise RuntimeError(f"No se pudo abrir {scene}")
        sensors = RoverSensors("/Surco/Rover/Base")
        omni.timeline.get_timeline_interface().play()
        for _ in range(120):
            app.update()
            frames = []
            for eye in sensors.eyes:
                rgb, _ = eye.get_data("rgb")
                depth, _ = eye.get_data("distance_to_image_plane")
                if rgb is None or depth is None:
                    break
                rgb = rgb.numpy()[..., :3]
                depth = depth.numpy().squeeze(-1)
                if rgb.size == 0 or rgb.min() == rgb.max() or not (np.isfinite(depth) & (depth > 0.)).any():
                    break
                frames.append((rgb, depth))
            if len(frames) == 2:
                break
        else:
            raise RuntimeError("Las dos cámaras no entregaron una muestra válida")
        args.output.mkdir(parents=True, exist_ok=False)
        for name, (rgb, depth) in zip(("izquierda", "derecha"), frames, strict=True):
            Image.fromarray(rgb).save(args.output / f"{name}_rgb.png")
            np.save(args.output / f"{name}_profundidad_m.npy", depth)
        print(args.output.resolve(), flush=True)
        exit_code = 0
    except Exception:
        # Kit puede finalizar el intérprete durante el cierre. Conserva el diagnóstico.
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
