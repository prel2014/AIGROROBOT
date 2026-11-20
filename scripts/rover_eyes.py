"""Captura ambos ojos RGB-D y el estado móvil con los ángulos físicos del cabezal."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--pan-deg", type=float, default=0., help="Giro desde la vista de cosecha; positivo a la izquierda")
    parser.add_argument("--tilt-deg", type=float, default=0., help="Elevación desde la vista de cosecha; positivo hacia arriba")
    parser.add_argument("--instruction", default="Observa el surco.", help="Orden en español guardada con la muestra")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("La carpeta de salida ya existe. Elige una carpeta nueva.")
    from dexman.tasks.agriculture.mobile_state import HEAD_JOINTS, HEAD_LIMITS_DEG
    target = (args.pan_deg, args.tilt_deg)
    for name, value, (low, high) in zip(HEAD_JOINTS, target, HEAD_LIMITS_DEG, strict=True):
        if not low <= value <= high:
            parser.error(f"{name} admite {low} a {high} grados")
    if not args.instruction.strip():
        parser.error("La orden requiere texto")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "multi_gpu": False})
    exit_code = 1
    try:
        import numpy as np
        import h5py
        import omni.timeline
        from PIL import Image
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import open_stage
        from dexman.experiments import sha256
        from dexman.paths import ROOT
        from dexman.sensors.mobile import RoverArticulation, RoverSensors
        from dexman.vla.bimanual_contract import bimanual_contract, mobile_metadata

        enable_extension("isaacsim.sensors.experimental.physics")
        enable_extension("isaacsim.sensors.experimental.rtx")
        scene = ROOT / "assets/agriculture/robot/rover/surco.usdc"
        rover_digest = sha256(scene.with_name("rover.usdc"))
        if not open_stage(str(scene)):
            raise RuntimeError(f"No se pudo abrir {scene}")
        sensors = RoverSensors("/Surco/Rover")
        omni.timeline.get_timeline_interface().play()
        app.update()
        robot = RoverArticulation(sensors.root)
        robot.set_head(target)
        settled = 0
        for _ in range(360):
            app.update()
            imu = sensors.imu.get_data()
            if imu["time"] <= 0:
                continue
            state = robot.state(imu).numpy()
            measured = np.rad2deg(state[0, -2:])
            settled = settled + 1 if np.all(np.abs(measured-target) < 1.) else 0
            if settled < 5:
                continue
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
            raise RuntimeError("El cabezal no se asentó a 1° de la consigna o faltan lecturas RGB-D/IMU")
        metadata = bimanual_contract({**mobile_metadata(), "joint_limits_rad": robot.action_limits(),
                                      "sample_kind": "robot_observation"})
        metadata["sample_kind"] = "robot_observation"
        metadata["rover_asset_sha256"] = rover_digest
        depth = np.stack([frame[1] for frame in frames])[None].astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0.)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent, prefix=f".{args.output.name}-") as temporary:
            sample = Path(temporary)
            for name, (rgb, raw_depth) in zip(("izquierda", "derecha"), frames, strict=True):
                Image.fromarray(rgb).save(sample / f"{name}_rgb.png")
                np.save(sample / f"{name}_profundidad_m.npy", raw_depth)
            with h5py.File(sample / "observacion.h5", "x") as capture:
                capture.attrs["metadata"] = json.dumps(metadata, ensure_ascii=False)
                capture.attrs["instruction"] = args.instruction
                capture["rgb"] = np.stack([frame[0] for frame in frames])[None]
                capture["depth_m"] = np.where(valid, depth, 0.)
                capture["depth_valid"] = valid
                capture["mobile_state"] = state
                capture["time_s"] = np.asarray([imu["time"]], dtype=np.float64)
                capture["head_target_rad"] = np.deg2rad(target)[None]
            sample.rename(args.output)
        print(f"{args.output.resolve()} · pan/tilt medidos {measured.round(2)} grados", flush=True)
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
