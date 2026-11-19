"""Fotografía la vid y el racimo del surco real con cámaras RTX de Isaac Sim."""
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLANT = "/Surco/Plantas/Izq2"
FRUIT = "/Surco/Frutos/Izq2"


def render(app, scene, output):
    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.stage import open_stage
    from PIL import Image
    from pxr import Gf, UsdGeom

    enable_extension("isaacsim.sensors.experimental.rtx")
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

    if not open_stage(str(scene)):
        raise RuntimeError(f"No se pudo abrir {scene}")
    stage = omni.usd.get_context().get_stage()
    bounds = UsdGeom.BBoxCache(0., [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def box(path):
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError(f"Falta {path}")
        result = bounds.ComputeWorldBound(prim).ComputeAlignedBox()
        if result.IsEmpty():
            raise ValueError(f"Geometría vacía en {path}")
        return result

    plant = box(PLANT).GetMidpoint()
    fruit_box = box(FRUIT)
    fruit = fruit_box.GetMidpoint()
    focus = fruit + Gf.Vec3d(0., 0., .016)
    height = max(.12, fruit_box.GetSize()[2])
    poses = [
        ("vid_completa", plant + Gf.Vec3d(.62, -2.60, .67), plant, 11.1),
        ("racimo_en_vid", focus + height*Gf.Vec3d(-1.65, -3., .54), focus, 34.),
    ]
    sensors = []
    for name, position, target, focal in poses:
        rig = UsdGeom.Xform.Define(stage, "/VistasVid/" + name)
        rig.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(position, target, Gf.Vec3d(0,0,1)).GetInverse())
        cam = UsdGeom.Camera.Define(stage, rig.GetPath().AppendChild("RGB"))
        cam.CreateHorizontalApertureAttr(24.)
        cam.CreateFocalLengthAttr(focal)
        cam.CreateClippingRangeAttr((.008, 30.))
        sensor = CameraSensor(RtxCamera(str(cam.GetPath())), resolution=(768,1280), annotators=["rgb"])
        sensors.append((name, sensor))

    # Sólo se agregan cámaras en memoria. Los materiales y la luz proceden del surco.
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(48):
        app.update()
    output.mkdir(parents=True, exist_ok=True)
    for name, sensor in sensors:
        raw, _ = sensor.get_data("rgb")
        if raw is None:
            raise RuntimeError(f"{name} sin imagen")
        pixels = raw.numpy()[..., :3]
        if pixels.shape != (768, 1280, 3) or pixels.std() < 1.:
            raise RuntimeError(f"{name} imagen no válida: {pixels.shape}")
        Image.fromarray(pixels).save(output/(name+".png"))
    timeline.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT/"assets/agriculture/robot/rover/surco.usdc")
    parser.add_argument("--output", type=Path, default=ROOT/"assets/agriculture/previews/vineyard")
    args = parser.parse_args()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "multi_gpu": False})
    status = 1
    try:
        render(app, args.scene.resolve(strict=True), args.output)
        status = 0
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        app.close(exit_code=status)


if __name__ == "__main__":
    main()
