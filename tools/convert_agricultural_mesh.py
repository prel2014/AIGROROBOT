"""Convert textured GLB/glTF assets to visual USD using the Isaac Sim 6.2 importer."""
import argparse
import asyncio
import sys
from pathlib import Path


async def convert(assets):
    import omni.kit.asset_converter
    from pxr import Usd, UsdGeom

    converter = omni.kit.asset_converter.get_instance()
    for source, output in assets:
        context = omni.kit.asset_converter.AssetConverterContext()
        context.ignore_animation = True
        context.ignore_cameras = True
        context.ignore_lights = True
        context.export_preview_surface = True
        context.use_meter_as_world_unit = True
        context.convert_stage_up_y = True
        output.parent.mkdir(parents=True, exist_ok=True)
        geometry = output.parent / "geometry.usdc"
        task = converter.create_converter_task(str(source), str(geometry), None, context)
        if not await task.wait_until_finished():
            raise RuntimeError(f"USD conversion failed for {source.name}: {task.get_error_message()}")
        stage = Usd.Stage.Open(str(geometry))
        if not stage or not stage.GetDefaultPrim() or not any(
                prim.IsA(UsdGeom.Mesh) for prim in stage.Traverse()):
            raise RuntimeError(f"No mesh imported from {source.name}. Check glTF extensions and re-export if needed.")
        if UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise RuntimeError(f"Expected meter units in {geometry}")
        visual = Usd.Stage.CreateNew(str(output))
        UsdGeom.SetStageUpAxis(visual, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(visual, 1)
        root = UsdGeom.Xform.Define(visual, "/Asset")
        visual.SetDefaultPrim(root.GetPrim())
        # glTF stores Y-up geometry. Set a transform on the reference, not just upAxis metadata.
        root.AddRotateXOp().Set(90)
        prim = UsdGeom.Xform.Define(visual, "/Asset/Geometry").GetPrim()
        prim.GetReferences().AddReference("./geometry.usdc")
        visual.GetRootLayer().Save()
        print(f"Converted {source.name} -> {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="Original .glb or .gltf files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assets = []
    for source in args.sources:
        if not source.is_file() or source.suffix.lower() not in (".glb", ".gltf"):
            parser.error(f"Expected a GLB/glTF file: {source}")
        output = args.output / source.stem / "visual.usda"
        if output.parent.exists() or any(existing == output.resolve() for _, existing in assets):
            parser.error(f"Duplicate or existing destination: {output}")
        assets.append((source.resolve(), output.resolve()))

    from isaacsim import SimulationApp

    del sys.argv[1:]
    app = SimulationApp({"headless": True})
    try:
        from isaacsim.core.experimental.utils.app import enable_extension
        enable_extension("omni.kit.asset_converter")
        asyncio.get_event_loop().run_until_complete(convert(assets))
    finally:
        app.close()


if __name__ == "__main__":
    main()
