"""Open one field in headless Isaac Sim and photograph its three authored views."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from dexman.tasks.agriculture.field_layout import FIELDS


def photograph(app, crop, output):
    import numpy as np
    import omni.physx
    import omni.timeline
    import omni.usd
    from PIL import Image
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.stage import open_stage
    from pxr import Usd, UsdGeom, UsdPhysics

    enable_extension("isaacsim.sensors.experimental.rtx")
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
    field = FIELDS[crop]
    asset = ROOT/"assets/agriculture/parcelas"/(field.name+".usdc")
    if not open_stage(str(asset)):
        raise RuntimeError(f"Isaac Sim could not load {asset}")
    context = omni.usd.get_context()
    stage = context.get_stage()
    rows = stage.GetPrimAtPath("/Parcela/Hileras").GetChildren()
    if len(rows)!=3 or stage.GetCompositionErrors():
        raise ValueError(f"Incomplete USD composition: {asset}")
    for row in rows:
        if len(row.GetChildren())!=field.plants_per_row:
            raise ValueError(f"Incomplete plant row: {row.GetPath()}")
    if not stage.GetPrimAtPath("/Parcela/Suelo").HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError("Missing terrain collider")
    sensors = {}
    for view in ("General","Calle","Fruto"):
        prim = stage.GetPrimAtPath("/Parcela/Vistas/"+view)
        if not prim.IsA(UsdGeom.Camera):
            raise ValueError(f"Missing camera: {view}")
        sensors[view] = CameraSensor(RtxCamera(str(prim.GetPath())),resolution=(640,960),annotators=["rgb"])
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(72):
        app.update()
    query = omni.physx.get_physx_scene_query_interface()
    def hit(x,y):
        return query.raycast_closest((float(x),float(y),6.),(0.,0.,-1.),12.)
    for offset in (0.,.51):
        x = 5.7
        y = field.row_y(x,-.5)+offset
        contact = hit(x,y)
        if contact.get("collision")!="/Parcela/Suelo" or abs(contact["position"][2]-field.height(x,y))>.025:
            raise ValueError(f"PhysX terrain mismatch at {x,y}: {contact}")
    if hit(-field.headland,-field.fence_y).get("collision")!="/Parcela/Cerca/Postes":
        raise ValueError("PhysX failed to detect the entrance fence post")
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(),["default","proxy","render"])
    box = bounds.ComputeWorldBound(stage.GetPrimAtPath("/Parcela/Obstaculos")).ComputeAlignedBox()
    x,y,_ = (np.linspace(a,b,9) for a,b in zip(box.GetMin(),box.GetMax()))
    if not any(str(hit(i,j).get("collision","")).startswith("/Parcela/Obstaculos/") for i in x for j in y):
        raise ValueError("PhysX failed to detect the lane obstacle")
    output.mkdir(parents=True,exist_ok=True)
    for view,sensor in sensors.items():
        data,_ = sensor.get_data("rgb")
        if data is None:
            raise ValueError(f"Missing RGB data: {view}")
        pixels = data.numpy()[...,:3]
        if pixels.shape!=(640,960,3) or pixels.std()<3:
            raise ValueError(f"Invalid capture: {view} {pixels.shape}")
        Image.fromarray(pixels).save(output/f"{field.name}_{view.lower()}.jpg",quality=93)
    timeline.stop()
    print(f"Captured {field.name} with Isaac Sim RTX",flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop",choices=tuple(FIELDS),required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"assets/agriculture/previews/parcelas")
    args = parser.parse_args()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless":True,"multi_gpu":False})
    status = 1
    try:
        photograph(app,args.crop,args.output)
        status = 0
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        app.close(exit_code=status)
