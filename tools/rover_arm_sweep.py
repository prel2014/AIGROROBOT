"""Sondea el brazo izquierdo con drives de Isaac Sim y consultas de colisión de PhysX."""
import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARM = "/World/Rover/LeftArm"
BASE = "/World/Rover/Base"


def run(asset):
    import carb
    import omni.physx
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdUtils

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetEndTimeCode(60000)
    scene = UsdPhysics.Scene.Define(stage, "/World/Physics")
    scene.CreateGravityDirectionAttr((0., 0., -1.))
    scene.CreateGravityMagnitudeAttr(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateTimeStepsPerSecondAttr(60)
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.)
    ground.AddTranslateOp().Set((0., 0., -.05))
    ground.AddScaleOp().Set((10., 10., .1))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    rover = UsdGeom.Xform.Define(stage, "/World/Rover")
    root_digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    rover.GetPrim().GetReferences().AddReference(str(asset))
    rear = {}
    for fixture in (BASE, "/World/Rover/HeadPan", "/World/Rover/HeadTilt"):
        body = stage.GetPrimAtPath(fixture)
        if not body:
            continue
        PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.)
        for prim in Usd.PrimRange(body):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                path = prim.GetPath()
                rear[str(path)] = PhysicsSchemaTools.encodeSdfPath(path)
    if not rear:
        raise ValueError("No se han encontrado colisiones del bastidor")
    names = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    joints = [UsdPhysics.RevoluteJoint.Get(stage, ARM+"/Arm/joints/"+name) for name in names]
    drives = [UsdPhysics.DriveAPI.Get(j.GetPrim(), "angular") for j in joints]
    limits = {name: [j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()] for name, j in zip(names, joints)}
    def actual_angle(j, cache):
        frames = []
        for body, rotation in ((j.GetBody0Rel(), j.GetLocalRot0Attr()), (j.GetBody1Rel(), j.GetLocalRot1Attr())):
            local = Gf.Matrix4d().SetRotate(Gf.Quatd(rotation.Get()))
            frames.append(local * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body.GetTargets()[0])))
        quat = (frames[1] * frames[0].GetInverse()).ExtractRotationQuat()
        axis = "XYZ".index(j.GetAxisAttr().Get())
        angle = math.degrees(2*math.atan2(quat.GetImaginary()[axis], quat.GetReal()))
        return (angle+180.)%360.-180.

    def moving(path):
        return path.startswith(ARM+"/") and not path.startswith(ARM+"/Arm/base_link/")

    carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
    query = omni.physx.get_physx_scene_query_interface()
    physics = omni.physx.get_physx_simulation_interface()
    stage_id = UsdUtils.StageCache.Get().Insert(stage)
    if not physics.attach_stage(stage_id.ToLongInt()):
        raise RuntimeError("PhysX no pudo cargar el rover")
    dt, elapsed = 1./60., 0.
    rows = []
    neutral = (0.,) * 6
    previous = tuple(drive.GetTargetPositionAttr().Get() for drive in drives)
    path = [("home", neutral), ("frontal", (20,-20,35,0,0,0)), ("recto", neutral),
            ("pan_lateral", (85,0,0,0,0,0)), ("pan_posterior", (145,0,0,0,0,0)),
            ("plegado", (90,-70,105,0,0,0)), ("retorno", neutral),
            ("plegado_posterior", (60,20,90,90,90,0)), ("retorno_neutro", neutral),
            ("fruta", (23.75,-41.74,67.13,-102.06,-54.45,18.54)),
            ("paso_neutro", neutral), ("canasta", (-31.45,-44.97,77.61,-92.13,-93.39,20.52)),
            ("frontal_final", (20,-20,35,0,0,0))]
    for label, target in path:
        found, touched = {}, {}
        for step in range(1, 121):
            t = min(1., step/90)
            for drive, a, b in zip(drives, previous, target):
                drive.GetTargetPositionAttr().Set((1-t)*a+t*b)
            physics.simulate(dt, elapsed)
            physics.fetch_results()
            elapsed += dt
            cache = UsdGeom.XformCache()
            pan = actual_angle(joints[0], cache)
            for obstacle, encoded in rear.items():
                def hit(result):
                    collider = str(result.collision)
                    if moving(collider):
                        found.setdefault((obstacle, collider), round(pan, 2))
                    return True
                query.overlap_shape(*encoded, hit, False)
            headers, _ = physics.get_contact_report()
            for header in headers:
                if not header.num_contact_data:
                    continue
                a,b = (str(PhysicsSchemaTools.intToSdfPath(i)) for i in (header.collider0, header.collider1))
                if a in rear and moving(b):
                    touched.setdefault((a,b), round(pan, 2))
                if b in rear and moving(a):
                    touched.setdefault((b,a), round(pan, 2))
        def pairs(observed):
            return [
                {"fixture": obstacle.removeprefix(BASE + "/"), "link": link.removeprefix(ARM + "/"),
                 "first_pan_deg": angle} for (obstacle, link), angle in observed.items()
            ]
        angles = [round(actual_angle(j,cache),2) for j in joints]
        row = dict(pose=label, request_deg=target, measured_deg=angles,
                   overlap=pairs(found), contact=pairs(touched))
        rows.append(row)
        print(f"{label}: q={angles} overlaps={len(found)} contacts={len(touched)}",flush=True)
        previous = target
    physics.detach_stage()
    UsdUtils.StageCache.Get().Erase(stage_id)
    if abs(rows[1]["measured_deg"][0]-rows[0]["measured_deg"][0])<5:
        raise RuntimeError("El brazo no realizó el desplazamiento de control")
    all_clear = all(not row["overlap"] and not row["contact"] for row in rows)
    tracking = [abs(measured - target) for row in rows if row["pose"] in ("fruta","canasta")
                for measured, target in zip(row["measured_deg"], row["request_deg"])]
    return {"asset": str(asset), "root_layer_sha256": root_digest, "physics_hz": round(1./dt), "joint_order": names,
            "limits_deg": limits, "contact_free": all_clear, "work_pose_max_error_deg": max(tracking),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=ROOT/"assets/agriculture/robot/rover/rover.usdc")
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "multi_gpu": False})
    status = 1
    try:
        report = run(opts.asset.resolve(strict=True))
        opts.output.parent.mkdir(parents=True, exist_ok=True)
        opts.output.write_text(json.dumps(report, indent=2),encoding="utf-8")
        status = 0 if report["contact_free"] and report["work_pose_max_error_deg"] <= 2. else 2
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        app.close(exit_code=status)


if __name__ == "__main__":
    main()
