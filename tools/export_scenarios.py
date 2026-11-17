"""Export portable procedural USD props and grounded Spanish plans; starts no simulator."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from usd_python import configure_usd


def main():
    configure_usd()
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    from dexman.tasks.manipulation.scenarios import SITUATIONS, scenario_catalog, push_offset
    from dexman.tasks.manipulation.planner import pick_place_plan, lift_plan, push_plan

    output = ROOT / "assets/procedural"
    output.mkdir(parents=True, exist_ok=True)

    def primitive(stage, path, shape, dimensions, position, color, mass=None):
        root = UsdGeom.Xform.Define(stage, path)
        root.AddTranslateOp().Set(Gf.Vec3d(*position))
        if mass is not None:
            body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            if mass == -1:
                body.CreateKinematicEnabledAttr(True)
                mass = 1.
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
        classes = {"box":UsdGeom.Cube, "sphere":UsdGeom.Sphere,
                   "cylinder":UsdGeom.Cylinder, "capsule":UsdGeom.Capsule}
        geometry = classes[shape].Define(stage, path+"/Geometry")
        if shape == "box":
            geometry.CreateSizeAttr(1.)
            geometry.AddScaleOp().Set(Gf.Vec3f(*dimensions))
        else:
            geometry.CreateRadiusAttr(dimensions[0])
            if shape != "sphere":
                geometry.CreateHeightAttr(dimensions[1])
                geometry.CreateAxisAttr("Z")
        geometry.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
        material = UsdShade.Material.Define(stage, path+"/Material")
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(.7)
        physics.CreateDynamicFrictionAttr(.5)
        physics.CreateRestitutionAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material, materialPurpose="physics")

    catalog = scenario_catalog()
    for s, record in zip(SITUATIONS, catalog, strict=True):
        stage = Usd.Stage.CreateNew(str(output / (s.name+".usda")))
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Scene").GetPrim())
        half = (s.dimensions[2]/2 if s.shape == "box" else s.dimensions[0] if s.shape == "sphere"
                else s.dimensions[1]/2 + (s.dimensions[0] if s.shape == "capsule" else 0.))
        seed = json.loads((ROOT/"configs/grasp_seed.json").read_text(encoding="utf-8"))["cube_local"]
        start = (seed[0],seed[1],.6+half)
        primitive(stage,"/Scene/Object",s.shape,s.dimensions,start,s.color,s.mass)
        primitive(stage,"/Scene/Table","box",(2.2,1.4,.1),(0.,0.,.55),s.table)
        primitive(stage,"/Scene/Floor","box",(4.,4.,.05),(0.,0.,-.025),tuple(v*.55 for v in s.table))
        for i,(x,y,sx,sy,sz) in enumerate(s.obstacles):
            primitive(stage,f"/Scene/Obstacle{i}","box",(sx,sy,sz),(x,y,.6+sz/2),(.38,.42,.48),-1)
        for i,(position,color) in enumerate((((.05,.35,.625),(.05,.75,.8)),((.65,-.30,.625),(.85,.2,.55)))):
            primitive(stage,f"/Scene/Distractor{i}","box",(.045,.045,.05),position,color,.09)
        stage.GetRootLayer().Save()
        usd_path = output / (s.name+".usda")
        usd_path.write_text(usd_path.read_text(encoding="utf-8").rstrip()+"\n",encoding="utf-8")
        goal = (start[0]+s.destination[0], start[1]+s.destination[1],start[2])
        obstacles = [o[:4] for o in s.obstacles] + [(.05,.35,.045,.045),(.65,-.30,.045,.045)]
        push_dx,push_dy,_ = push_offset(s)
        push_goal = (start[0]+push_dx,start[1]+push_dy,start[2])
        record["task_plans"] = []
        for kind,text,builder in (("lift",record["instructions"][0],lambda:lift_plan(start)),
                                  ("place",record["instructions"][1],lambda:pick_place_plan(start,goal,obstacles)),
                                  ("push",record["instructions"][2],lambda:push_plan(start,push_goal,obstacles))):
            entry = {"kind":kind,"instruction":text}
            try:
                entry.update(steps=[asdict(step) for step in builder()],route_available=True)
            except ValueError as error:
                entry.update(steps=[{"name":"stop","text":"Detente y solicita otro destino; no fuerces el paso."}],
                             route_available=False,reason=str(error))
            record["task_plans"].append(entry)
        record["plan"] = record["task_plans"][1]["steps"]
        record["route_available"] = record["task_plans"][1]["route_available"]
        record.update(start_xyz=start, destination_xyz=goal, frame="robot_base_X_forward_Y_left_Z_up",
                      usd=f"assets/procedural/{s.name}.usda", planner="teacher_ground_truth_XY_inflated_boxes",
                      controller_integrated=False)
        preview = ROOT / "assets/previews" / s.name / "scene.json"
        if preview.exists():
            capture = json.loads(preview.read_text(encoding="utf-8"))
            capture["orders"] = record["instructions"]
            preview.write_text(json.dumps(capture,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (output/"catalog.json").write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines = ["# Órdenes y subtareas", "", "24 órdenes propuestas; ninguna ha sido entrenada con estos escenarios.",
             "X: delante del robot; Y: izquierda; Z: arriba. Las rutas usan geometría conocida del simulador.", ""]
    for record in catalog:
        lines += ["## "+record["theme"], ""] + [f"- {text}" for text in record["instructions"]]
        lines += [""]
        for task in record["task_plans"]:
            lines += ["Subtareas ("+task["kind"]+"): "+" → ".join(step["text"] for step in task["steps"]), ""]
    (ROOT/"docs/robot-orders.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"Exported {len(catalog)} portable scenes and {sum(len(r['instructions']) for r in catalog)} orders; no training")


if __name__ == "__main__":
    main()
