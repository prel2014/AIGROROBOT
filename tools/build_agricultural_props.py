"""Build the fixed torso shell and fruit wrappers from the agricultural USD library."""
import math
from pathlib import Path

from usd_python import configure_usd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/agriculture"


def build():
    configure_usd()
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    def stage_for(path, root_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        layer = Sdf.Layer.FindOrOpen(str(path)) if path.exists() else Sdf.Layer.CreateNew(str(path))
        layer.Clear()
        stage = Usd.Stage.Open(layer)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        root = UsdGeom.Xform.Define(stage, root_path)
        stage.SetDefaultPrim(root.GetPrim())
        return stage

    def material(stage, path, color, metallic, roughness):
        result = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return result

    stage = stage_for(ASSETS / "robot/torso.usda", "/Torso")
    ivory = material(stage, "/Torso/Materials/Shell", (.62, .69, .67), .28, .35)
    charcoal = material(stage, "/Torso/Materials/Frame", (.075, .11, .13), .52, .42)
    teal = material(stage, "/Torso/Materials/Accent", (.06, .28, .25), .18, .32)

    def shell(name, center, size, bevel, surface):
        """An octagonal cross section with planar faces and chamfered corners."""
        mesh = UsdGeom.Mesh.Define(stage, "/Torso/" + name)
        x, y, z = (extent / 2 for extent in size)
        profile = [(-x+bevel, -y), (x-bevel, -y), (x, -y+bevel), (x, y-bevel),
                   (x-bevel, y), (-x+bevel, y), (-x, y-bevel), (-x, -y+bevel)]
        points = [Gf.Vec3f(center[0]+px, center[1]+py, center[2]+pz)
                  for pz in (-z, z) for px, py in profile]
        faces = [list(reversed(range(8))), list(range(8, 16))]
        for i in range(8):
            j = (i+1) % 8
            faces.append([i, j, j+8, i+8])
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
        mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(surface)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")

    shell("Foundation", (-.77, 0, .055), (.58, 1.12, .11), .10, charcoal)
    shell("Pedestal", (-.77, 0, .55), (.34, .41, .91), .055, charcoal)
    shell("Chest", (-.77, 0, 1.14), (.39, .50, .54), .065, ivory)
    shell("LeftShoulder", (-.77, .47, 1.035), (.30, .52, .09), .04, charcoal)
    shell("RightShoulder", (-.77, -.47, 1.035), (.30, .52, .09), .04, charcoal)
    shell("SensorBar", (-.63, 0, 1.48), (.21, .39, .16), .035, teal)
    stage.GetRootLayer().Save()

    # Put the imported orientation on a child: Isaac Lab owns the spawn transform
    # on the asset root so that each scene may position the same plant differently.
    for name, structure in (
        ("banana_plant", ("Pseudostem", "Peduncle_and_rachis")),
        ("vineyard", ("oak", "timber", "steel")),
        ("berry_bush", ()),
        ("blueberries", ()),
    ):
        stage = stage_for(ASSETS / f"props/{name}.usda", "/Model")
        model = UsdGeom.Xform.Define(stage, "/Model/Visual")
        model.GetPrim().GetReferences().AddReference(f"../models/{name}/visual.usda")
        for item in Usd.PrimRange(model.GetPrim()):
            if item.IsA(UsdGeom.Mesh) and any(part in item.GetName() for part in structure):
                UsdPhysics.CollisionAPI.Apply(item)
                UsdPhysics.MeshCollisionAPI.Apply(item).CreateApproximationAttr("none")
        stage.GetRootLayer().Save()

    for name, source, approximation in (
        ("orange", "orange/visual.usda", "convexHull"),
        ("banana", "bananas/bananas_1k.usdc", "convexDecomposition"),
        ("grapes", "grapes/visual.usda", "convexHull"),
        ("blueberry", "blueberry/visual.usda", "convexHull"),
    ):
        stage = stage_for(ASSETS / f"props/{name}.usda", "/Fruit")
        placement = UsdGeom.Xform.Define(stage, "/Fruit/Placement")
        model = UsdGeom.Xform.Define(stage, "/Fruit/Placement/Model")
        model.GetPrim().GetReferences().AddReference("../models/" + source)
        if name == "banana":
            for child in model.GetPrim().GetChildren():
                if child.GetName().startswith("bananas_") and child.GetName() != "bananas_a":
                    child.SetActive(False)
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
            model.GetPrim()).ComputeAlignedRange()
        center = bounds.GetMidpoint()
        # Keep the model's own transform stack intact and put the root at the footprint.
        placement.AddTranslateOp().Set(Gf.Vec3d(-center[0], -center[1], -bounds.GetMin()[2]))
        for item in Usd.PrimRange(model.GetPrim()):
            if item.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(item)
                UsdPhysics.MeshCollisionAPI.Apply(item).CreateApproximationAttr(approximation)
        stage.GetRootLayer().Save()

    def tube_segment(stage, path, a, b, radius, surface):
        start, end = Gf.Vec3d(*a), Gf.Vec3d(*b)
        segment = UsdGeom.Capsule.Define(stage, path)
        segment.CreateRadiusAttr(radius)
        segment.CreateHeightAttr((end-start).GetLength())
        segment.CreateAxisAttr(UsdGeom.Tokens.z)
        segment.AddTranslateOp().Set((start+end)/2)
        segment.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Rotation(Gf.Vec3d(0, 0, 1), end-start).GetQuat())
        UsdPhysics.CollisionAPI.Apply(segment.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(segment.GetPrim()).Bind(surface)

    stage = stage_for(ASSETS / "props/branch.usda", "/Obstacle")
    bark = material(stage, "/Obstacle/Bark", (.18, .092, .044), 0., .85)
    for i, (start, end, radius) in enumerate((
        ((-.090,-.016,.014), (-.026,.008,.014), .014),
        ((-.026,.008,.014), (.035,-.012,.014), .012),
        ((.035,-.012,.014), (.092,.011,.012), .009),
        ((.009,-.004,.014), (.050,.038,.035), .008),
        ((.050,.038,.035), (.072,.051,.047), .006),
    )):
        tube_segment(stage, f"/Obstacle/Segment{i}", start, end, radius, bark)
    stage.GetRootLayer().Save()

    stage = stage_for(ASSETS / "props/irrigation_hose.usda", "/Obstacle")
    rubber = material(stage, "/Obstacle/Rubber", (.020, .041, .036), 0., .55)
    points = [(i*.016-.088, .028*math.sin(i*.46), .009) for i in range(12)]
    for i, (a, b) in enumerate(zip(points, points[1:])):
        tube_segment(stage, f"/Obstacle/Hose{i}", a, b, .009, rubber)
    stage.GetRootLayer().Save()

    stage = stage_for(ASSETS / "props/nursery_pot.usda", "/Obstacle")
    clay = material(stage, "/Obstacle/Clay", (.28, .095, .044), 0., .72)
    soil = material(stage, "/Obstacle/Soil", (.071, .046, .026), 0., .96)
    body = UsdGeom.Mesh.Define(stage, "/Obstacle/PotBody")
    sides = 24
    rings = ((.041,0.), (.056,.078), (.061,.079), (.061,.097))
    points = [Gf.Vec3f(radius*math.cos(i*math.tau/sides), radius*math.sin(i*math.tau/sides), z)
              for radius, z in rings for i in range(sides)]
    faces = [tuple(reversed(range(sides))), tuple(range(3*sides,4*sides))]
    for ring in range(3):
        a, b = ring*sides, (ring+1)*sides
        for i in range(sides):
            j = (i+1)%sides
            faces.append((a+i, a+j, b+j, b+i))
    body.CreatePointsAttr(points)
    body.CreateFaceVertexCountsAttr([len(face) for face in faces])
    body.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    body.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(body.GetPrim()).CreateApproximationAttr("convexHull")
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(clay)
    fill = UsdGeom.Cylinder.Define(stage, "/Obstacle/SoilSurface")
    fill.CreateRadiusAttr(.052)
    fill.CreateHeightAttr(.003)
    fill.AddTranslateOp().Set(Gf.Vec3d(0., 0., .097))
    UsdShade.MaterialBindingAPI.Apply(fill.GetPrim()).Bind(soil)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    build()
