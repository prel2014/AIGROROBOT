"""Generate meter-scale agricultural props with colliders for the scene randomizer."""
import math
from pathlib import Path
from random import Random

from usd_python import configure_usd


OUTPUT = Path(__file__).resolve().parents[1] / "assets/agriculture/props"


def build():
    configure_usd()
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    def stage_for(name):
        path = OUTPUT / f"{name}.usda"
        layer = Sdf.Layer.FindOrOpen(str(path)) if path.exists() else Sdf.Layer.CreateNew(str(path))
        layer.Clear()
        stage = Usd.Stage.Open(layer)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Obstacle").GetPrim())
        return stage

    def material(stage, name, color, roughness=.54, metal=0.):
        path = f"/Obstacle/Materials/{name}"
        surface = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        surface.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return surface

    def finish(geom, surface, *, collide=True):
        prim = geom.GetPrim()
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(surface)
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim)
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")

    def mesh(stage, name, points, faces, surface, *, collide=True):
        geom = UsdGeom.Mesh.Define(stage, f"/Obstacle/{name}")
        geom.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        geom.CreateFaceVertexCountsAttr([len(face) for face in faces])
        geom.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
        geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        finish(geom, surface, collide=collide)

    def loft(stage, name, rings, surface, *, caps=True, collide=True):
        sides = len(rings[0])
        faces = []
        for ring in range(len(rings)-1):
            first = ring*sides
            second = first+sides
            for i in range(sides):
                j = (i+1) % sides
                faces.append((first+i, first+j, second+j, second+i))
        if caps:
            faces.extend((tuple(reversed(range(sides))), tuple(range(second, second+sides))))
        mesh(stage, name, [point for ring in rings for point in ring], faces, surface, collide=collide)

    def circular(radius, height, sides=24):
        return [(radius*math.cos(i*math.tau/sides), radius*math.sin(i*math.tau/sides), height)
                for i in range(sides)]

    def capsule(stage, name, start, end, radius, surface, *, collide=True):
        a, b = Gf.Vec3d(*start), Gf.Vec3d(*end)
        geom = UsdGeom.Capsule.Define(stage, f"/Obstacle/{name}")
        geom.CreateRadiusAttr(radius)
        geom.CreateHeightAttr((b-a).GetLength())
        geom.CreateAxisAttr(UsdGeom.Tokens.z)
        geom.AddTranslateOp().Set((a+b)*.5)
        geom.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Rotation(Gf.Vec3d(0., 0., 1.), b-a).GetQuat())
        finish(geom, surface, collide=collide)

    def extrusion(stage, name, outline, z0, z1, surface, *, collide=True):
        # Keep outward normals consistent for both handed contours.
        area = sum(x*q-y*p for (x, y), (p, q) in zip(outline, outline[1:]+outline[:1]))
        polygon = outline if area > 0 else outline[::-1]
        loft(stage, name, [[(x, y, z) for x, y in polygon] for z in (z0, z1)], surface, collide=collide)

    stage = stage_for("harvest_bucket")
    plastic = material(stage, "Body", (.056, .19, .157), .49)
    handle = material(stage, "Handle", (.37, .41, .42), .33, .6)
    # The thick wall is closed at the bottom and open at the top for rendering.
    profile = ((.077, .0), (.081, .011), (.108, .186), (.115, .190), (.115, .205),
               (.104, .205), (.103, .192), (.073, .014))
    loft(stage, "BucketShell", [circular(r, z) for r, z in profile], plastic, collide=False)
    # Twelve overlapping wall panels form a hollow collision body with a separate floor.
    for i in range(12):
        angle = i*math.tau/12
        angles = (angle-.01, angle+math.tau/12+.01)
        panels = []
        for low, high, z in ((.073,.081,.010), (.102,.114,.205)):
            lo, hi = angles
            panels.append([(r*math.cos(t), r*math.sin(t), z)
                           for r,t in ((low,lo),(high,lo),(high,hi),(low,hi))])
        # Same color as the exterior. Hidden geometry remains present for PhysX.
        name = f"CollisionWall{i}"
        loft(stage, name, panels, plastic)
        UsdGeom.Imageable(stage.GetPrimAtPath(f"/Obstacle/{name}")).MakeInvisible()
    floor = UsdGeom.Cylinder.Define(stage, "/Obstacle/CollisionFloor")
    floor.CreateRadiusAttr(.079)
    floor.CreateHeightAttr(.014)
    floor.AddTranslateOp().Set(Gf.Vec3d(0., 0., .007))
    finish(floor, plastic)
    floor.MakeInvisible()
    arc = [(.118*math.cos(i*math.pi/12), 0., .18+.112*math.sin(i*math.pi/12)) for i in range(13)]
    for i, (a, b) in enumerate(zip(arc, arc[1:])):
        capsule(stage, f"Handle{i}", a, b, .0036, handle)
    stage.GetRootLayer().Save()

    for name, width, depth, height, color, seed in (
        ("field_rock", .083, .062, .069, (.25,.24,.21), 16),
        ("flat_rock", .098, .070, .038, (.20,.24,.255), 27),
    ):
        stage = stage_for(name)
        stone = material(stage, "WeatheredStone", color, .91)
        rng = Random(seed)
        sides = 16
        angles = [i*math.tau/sides for i in range(sides)]
        ribs = [rng.uniform(.84, 1.04) for _ in angles]
        rings = []
        for z, scale, shift in ((0.,.59,-.009), (.31,.99,0.), (.69,.88,.006), (.99,.37,.018)):
            rings.append([(shift+width*scale*b*math.cos(t), depth*scale*b*math.sin(t),
                           height*z*(1. + rng.uniform(-.10,.08))) for t,b in zip(angles,ribs)])
        loft(stage, "Rock", rings, stone)
        stage.GetRootLayer().Save()

    stage = stage_for("brick")
    clay = material(stage, "Clay", (.37,.117,.060), .90)
    worn = material(stage, "Recess", (.23,.075,.039), .94)
    outline = [(-.078,-.041),(.075,-.041),(.089,-.032),(.085,.034),
               (.074,.041),(-.077,.038),(-.087,.027),(-.087,-.029)]
    loft(stage, "Brick", [[(x*factor,y*factor,z) for x,y in outline]
                          for z, factor in ((0.,.94),(.005,1.),(.047,1.),(.054,.94))], clay)
    recess = [(-.062,-.024),(.061,-.024),(.067,-.016),(.067,.016),
              (.061,.024),(-.062,.024),(-.067,.016),(-.067,-.016)]
    extrusion(stage, "Inset", recess, .0541, .0544, worn, collide=False)
    stage.GetRootLayer().Save()

    stage = stage_for("pruning_shears")
    steel = material(stage, "Steel", (.43,.49,.50), .34, .67)
    grip = material(stage, "Grip", (.69,.105,.035), .56)
    dark = material(stage, "Frame", (.073,.087,.096), .49, .28)
    for side in (-1, 1):
        capsule(stage, f"Grip{side+1}", (-.093,side*.033,.012), (-.027,side*.019,.012), .011, grip)
        capsule(stage, f"Lever{side+1}", (-.045,side*.021,.010), (.016,side*.009,.011), .0075, dark)
    extrusion(stage, "Blade", [(.004,-.012),(.046,-.014),(.074,-.009),(.099,.007),
                                (.081,.011),(.045,.008),(.020,.014)], .010, .017, steel)
    extrusion(stage, "Hook", [(.008,.003),(.055,.003),(.083,.013),(.089,.022),
                               (.067,.022),(.049,.014),(.006,.019)], .005, .011, dark)
    pin = UsdGeom.Cylinder.Define(stage, "/Obstacle/Pivot")
    pin.CreateRadiusAttr(.0095)
    pin.CreateHeightAttr(.010)
    pin.AddTranslateOp().Set(Gf.Vec3d(.013,.003,.016))
    finish(pin, steel)
    stage.GetRootLayer().Save()

    stage = stage_for("hand_trowel")
    steel = material(stage, "Scoop", (.38,.43,.44), .35, .66)
    wood = material(stage, "WoodGrip", (.39,.23,.091), .62)
    capsule(stage, "Handle", (-.108,0.,.018), (-.043,0.,.019), .015, wood)
    capsule(stage, "Shank", (-.041,0.,.013), (.022,0.,.014), .006, steel)
    spoon = [(-.006,-.014),(.035,-.042),(.084,-.037),(.114,-.019),
             (.133,.0),(.115,.020),(.084,.037),(.035,.041),(-.006,.014)]
    loft(stage, "Scoop", [[(x,y,z+.38*abs(y)) for x,y in spoon] for z in (.003,.006)], steel)
    stage.GetRootLayer().Save()

    stage = stage_for("hand_cultivator")
    steel = material(stage, "Tines", (.19,.21,.21), .46, .6)
    grip = material(stage, "Handle", (.17,.30,.14), .6)
    capsule(stage, "Grip", (-.105,0.,.018), (-.032,0.,.018), .015, grip)
    capsule(stage, "Shank", (-.037,0.,.016), (.029,0.,.022), .006, steel)
    capsule(stage, "Crossbar", (.015,-.039,.021), (.015,.039,.021), .006, steel)
    for i, (y, length) in enumerate(((-.038,.101),(0.,.118),(.038,.101))):
        capsule(stage, f"Stem{i}", (.015,y,.021), (length-.025,y,.022), .0042, steel)
        capsule(stage, f"Hook{i}", (length-.025,y,.022), (length,y,.005), .0042, steel)
    stage.GetRootLayer().Save()

    stage = stage_for("field_hoe")
    wood = material(stage, "Handle", (.39,.248,.105), .70)
    steel = material(stage, "Blade", (.15,.17,.18), .57, .47)
    capsule(stage, "Handle", (-.295,0.,.016), (.209,0.,.016), .015, wood)
    capsule(stage, "Socket", (.180,0.,.018), (.247,0.,.018), .017, steel)
    extrusion(stage, "Blade", [(.216,-.024),(.256,-.077),(.309,-.086),(.323,-.068),
                                (.323,.072),(.307,.084),(.256,.072),(.216,.026)], .001, .007, steel)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build()
