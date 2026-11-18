"""Author an open citrus canopy with individual bent leaves and textured fruit references."""
import math
from pathlib import Path

import numpy as np

from usd_python import configure_usd


OUTPUT = Path(__file__).resolve().parents[1] / "assets/agriculture/props/orange_tree.usdc"


def unit(v):
    return v / np.linalg.norm(v)


class Surface:
    def __init__(self):
        self.points, self.faces, self.colors = [], [], []

    def add(self, points, faces, colors):
        base = len(self.points)
        self.points.extend(points)
        self.faces.extend([[base + i for i in face] for face in faces])
        self.colors.extend(colors)

    def tube(self, curve, radii, rng, sides=7):
        points, colors, faces = [], [], []
        ridges = rng.uniform(.88, 1.13, sides)
        for ring, center in enumerate(curve):
            direction = unit(curve[min(ring+1, len(curve)-1)] - curve[max(0, ring-1)])
            ref = np.array([1., 0., 0.]) if abs(direction[0]) < .8 else np.array([0., 1., 0.])
            side = unit(np.cross(direction, ref))
            normal = np.cross(direction, side)
            for i in range(sides):
                angle = 2*math.pi*i/sides
                points.append(center + radii[ring]*ridges[i]*(math.cos(angle)*side + math.sin(angle)*normal))
                colors.append(np.array([.19, .116, .062])*ridges[i]*rng.uniform(.78, 1.22))
            if ring:
                offset = (ring-1)*sides
                for i in range(sides):
                    j = (i+1)%sides
                    faces.append((offset+i, offset+j, offset+j+sides, offset+i+sides))
        faces.extend([tuple(reversed(range(sides))), tuple(range(len(points)-sides, len(points)))])
        self.add(points, faces, colors)

    def leaf(self, stem, direction, rng):
        axis = unit(direction)
        lateral = unit(np.cross(axis, [.19, .34, 1.]))
        normal = np.cross(lateral, axis)
        twist = rng.uniform(-.7, .7)
        lateral, normal = (math.cos(twist)*lateral + math.sin(twist)*normal,
                           math.cos(twist)*normal - math.sin(twist)*lateral)
        length = rng.uniform(.055, .105)
        width = length*rng.uniform(.16, .25)
        shade = rng.uniform(.68, 1.27)
        color = np.array([.115, .242, .038])*shade + rng.uniform(0., .025, 3)
        points, values, faces = [], [], []
        for i, t in enumerate(np.linspace(0., 1., 7)):
            curve = stem + axis*(t*length) + normal*(length*(.18*math.sin(math.pi*t)-.25*t*t))
            half = width*(.013 + math.sin(math.pi*t)**.9)
            for edge in (-1, 0, 1):
                points.append(curve + edge*half*lateral - abs(edge)*half*.29*normal)
                values.append(color*(1.17 if edge == 0 else .83))
            if i:
                j = (i-1)*3
                faces.extend(((j, j+3, j+4, j+1), (j+1, j+4, j+5, j+2)))
        self.add(points, faces, values)


def build():
    configure_usd()
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    rng = np.random.default_rng(1811)
    wood, leaves = Surface(), Surface()
    trunk = np.array([[0.,0.,0.], [-.035,.025,.24], [.015,.04,.52], [.06,-.012,.79],
                      [.023,.03,1.05], [-.04,.01,1.29], [-.03,.06,1.58]])
    wood.tube(trunk, [.073,.065,.054,.042,.031,.021,.008], rng, sides=12)
    hanging = []
    for i in range(11):
        heading = i*2.39996323 + rng.uniform(-.22,.22)
        radial = np.array([math.cos(heading), math.sin(heading), 0.])
        cross = np.array([-radial[1], radial[0], 0.])
        base = np.array([.03, .02, .68 + .061*i])
        reach = rng.uniform(.43, .63)*(1.12-.027*i)
        tip = base + radial*reach + cross*rng.uniform(-.18,.18) + [0.,0.,rng.uniform(.36,.59)]
        bend = base + .48*(tip-base) + cross*rng.uniform(-.10,.10) - [0.,0.,.085]
        wood.tube([base,bend,tip], [.029-i*.0012,.013,.0035], rng)
        for branch in range(7):
            t = .23 + branch*.12
            stem = (1-t)**2*base + 2*(1-t)*t*bend + t*t*tip
            side = -1 if branch%2 else 1
            shoot = stem + radial*rng.uniform(.07,.20) + cross*(side*rng.uniform(.085,.24))
            shoot += [0.,0.,rng.uniform(.10,.31)]
            wood.tube([stem, .48*stem+.52*shoot+cross*.023, shoot], [.0085,.0047,.0015], rng)
            for fork in range(4):
                at = .29 + .18*fork
                start = stem + at*(shoot-stem)
                bearing = heading + rng.uniform(-1.6,1.6)
                direction = np.array([math.cos(bearing),math.sin(bearing),rng.uniform(-.13,.50)])
                end = start + direction*rng.uniform(.075,.18)
                wood.tube([start,end], [.0034,.00075], rng, sides=5)
                for n in range(8):
                    blend = (n+1)/9
                    angle = bearing + (-1 if n%2 else 1)*rng.uniform(.42,1.30)
                    leaf_direction = np.array([math.cos(angle),math.sin(angle),rng.uniform(-.75,.52)])
                    leaves.leaf(start+blend*(end-start), leaf_direction, rng)
            if branch in (3, 6) and i != 8:
                near = .38*stem+.62*shoot
                fruit_tops = near + .057*radial - np.array([0.,0.,rng.uniform(.055,.095)])
                wood.tube([near, fruit_tops], [.0035,.0020], rng, sides=5)
                hanging.append(fruit_tops)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.FindOrOpen(str(OUTPUT)) if OUTPUT.exists() else Sdf.Layer.CreateNew(str(OUTPUT))
    layer.Clear()
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, "/OrangeTree")
    stage.SetDefaultPrim(root.GetPrim())

    def author(surface, name, roughness, leaf=False):
        mesh = UsdGeom.Mesh.Define(stage, f"/OrangeTree/{name}")
        points = Vt.Vec3fArray.FromNumpy(np.asarray(surface.points, dtype=np.float32))
        colors = Vt.Vec3fArray.FromNumpy(np.asarray(surface.colors, dtype=np.float32))
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([len(face) for face in surface.faces])
        mesh.CreateFaceVertexIndicesAttr([idx for face in surface.faces for idx in face])
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(leaf)
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(colors)
        mat_path = f"/OrangeTree/Materials/{name}"
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path+"/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        reader = UsdShade.Shader.Define(stage, mat_path+"/Color")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), "result")
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        return mesh.GetPrim()

    # Only woody parts are static physical obstacles. The leaf mesh is a visual occluder.
    branches = author(wood, "Branches", .82)
    UsdPhysics.CollisionAPI.Apply(branches)
    UsdPhysics.MeshCollisionAPI.Apply(branches).CreateApproximationAttr("none")
    author(leaves, "Leaves", .34, leaf=True)
    UsdGeom.Xform.Define(stage, "/OrangeTree/Fruit")
    for i, top in enumerate(hanging):
        fruit = UsdGeom.Xform.Define(stage, f"/OrangeTree/Fruit/Orange{i:02d}")
        scale = rng.uniform(.86,1.22)
        fruit.AddTranslateOp().Set(Gf.Vec3d(*(top - [0.,0.,.0763*scale])))
        fruit.AddRotateZOp().Set(float(rng.uniform(0.,360.)))
        fruit.AddScaleOp().Set(Gf.Vec3f(scale))
        shape = UsdGeom.Xform.Define(stage, fruit.GetPath().AppendChild("Model")).GetPrim()
        shape.GetReferences().AddReference("../models/orange/visual.usda")
        shape.SetInstanceable(True)
    layer.Save()
    print(f"Built {OUTPUT.name}: {len(leaves.faces)//12} leaves and {len(hanging)} oranges")


if __name__ == "__main__":
    build()
