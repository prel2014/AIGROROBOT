"""Comprueba el USD guardado, sus referencias, las mallas CAD y el árbol físico."""
from pathlib import Path
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

HERE = Path(__file__).resolve().parent


def geometria(stage, cad, cache):
    def index(st):
        meshes = [p for p in st.Traverse() if p.IsA(UsdGeom.Mesh) and p.HasAttribute("cad:ocurrencia")]
        result = {p.GetAttribute("cad:ocurrencia").Get(): p for p in meshes}
        assert len(result) == len(meshes)
        return result
    original, visual = index(cad), index(stage)
    assert original.keys() == visual.keys()
    for key, source in original.items():
        copy = visual[key]
        assert source.GetDisplayName() == copy.GetDisplayName()
        np.testing.assert_allclose(cache.GetLocalToWorldTransform(source), cache.GetLocalToWorldTransform(copy),
                                   atol=1e-8, rtol=0)
        for attr in ("points", "faceVertexIndices", "faceVertexCounts"):
            assert source.GetAttribute(attr).Get() == copy.GetAttribute(attr).Get(), (key, attr)
        assert UsdShade.MaterialBindingAPI(copy).ComputeBoundMaterial()[0]
    return len(visual)


def comprobar():
    cad = Usd.Stage.Open(str(HERE.parent / "cad/humanoide_xavier.usd"))
    stage = Usd.Stage.Open(str(HERE / "humanoide_xavier_isaac.usd"))
    cache = UsdGeom.XformCache()
    meshes = geometria(stage, cad, cache)
    bodies = {p.GetPath(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    parent = {}
    driven, loops, total = 0, 0, 0.
    for p in bodies.values():
        mass = UsdPhysics.MassAPI(p)
        eig = sorted(mass.GetDiagonalInertiaAttr().Get())
        assert 0 < eig[0] and eig[2] <= (eig[0] + eig[1]) * 1.0001
        m = mass.GetMassAttr().Get()
        assert m > 0
        total += m
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        a, = j.GetBody0Rel().GetTargets()
        b, = j.GetBody1Rel().GetTargets()
        point0 = cache.GetLocalToWorldTransform(bodies[a]).Transform(Gf.Vec3d(j.GetLocalPos0Attr().Get()))
        point1 = cache.GetLocalToWorldTransform(bodies[b]).Transform(Gf.Vec3d(j.GetLocalPos1Attr().Get()))
        np.testing.assert_allclose(point0, point1, atol=2e-7, rtol=0)
        if j.GetExcludeFromArticulationAttr().Get():
            loops += 1
        else:
            assert b not in parent
            parent[b] = a
        if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            assert prim.GetAttribute("physics:lowerLimit").Get() <= 0 <= prim.GetAttribute("physics:upperLimit").Get()
            driven += 1
    for link in bodies:
        visited = set()
        while link in parent:
            assert link not in visited
            visited.add(link)
            link = parent[link]
        assert str(link) == "/Humanoide/pelvis"
    assert len(bodies) == 98 and loops == 30 and driven == 37
    for side, sign in (("izq",-1),("der",1)):
        indice = 2 if sign == -1 else 4
        pierna = 7 if sign == -1 else 9
        for link, prefix in ((f"bastidor_giro_{side}",f"o1.20.{indice}."),
                             (f"bastidor_apertura_{side}",f"o1.20.{indice+1}."),
                             (f"muslo_{side}",f"o1.{pierna}.6.")):
            visual = stage.GetPrimAtPath(f"/Humanoide/{link}/Visual")
            assert any(m.GetAttribute("cad:ocurrencia").Get().startswith(prefix) for m in visual.GetChildren())
        for name, axis, limits, link0, link1 in (
            ("giro_cadera",(0,0,-sign),(-20,20),"pelvis","bastidor_giro_"+side),
            ("apertura_cadera",(0,-sign,0),(-8,18),"bastidor_giro_"+side,"bastidor_apertura_"+side),
            ("cadera",(1,0,0),(-15,35),"bastidor_apertura_"+side,"muslo_"+side),
        ):
            j = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"/Humanoide/Juntas/{name}_{side}"))
            assert str(j.GetBody0Rel().GetTargets()[0]) == "/Humanoide/"+link0
            assert str(j.GetBody1Rel().GetTargets()[0]) == "/Humanoide/"+link1
            np.testing.assert_allclose(j.GetLocalRot0Attr().Get().Transform(Gf.Vec3f(1,0,0)),axis,atol=1e-6)
            assert (j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get()) == limits
            assert j.GetPrim().GetAttribute("drive:angular:physics:maxForce").Get() == 24.
            assert j.GetPrim().GetAttribute("physxJoint:maxJointVelocity").Get() == 420.
    for file, expected in (("humanoide_xavier_isaac.usd", "/Humanoide/pelvis"),
                           ("banco_xavier.usda", "/World/Humanoide/Anclaje")):
        saved = Usd.Stage.Open(str(HERE / file))
        assert not saved.GetCompositionErrors() and not UsdUtils.ComputeAllDependencies(str(HERE / file))[2]
        assert UsdGeom.GetStageMetersPerUnit(saved) == 1 and UsdGeom.GetStageUpAxis(saved) == "Z"
        roots = [str(p.GetPath()) for p in saved.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        assert roots == [expected], roots
        if file == "banco_xavier.usda":
            scene = saved.GetPrimAtPath("/World/Fisica")
            assert scene.GetAttribute("physxScene:enableGPUDynamics").Get() is False
            assert scene.GetAttribute("physxScene:broadphaseType").Get() == "MBP"
    print(f"{meshes} mallas CAD, {len(bodies)} cuerpos, {driven} mandos, {loops} cierres. "
          f"Masa asignada: {total:.3f} kg.")


if __name__ == "__main__":
    comprobar()
