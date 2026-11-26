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
    assert len(bodies) == 94 and loops == 30 and driven == 33
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
