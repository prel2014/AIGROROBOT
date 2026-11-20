"""Bake botanical point instances once per species for the replicated RTX/PhysX scenes."""
import os

import numpy as np


def bake_mesh(stage,path,source,transforms):
    from pxr import UsdGeom,UsdShade

    src = UsdGeom.Mesh(source)
    mesh = UsdGeom.Mesh.Define(stage,path)
    points = np.asarray(src.GetPointsAttr().Get())
    # USD Gf matrices transform row vectors. Keep the authored geometry, UVs and normals.
    rotations = transforms[:,:3,:3]
    xyz = np.matmul(points[None],rotations)+transforms[:,None,3,:3]
    mesh.CreatePointsAttr(xyz.reshape(-1,3).astype("f"))
    count = len(transforms)
    mesh.CreateFaceVertexCountsAttr(np.tile(src.GetFaceVertexCountsAttr().Get(),count))
    faces = np.asarray(src.GetFaceVertexIndicesAttr().Get())[None]+len(points)*np.arange(count)[:,None]
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().astype("i"))
    mesh.CreateSubdivisionSchemeAttr(src.GetSubdivisionSchemeAttr().Get())
    mesh.CreateDoubleSidedAttr(src.GetDoubleSidedAttr().Get())
    normals = src.GetNormalsAttr().Get()
    if normals:
        normal = np.matmul(np.array(normals)[None],np.linalg.inv(rotations).transpose(0,2,1))
        normal /= np.maximum(1e-9,np.linalg.norm(normal,axis=-1,keepdims=True))
        mesh.CreateNormalsAttr(normal.reshape(-1,3).astype("f"))
        mesh.SetNormalsInterpolation(src.GetNormalsInterpolation())
    for name in ("st","displayColor","displayOpacity"):
        data = UsdGeom.PrimvarsAPI(src).GetPrimvar(name)
        if not data or not data.HasValue():
            continue
        values = data.ComputeFlattened()
        authored = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(name,data.GetTypeName(),data.GetInterpolation())
        authored.Set(values if data.GetInterpolation()=="constant" else type(values)(list(values)*count))
    material,_ = UsdShade.MaterialBindingAPI(src).ComputeBoundMaterial()
    if material:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(UsdShade.Material.Get(stage,material.GetPath()))


def compose_plants(field,out,asset):
    from pxr import Sdf,Usd,UsdGeom

    original = Usd.Stage.Open(str(asset))
    root_path = original.GetDefaultPrim().GetPath()
    layer = Usd.Stage.CreateNew(str(out/"planta.usdc"))
    root = UsdGeom.Xform.Define(layer,root_path).GetPrim()
    layer.SetDefaultPrim(root)
    root.GetReferences().AddReference(os.path.relpath(asset,out).replace(os.sep,"/"))
    time = Usd.TimeCode.Default()
    cache = UsdGeom.XformCache(time)
    for source in original.Traverse():
        if not source.IsA(UsdGeom.PointInstancer):
            continue
        inst = UsdGeom.PointInstancer(source)
        poses = np.array(inst.ComputeInstanceTransformsAtTime(time,time,UsdGeom.PointInstancer.IncludeProtoXform))
        indices = np.asarray(inst.GetProtoIndicesAttr().Get())
        parent = source.GetPath()
        UsdGeom.Xform.Define(layer,parent)
        for kind,path in enumerate(inst.GetPrototypesRel().GetTargets()):
            proto = original.GetPrimAtPath(path)
            proto_inverse = cache.GetLocalToWorldTransform(proto).GetInverse()
            subset = poses[indices==kind]
            if len(subset):
                for i,prim in enumerate(p for p in Usd.PrimRange(proto) if p.IsA(UsdGeom.Mesh)):
                    local = np.array(cache.GetLocalToWorldTransform(prim)*proto_inverse)
                    bake_mesh(layer,parent.AppendChild(f"Botanica{kind}_{i}"),prim,local[None]@subset)
            layer.OverridePrim(path.GetParentPath()).SetActive(False)
    layer.GetRootLayer().Save()
    for row in field.GetPrimAtPath("/Parcela/Hileras").GetChildren():
        for plant in row.GetChildren():
            plant.GetReferences().SetReferences([Sdf.Reference("planta.usdc")])
