"""Guarda las poses físicas medidas como una capa USD de reproducción a 10 Hz."""
from pathlib import Path
import os
import numpy as np


def guardar(destino, escena, enlaces, muestras):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    destino=Path(destino).resolve()
    stage=Usd.Stage.CreateNew(str(destino))
    stage.GetRootLayer().subLayerPaths=[Path(os.path.relpath(escena,destino.parent)).as_posix()]
    stage.SetTimeCodesPerSecond(10)
    stage.SetFramesPerSecond(10)
    stage.SetEndTimeCode(len(muestras)-1)
    stage.SetStartTimeCode(0)
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    stage.GetPrimAtPath('/World/Fisica').SetActive(False)
    stage.GetPrimAtPath('/World/Xavier/Juntas').SetActive(False)
    stage.GetPrimAtPath('/World/ArbolObjetivo/UnionPedunculo').SetActive(False)
    rutas=['/World/Xavier/'+n for n in enlaces]+['/World/Naranja']
    ops=[]
    for ruta in rutas:
        prim=stage.GetPrimAtPath(ruta)
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        ops.append(UsdGeom.Xformable(prim).MakeMatrixXform())
    for frame,poses in enumerate(muestras):
        for idx,(pose,op) in enumerate(zip(poses,ops)):
            x,y,z,qx,qy,qz,qw=map(float,pose)
            matriz=Gf.Matrix4d(1.).SetRotate(Gf.Quatd(qw,Gf.Vec3d(qx,qy,qz)).GetNormalized())
            matriz.SetTranslateOnly(Gf.Vec3d(x,y,z-.003 if idx<len(enlaces) else z))
            op.Set(matriz,frame)
    stage.GetRootLayer().Save()
