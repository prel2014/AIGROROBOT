"""Comprueba composición y contratos USD en memoria, sin SimulationApp ni pasos físicos."""
from pxr import Usd, UsdGeom, UsdPhysics
from curriculo_xavier import ACCIONES, PLANES
from estaciones_xavier import construir


def revisar():
    total = 0
    for tarea, plan in PLANES.items():
        for entorno in plan.entornos:
            for nivel in range(3):
                stage = Usd.Stage.CreateInMemory()
                cantidad = 16 if (tarea,entorno,nivel)==("mision","vivienda",2) else 1
                construir(stage,tarea,nivel,entorno,cantidad)
                for i in range(cantidad):
                    raiz=f"/World/E{i:02}"
                    prim=stage.GetPrimAtPath
                    assert prim(raiz+"/Xavier/pelvis").HasAPI(UsdPhysics.ArticulationRootAPI)
                    assert prim(raiz+"/Xavier/cabeza/Camara").IsA(UsdGeom.Camera)
                    assert prim(raiz+"/Decorado").GetChildren()
                    for nombre in ACCIONES:
                        j=prim(raiz+"/Xavier/Juntas/"+nombre)
                        drive=UsdPhysics.DriveAPI.Get(j,"angular")
                        assert drive and drive.GetMaxForceAttr().Get()>0, nombre
                        junta=UsdPhysics.RevoluteJoint(j)
                        assert junta.GetLowerLimitAttr().Get()<junta.GetUpperLimitAttr().Get()
                        assert all(prim(p) for p in junta.GetBody1Rel().GetTargets())
                    if plan.objeto:
                        assert prim(raiz+"/Objeto").HasAPI(UsdPhysics.RigidBodyAPI)
                        assert prim(raiz+"/Fuente").HasAPI(UsdPhysics.CollisionAPI)
                    if tarea=="tijeras":
                        assert prim(raiz+"/Objeto/Anilla0/Segmento0").HasAPI(UsdPhysics.CollisionAPI)
                    if tarea=="encestar":
                        assert prim(raiz+"/Aro/Segmento0").HasAPI(UsdPhysics.CollisionAPI)
                total += 1
    print(f"{total} combinaciones USD verificadas, incluida la composición con 16 Xavier")


if __name__=="__main__":
    revisar()
