"""Genera la pista compartida y los sensores del Xavier en una capa USD portátil."""
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from cargas_plantares import ANCHO, LARGO, ZONAS, centro_zona

HERE = Path(__file__).resolve().parent
ASSET = "humanoide_xavier_isaac.usd"
PISTA = HERE / "pista_marcha_16.usda"
N = 16
DEDOS = ("indice_", "medio_", "anular_", "menique_", "pulgar_")


def agrupar_mano(stage, source, side):
    """Agrega las masas e inercias de los dedos abiertos en el marco de la palma."""
    from scipy.spatial.transform import Rotation

    palm = source.GetPrimAtPath("/Humanoide/mano_" + side)
    dest = "/XavierMarcha/mano_" + side
    origin = np.array(palm.GetAttribute("xformOp:translate").Get())
    parts = [p for p in source.GetDefaultPrim().GetChildren()
             if p.GetName().startswith(DEDOS) and f"_{side}_" in p.GetName()]
    masses, centers, inertias = [], [], []
    for part in [palm, *parts]:
        offset = np.array(part.GetAttribute("xformOp:translate").Get()) - origin
        api = UsdPhysics.MassAPI(part)
        masses.append(api.GetMassAttr().Get())
        centers.append(offset + api.GetCenterOfMassAttr().Get())
        axes = api.GetPrincipalAxesAttr().Get()
        rotation = Rotation.from_quat([*axes.GetImaginary(), axes.GetReal()]).as_matrix()
        inertias.append(rotation @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ rotation.T)
        if part != palm:
            child = UsdGeom.Xform.Define(stage, dest + "/" + part.GetName()).GetPrim()
            child.GetReferences().AddReference("./" + ASSET, part.GetPath())
            for schema in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.FilteredPairsAPI):
                child.RemoveAPI(schema)
            child.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*offset))
            # Al referenciar solo un dedo sus materiales quedan fuera de esa referencia.
            # Volver a enlazarlos a la biblioteca común del Xavier.
            for mesh in part.GetChild("Visual").GetChildren():
                binding = mesh.GetRelationship("material:binding")
                if binding and binding.GetTargets():
                    dst = stage.OverridePrim(child.GetPath().AppendPath("Visual/" + mesh.GetName()))
                    dst.GetRelationship("material:binding").SetTargets([
                        binding.GetTargets()[0].ReplacePrefix("/Humanoide", "/XavierMarcha")])
            stage.OverridePrim("/XavierMarcha/" + part.GetName()).SetActive(False)
    center = np.average(centers, axis=0, weights=masses)
    inertia = np.sum(inertias, axis=0)
    for mass, delta in zip(masses, centers - center):
        inertia += mass * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
    moments, axes = np.linalg.eigh(inertia)
    axes[:, 0] *= np.linalg.det(axes)
    q = Rotation.from_matrix(axes).as_quat()
    palm_dest = stage.GetPrimAtPath(dest)
    palm_dest.RemoveAPI(UsdPhysics.FilteredPairsAPI)
    api = UsdPhysics.MassAPI(palm_dest)
    api.GetMassAttr().Set(sum(masses))
    api.GetCenterOfMassAttr().Set(Gf.Vec3f(*center))
    api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*moments))
    api.GetPrincipalAxesAttr().Set(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))


def recorrido(i):
    """Cuatro cruces en un único suelo, dos filas de dos robots en cada cruce."""
    group, slot = divmod(i, 4)
    center = np.array([3.2 * (-1 if group % 2 == 0 else 1),
                       3.2 * (-1 if group < 2 else 1)])
    heading = np.array([(np.sign(center[0]), 0), (0, np.sign(center[1]))][slot % 2])
    start = center - (.80 + 1.20 * (slot // 2)) * heading
    # Las salidas apuntan hacia el perímetro para que cada fila complete su propio cruce.
    return start, heading, center + (1.95 if slot < 2 else .85) * heading


def crear(destino=PISTA):
    # Identificadores relativos al archivo de salida para poder mover la carpeta.
    import os
    import tempfile

    fd, filename = tempfile.mkstemp(dir=HERE, suffix=".usda")
    os.close(fd)
    outfile = Path(filename)
    try:
        stage = Usd.Stage.CreateNew(str(outfile))
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
        bench = Sdf.Layer.FindOrOpen(str(HERE / "banco_xavier.usda"))
        for path in ("Fisica", "Suelo", "Luz", "LuzPrincipal"):
            path = "/World/" + path
            Sdf.CopySpec(bench, path, stage.GetRootLayer(), path)
        stage.GetPrimAtPath("/World/Fisica").GetAttribute("physxScene:timeStepsPerSecond").Set(120)
        stage.GetPrimAtPath("/World/Suelo").GetAttribute("xformOp:scale").Set((24., 24., .07))
        stage.SetTimeCodesPerSecond(20.)
        stage.SetEndTimeCode(20. * 86400)
        stage.GetRootLayer().customLayerData = {"cameraSettings": {"Perspective": {
            "position": Gf.Vec3d(11., -14., 12.), "target": Gf.Vec3d(0, 0, .6)}}}

        # La clase abstracta aporta el mismo robot y los mismos marcos locales a las 16 copias.
        template = stage.CreateClassPrim("/XavierMarcha")
        template.GetReferences().AddReference("./" + ASSET)
        source = Usd.Stage.Open(str(HERE / ASSET))
        for side in ("izq", "der"):
            agrupar_mano(stage, source, side)
            for i, nombre in enumerate(ZONAS):
                # Guías de las áreas de integración. La colisión sigue en la suela original.
                zona = UsdGeom.Cube.Define(stage, f"/XavierMarcha/pie_{side}/Cargas/{nombre}")
                zona.CreateSizeAttr(1.)
                zona.AddTranslateOp().Set(centro_zona(side, i))
                zona.AddScaleOp().Set((ANCHO/2, LARGO/2, .0005))
                zona.CreatePurposeAttr(UsdGeom.Tokens.guide)
                zona.CreateDisplayColorAttr([(0.1, .65, .95) if i%2 == 0 else (.95, .5, .12)])
                zona.GetPrim().CreateAttribute("carga:canal", Sdf.ValueTypeNames.Int).Set(i)
        active_links = []
        for link in source.GetDefaultPrim().GetChildren():
            if not link.HasAPI(UsdPhysics.RigidBodyAPI) or link.GetName().startswith(DEDOS):
                continue
            body = stage.OverridePrim("/XavierMarcha/" + link.GetName())
            active_links.append(body.GetPath())
            body.AddAppliedSchema("PhysxContactReportAPI")
            body.CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float).Set(0.)
            visual = stage.OverridePrim(str(body.GetPath()) + "/Visual")
            visual.GetReferences().AddReference("./" + ASSET, str(link.GetPath()) + "/Visual")
            visual.SetInstanceable(True)
        template.GetRelationship("isaac:physics:robotLinks").SetTargets(active_links)
        pelvis = stage.GetPrimAtPath("/XavierMarcha/pelvis")
        pelvis.GetAttribute("physxArticulation:solverPositionIterationCount").Set(12)
        # Los apoyos quietos también necesitan impulsos nuevos en cada tick de muestreo.
        pelvis.CreateAttribute("physxArticulation:sleepThreshold", Sdf.ValueTypeNames.Float, custom=False).Set(0.)
        camera = UsdGeom.Camera.Define(stage, "/XavierMarcha/cabeza/Camara")
        camera.AddTranslateOp().Set((0., .080, .048))
        camera.AddRotateXOp().Set(75.)  # -Z óptico mira a +Y con 15° de inclinación.
        camera.CreateHorizontalApertureAttr(20.)
        camera.CreateVerticalApertureAttr(20.)
        camera.CreateFocalLengthAttr(8.4)
        camera.CreateClippingRangeAttr((.02, 30.))
        stage.DefinePrim("/XavierMarcha/pelvis/IMU", "IsaacImuSensor")

        active_joints = []
        for prim in source.GetPrimAtPath("/Humanoide/Juntas").GetChildren():
            path = "/XavierMarcha/Juntas/" + prim.GetName()
            if prim.GetName().startswith(DEDOS):
                stage.OverridePrim(path).SetActive(False)
                continue
            active_joints.append(path)
        template.GetRelationship("isaac:physics:robotJoints").SetTargets(active_joints)
        for i in range(N):
            start, heading, _ = recorrido(i)
            robot = UsdGeom.Xform.Define(stage, f"/World/Xavier_{i:02}")
            robot.GetPrim().GetInherits().AddInherit(template.GetPath())
            robot.AddTranslateOp().Set((*start, .003))
            robot.AddRotateZOp().Set(float(-np.rad2deg(np.arctan2(heading[0], heading[1]))))
        stage.GetRootLayer().Save()
        os.replace(outfile, destino)
    finally:
        outfile.unlink(missing_ok=True)


if __name__ == "__main__":
    crear()
    print(PISTA)
