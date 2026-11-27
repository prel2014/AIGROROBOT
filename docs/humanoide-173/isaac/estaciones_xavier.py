"""Adaptador de geometría USD de las estaciones, creado sólo al solicitar una corrida."""
from pathlib import Path
from math import pi, sin, cos
import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics, UsdLux, Sdf

from curriculo_xavier import diseno, PLANES

HERE = Path(__file__).resolve().parent
MADERA = (.50, .30, .14)
AZUL = (.08, .36, .65)
NARANJA = (.94, .34, .07)


def cubo(stage, path, xyz, tam, color, colision=True, angulo=0):
    cub = UsdGeom.Cube.Define(stage, path)
    cub.CreateSizeAttr(1.)
    cub.AddTranslateOp().Set(Gf.Vec3d(*map(float, xyz)))
    cub.AddRotateZOp().Set(float(angulo))
    cub.AddScaleOp().Set(Gf.Vec3f(*map(float, tam)))
    cub.CreateDisplayColorAttr([color])
    if colision:
        UsdPhysics.CollisionAPI.Apply(cub.GetPrim())
    return cub


def esfera(stage, path, xyz, radio, color, colision=False):
    forma = UsdGeom.Sphere.Define(stage, path)
    forma.AddTranslateOp().Set(Gf.Vec3d(*map(float, xyz)))
    forma.CreateRadiusAttr(radio)
    forma.CreateDisplayColorAttr([color])
    if colision:
        UsdPhysics.CollisionAPI.Apply(forma.GetPrim())
    return forma


def anilla(stage, path, centro, radio, grosor, color, partes):
    for j in range(partes):
        a = 2*pi*j/partes
        pos = np.array(centro) + [radio*cos(a), radio*sin(a), 0]
        cubo(stage, f"{path}/Segmento{j}", pos,
             (2*radio*sin(pi/partes)*1.04, grosor, grosor), color, angulo=np.rad2deg(a)+90)


def base(i):
    return np.array([(i%4)*64., (i//4)*64., 0.])


def silla_z(nivel):
    # Banco alto compatible con los 35 grados de flexión de cadera del USD.
    return (.81, .79, .785)[nivel]


def construir(stage, tarea, nivel, entorno, cantidad):
    mundo = HERE / "escenarios" / (entorno+".usda")
    stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.Xform.Define(stage, "/World")
    for nombre in ("Fisica", "Luz", "LuzPrincipal"):
        stage.DefinePrim("/World/"+nombre).GetReferences().AddReference(str(mundo), "/World/"+nombre)
    origen, destino, final = diseno(tarea, nivel)
    for i in range(cantidad):
        path = f"/World/E{i:02}"
        estacion = UsdGeom.Xform.Define(stage, path)
        estacion.AddTranslateOp().Set(Gf.Vec3d(*base(i).tolist()))
        stage.DefinePrim(path+"/Decorado").GetReferences().AddReference(str(mundo), "/World/Escenario")
        cubo(stage, path+"/Suelo", (0,0,-.04), (57,54,.08), (.48,.46,.43))
        robot = stage.DefinePrim(path+"/Xavier")
        robot.GetReferences().AddReference(str(HERE/"humanoide_xavier_isaac.usd"), "/Humanoide")
        pelvis = stage.GetPrimAtPath(path+"/Xavier/pelvis")
        pelvis.CreateAttribute("physxArticulation:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.)
        for cuerpo in robot.GetChildren():
            if cuerpo.HasAPI(UsdPhysics.RigidBodyAPI):
                cuerpo.AddAppliedSchema("PhysxContactReportAPI")
                cuerpo.CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float).Set(0.)
        cam = UsdGeom.Camera.Define(stage, path+"/Xavier/cabeza/Camara")
        cam.AddTranslateOp().Set((0., .080, .048))
        cam.AddRotateXOp().Set(73.)
        cam.CreateFocalLengthAttr(6.)
        cam.CreateHorizontalApertureAttr(20.)
        cam.CreateVerticalApertureAttr(20.)
        cam.CreateClippingRangeAttr((.02, 25.))
        if tarea in ("levantarse","mision") and not (tarea=="levantarse" and nivel==2):
            z = silla_z(nivel)
            cubo(stage, path+"/Asiento", (0,-.17,z-.04), (.64,.45,.08), MADERA)
            cubo(stage, path+"/PieAsiento", (0,-.26,(z-.08)/2), (.32,.20,z-.08), MADERA)
            cubo(stage, path+"/Respaldo", (0,-.40,z+.20), (.64,.04,.40), MADERA)
        if tarea in ("caminar", "mision", "transportar"):
            y = .72 if tarea=="caminar" else destino[1]-.72
            x = 0 if tarea=="caminar" else .73
            for j, lado in enumerate((-1,1)):
                cubo(stage,path+f"/Puerta{j}",(x+lado*.55,y,1.),(.10,.12,2.),MADERA)
            if tarea=="caminar":
                cubo(stage,path+"/Silla",(-.12,1.55,.25),(.42,.38,.5),MADERA)
                if entorno=="calle":
                    cubo(stage,path+"/Bordillo",(0,final[1]+.55,.065),(2.,.16,.13),MADERA)
                else:
                    cubo(stage,path+"/MesaFinal",(0,final[1]+.73,.78),(1.1,.62,.08),MADERA)
        if tarea=="pirueta":
            cubo(stage,path+"/Colchoneta",(0,0,.012),(2.,2.,.024),AZUL)
        if tarea=="correr":
            if nivel:
                cubo(stage,path+"/MarcaCurva",(-.60*nivel,1.65,.012),(.50,.12,.006),NARANJA,False)
            for j,y in enumerate((.75,final[1]-.65,final[1])):
                cubo(stage,path+f"/Linea{j}",(final[0]*j/2,y,.012),(1.2,.045,.006),AZUL,False)
        if tarea=="imitar":
            UsdGeom.Xform.Define(stage,path+"/Modelo")
            cubo(stage,path+"/Modelo/Torso",(0,1.65,1.20),(.39,.20,.52),AZUL,False)
            esfera(stage,path+"/Modelo/Cabeza",(0,1.65,1.62),.12,MADERA)
            for j in range(2):
                pierna = UsdGeom.Xform.Define(stage,path+f"/Modelo/Pierna{j}")
                pierna.AddTranslateOp().Set(((j-.5)*.22,1.65,1.))
                pierna.AddRotateXOp().Set(0.)
                cubo(stage,str(pierna.GetPath())+"/Muslo",(0,0,-.25),(.14,.16,.5),AZUL,False)
                rodilla = UsdGeom.Xform.Define(stage,str(pierna.GetPath())+"/Rodilla")
                rodilla.AddTranslateOp().Set((0,0,-.5))
                rodilla.AddRotateXOp().Set(0.)
                cubo(stage,str(rodilla.GetPath())+"/Pantorrilla",(0,0,-.25),(.13,.14,.5),AZUL,False)
                brazo = UsdGeom.Xform.Define(stage,path+f"/Modelo/Brazo{j}")
                brazo.AddTranslateOp().Set(((.5-j)*.49,1.65,1.41))
                brazo.AddRotateXOp().Set(0.)
                cubo(stage,str(brazo.GetPath())+"/Superior",(0,0,-.15),(.085,.085,.30),MADERA,False)
                codo = UsdGeom.Xform.Define(stage,str(brazo.GetPath())+"/Codo")
                codo.AddTranslateOp().Set((0,0,-.30))
                codo.AddRotateXOp().Set(0.)
                cubo(stage,str(codo.GetPath())+"/Inferior",(0,0,-.125),(.08,.08,.25),NARANJA,False)
        objeto = PLANES[tarea].objeto
        if not objeto:
            continue
        rigido = UsdGeom.Xform.Define(stage,path+"/Objeto")
        rigido.AddTranslateOp().Set(Gf.Vec3d(*origen.tolist()))
        UsdPhysics.RigidBodyAPI.Apply(rigido.GetPrim())
        rigido.GetPrim().AddAppliedSchema("PhysxContactReportAPI")
        rigido.GetPrim().CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float).Set(0.)
        masa = {"caja":.25+.12*nivel, "tijeras":.055, "pelota":.42}[objeto]
        UsdPhysics.MassAPI.Apply(rigido.GetPrim()).CreateMassAttr(masa)
        ruta = str(rigido.GetPath())
        if objeto=="caja":
            cubo(stage,ruta+"/Caja",(0,0,0),(.44,.16,.14),NARANJA)
            altura = .07
        elif objeto=="tijeras":
            for j, x in enumerate((-.022,.022)):
                anilla(stage,ruta+f"/Anilla{j}",(x,-.030,0),.020,.009,AZUL,12)
            cubo(stage,ruta+"/HojasCerradas",(0,.047,0),(.018,.11,.009),(.55,.58,.61))
            altura = .0045
        else:
            esfera(stage,ruta+"/Pelota",(0,0,0),.11,NARANJA,True)
            altura = .11
            anilla(stage,path+"/Aro",destino,.235,.02,NARANJA,24)
            cubo(stage,path+"/Tablero",(0,destino[1]+.30,destino[2]+.20),(.85,.04,.55),MADERA)
        cubo(stage,path+"/Fuente",(origen[0],origen[1],origen[2]-altura-.04),(.57,.28,.08),MADERA)
        h = origen[2]-altura-.08
        cubo(stage,path+"/PedestalFuente",(origen[0],origen[1]+.07,h/2),(.18,.12,h),MADERA)
        if objeto!="pelota":
            h = destino[2]-altura-.07
            cubo(stage,path+"/PedestalDestino",(destino[0],destino[1],h/2),(.16,.12,h),AZUL)
            cubo(stage,path+"/Destino",(destino[0],destino[1],destino[2]-altura-.035),
                 ((.65 if objeto=="caja" else .26),(.40 if objeto=="caja" else .22),.07),AZUL)
            if objeto=="tijeras":
                for j,signo in enumerate((-1,1)):
                    cubo(stage,path+f"/BordeX{j}",destino+[signo*.13,0,.012],(.009,.22,.035),AZUL)
                    cubo(stage,path+f"/BordeY{j}",destino+[0,signo*.11,.012],(.26,.009,.035),AZUL)
