"""Naranjal con Xavier de base libre, árbol frutal y naranja dinámica con soporte liberable."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
ESCENA=HERE/'escenarios/naranjal.usda'
FRUTO=(.250, .75, .87)
RADIO=.029


def rama_visible(t):
    """Ramificación visual entre el tronco y el pedúnculo del fruto de ensayo."""
    import math
    import numpy as np
    from geometria_escenarios import material

    p = t.grupo(t.raiz, "RamaFructifera")
    materiales = t.raiz + "/Materiales"
    t.materiales["corteza"] = material(t.stage, materiales, "corteza",
            "../objetos/tree_small_02/textures/tree_small_02_diff_1k.jpg")
    t.materiales["hoja_frutal"] = material(t.stage, materiales, "hoja_frutal",
            "../objetos/orange_tree/textures/hoja_huerto.png")
    centros = np.array([(1.10, 1.33, 1.04), (.90,1.34,1.01), (.76,1.25,.985),
                        (.57,1.09,.98), (.40,.90,.966), (.25,.75,.965)])
    radios = (.035, .029, .022, .015, .009, .0035)
    anillos = []
    for i, (centro, radio) in enumerate(zip(centros, radios)):
        eje = centros[min(i+1, len(centros)-1)] - centros[max(i-1, 0)]
        u = np.cross(eje, [0,0,1]); u /= np.linalg.norm(u)
        v = np.cross(eje, u); v /= np.linalg.norm(v)
        anillos.append([centro + radio * (math.cos(k*math.tau/9)*u + math.sin(k*math.tau/9)*v)
                        for k in range(9)])
    caras, uv = [], []
    for i in range(len(centros)-1):
        for k in range(9):
            h=(k+1)%9
            caras.append([anillos[i][k], anillos[i][h], anillos[i+1][h], anillos[i+1][k]])
            uv += [(k/9,i/5), ((k+1)/9,i/5), ((k+1)/9,(i+1)/5), (k/9,(i+1)/5)]
    t.malla(p, "Corteza", caras, "corteza", uv=uv)
    for i, inicio in enumerate(centros[1:-1]):
        for lado in (-1, 1):
            extremo = inicio + np.array((-.075*lado, .096*lado, .06 + i*.008))
            t.varilla(p,"Ramilla",inicio,extremo,.0027,"oliva")
            for paso in range(3):
                base = inicio + (extremo-inicio)*(.46+.25*paso)
                direccion = np.array((-.072, (.049 if paso%2 else -.067)*lado, .014-.012*paso))
                ancho = np.array((-direccion[1],direccion[0],0))*.25
                punta = base + direccion
                medio = base + direccion*.49
                vena = medio + np.array((0,0,.006))
                t.malla(p,"Hoja",[[base,medio-ancho,vena],[base,vena,medio+ancho],
                        [vena,medio-ancho,punta],[vena,punta,medio+ancho]],"hoja_frutal",
                        uv=[(.5,0),(0,.5),(.5,.5),(.5,0),(.5,.5),(1,.5),
                            (.5,.5),(0,.5),(.5,1),(.5,.5),(.5,1),(1,.5)])


def crear():
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
    from geometria_escenarios import Taller
    from fondos_escenarios import callejones_huerto, frutos_de_fondo
    stage=Usd.Stage.CreateNew(str(ESCENA))
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,'Z')
    world=UsdGeom.Xform.Define(stage,'/World')
    stage.SetDefaultPrim(world.GetPrim())
    base=Sdf.Layer.FindOrOpen(str(HERE/'banco_xavier.usda'))
    for nombre in ('Fisica','Suelo','Luz','LuzPrincipal'):
        path='/World/'+nombre
        Sdf.CopySpec(base,path,stage.GetRootLayer(),path)
    stage.GetPrimAtPath('/World/Suelo').GetAttribute('xformOp:scale').Set((185.,185.,.07))
    stage.GetPrimAtPath('/World/Suelo').GetAttribute('primvars:displayColor').Set([(0.19,.11,.055)])
    r=UsdGeom.Xform.Define(stage,'/World/Xavier')
    r.GetPrim().GetReferences().AddReference('../humanoide_xavier_isaac.usd')
    r.AddTranslateOp().Set((0.,0.,.003))
    camara = UsdGeom.Camera.Define(stage, '/World/Xavier/cabeza/Camara')
    camara.GetPrim().GetReferences().AddReference('../pista_marcha_16.usda', '/XavierMarcha/cabeza/Camara')
    camara.CreateClippingRangeAttr((.02, 220.))
    for lado in ('izq','der'):
        pie=stage.OverridePrim('/World/Xavier/pie_'+lado)
        pie.AddAppliedSchema('PhysxContactReportAPI')
        pie.CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0.)
    stage.GetPrimAtPath('/World/Fisica').GetAttribute('physxScene:timeStepsPerSecond').Set(120)
    material=UsdShade.Material.Define(stage,'/World/Cascara')
    api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(.85)
    api.CreateDynamicFrictionAttr(.65)
    api.CreateRestitutionAttr(0.)
    goma=UsdShade.Material.Define(stage,'/World/Almohadilla')
    contacto=UsdPhysics.MaterialAPI.Apply(goma.GetPrim())
    contacto.CreateStaticFrictionAttr(1.0)
    contacto.CreateDynamicFrictionAttr(.85)
    for prim in Usd.PrimRange(r.GetPrim()):
        if prim.GetName()=='Contacto' and '_der' in prim.GetParent().GetName() and any(
            key in prim.GetParent().GetName() for key in ('distal','proximal','mano')):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(goma,materialPurpose='physics')
    f=UsdGeom.Xform.Define(stage,'/World/Naranja')
    f.AddTranslateOp().Set(FRUTO)
    UsdPhysics.RigidBodyAPI.Apply(f.GetPrim())
    UsdPhysics.MassAPI.Apply(f.GetPrim()).CreateMassAttr(.095)
    f.GetPrim().AddAppliedSchema('PhysxContactReportAPI')
    f.GetPrim().CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0.)
    col=UsdGeom.Sphere.Define(stage,'/World/Naranja/Cascara')
    col.CreateRadiusAttr(RADIO)
    col.CreatePurposeAttr('guide')
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    col.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
    col.GetPrim().CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.001)
    UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(material,materialPurpose='physics')
    base_model='./objetos'
    modelo=UsdGeom.Xform.Define(stage,'/World/Naranja/Visual')
    # Normalizar el radio de la geometría original a la misma escala que su contacto.
    child=UsdGeom.Xform.Define(stage,str(modelo.GetPath())+'/Modelo')
    child.GetPrim().GetReferences().AddReference(base_model+'/orange/visual.usda')
    caja=UsdGeom.BBoxCache(0,['default']).ComputeLocalBound(child.GetPrim()).ComputeAlignedRange()
    factor=2*RADIO/max(caja.GetSize())
    modelo.AddScaleOp().Set((factor,)*3)
    modelo.AddTranslateOp().Set(-caja.GetMidpoint())
    def rama(nombre,a,b,radio):
        a,b=Gf.Vec3d(*a),Gf.Vec3d(*b)
        prim=UsdGeom.Capsule.Define(stage,'/World/ArbolObjetivo/'+nombre)
        prim.CreateHeightAttr((b-a).GetLength())
        prim.CreateRadiusAttr(radio)
        prim.AddTranslateOp().Set((a+b)/2)
        prim.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Rotation(Gf.Vec3d(0,0,1),b-a).GetQuat())
        prim.CreateDisplayColorAttr([(.20,.095,.036)])
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    UsdGeom.Xform.Define(stage,'/World/ArbolObjetivo')
    x,y,z=FRUTO
    rama('Rama',(x,y,z+.095),(.85,1.33,.98),.015)
    rama('Pedunculo',(x,y,z+.030),(x,y,z+.095),.0025)
    stage.GetPrimAtPath('/World/ArbolObjetivo/Pedunculo').RemoveAPI(UsdPhysics.CollisionAPI)
    ancla=UsdGeom.Xform.Define(stage,'/World/ArbolObjetivo/Anclaje')
    ancla.AddTranslateOp().Set((x,y,z+.030))
    UsdPhysics.RigidBodyAPI.Apply(ancla.GetPrim()).CreateKinematicEnabledAttr(True)
    tallo=UsdPhysics.FixedJoint.Define(stage,'/World/ArbolObjetivo/UnionPedunculo')
    tallo.CreateBody0Rel().SetTargets([ancla.GetPath()])
    tallo.CreateBody1Rel().SetTargets([f.GetPath()])
    tallo.CreateLocalPos0Attr((0.,0.,0.))
    tallo.CreateLocalPos1Attr((0.,0.,.030))
    tallo.CreateJointEnabledAttr(True)
    tallo.GetPrim().SetDocumentation("Soporte del ensayo. Se libera al comprobar contacto opuesto sostenido para medir la retención de la mano bajo gravedad.")
    taller = Taller(stage)
    UsdGeom.Imageable(stage.GetPrimAtPath('/World/ArbolObjetivo/Rama')).MakeInvisible()
    rama_visible(taller)
    taller.suelo(taller.raiz, "Tierra", (0, 0), (185, 185), "suelo_seco")
    callejones_huerto(taller)
    cielo = UsdLux.DomeLight(stage.GetPrimAtPath('/World/Luz'))
    cielo.CreateTextureFileAttr(Sdf.AssetPath('./texturas/clarens_midday_4k.hdr'))
    cielo.CreateTextureFormatAttr('latlong')
    cielo.CreateIntensityAttr(470.)
    sol = UsdLux.DistantLight(stage.GetPrimAtPath('/World/LuzPrincipal'))
    sol.CreateIntensityAttr(880.)
    sol.CreateAngleAttr(3.2)
    posiciones = [(.95,1.45),(-2.,1.5),(.95,3.9),(-2.,3.9)]
    posiciones += [(x,y) for x in (-14.,-11.,-8.,-5.,-2.,1.,4.,7.,10.,13.,16.)
                   for y in (7.,10.,13.,16.,19.,31.,36.,41.,46.,51.)]
    for i,(tx,ty) in enumerate(posiciones):
        arbol=UsdGeom.Xform.Define(stage,'/World/Huerto/Arbol'+str(i))
        arbol.AddTranslateOp().Set((tx,ty,0.))
        arbol.AddRotateZOp().Set(float((i * 67) % 360))
        escala = .63 if i < 4 else .64 + .065 * ((i * 11) % 5)
        arbol.AddScaleOp().Set((escala,) * 3)
        child=UsdGeom.Xform.Define(stage,str(arbol.GetPath())+'/Modelo')
        child.GetPrim().GetReferences().AddReference(base_model+'/tree_small_02/visual.usdc')
        child.GetPrim().SetInstanceable(True)
        if ty <= 16. and abs(tx) < 10.:
            frutos_de_fondo(taller, str(arbol.GetPath()), i)
    ojo, centro = Gf.Vec3d(1.9,-1.75,1.48), Gf.Vec3d(.05,.56,.96)
    cam = UsdGeom.Camera.Define(stage, taller.raiz + "/Vista")
    cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(ojo, centro, Gf.Vec3d(0,0,1)).GetInverse())
    cam.CreateHorizontalApertureAttr(32.)
    cam.CreateFocalLengthAttr(25.)
    stage.GetRootLayer().customLayerData={'cameraSettings':{'Perspective':{'position':ojo,'target':centro}}}
    stage.GetRootLayer().Save()


if __name__=='__main__':
    crear()
