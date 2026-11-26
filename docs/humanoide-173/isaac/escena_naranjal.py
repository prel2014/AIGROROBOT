"""Naranjal con Xavier de base libre, árbol frutal y naranja dinámica con soporte liberable."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
ESCENA=HERE/'escenarios/naranjal.usda'
FRUTO=(.250, .75, .87)
RADIO=.029


def crear():
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    stage=Usd.Stage.CreateNew(str(ESCENA))
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,'Z')
    world=UsdGeom.Xform.Define(stage,'/World')
    stage.SetDefaultPrim(world.GetPrim())
    base=Sdf.Layer.FindOrOpen(str(HERE/'banco_xavier.usda'))
    for nombre in ('Fisica','Suelo','Luz','LuzPrincipal'):
        path='/World/'+nombre
        Sdf.CopySpec(base,path,stage.GetRootLayer(),path)
    stage.GetPrimAtPath('/World/Suelo').GetAttribute('xformOp:scale').Set((20.,20.,.07))
    stage.GetPrimAtPath('/World/Suelo').GetAttribute('primvars:displayColor').Set([(0.19,.11,.055)])
    r=UsdGeom.Xform.Define(stage,'/World/Xavier')
    r.GetPrim().GetReferences().AddReference('../humanoide_xavier_isaac.usd')
    r.AddTranslateOp().Set((0.,0.,.003))
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
    for i,(tx,ty) in enumerate([(.95,1.45),(-2.,1.5),(.95,3.9),(-2.,3.9)]):
        arbol=UsdGeom.Xform.Define(stage,'/World/Huerto/Arbol'+str(i))
        arbol.AddTranslateOp().Set((tx,ty,0.))
        child=UsdGeom.Xform.Define(stage,str(arbol.GetPath())+'/Modelo')
        child.GetPrim().GetReferences().AddReference(base_model+'/orange_tree/visual.usda')
        child.GetPrim().SetInstanceable(True)
    stage.GetRootLayer().customLayerData={'cameraSettings':{'Perspective':{'position':Gf.Vec3d(2.2,-1.7,1.6),'target':Gf.Vec3d(0,.6,.95)}}}
    stage.GetRootLayer().Save()


if __name__=='__main__':
    crear()
