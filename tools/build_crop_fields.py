"""Generate the four standalone, metre-scale Isaac Sim rover fields."""
import argparse
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from build_grapevine import Surface, build as build_vine, material
from design_harvest_rover import ROOT, attr, bind, friction, granular_patch, schemas
from dexman.tasks.agriculture.field_layout import FIELDS
from field_foliage import build_bush, crop_relief, instances


ASSETS = ROOT/"assets/agriculture"
DIRECTORY = ASSETS/"parcelas"
WORLD = "/Parcela"
CROPS = {
    "citrus": "props/orange_tree.usdc",
    "banana": "models/banana_plant/cultivo.usda",
    "vineyard": "models/vineyard/cepa.usdc",
    "berries": "models/arandano/mata.usdc",
}


def reference(stage, path, asset, xyz, angle=0., scale=(1.,1.,1.), instance=True):
    node = UsdGeom.Xform.Define(stage,path)
    rel = os.path.relpath(asset,DIRECTORY).replace(os.sep,"/")
    node.GetPrim().GetReferences().AddReference(rel)
    node.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    node.AddRotateZOp().Set(float(angle))
    node.AddScaleOp().Set(Gf.Vec3f(*scale))
    node.GetPrim().SetInstanceable(instance)
    return node.GetPrim()


def soil_map(field, bounds, rng):
    """An orthophoto-style albedo, registered to the actual ridges and depressions."""
    x0,x1,y0,y1 = bounds
    width = 2048
    height = round(width*(y1-y0)/(x1-x0))
    x,y = np.meshgrid(np.linspace(x0,x1,width),np.linspace(y1,y0,height))
    lateral = y - field.bend*(.72*np.sin(.23*x-.6)+.28*np.sin(.47*x+.3))
    nearest = np.minimum(abs(lateral), abs(abs(lateral)-field.row_spacing))
    planted = .5*(np.tanh(1.9*x)-np.tanh(1.9*(x-field.length)))
    lane = abs(abs(lateral)-field.row_spacing*.5)
    def noise(nx,ny):
        image = Image.fromarray(rng.integers(0,256,(ny,nx),dtype=np.uint8))
        return np.asarray(image.resize((width,height),Image.Resampling.BICUBIC))/255-.5
    broad = noise(35,21)
    patch = noise(197,103)
    fine = rng.normal(0,.75,(height,width))
    bed = np.exp(-(nearest/(field.bed_width*.60))**4)*planted
    rut = np.exp(-((lane-.51)/.17)**2)*planted
    grass = np.clip(.32 + broad*.93 + patch*.56 + .32*np.exp(-(lane/.23)**2)
                    -.73*bed-.76*rut,0,1)
    dirt = np.array([129,101,72]) if field.name!="arandanal" else np.array([107,83,62])
    color = dirt + grass[...,None]*(np.array([88,105,55])-dirt)
    color += (broad*20 + patch*15 + fine*8 - rut*15 - bed*10)[...,None]
    edge = np.minimum(np.minimum(x-x0,x1-x),np.minimum(y-y0,y1-y))
    field_mix = np.clip(edge/3.,0,1)[...,None]
    meadow = np.array([103,114,75])*(.95+.05*np.sin(.41*x+.23*y)+.045*np.sin(.19*y-.32*x))[...,None]
    color = field_mix*color+(1-field_mix)*meadow
    path = DIRECTORY/"textures"/(field.name+".jpg")
    Image.fromarray(np.uint8(np.clip(color,0,255))).save(path,quality=90)
    return path.name


def terrain(stage, field, rng):
    bounds = (-field.headland-4,field.length+field.headland+4,-field.fence_y-4,field.fence_y+4)
    x0,x1,y0,y1 = bounds
    xs = np.linspace(x0,x1,math.ceil((x1-x0)/.12)+1)
    ys = np.linspace(y0,y1,math.ceil((y1-y0)/.105)+1)
    xy = np.array([(x,y) for x in xs for y in ys])
    points = np.c_[xy,[field.height(x,y) for x,y in xy]]
    ny = len(ys)
    idx = np.arange(len(points)).reshape(len(xs),ny)[:-1,:-1].ravel()
    indices = np.stack((idx,idx+ny,idx+ny+1,idx+1),axis=1)
    mesh = UsdGeom.Mesh.Define(stage,WORLD+"/Suelo")
    mesh.CreatePointsAttr(points.astype("f"))
    mesh.CreateFaceVertexCountsAttr(np.full(len(idx),4,np.int32))
    mesh.CreateFaceVertexIndicesAttr(indices.ravel().astype("i"))
    mesh.CreateSubdivisionSchemeAttr("none")
    uv = (xy-[x0,y0])/[x1-x0,y1-y0]
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,"vertex").Set(uv.astype("f"))
    skin = material(stage,"Suelo",(.18,.12,.07),.96,soil_map(field,bounds,rng),root=WORLD)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(skin)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    bind(mesh.GetPrim(),friction(stage,WORLD+"/Materials/TierraFirme",.75,.58))
    # Continue from the four field edges to a rolling horizon outside the fence.
    boundary = np.concatenate([np.c_[xs,np.full(len(xs),y0)],np.c_[np.full(len(ys)-1,x1),ys[1:]],
                               np.c_[xs[-2::-1],np.full(len(xs)-1,y1)],np.c_[np.full(len(ys)-2,x0),ys[-2:0:-1]]])
    boundary = boundary[::3]
    c = np.array([(x0+x1)/2,0.])
    direction = (boundary-c)/np.linalg.norm(boundary-c,axis=1)[:,None]
    base = np.array([field.height(x,y) for x,y in boundary])-.030
    rings,colors = [],[]
    for radius in (0,1.5,3.5,6,10,17,25,36,48,63,80,105,140,180):
        xy = boundary+radius*direction
        x,y = xy.T
        h = (2.4*np.sin(.078*x+.031*y)+1.2*np.cos(.096*y-.075*x)+2.0*np.sin(.056*y+.034*x))
        z = base + min(radius/34,1)*h + .014*radius
        rings.append(np.c_[xy,z])
        mottling = .85+.10*np.sin(.41*x+.23*y)+.09*np.sin(.19*y-.32*x)
        colors.append(mottling[:,None]*[.135,.167,.071])
    outer = UsdGeom.Mesh.Define(stage,WORLD+"/Linderos/Lomas")
    outer.CreatePointsAttr(np.vstack(rings).astype("f"))
    n = len(base)
    a = np.arange((len(rings)-1)*n).reshape(-1,n)
    b = np.roll(a,-1,axis=1)
    outer.CreateFaceVertexCountsAttr(np.full(a.size,4,np.int32))
    outer.CreateFaceVertexIndicesAttr(np.stack((a,a+n,b+n,b),axis=-1).ravel().astype("i"))
    outer.CreateSubdivisionSchemeAttr("none")
    ring_grid = np.array(rings)
    radial = np.gradient(ring_grid,axis=0)
    tangent = np.roll(ring_grid,-1,axis=1)-np.roll(ring_grid,1,axis=1)
    normals = np.cross(radial,tangent)
    normals /= np.maximum(1.e-9,np.linalg.norm(normals,axis=2,keepdims=True))
    outer.CreateNormalsAttr(normals.reshape(-1,3).astype("f"))
    outer.SetNormalsInterpolation("vertex")
    outer.CreateDisplayColorPrimvar("vertex").Set(np.vstack(colors).astype("f"))
    # Shelterbelt beyond the far fence. Anchor roots to the near landscape rings.
    for i in range(32):
        idx = np.argmin(np.linalg.norm(boundary-[x1,rng.uniform(y0,y1)],axis=1))
        p = rings[2+i%3][idx]
        size = rng.uniform(1.6,3.3)
        reference(stage,WORLD+f"/Linderos/Seto/Arbusto{i:02}",ASSETS/CROPS["berries"],
                  p,rng.uniform(0,360),(1.22*size,1.18*size,size))


def sky_texture():
    # Lat-long sky for the sensor background and skylight. Keep it seamless in azimuth.
    rng = np.random.default_rng(211)
    w,h = 1024,512
    y,x = np.mgrid[:h,:w]
    altitude = np.sin((.5-y/(h-1))*np.pi)
    t = np.maximum(altitude,0)**.55
    color = np.array([205,222,226]) + t[...,None]*np.array([-116,-70,-23])
    def cloud(scale):
        tile = rng.integers(0,256,(scale,scale*2),dtype=np.uint8)
        img = Image.fromarray(np.tile(tile,(1,3))).resize((3*w,h),Image.Resampling.BICUBIC)
        return np.asarray(img)[:,w:2*w]/255
    veil = np.clip((.64*cloud(8)+.25*cloud(22)+.11*cloud(60)-.48)*3.,0,1)
    veil *= np.clip(altitude*5,0,1)*.64
    color = color*(1-veil[...,None])+veil[...,None]*np.array([242,244,239])
    Image.fromarray(np.uint8(color)).save(DIRECTORY/"textures/cielo.jpg",quality=94)


def fence(stage, field, rng):
    wood, wire, nails = Surface(), Surface(), Surface()
    def span(start, end, spacing=2.65):
        a,b = np.array(start),np.array(end)
        segs = math.ceil(np.linalg.norm(b-a)/spacing)
        posts = []
        for i in range(segs+1):
            xy = a+(b-a)*(i/segs)
            z = field.height(*xy)
            foot = np.r_[xy,z-.12]
            tip = np.r_[xy+rng.uniform(-.021,.021,2),z+rng.uniform(1.05,1.19)]
            wood.tube([foot,foot*.5+tip*.5,tip],[.055,.048,.042],7,.065)
            posts.append(foot)
        for i,(p,q) in enumerate(zip(posts,posts[1:])):
            for h in (.31,.62,.99):
                v = np.linspace(0,1,6)
                line = p+(q-p)*v[:,None]
                line[:,2] += h-.037*np.sin(np.pi*v)
                wire.tube(line,.004,5)
                # Staple straddles the wire at the post face.
                v0 = p+[.045,0,h]
                nails.tube([v0+[0,-.012,.009],v0+[.005,0,0],v0+[0,.012,-.009]],.004,5)
            if i%4==0 and len(posts)>2:
                wood.tube([p+[0,0,.16],q+[0,0,.94]],[.025,.023],6,.055)
    near,far = -field.headland,field.length+field.headland
    edge = field.fence_y
    for sign in (-1,1):
        span((near,sign*edge),(far,sign*edge))
    span((far,-edge),(far,edge))
    # An open entrance for each inter-row lane at the near headland.
    half_gate = min(1.36,field.row_spacing*.39)
    gates = [field.row_y(near,lane) for lane in (-.5,.5)]
    for y0,y1 in zip([-edge]+[y+half_gate for y in gates],[y-half_gate for y in gates]+[edge]):
        span((near,y0),(near,y1))
    wood.write(stage,WORLD+"/Cerca/Postes",material(stage,"Postes",(.26,.18,.11),.85,
               "../models/vineyard/textures/wood.png",root=WORLD),collider=True)
    steel = material(stage,"Alambre",(.32,.33,.29),.57,root=WORLD)
    wire.write(stage,WORLD+"/Cerca/Alambres",steel,collider=True)
    nails.write(stage,WORLD+"/Cerca/Grapas",steel)


def trellis(stage,field):
    posts, wires = Surface(), Surface()
    for row in (-1,0,1):
        anchors = []
        for x in np.linspace(.16,field.length-.16,8):
            xy = (x,field.row_y(x,row))
            base = np.array([*xy,field.height(*xy)])
            posts.tube([base-[0,0,.12],base+[.012,0,1.83]],[.048,.038],7,.055)
            anchors.append(base)
        for p,q in zip(anchors,anchors[1:]):
            for z,yoff in ((.84,0),(1.17,0),(1.46,.07),(1.46,-.07),(1.74,.07),(1.74,-.07)):
                t = np.linspace(0,1,9)
                wire = p+(q-p)*t[:,None]+[0,yoff,z]
                wire[:,2] -= .026*np.sin(np.pi*t)
                wires.tube(wire,.003,5)
    posts.write(stage,WORLD+"/Espaldera/Postes",UsdShade.Material.Get(stage,WORLD+"/Looks/Postes"),collider=True)
    wires.write(stage,WORLD+"/Espaldera/Alambres",UsdShade.Material.Get(stage,WORLD+"/Looks/Alambre"),collider=True)


def understory(stage,field,rng):
    # Model low vegetation at bed edges, with sparse grass in the middle of the lanes.
    root = WORLD+"/Cubresuelo"
    types, items = [], []
    for i,shade in enumerate(((.11,.15,.034),(.18,.20,.055),(.22,.165,.073))):
        blades = Surface()
        for _ in range(14):
            a = rng.uniform(0,2*math.pi)
            d = np.array([math.cos(a),math.sin(a),0])
            w = np.array([-d[1],d[0],0])*rng.uniform(.003,.007)
            base = d*rng.uniform(0,.03)
            length = rng.uniform(.04,.14)
            mid = base+d*.022+[0,0,length*.68]
            top = base+d*rng.uniform(.035,.075)+[0,0,length]
            k = len(blades.points)
            blades.points.extend([base-w,base+w,mid-w*.62,mid+w*.62,top])
            blades.faces.extend([(k,k+1,k+3,k+2),(k+2,k+3,k+4)])
            blades.uv.extend([(0,0),(1,0),(0,.6),(1,.6),(.5,1)])
        p = root+f"/Tipos/Hierba{i}"
        blades.write(stage,p,material(stage,f"Pasto{i}",shade,.88,root=WORLD))
        types.append(p)
    for _ in range(4600):
        x = rng.uniform(-field.headland-2,field.length+field.headland+2)
        y = rng.uniform(-field.fence_y-2,field.fence_y+2)
        bed,track = field.bed_and_tracks(x,y)
        if track>.2 or (bed>.65 and rng.random()<.82):
            continue
        size = rng.uniform(.48,1.25)
        q = Gf.Rotation(Gf.Vec3d.ZAxis(),rng.uniform(0,360)).GetQuat()
        items.append((int(rng.choice(3,p=[.54,.35,.11])),(x,y,field.height(x,y)-.006),
                      Gf.Quath(q),(size,)*3))
    instances(stage,root,types,items)


def cameras(stage,field,focus):
    def cam(name,eye,target,focal):
        lens = UsdGeom.Camera.Define(stage,WORLD+"/Vistas/"+name)
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d.ZAxis())
        lens.AddTransformOp().Set(view.GetInverse())
        lens.CreateHorizontalApertureAttr(24.)
        lens.CreateFocalLengthAttr(focal)
        lens.CreateClippingRangeAttr((.02,220.))
    eye = (-14.,-field.fence_y-14,19.5)
    center = (10.,0.,.9)
    cam("General",eye,center,16.)
    x = -.65
    cam("Calle",(x,field.row_y(x,-.5),field.height(x,field.row_y(x,-.5))+1.52),
        (8.,field.row_y(8.,-.5),field.height(8.,field.row_y(8.,-.5))+.96),17.)
    dist = dict(naranjal=.96,platanal=1.47,vinedo=1.10,arandanal=.39)[field.name]
    cam("Fruto",focus+np.array([-.35,-1.,.24])*dist,focus,27.)
    stage.GetRootLayer().customLayerData = {"cameraSettings":{
        "Perspective":{"position":Gf.Vec3d(*eye),"target":Gf.Vec3d(*center)},
        "boundCamera":WORLD+"/Vistas/General"}}


def fruit_focus(stage, key, plant):
    path = str(plant.GetPath())
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),["default","render"])
    def center(p):
        bbox = cache.ComputeWorldBound(p).ComputeAlignedBox()
        if bbox.IsEmpty():
            raise ValueError(f"Empty fruit geometry: {p}")
        return np.array(bbox.GetMidpoint())
    if key=="berries":
        inst = UsdGeom.PointInstancer.Get(stage,path+"/Bayas")
        pose = UsdGeom.XformCache().GetLocalToWorldTransform(inst.GetPrim())
        pts = np.array([pose.Transform(Gf.Vec3d(p)) for p in inst.GetPositionsAttr().Get()])
    elif key=="banana":
        return center(stage.GetPrimAtPath(path+"/Visual/Geometry/Bananas/Bananas"))
    else:
        folder = path+("/Fruit" if key=="citrus" else "/Cosecha")
        pts = np.array([center(p) for p in stage.GetPrimAtPath(folder).GetFilteredChildren(Usd.TraverseInstanceProxies())])
    return pts[pts[:,1].argmin()]


def build(key):
    field = FIELDS[key]
    rng = np.random.default_rng(field.seed)
    (DIRECTORY/"textures").mkdir(parents=True,exist_ok=True)
    output = DIRECTORY/(field.name+".usdc")
    stage = Usd.Stage.CreateNew(str(output))
    root = UsdGeom.Xform.Define(stage,WORLD).GetPrim()
    stage.SetDefaultPrim(root)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(36000)
    stage.SetTimeCodesPerSecond(60)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    root.SetCustomDataByKey("crop",key)
    root.SetCustomDataByKey("fieldSeed",field.seed)
    navigation = UsdGeom.Scope.Define(stage,WORLD+"/Navegacion").GetPrim()
    navigation.CreateAttribute("ruta:centroNominal",Sdf.ValueTypeNames.Point3fArray).Set(field.reference_route())
    phys = UsdPhysics.Scene.Define(stage,WORLD+"/Physics")
    phys.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
    phys.CreateGravityMagnitudeAttr(9.81)
    schemas(phys.GetPrim(),"PhysxSceneAPI")
    attr(phys.GetPrim(),"physxScene:gpuFoundLostAggregatePairsCapacity",Sdf.ValueTypeNames.UInt,16384)
    terrain(stage,field,rng)
    fence(stage,field,rng)
    understory(stage,field,rng)
    if key=="vineyard":
        trellis(stage,field)
    if key in ("banana","berries"):
        attr(phys.GetPrim(),"physxScene:enableGPUDynamics",Sdf.ValueTypeNames.Bool,True)
        granular_patch(stage,WORLD,field.height,9.,field.row_y(9.6,-.5)-.75,field.seed,irregular=True)
    focus = None
    for row in (-1,0,1):
        for i in range(field.plants_per_row):
            x = field.plant_x(i)
            y = field.row_y(x,row)
            yaw = field.row_yaw(x)
            size = rng.uniform(.91,1.06)
            scale = (size,)*3
            if key=="citrus":
                yaw += rng.uniform(-180,180)
                scale = (1.50*size,1.50*size,1.22*size)
            elif key=="banana":
                yaw += (-106 if i%2 or row==0 else 64) + rng.uniform(-18,18)
            elif key=="berries":
                yaw += rng.uniform(-180,180)
            else:
                yaw += 180*(i%2)
            prim = reference(stage,WORLD+f"/Hileras/Fila{row+2}/Planta{i+1:02}",ASSETS/CROPS[key],
                             (x,y,field.height(x,y)-.012),yaw,scale)
            if row==0 and i==1:
                focus = fruit_focus(stage,key,prim)
    # Objects intrude into the lane, away from the spawn and the headlands.
    x = 12.75
    y = field.row_y(x,-.5)-.42
    path = WORLD+"/Obstaculos"
    obstacles = {"citrus":("Roca","field_rock",4.4),"banana":("Balde","harvest_bucket",1.35),
                 "berries":("Rama","branch",2.8)}
    if key in obstacles:
        name,source,size = obstacles[key]
        reference(stage,path+"/"+name,ASSETS/"props"/(source+".usda"),(x,y,field.height(x,y)),24.,(size,)*3)
    else:
        log = Surface()
        ends = [(x-.72,y-.36),(x+.72,y+.36)]
        log.tube([(a,b,field.height(a,b)+.065) for a,b in ends],[.069,.052],8,.12)
        log.write(stage,path+"/PosteCaido",UsdShade.Material.Get(stage,WORLD+"/Looks/Postes"),collider=True)
    start = -2.9
    y = field.row_y(start,-.5)
    reference(stage,WORLD+"/Rover",ASSETS/"robot/rover/rover.usdc",
              (start,y,field.height(start,y)+.07),field.row_yaw(start),instance=False)
    sun = UsdLux.DistantLight.Define(stage,WORLD+"/Sol")
    sun.CreateIntensityAttr(2700.)
    sun.CreateAngleAttr(.85)
    sun.AddRotateXYZOp().Set((-27.,-38.,16.))
    sky = UsdLux.DomeLight.Define(stage,WORLD+"/Cielo")
    sky.CreateIntensityAttr(490.)
    sky.CreateColorAttr((.83,.90,1.))
    sky.CreateTextureFileAttr("./textures/cielo.jpg")
    sky.CreateTextureFormatAttr("latlong")
    cameras(stage,field,focus)
    stage.GetRootLayer().Save()
    print(f"{output.name}: {3*field.plants_per_row} plants; seed {field.seed}",flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop",choices=tuple(FIELDS))
    args = parser.parse_args()
    (DIRECTORY/"textures").mkdir(parents=True,exist_ok=True)
    sky_texture()
    keys = [args.crop] if args.crop else list(FIELDS)
    if "citrus" in keys:
        crop_relief("orange",["Original_orange_citrus_peel"])
    if "banana" in keys:
        crop_relief("banana_plant",["Green_bananas","Banana_leaf__5_mm_blade_",
                                    "Pseudostem_and_petioles__mottled_leaf_sheaths"])
    if "berries" in keys or not (ASSETS/CROPS["berries"]).exists():
        build_bush()
    if "vineyard" in keys:
        build_vine(single_vine=True)
    for key in keys:
        build(key)
