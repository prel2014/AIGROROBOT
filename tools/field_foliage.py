"""Author leaf venation maps and blueberry bushes for close-range RGB-D views."""
from math import cos, pi, sin
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from build_grapevine import Surface, bump_image, material, normalized


MODELS = Path(__file__).resolve().parents[1]/"assets/agriculture/models"


def crop_relief(model, materials):
    """Add tangent-space detail to the visual wrapper and retain the source meshes."""
    directory = MODELS/model
    stage = Usd.Stage.Open(str(directory/"visual.usda"))
    for name in materials:
        path = "/Asset/Geometry/Looks/"+name
        base_tex = UsdShade.Shader.Get(stage,path+"/diffuseTex")
        source = Path(base_tex.GetInput("file").Get().resolvedPath)
        filename = source.stem+"_relieve.png"
        size = 768
        luma = Image.open(source).convert("L").resize((size,size),Image.Resampling.LANCZOS)
        relief = (np.asarray(luma,dtype=float)-np.asarray(luma.filter(ImageFilter.GaussianBlur(4))))/40
        if model=="orange":
            pores = Image.new("L",(size,size))
            pen = ImageDraw.Draw(pores)
            rng = np.random.default_rng(1811)
            for x,y in rng.integers(0,size,(8300,2)):
                r = rng.uniform(.8,2.2)
                pen.ellipse((x-r,y-r,x+r,y+r),fill=int(rng.integers(95,240)))
            relief -= .9*np.asarray(pores.filter(ImageFilter.GaussianBlur(.72)))/255
        bump_image(relief,2.4,directory/"textures"/filename)
        tex = UsdShade.Shader.Define(stage,path+"/Relieve")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file",Sdf.ValueTypeNames.Asset).Set("./textures/"+filename)
        tex.CreateInput("sourceColorSpace",Sdf.ValueTypeNames.Token).Set("raw")
        tex.CreateInput("scale",Sdf.ValueTypeNames.Float4).Set((2.,2.,2.,1.))
        tex.CreateInput("bias",Sdf.ValueTypeNames.Float4).Set((-1.,-1.,-1.,0.))
        tex.CreateInput("st",Sdf.ValueTypeNames.Float2).GetAttr().SetConnections(
            base_tex.GetInput("st").GetAttr().GetConnections())
        tex.CreateOutput("rgb",Sdf.ValueTypeNames.Float3)
        surface = UsdShade.Shader.Get(stage,path+"/"+name)
        surface.GetInput("normal").ConnectToSource(tex.ConnectableAPI(),"rgb")
    stage.GetRootLayer().Save()


def leaf_map(path, seed, base):
    path.parent.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(seed)
    size = 256
    marks = Image.new("L", (size, size))
    pen = ImageDraw.Draw(marks)
    pen.line([(128,255),(129,183),(127,96),(128,0)], fill=242, width=3)
    for start in range(33, 255, 23):
        for sign in (-1, 1):
            dy = rng.integers(23, 51)
            arc = [(128+sign*int(120*t), start-int(dy*(1.5*t-.5*t*t))) for t in np.linspace(0,1,14)]
            pen.line(arc, fill=int(rng.integers(96,164)), width=2)
            for x,y in arc[3:-1:3]:
                pen.line([(x,y),(x+sign*17,y-13),(x+sign*25,y-15)], fill=78)
    ribs = np.asarray(marks.filter(ImageFilter.GaussianBlur(.6)))/255
    y,x = np.mgrid[:size,:size]/size
    speck = rng.normal(0,1.8,(size,size)) + 3.2*np.sin(12*x+5*y) + 1.9*np.cos(21*y-5*x)
    shade = 1-.22*np.abs(2*x-1)**2 + .11*y
    rgb = np.array(base)*shade[...,None] + ribs[...,None]*[40,46,21] + speck[...,None]
    Image.fromarray(np.uint8(np.clip(rgb,0,255))).save(path)


def blade():
    surf = Surface()
    for i,t in enumerate(np.linspace(0,1,11)):
        width = .22*(.012+sin(pi*t)**.88)*(1.15-.27*t)
        for j,u in enumerate((-1,-.48,0,.48,1)):
            surf.points.append((width*u,t,.13*sin(pi*t)-.18*t*t-.10*abs(u)*sin(pi*t)))
            surf.normals.append(normalized(np.array([.28*u, -.13*pi*cos(pi*t)+.36*t,1.])))
            surf.uv.append(((u+1)/2,t))
            if i and j:
                k = len(surf.points)-1
                surf.faces.append((k-6,k-5,k,k-1))
    return surf


def berry(stage, path, skin, calyx):
    # Flattened berry with a sunken blossom end and five persistent sepals.
    s = Surface()
    rings = [(.0004,.0077),(.005,.0067),(.009,.0037),(.010,-.0008),
             (.0087,-.0047),(.0053,-.0066),(.0026,-.0063),(.001,-.0054)]
    for i,(radius,z) in enumerate(rings):
        dr,dz = np.subtract(rings[min(i+1,len(rings)-1)],rings[max(i-1,0)])
        for j in range(25):
            a = 2*pi*j/24
            s.points.append((radius*cos(a), radius*sin(a), z))
            s.normals.append(normalized(np.array([-dz*cos(a),-dz*sin(a),dr])))
            s.uv.append((j/24,i/(len(rings)-1)))
            if i and j:
                k = len(s.points)-1
                s.faces.append((k-26,k-1,k,k-25))
    s.write(stage,path+"/Pruina",skin)
    star = Surface()
    for j in range(5):
        a = 2*pi*j/5
        k = len(star.points)
        for r,da,z in ((.0012,0,-.0058),(.003,-.44,-.0066),(.0047,.08,-.0083),(.003,.44,-.0066)):
            star.points.append((r*cos(a+da),r*sin(a+da),z))
            star.uv.append((0.,0.))
        star.faces.extend([(k,k+1,k+2),(k,k+2,k+3)])
    star.write(stage,path+"/Caliz",calyx)


def instances(stage, path, prototypes, items):
    inst = UsdGeom.PointInstancer.Define(stage,path)
    inst.CreatePrototypesRel().SetTargets(prototypes)
    kinds, xyz, orientations, scales = zip(*items)
    inst.CreateProtoIndicesAttr(kinds)
    inst.CreatePositionsAttr(np.asarray(xyz,dtype='f'))
    inst.CreateOrientationsAttr(orientations)
    inst.CreateScalesAttr(np.asarray(scales,dtype='f'))
    t = Usd.TimeCode.Default()
    inst.CreateExtentAttr(inst.ComputeExtentAtTime(t,t))


def build_bush():
    target = MODELS/"arandano"
    target.mkdir(exist_ok=True)
    leaf_map(target/"textures/hoja.png",751,(67,106,58))
    stage = Usd.Stage.CreateNew(str(target/"mata.usdc"))
    root = "/Geometry"
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,root).GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    leaf = material(stage,"Follaje",(.09,.20,.06),.55,"hoja.png")
    wood = material(stage,"Corteza",(.15,.083,.048),.88,"../vineyard/textures/vine_bark.png",
                    "../vineyard/textures/vine_bark_normal.png")
    green = material(stage,"Ramilla",(.19,.23,.070),.68)
    fruit_colors = [(66,79,107),(82,99,126),(137,82,113),(119,151,87)]
    sepals = material(stage,"Caliz",(.048,.056,.062),.79)
    leaf_proto = root+"/Hojas/Tipos/Lamina"
    blade().write(stage,leaf_proto,leaf)
    berries = []
    for i,color in enumerate(fruit_colors):
        p = root+f"/Bayas/Tipos/Madurez{i}"
        UsdGeom.Xform.Define(stage,p)
        rng = np.random.default_rng(751+i)
        noise = rng.normal(0,2.9,(128,256))
        clouds = Image.fromarray(rng.integers(0,256,(8,16),dtype=np.uint8)).resize((256,128),Image.Resampling.BICUBIC)
        bloom = (np.asarray(clouds,dtype=float)/255-.5)*22+noise
        skin = np.array(color)+bloom[...,None]
        filename = f"pruina{i}.png"
        Image.fromarray(np.uint8(np.clip(skin,0,255))).save(target/"textures"/filename)
        berry(stage,p,material(stage,f"Pruina{i}",(.08,.10,.15),.74,filename),sepals)
        berries.append(p)
    stems, shoots = Surface(), Surface()
    leaves, fruits = [], []
    rng = np.random.default_rng(751)
    for n in range(17):
        a = n*2.39996 + rng.uniform(-.30,.30)
        out = np.array([cos(a),sin(a),0.])
        side = np.array([-sin(a),cos(a),0.])
        t = np.linspace(0,1,10)
        cane = ((.045+rng.uniform(.13,.36)*t**1.5)[:,None]*out
                + (rng.uniform(-.06,.06)*np.sin(pi*t))[:,None]*side
                + np.c_[0*t,0*t,rng.uniform(.62,1.07)*t])
        stems.tube(cane,.013-.009*t,8,.09)
        for j in (3,5,7,9):
            base = cane[j]
            flank = (-1)**(n+j//2)
            tip = base + out*rng.uniform(.05,.12) + flank*side*rng.uniform(.065,.18) + [0,0,rng.uniform(.05,.16)]
            shoots.tube([base,(base+tip)/2+[0,0,.012],tip],[.0042,.0029,.0014],6)
            for k in range(9):
                t = (k+.65)/9
                node = base+(tip-base)*t+[0,0,.012*(1-abs(2*t-1))]
                yaw = a+flank*.8+(-1)**k*.95
                forward = normalized(np.array([cos(yaw),sin(yaw),rng.uniform(-.43,.65)]))
                edge = normalized(np.cross(forward,[0,0,1]))
                rotation = Gf.Matrix3d(*(float(v) for v in np.r_[edge,forward,np.cross(edge,forward)]))
                length = rng.uniform(.040,.072)
                leaves.append((0,node,Gf.Quath(rotation.ExtractRotation().GetQuat()),(length,)*3))
            if j==3 or rng.random()<.20:
                continue
            rachis = tip+[0,0,-.029]
            shoots.tube([tip,rachis],[.0016,.0011],5)
            for k in range(6):
                yaw = 2*pi*k/6+.4*n
                r = .019+rng.uniform(0,.008)
                end = rachis+[r*cos(yaw),r*sin(yaw),rng.uniform(-.034,-.008)]
                shoots.tube([rachis,end],[.0011,.0007],5)
                size = rng.uniform(.70,1.08)
                fruits.append((int(rng.choice(4,p=[.5,.3,.13,.07])),end-[0,0,.0077*size],
                               Gf.Quath(1), (size,)*3))
    stems.write(stage,root+"/Canas",wood,collider=True)
    shoots.write(stage,root+"/Brotes",green)
    instances(stage,root+"/Hojas",[leaf_proto],leaves)
    instances(stage,root+"/Bayas",berries,fruits)
    stage.GetRootLayer().Save()
