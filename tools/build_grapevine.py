"""Build a four-metre fruiting Vitis trellis, in metres with Z up."""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


MODEL = Path(__file__).resolve().parents[1] / "assets/agriculture/models/vineyard"
SEED = 641307
HARVEST = np.array((.080, -.275, 1.100))
ROOT = "/Geometry"
SIDE = np.array([(0, .04), (.12, -.12), (.34, -.19), (.39, -.12),
                 (.46, -.07), (.48, .06), (.37, .15), (.52, .145),
                 (.65, .27), (.70, .44), (.64, .53), (.47, .57),
                 (.35, .51), (.43, .71), (.35, .83), (.21, .84),
                 (.16, 1.00), (.065, 1.09), (0., 1.19)])


def normalized(v):
    return v / max(np.linalg.norm(v), 1e-9)


def noise(rng, width, height, size):
    values = rng.integers(0, 256, (height, width), dtype=np.uint8)
    # The maps wrap around the circumference and repeat along each cane.
    values = np.tile(values, (3, 3))
    tile = Image.fromarray(values).resize((size*3, size*3), Image.Resampling.BICUBIC)
    return np.asarray(tile.crop((size,size,size*2,size*2)))/255. - .5


def color_image(array, path):
    Image.fromarray(np.uint8(np.clip(array, 0, 255))).save(path, optimize=True)


def bump_image(height, strength, path):
    dy, dx = np.gradient(height)
    normal = np.stack((-dx*strength, -dy*strength, np.ones_like(height)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    color_image((normal + 1)*127.5, path)


def leaf_uv(x, y):
    return (.5+x/1.52, .04+(y+.20)/1.49)


def textures(rng):
    directory = MODEL / "textures"
    directory.mkdir(exist_ok=True)
    size = 512
    y, x = np.mgrid[:size, :size] / size
    warp = noise(rng, 13, 4, size)
    grooves = (.5+.5*np.cos(2*np.pi*(18*x + .82*warp + .14*np.sin(2*np.pi*(y+x)))))**8
    bark = noise(rng, 82, 12, size)*.71 + noise(rng, 219, 43, size)*.45 + noise(rng, 80, 80, size)*.22 - grooves*.43
    color_image(np.array([113, 93, 74])+bark[:,:,None]*[52,50,46], directory/"vine_bark.png")
    bump_image(bark, 2.4, directory/"vine_bark_normal.png")
    grain=noise(rng,94,4,size)*.68+noise(rng,221,12,size)*.34+noise(rng,34,35,size)*.21
    color_image(np.array([136, 127, 110])+grain[:,:,None]*[58,55,52], directory/"wood.png")
    veins = Image.new("L", (size, size))
    draw = ImageDraw.Draw(veins)
    def uvpt(p):
        u,v = leaf_uv(*p)
        return (int(u*(size-1)), int((1-v)*(size-1)))
    origin = np.array([0., .055])
    for target in [(0,1.19), (.39,.78), (.67,.42), (.46,.0), (.32,-.15),
                   (-.39,.78),(-.67,.42),(-.46,.0),(-.32,-.15)]:
        target=np.array(target)
        curve=[origin*(1-t)+target*t+np.array([.018*math.sin(3*t),.015*math.sin(t*4)]) for t in np.linspace(0,1,18)]
        for i in range(17):
            draw.line([uvpt(curve[i]),uvpt(curve[i+1])], fill=220-i*4, width=max(1, 5-i//4))
        d=target-origin
        perp=normalized(np.array([-d[1],d[0]]))
        for k in (4,7,10,13):
            for side in (-1,1):
                start=curve[k]
                end=start+.13*d+perp*side*(.07+(1-k/18)*.043)
                mid=start*.45+end*.55-.018*d
                draw.line([uvpt(start),uvpt(mid),uvpt(end)],fill=115,width=2)
                draw.line([uvpt(mid),uvpt(mid+.05*d+side*.035*perp)],fill=88,width=1)
    relief=np.asarray(veins.filter(ImageFilter.GaussianBlur(.65)))/255.
    flecks=noise(rng,36,36,size)*.62+noise(rng,140,140,size)*.3+noise(rng,11,11,size)*.34
    for name, shade in [("leaf_forest",[64,104,46]),("leaf_sun",[87,129,47]),("leaf_tip",[112,149,60])]:
        data=np.array(shade)+flecks[:,:,None]*[35,36,18]+relief[:,:,None]*[40,37,18]
        color_image(data,directory/(name+".png"))
    bump_image(relief + flecks*.1,1.65,directory/"leaf_normal.png")


def material(stage, name, color, roughness, texture=None, normal=None):
    path=ROOT+"/Looks/"+name
    mat=UsdShade.Material.Define(stage,path)
    surface=UsdShade.Shader.Define(stage,path+"/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    surface.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    if texture:
        reader=UsdShade.Shader.Define(stage,path+"/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname",Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result",Sdf.ValueTypeNames.Float2)
        for channel,filename in [("diffuseColor",texture),("normal",normal)]:
            if not filename:
                continue
            tex=UsdShade.Shader.Define(stage,path+"/"+channel)
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file",Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./textures/"+filename))
            tex.CreateInput("sourceColorSpace",Sdf.ValueTypeNames.Token).Set("raw" if channel=="normal" else "sRGB")
            tex.CreateInput("st",Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),"result")
            tex.CreateInput("wrapS",Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT",Sdf.ValueTypeNames.Token).Set("repeat")
            if channel=="normal":
                tex.CreateInput("scale",Sdf.ValueTypeNames.Float4).Set((2.,2.,2.,1.))
                tex.CreateInput("bias",Sdf.ValueTypeNames.Float4).Set((-1.,-1.,-1.,0.))
                surface.CreateInput(channel,Sdf.ValueTypeNames.Normal3f)
            tex.CreateOutput("rgb",Sdf.ValueTypeNames.Float3)
            surface.GetInput(channel).ConnectToSource(tex.ConnectableAPI(),"rgb")
    return mat


class Surface:
    def __init__(self):
        self.points, self.faces, self.uv, self.normals = [], [], [], []

    def tube(self, path, radii, steps=8, ridges=0):
        path=np.asarray(path)
        if np.isscalar(radii):
            radii=np.full(len(path),radii)
        base=len(self.points)
        dist=0.
        for i,(p,r) in enumerate(zip(path,radii,strict=True)):
            tangent=normalized(path[min(i+1,len(path)-1)]-path[max(i-1,0)])
            if i==0:
                seed=np.eye(3)[np.argmin(abs(tangent))]
                normal=normalized(np.cross(tangent,seed))
            else:
                normal=normalized(normal-np.dot(tangent,normal)*tangent)
                dist+=np.linalg.norm(p-path[i-1])
            b=np.cross(tangent,normal)
            for j in range(steps+1):
                t=2*math.pi*j/steps
                radial=normal*math.cos(t)+b*math.sin(t)
                diameter=r*(1+ridges*(.6*math.cos(5*t+.9*i)+.4*math.cos(3*t-.3*i)))
                self.points.append(p+radial*diameter)
                self.normals.append(radial)
                self.uv.append((j/steps,dist/.32))
                if i and j:
                    k=base+i*(steps+1)+j
                    self.faces.append((k-steps-2,k-steps-1,k,k-1))
        self.faces.append(tuple(base+j for j in reversed(range(steps))))
        self.faces.append(tuple(base+(len(path)-1)*(steps+1)+j for j in range(steps)))

    def write(self,stage,path,mat,collider=False):
        mesh=UsdGeom.Mesh.Define(stage,path)
        mesh.CreatePointsAttr(np.asarray(self.points,dtype=np.float32))
        mesh.CreateFaceVertexCountsAttr([len(f) for f in self.faces])
        mesh.CreateFaceVertexIndicesAttr([i for f in self.faces for i in f])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        if self.normals:
            mesh.CreateNormalsAttr(np.asarray(self.normals,dtype=np.float32))
            mesh.SetNormalsInterpolation("vertex")
        attr=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,"vertex")
        attr.Set(np.asarray(self.uv,dtype=np.float32))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        if collider:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        return mesh


def leaf_model(rng):
    """A palmate blade with a petiolar sinus, five lobes and serrated edge."""
    leaf=Surface()
    boundary=np.vstack((SIDE,SIDE[-2:0:-1]*[-1,1]))
    edge=[]
    for i,a in enumerate(boundary):
        b=boundary[(i+1)%len(boundary)]
        d=b-a
        outward=normalized(np.array([d[1],-d[0]]))
        n=max(1,round(np.linalg.norm(d)/.064))
        for j in range(n):
            edge.append(a+d*j/n)
            edge.append(a+d*(j+.67)/n+outward*.020*rng.uniform(.75,1.2))
    center=np.array([0.,.46])
    leaf.points.append((0,.46,0))
    leaf.uv.append(leaf_uv(*center))
    phase=rng.uniform(0,6.28)
    for ring,t in enumerate((.34,.67,1.)):
        for j,p in enumerate(edge):
            u,v=center*(1-t)+p*t
            z=.11*math.sin(v*3.2)+.15*u*u-.11*v*v+.05*t*t*math.sin(u*12+2*v+phase)
            leaf.points.append((u,v,z))
            leaf.uv.append(leaf_uv(u,v))
            k=1+ring*len(edge)+j
            nxt=1+ring*len(edge)+(j+1)%len(edge)
            if ring==0:
                leaf.faces.append((0,k,nxt))
            else:
                leaf.faces.append((k-len(edge),k,nxt,nxt-len(edge)))
    normals=np.zeros((len(leaf.points),3))
    points=np.array(leaf.points)
    for face in leaf.faces:
        p=points[list(face)]
        normals[list(face)]+=np.cross(p[1]-p[0],p[-1]-p[0])
    leaf.normals=[normalized(n) for n in normals]
    return leaf


def bunch(stage,path,tip,scale,yaw):
    anchor=UsdGeom.Xform.Define(stage,path)
    anchor.AddTranslateOp().Set(Gf.Vec3d(*tip))
    if yaw:
        anchor.AddRotateZOp().Set(yaw)
    if scale!=1.:
        anchor.AddScaleOp().Set(Gf.Vec3f(*([scale]*3)))
    fruit=UsdGeom.Xform.Define(stage,path+"/Racimo")
    fruit.GetPrim().GetReferences().AddReference("../grapes/bunch.usdc")
    fruit.GetPrim().SetInstanceable(True)
    stem_tip = fruit.GetPrim().GetAttribute("grape:stemTip").Get()
    if stem_tip is None:
        raise ValueError("El racimo debe declarar el punto de unión grape:stemTip")
    fruit.AddTranslateOp().Set(-Gf.Vec3d(stem_tip))


def build():
    rng=np.random.default_rng(SEED)
    textures(rng)
    stage=Usd.Stage.CreateNew(str(MODEL/"geometry.usdc"))
    root=UsdGeom.Xform.Define(stage,ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    bark=material(stage,"OldWood",(.19,.11,.07),.92,"vine_bark.png","vine_bark_normal.png")
    timber=material(stage,"TrellisPosts",(.24,.17,.12),.82,"wood.png")
    steel=material(stage,"Wire",(.19,.18,.14),.66)
    fresh=material(stage,"Canes",(.18,.22,.052),.78)
    young=material(stage,"Tendrils",(.27,.31,.077),.82)
    leaf_mats=[material(stage,n,(.10,.24,.05),.73,n+".png","leaf_normal.png")
               for n in ("leaf_forest","leaf_sun","leaf_tip")]
    leaves=UsdGeom.PointInstancer.Define(stage,ROOT+"/Canopy")
    prototypes=[]
    for i in range(6):
        p=str(leaves.GetPath())+"/Prototypes/Blade"+str(i)
        leaf_model(rng).write(stage,p,leaf_mats[i%3])
        prototypes.append(p)
    leaves.CreatePrototypesRel().SetTargets(prototypes)
    indices,scales,positions,orientations=[],[],[],[]
    wood,canes,petioles,tendrils,posts,wires=(Surface() for _ in range(6))

    def add_leaf(node,a,length,roll,slope,tender=False):
        direction=np.array([math.cos(a),math.sin(a),rng.uniform(-.1,.47)])
        attachment=node+direction*rng.uniform(.042,.079)
        if np.linalg.norm(attachment-HARVEST)<.15 and attachment[1]<HARVEST[1]+.04:
            return
        petioles.tube([node,(node+attachment)/2+[0,0,.006],attachment], [.0026,.0021,.0015],6)
        forward=normalized(np.array([math.cos(a),math.sin(a),slope]))
        across=normalized(np.cross(forward,[0,0,1]))
        across=across*math.cos(roll)+np.cross(forward,across)*math.sin(roll)
        normal=np.cross(across,forward)
        rotation=Gf.Matrix3d(*(float(v) for v in np.r_[across,forward,normal]))
        orientations.append(Gf.Quath(rotation.ExtractRotation().GetQuat()))
        positions.append(attachment)
        scales.append((length*rng.uniform(.87,1.12),length,length))
        indices.append(int(rng.choice((2,5) if tender else (0,1,3,4))))

    for i,x in enumerate((-1.99,.45,2.015)):
        posts.tube([(x,0,-.017),(x+.014,0,1.88 if i!=1 else 1.84)], [.045,.036],6,.04)
    for z in (.85,1.17,1.47,1.76):
        for y in ((-.072,.072) if z>1.2 else (0.,)):
            t=np.linspace(0,1,24)
            wire=np.c_[-1.99+4.01*t,np.full(len(t),y),z-.016*np.sin(t*math.pi*2)**2]
            wires.tube(wire,.0027,6)
    n_bunch=0
    for vi,center in enumerate((-1.34,.01,1.34)):
        phase=rng.uniform(-math.pi,math.pi)
        z=np.linspace(-.035,.98+rng.uniform(-.045,.055),20)
        trunk=np.c_[center+.053*np.sin(z*5+phase)+.023*np.sin(z*11),
                    .034*np.sin(z*6+phase),z]
        wood.tube(trunk,.054-.023*z+ .017*np.exp(-14*np.maximum(z,0)),18,.11)
        for side in (-1,1):
            t=np.linspace(0,1,16)
            origin=trunk[-1]
            cordon=origin+np.c_[side*(.64*t),.036*np.sin(2*np.pi*t),
                                .047*(np.sin(t*6+phase)-math.sin(phase))+rng.uniform(-.12,.035)*t]
            wood.tube(cordon,.033-.020*t,12,.10)
            for idx in range(4):
                base=cordon[2+idx*4]
                spur=base+np.array([rng.uniform(-.055,.055), rng.uniform(-.04,.04),rng.uniform(.04,.09)])
                u=np.linspace(0,1,6)
                spur_path=base+u[:,None]*(spur-base)+np.sin(math.pi*u)[:,None]*[.013,.010,0]
                wood.tube(spur_path,.020-.011*u,10,.07)
                t=np.linspace(0,1,12)
                h=rng.uniform(.62,.91)
                lean=rng.uniform(-.16,.16)
                bend=rng.uniform(-.19,.19)
                undulation=rng.uniform(0,2*np.pi)
                shoot=spur+np.c_[lean*t+.050*(np.sin(5*t+undulation)-math.sin(undulation)),
                                  bend*t+.045*np.sin(5*t),h*t]
                canes.tube(shoot,.007-.0048*t,7)
                if idx in (0,2,3):
                    node=shoot[int(rng.choice((1,2,3)))]
                    tip=node+np.array([rng.uniform(-.069,.069),rng.choice([-1,1])*rng.uniform(.09,.18),rng.uniform(-.12,-.044)])
                    main=vi==1 and side==1 and idx==0
                    if main: tip=HARVEST
                    route=np.array([node,node+(tip-node)*.43+[.008,0,.022],tip])
                    petioles.tube(route,[.0035,.003,.0017],7)
                    name="PedunculoObjetivo" if main else f"RacimoLateral{n_bunch:02d}"
                    bunch(stage,ROOT+"/Cosecha/"+name,tip,1. if main else rng.uniform(.97,1.48),
                          0. if main else rng.uniform(-160,160))
                    n_bunch+=1
                for k in range(1,11):
                    node=shoot[k]
                    a=rng.uniform(-.70,.70) + math.pi/2 + (math.pi if (k+idx+vi)%2 else 0)
                    slope=rng.uniform(-.84,-.24) if k<9 else rng.uniform(-.43,.05)
                    length=rng.uniform(.108,.166)*(1 - .025*k)
                    add_leaf(node,a,length,rng.uniform(-.62,.62),slope,k>=9)
                # Axillary laterals fill the curtain, including a few overhanging outer shoots.
                for k in rng.choice([2,4,6,8],size=2,replace=False):
                    node=shoot[k]
                    a=rng.uniform(-math.pi,math.pi)
                    offset=np.array([math.cos(a)*.20,math.sin(a)*.10,rng.uniform(-.16,.14)])
                    v=np.linspace(0,1,7)
                    lateral=node+v[:,None]*offset+np.sin(v*math.pi)[:,None]*[.012,.018,.046]
                    canes.tube(lateral,.0040-.0025*v,6)
                    for j in (2,4,6):
                        add_leaf(lateral[j],a+rng.uniform(-1.5,1.5),rng.uniform(.070,.121),
                                 rng.uniform(-.7,.7),rng.uniform(-.90,-.06),tender=j==6)
                if idx%2==0:
                    node=shoot[6]
                    u=np.linspace(0,1,33)
                    curl=node+np.c_[side*(.11*u),.020*np.sin(u*math.pi*6)*u,.028*u+.020*np.cos(u*math.pi*6)*u]
                    tendrils.tube(curl,.0016-.00075*u,5)
    leaves.CreatePositionsAttr(np.asarray(positions,dtype=np.float32))
    leaves.CreateProtoIndicesAttr(indices)
    leaves.CreateOrientationsAttr(orientations)
    leaves.CreateScalesAttr(np.asarray(scales,dtype=np.float32))
    leaves.CreateExtentAttr(leaves.ComputeExtentAtTime(Usd.TimeCode.Default(),Usd.TimeCode.Default()))
    for surf,part,mat in [(wood,"oak",bark),(posts,"timber",timber),(wires,"steel",steel)]:
        name="vine_row_fruiting_"+part
        surf.write(stage,f"{ROOT}/{name}/{name}",mat,collider=True)
    for surf,name,mat in [(canes,"CurrentSeasonShoots",fresh),(petioles,"Petioles",fresh),(tendrils,"Tendrils",young)]:
        surf.write(stage,ROOT+"/"+name,mat)
    stage.GetRootLayer().Save()


if __name__=="__main__":
    build()
