"""Build the render-only grape bunch using the centers of its calibrated CC0 colliders."""
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


ASSETS = Path(__file__).resolve().parents[1] / "assets/agriculture/models/grapes"
STEM_TIP = (.00760457, .00025927, .126628)
SECTORS, RINGS = 32, 20
ATLAS_COLS, ATLAS_ROWS = 6, 4


def skin_atlas():
    """A periodic spherical wax pattern. The same wax coverage drives color and roughness."""
    tile = 256
    rng = np.random.default_rng(818)
    y, x = np.mgrid[:tile, :tile] / (tile-1)
    theta, phi = y*np.pi, x*2*np.pi
    sphere = np.stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)), -1)
    albedo = np.zeros((tile*ATLAS_ROWS, tile*ATLAS_COLS, 3), np.uint8)
    roughness = np.zeros(albedo.shape[:2], np.uint8)
    original = np.asarray(Image.open(ASSETS/"textures/grape_purple_food_2k.png").convert("RGB")
                          .resize((tile, tile)), dtype=float)
    for index in range(ATLAS_ROWS*ATLAS_COLS):
        def noise(frequency):
            directions = rng.normal(size=(3, 6))
            directions /= np.linalg.norm(directions, axis=0)
            return np.cos(sphere @ directions * frequency + rng.uniform(0, 2*np.pi, 6)).mean(axis=2)

        # A mostly continuous wax film is thinner where neighboring berries rub.
        # Locate the scuffs on the sphere so a highlight stays continuous across the UV seam.
        rubbed = np.zeros_like(x)
        for _ in range(2):
            contact = rng.normal(size=3)
            contact /= np.linalg.norm(contact)
            width = rng.uniform(.028, .060)
            rubbed = np.maximum(rubbed, np.exp(-(1 - sphere @ contact)/width))
        fine = .17*noise(49) + .09*noise(113)
        wax = np.clip(rng.uniform(.55,.72) + .13*np.cos(theta) + .26*noise(13)
                      + fine - .47*rubbed, .08,.94)
        maturity = rng.uniform(0., 1.)
        pigment = np.array([77,32,53])*(1-maturity) + np.array([47,31,64])*maturity
        pigment += rng.uniform(-4., 4.)
        bloom = pigment + np.array([44,48,55])
        blush = 2.*sphere[:,:,0] + 2.5*sphere[:,:,2]
        rgb = (pigment + (bloom-pigment)*wax[:,:,None] + blush[:,:,None] +
               .23*(original-original.mean(axis=(0,1))))
        # The tiny pistil scar is at the blossom end, opposite the pedicel.
        rad = (1-y)*np.pi
        scar = np.clip(1-rad/.085, 0, 1) * (.83+.15*np.cos(5*phi))
        rgb = rgb*(1-scar[:,:,None]) + np.array([96,65,54])*scar[:,:,None]
        a, b = divmod(index, ATLAS_COLS)
        region = np.s_[a*tile:(a+1)*tile,b*tile:(b+1)*tile]
        albedo[region] = np.uint8(np.clip(rgb,0,255))
        roughness[region] = np.uint8(255*np.clip(.23+.34*wax+.07*scar,0,1))
    Image.fromarray(albedo).save(ASSETS/"textures/bunch_skin.png")
    Image.fromarray(roughness).save(ASSETS/"textures/bunch_roughness.png")


def material(stage, label, texture, roughness):
    base = "/Bunch/Materials/" + label
    mat = UsdShade.Material.Define(stage, base)
    shader = UsdShade.Shader.Define(stage, base+"/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    reader = UsdShade.Shader.Define(stage, base+"/Coordinates")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    def sampler(name, file, space):
        node = UsdShade.Shader.Define(stage, base+"/"+name)
        node.CreateIdAttr("UsdUVTexture")
        node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./textures/"+file))
        node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(space)
        return node
    color = sampler("Color", texture, "sRGB")
    color.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(color.ConnectableAPI(), "rgb")
    value = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    if isinstance(roughness, str):
        gloss = sampler("Wax", roughness, "raw")
        gloss.CreateOutput("r", Sdf.ValueTypeNames.Float)
        value.ConnectToSource(gloss.ConnectableAPI(), "r")
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.46)
        shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(.10)
        shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(.24)
    else:
        value.Set(roughness)
    return mat


def surface(stage, name, points, faces, uv, mat):
    mesh = UsdGeom.Mesh.Define(stage, "/Bunch/"+name)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([j for f in faces for j in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    normals = np.zeros_like(points)
    for f in faces:
        i, j, k = f[:3]
        normal = np.cross(points[j]-points[i],points[k]-points[i])
        for v in f:
            normals[v] += normal
    normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    mesh.CreateNormalsAttr(normals.astype(np.float32))
    mesh.SetNormalsInterpolation("vertex")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,"faceVarying").Set(uv)
    mesh.CreateExtentAttr([Gf.Vec3f(*points.min(axis=0)),Gf.Vec3f(*points.max(axis=0))])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def berry(stage, number, center, radii, rng, mat):
    # Keep the calibrated berry envelope, with modest asymmetry and an upturned pedicel end.
    axis = np.r_[-center[:2]*8, 1.]
    axis /= np.linalg.norm(axis)
    vx = np.cross((0.,1.,0.),axis)
    vx /= np.linalg.norm(vx)
    vy = np.cross(axis,vx)
    phase = rng.uniform(0, 2*np.pi)
    radii = radii*rng.uniform(.972,1.028,3)
    points = [center+axis*(radii[2]-.00065)]
    for row in range(1,RINGS):
        t = np.pi*row/RINGS
        for col in range(SECTORS):
            p = 2*np.pi*col/SECTORS
            d = np.sin(t)*(1 + .024*np.cos(t) + .016*np.sin(3*p+phase)*np.sin(t)**2)
            z = radii[2]*np.cos(t) - .00065*np.exp(-(t/.20)**2)
            points.append(center+vx*(radii[0]*d*np.cos(p))+vy*(radii[1]*d*np.sin(p))+axis*z)
    points.append(center-axis*(radii[2]-.00015))
    tile_row,tile_col = divmod(number,ATLAS_COLS)
    faces, uv = [], []
    def face(indices, coords):
        faces.append(indices)
        for col,row in coords:
            # Inset one texel to keep neighboring swatches out of the seam.
            u=(1+253*col/SECTORS)/256
            v=(1+253*row/RINGS)/256
            uv.append(((tile_col+u)/ATLAS_COLS,1-(tile_row+v)/ATLAS_ROWS))
    for col in range(SECTORS):
        nxt=(col+1)%SECTORS
        face([0,1+col,1+nxt], [(col+.5,0),(col,1),(col+1,1)])
        for row in range(1,RINGS-1):
            a=1+(row-1)*SECTORS
            face([a+col,a+col+SECTORS,a+nxt+SECTORS,a+nxt],
                 [(col,row),(col,row+1),(col+1,row+1),(col+1,row)])
        a=1+(RINGS-2)*SECTORS
        face([a+col,len(points)-1,a+nxt], [(col,RINGS-1),(col+.5,RINGS),(col+1,RINGS-1)])
    surface(stage,f"Berry_{number:02d}",np.asarray(points),faces,uv,mat)
    return points[0]


def branch(stage, name, path, radii, mat):
    path = np.asarray(path)
    sides=9
    directions=np.gradient(path,axis=0)
    points,faces,uv=[],[],[]
    for i,(center,direction,r) in enumerate(zip(path,directions,radii,strict=True)):
        unit=direction/np.linalg.norm(direction)
        cross=np.cross(unit,[1,0,0] if abs(unit[0])<.9 else [0,1,0])
        cross/=np.linalg.norm(cross)
        normal=np.cross(unit,cross)
        for j in range(sides):
            angle=2*np.pi*j/sides
            points.append(center+r*(np.cos(angle)*cross+np.sin(angle)*normal))
            if i:
                a=(i-1)*sides+j
                b=(i-1)*sides+(j+1)%sides
                faces.append([a,b,b+sides,a+sides])
                uv.extend([(j/sides,(i-1)/(len(path)-1)),((j+1)/sides,(i-1)/(len(path)-1)),
                           ((j+1)/sides,i/(len(path)-1)),(j/sides,i/(len(path)-1))])
    for i in (0,len(path)-1):
        ring=list(range(i*sides,(i+1)*sides))
        faces.append(ring if i else ring[::-1])
        uv.extend([(.5,.5)]*sides)
    surface(stage,name,np.asarray(points),faces,uv,mat)


def main():
    original = Usd.Stage.Open(str(ASSETS/"visual.usda"))
    cache = UsdGeom.XformCache()
    prior = sorted((p for p in original.Traverse() if p.IsA(UsdGeom.Mesh) and p.GetName().startswith("Grape_berry_")),
                   key=lambda p:p.GetName())
    if len(prior)!=24:
        raise ValueError("Expected the calibrated Innerscene bunch with 24 berries")
    skin_atlas()
    stage=Usd.Stage.CreateNew(str(ASSETS/"bunch.usdc"))
    root=UsdGeom.Xform.Define(stage,"/Bunch").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    root.CreateAttribute("grape:stemTip",Sdf.ValueTypeNames.Double3,custom=True).Set(Gf.Vec3d(*STEM_TIP))
    stage.GetRootLayer().documentation="Visual derivative of Innerscene Bunch of Grapes (CC0), fitted to the calibrated berry positions. Z up, meters."
    fruit=material(stage,"Skin","bunch_skin.png","bunch_roughness.png")
    stem=material(stage,"Peduncle","stem_brown_food_2k.png",.65)
    # The central lignified stem bends toward the attachment and tapers inside the cluster.
    zs=np.linspace(.018,STEM_TIP[2],20)
    u=(zs-zs[0])/(zs[-1]-zs[0])
    spine=np.column_stack((.00760457*u**3+.0018*np.sin(u*2*np.pi),
                           .00025927*u-.002*np.sin(np.pi*u),zs))
    branch(stage,"Rachis",spine,.00072+.00135*u,stem)
    rng=np.random.default_rng(9607)
    for index,old in enumerate(prior):
        points=np.asarray(UsdGeom.Mesh(old).GetPointsAttr().Get())
        points=(np.c_[points,np.ones(len(points))]@np.asarray(cache.GetLocalToWorldTransform(old)))[:,:3]
        center=(points.max(0)+points.min(0))/2
        end=berry(stage,index,center,np.ptp(points,axis=0)/2,rng,fruit)
        at=min(STEM_TIP[2]-.003,end[2]+rng.uniform(.005,.012))
        start=np.array([np.interp(at,zs,spine[:,i]) for i in range(3)])
        t=np.linspace(0,1,7)
        bend=np.array([rng.uniform(-.003,.003),rng.uniform(-.003,.003),.003])
        curve=(1-t[:,None])*start + t[:,None]*end + np.sin(np.pi*t)[:,None]*bend
        branch(stage,f"Pedicel_{index:02d}",curve,.00104-.00033*t+.00038*t**6,stem)
    stage.GetRootLayer().Save()


if __name__=="__main__":
    main()
