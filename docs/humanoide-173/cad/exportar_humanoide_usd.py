"""Exporta el CAD y la secuencia del MP4 a un solo USD. Requiere OpenUSD, NumPy y FFmpeg."""
from functools import lru_cache
from hashlib import file_digest
from itertools import product
from pathlib import Path
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unicodedata

import numpy as np
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt

CAD = Path(__file__).resolve().parent
ANIM = CAD.parent / "animacion"
sys.path.insert(0, str(ANIM))
from secuencia_guia import DETAILS, STEPS, contains, paths

PACKAGE = CAD / "__cadgen__/models/humanoide_xavier.step.py"
MOVIE = ANIM / "guia_montaje_en_banco.mp4"
OUTPUT = CAD / "humanoide_xavier.usd"
ROOT = "/Humanoide"
FPS = 24
ASPECT = 882 / 644
T = Sdf.ValueTypeNames


def executable(name):
    path = Path(shutil.which(name) or name)
    shim = path.with_suffix(".shim")
    if shim.is_file():
        for line in shim.read_text(encoding="utf8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "path":
                return os.path.expandvars(value.strip().strip('"'))
    return str(path)


def clock(value):
    h, m, s = value.replace(",", ".").split(":")
    return int(h)*3600 + int(m)*60 + float(s)


def edits():
    info = subprocess.check_output([executable("ffprobe"), "-v", "error", "-show_chapters",
                                    "-of", "json", str(MOVIE)])
    chapters = json.loads(info)["chapters"]
    subtitles = subprocess.check_output([executable("ffmpeg"), "-v", "error", "-i", str(MOVIE),
                                         "-map", "0:s:0", "-f", "srt", "-"]).decode("utf8")
    cues = [(clock(a), clock(b)) for a,b in re.findall(r"([\d:,]+) --> ([\d:,]+)", subtitles)]
    assert len(chapters) == len(STEPS) == len(DETAILS)
    spans = []
    for step, chapter in zip(STEPS, chapters):
        assert chapter["tags"]["title"].endswith(step.title)
        start, end = (float(chapter[k+"_time"]) for k in ("start","end"))
        within = [(a,b) for a,b in cues if start <= a < end]
        # The spoken instructions are separated by half a second in the MP4.
        cuts = [a for (_,b),(a,_) in zip(within,within[1:]) if a-b > .25]
        if len(cuts) != 1:
            raise ValueError(f"No se reconoce el cambio de plano en {step.title}")
        spans.append(tuple(round(t*FPS) for t in (start, cuts[0], end)))
    return spans


def slug(name):
    ascii_name = unicodedata.normalize("NFKD",name).encode("ascii","ignore").decode()
    return re.sub("[^A-Za-z0-9]+","_",ascii_name).strip("_")


def part_arrays(file):
    raw = file.read_bytes()
    if struct.unpack_from("<4sII",raw) != (b"glTF",2,len(raw)):
        raise ValueError(f"Malla GLB incompleta: {file.name}")
    size = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20:20+size])
    def accessor(i):
        a = gltf["accessors"][i]
        v = gltf["bufferViews"][a["bufferView"]]
        assert "byteStride" not in v and v["buffer"] == 0
        width = {"SCALAR":1, "VEC3":3}[a["type"]]
        dtype = {5121:"u1",5123:"<u2",5125:"<u4",5126:"<f4"}[a["componentType"]]
        return np.frombuffer(raw, dtype=dtype, count=a["count"]*width,
                             offset=28+size+v.get("byteOffset",0)+a.get("byteOffset",0)).reshape(-1,width)
    positions, normals, indices = [], [], []
    count = 0
    for p in gltf["meshes"][0]["primitives"]:
        if p.get("mode",4) != 4:
            continue
        pts = accessor(p["attributes"]["POSITION"])
        norm = accessor(p["attributes"]["NORMAL"])
        idx = accessor(p["indices"]).ravel().astype(np.int32)
        assert len(pts) == len(norm) and len(idx)%3 == 0 and 0 <= idx.min() <= idx.max() < len(pts)
        assert np.isfinite(pts).all() and np.isfinite(norm).all()
        positions.append(pts)
        normals.append(norm)
        indices.append(idx + count)
        count += len(pts)
    return np.concatenate(positions), np.concatenate(normals), np.concatenate(indices)


def cad_matrix(occ):
    matrix = np.asarray(occ["transform"],dtype=float).reshape(4,4).copy()
    matrix[:3,3] *= .001     # the GLB vertices are already in metres
    return matrix


def usd_matrix(matrix):
    return Gf.Matrix4d(*matrix.T.ravel().tolist())


def material(stage, rgb):
    tag = "".join(f"{round(c*255):02X}" for c in rgb)
    path = ROOT+"/Materiales/Color_"+tag
    mat = UsdShade.Material.Define(stage,path)
    shade = UsdShade.Shader.Define(stage,path+"/Superficie")
    shade.CreateIdAttr("UsdPreviewSurface")
    linear = Gf.Vec3f(*[c/12.92 if c <= .04045 else ((c+.055)/1.055)**2.4 for c in rgb])
    shade.CreateInput("diffuseColor",T.Color3f).Set(linear)
    shade.CreateInput("roughness",T.Float).Set(.53)
    shade.CreateOutput("surface",T.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shade.ConnectableAPI(),"surface")
    return mat, linear


def route_key(cad_id):
    return tuple(int(n) for n in cad_id.split(".")[1:])


def corners(lo, hi):
    return np.array(list(product(*zip(lo,hi))))


def geometry(stage, doc):
    root = UsdGeom.Xform.Define(stage,ROOT).GetPrim()
    root.SetDisplayName("Humanoide Xavier")
    Usd.ModelAPI(root).SetKind(Kind.Tokens.assembly)
    stage.SetDefaultPrim(root)
    UsdGeom.Scope.Define(stage, ROOT+"/Materiales")
    occurrences = {o["id"]:o for o in doc["occurrences"]}
    @lru_cache(None)
    def mesh_data(key):
        comp = doc["components"][key]
        p,n,i = part_arrays(PACKAGE/comp["glb"])
        return (p, Vt.Vec3fArray.FromNumpy(p), Vt.Vec3fArray.FromNumpy(n),
                Vt.IntArray.FromNumpy(i), Vt.IntArray.FromNumpy(np.full(len(i)//3,3,dtype=np.int32)),
                Vt.Vec3fArray.FromNumpy(np.stack((p.min(0),p.max(0)))))
    palette = lru_cache(None)(lambda rgb: material(stage,rgb))
    nodes, bounds = {}, {}
    moving = {prefix for s in STEPS for spec,*_ in s.moves for prefix in paths(spec)}
    def visit(node, parent):
        key = route_key(node["id"])
        path = parent + f"/p{key[-1]:02d}_" + slug(node["name"])
        if node["id"] in occurrences:
            o = occurrences[node["id"]]
            points, verts, normals, indices, counts, extent = mesh_data(o["component"])
            mesh = UsdGeom.Mesh.Define(stage,path)
            mesh.CreatePointsAttr(verts)
            mesh.CreateNormalsAttr(normals)
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            mesh.CreateFaceVertexIndicesAttr(indices)
            mesh.CreateFaceVertexCountsAttr(counts)
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateExtentAttr(extent)
            prim = mesh.GetPrim()
            mat, linear = palette(tuple(o["color"][:3]))
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
            mesh.CreateDisplayColorAttr([linear])
            transform = cad_matrix(o)
            world_points = points @ transform[:3,:3].T + transform[:3,3]
            bounds[key] = corners(world_points.min(0),world_points.max(0))
        else:
            prim = UsdGeom.Xform.Define(stage,path).GetPrim()
        prim.SetDisplayName(node["name"])
        prim.CreateAttribute("cad:ocurrencia",T.String,custom=True).Set(node["id"])
        nodes[key] = prim
        xf = UsdGeom.Xformable(prim)
        if key in moving:
            # Unauthored in the base layer so that the variant can supply motion.
            xf.AddTranslateOp(opSuffix="montaje")
        if node["id"] in occurrences and not np.array_equal(transform,np.eye(4)):
            xf.AddTransformOp(opSuffix="cad").Set(usd_matrix(transform))
        if key == (2,8):
            UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
        for child in node.get("children",[]):
            visit(child,path)
    for child in doc["assembly"]["root"]["children"]:
        visit(child,ROOT)
    return root, nodes, bounds


def position(u, xyz, begin, end):
    t = min(1.,max(0.,(u-begin)/(end-begin)))
    return np.array(xyz)*(1-t*t*(3-2*t))


def local_moves(step):
    moves = {}
    for spec,xyz,a,b in step.moves:
        for k in paths(spec):
            moves.setdefault(k,[]).append((xyz,a,b))
    return moves


def framing(step, bounds):
    show, hide, focus = (paths(text) for text in (step.show,step.hide,step.focus))
    keys = [k for k in bounds if contains(k,show) and not contains(k,hide) and (not focus or contains(k,focus))]
    initial = {k:np.zeros(3) for k in keys}
    for spec,xyz,*_ in step.moves:
        group = paths(spec)
        for k in keys:
            if contains(k,group):
                initial[k] += xyz
    cloud = np.concatenate([pts for k in keys for pts in (bounds[k],bounds[k]+initial[k])])
    direction = np.array(step.direction,dtype=float)
    direction /= np.linalg.norm(direction)
    right = np.cross(-direction,[0,0,1])
    right /= np.linalg.norm(right)
    up = np.cross(right,-direction)
    axes = np.stack([right,up,direction])
    q = cloud @ axes.T
    lo, hi = q.min(0), q.max(0)
    center = (lo+hi)*.5 @ axes
    span = max(.012, (hi[1]-lo[1])*1.14, (hi[0]-lo[0])*1.14/ASPECT)
    return Gf.Vec3d(*center), direction, span


def animate(stage, root, nodes, bounds, spans):
    variants = root.GetVariantSets().AddVariantSet("Vista")
    variants.AddVariant("Ensamblado")
    variants.AddVariant("Montaje")
    variants.SetVariantSelection("Montaje")
    with variants.GetVariantEditContext():
        cam = UsdGeom.Camera.Define(stage,ROOT+"/CamaraDeMontaje")
        cam.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
        cam.CreateClippingRangeAttr((.01,10))
        camera_op = cam.AddTransformOp()
        caption = root.CreateAttribute("montaje:capitulo",T.String,custom=True)
        check = root.CreateAttribute("montaje:comprobacion",T.String,custom=True)
        states = {}
        for i,(start,cut,end) in enumerate(spans):
            caption.Set(f"{i+1:02d} · {STEPS[i].title}",start)
            check.Set(STEPS[i].check,start)
            for step, first, stop, ticks in ((STEPS[i],start,cut,90),(DETAILS[i],cut,end,56)):
                show, hide = paths(step.show), paths(step.hide)
                # Affect the CAD subtrees at their top boundary. Reset children
                # when an earlier isolated view gives way to a full assembly.
                for k,prim in nodes.items():
                    def relevant(part):
                        return (contains(part,show) or any(p[:len(part)]==part for p in show)) and not contains(part,hide)
                    parent = k[:-1]
                    visible = relevant(k) or (parent and not relevant(parent))
                    value = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
                    if value != states.get(k,UsdGeom.Tokens.inherited):
                        attr = UsdGeom.Imageable(prim).CreateVisibilityAttr()
                        if k not in states and first:
                            attr.Set(UsdGeom.Tokens.inherited,0)
                        attr.Set(value,first)
                        states[k] = value
                for k,motions in local_moves(step).items():
                    attr = nodes[k].GetAttribute("xformOp:translate:montaje")
                    if first:
                        attr.Set(Gf.Vec3d(0),first-1)
                    for tick in range(0,ticks+1,3):
                        value = Gf.Vec3d(*sum(position(tick/ticks,*m) for m in motions))
                        if tick == 0:
                            attr.Set(value,first)
                        attr.Set(value, first+8+tick*2)
                center,direction,span = framing(step,bounds)
                for attr,size in ((cam.CreateVerticalApertureAttr(),span),
                                  (cam.CreateHorizontalApertureAttr(),span*ASPECT)):
                    attr.Set(size/Gf.Camera.APERTURE_UNIT,first)
                    attr.Set(size/Gf.Camera.APERTURE_UNIT,stop-1)
                for t,u in ((first,0),(first+8,0),(first+8+ticks, .5),
                            (first+8+2*ticks,1),(stop-1,1)):
                    ang = (.09 if step.moves else .21)*(u-.5)
                    x,y,z = direction
                    view = Gf.Vec3d(x*np.cos(ang)-y*np.sin(ang),x*np.sin(ang)+y*np.cos(ang),z)
                    look = Gf.Matrix4d().SetLookAt(center+view*2,center,Gf.Vec3d.ZAxis()).GetInverse()
                    camera_op.Set(look,t)
    variants.SetVariantSelection("Ensamblado")


def verify(file, doc, nodes, bounds, spans):
    stage = Usd.Stage.Open(str(file))
    root = stage.GetDefaultPrim()
    assert stage.GetCompositionErrors() == []
    assert root.GetPath() == Sdf.Path(ROOT)
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1 and UsdGeom.GetStageUpAxis(stage) == "Z"
    variants = root.GetVariantSets().GetVariantSet("Vista")
    assert variants.GetVariantSelection()=="Ensamblado"
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert len(meshes) == len(doc["occurrences"])
    occurrences = {o["id"]:o for o in doc["occurrences"]}
    read = lru_cache(None)(part_arrays)
    cache = UsdGeom.XformCache()
    for prim in meshes:
        cad_id = prim.GetAttribute("cad:ocurrencia").Get()
        occ = occurrences[cad_id]
        key = route_key(cad_id)
        mesh = UsdGeom.Mesh(prim)
        p = np.asarray(mesh.GetPointsAttr().Get())
        points,normals,faces = read(PACKAGE/doc["components"][occ["component"]]["glb"])
        assert np.array_equal(p,points) and np.array_equal(mesh.GetNormalsAttr().Get(),normals)
        assert np.array_equal(mesh.GetFaceVertexIndicesAttr().Get(),faces)
        assert mesh.GetFaceVertexCountsAttr().Get() == Vt.IntArray(len(faces)//3,3)
        matrix = np.array(cache.GetLocalToWorldTransform(prim))
        assert np.allclose(matrix,cad_matrix(occ).T,atol=1e-10,rtol=0)
        world = p @ matrix[:3,:3] + matrix[3,:3]
        assert np.allclose([world.min(0),world.max(0)], [bounds[key].min(0),bounds[key].max(0)],atol=1e-7,rtol=0)
        assert mesh.GetSubdivisionSchemeAttr().Get()=="none"
        assert UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    bbox = np.concatenate(list(bounds.values()))
    bmin,bmax = bbox.min(0),bbox.max(0)
    # The package stores a conservative bound for curved B-rep surfaces.
    cad_lo,cad_hi = np.array([doc["bbox"]["min"],doc["bbox"]["max"]])*.001
    assert np.all(bmin>=cad_lo-.0001) and np.all(bmax<=cad_hi+.0001)
    assert abs(bmax[2]-bmin[2]-1.730)<.0001
    layer = stage.GetRootLayer()
    assert not layer.subLayerPaths and not layer.GetExternalReferences()
    stage.SetEditTarget(stage.GetSessionLayer())
    variants.SetVariantSelection("Montaje")
    camera = UsdGeom.Camera(stage.GetPrimAtPath(ROOT+"/CamaraDeMontaje"))
    actual = {k:stage.GetPrimAtPath(v) for k,v in nodes.items()}
    for i,(start,cut,end) in enumerate(spans):
        assert root.GetAttribute("montaje:capitulo").Get(start).endswith(STEPS[i].title)
        for step,t in ((STEPS[i],start),(DETAILS[i],cut)):
            show,hide = paths(step.show),paths(step.hide)
            for k in bounds:
                expected = contains(k,show) and not contains(k,hide)
                assert (UsdGeom.Imageable(actual[k]).ComputeVisibility(t)!="invisible")==expected,(i,k,t)
            assert camera.GetCamera(t).frustum.GetWindow().GetSize()[1] > 0
        for k,motions in local_moves(STEPS[i]).items():
            attr = actual[k].GetAttribute("xformOp:translate:montaje")
            assert np.allclose(attr.Get(start),sum(np.array(m[0]) for m in motions),atol=1e-7)
            assert np.allclose(attr.Get(cut-1),0,atol=1e-7)
    layers,assets,missing = UsdUtils.ComputeAllDependencies(str(file))
    assert not missing and not assets and len(layers)==1
    print(f"{len(meshes)} piezas, {len(spans)} capítulos, medidas XYZ (m): {(bmax-bmin).round(5)}",flush=True)


def main():
    doc = json.loads((PACKAGE/"assembly.json").read_text(encoding="utf8"))
    with (CAD/doc["stepPath"]).open("rb") as src:
        if file_digest(src,"sha256").hexdigest()!=doc["stepHash"]:
            raise ValueError("Regenera el paquete de mallas del STEP antes de exportar.")
    spans = edits()
    with tempfile.TemporaryDirectory(dir=OUTPUT.parent) as tmp:
        file = Path(tmp)/OUTPUT.name
        stage = Usd.Stage.CreateNew(str(file))
        UsdGeom.SetStageMetersPerUnit(stage,1)
        UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        stage.SetFramesPerSecond(FPS)
        stage.SetTimeCodesPerSecond(FPS)
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(spans[-1][-1]-1)
        root,nodes,bounds = geometry(stage,doc)
        root.SetDocumentation("Humanoide Xavier de 1,73 m. Selecciona Vista = Montaje y CamaraDeMontaje para recorrer la guía.")
        root.SetCustomDataByKey("cad:sha256",doc["stepHash"])
        root.SetCustomDataByKey("cad:archivo",doc["stepPath"])
        root.SetCustomDataByKey("montaje:inicios",Vt.IntArray([s[0] for s in spans]))
        root.SetCustomDataByKey("montaje:titulos",Vt.StringArray([s.title for s in STEPS]))
        animate(stage,root,nodes,bounds,spans)
        stage.GetRootLayer().Save()
        # Release file mappings before replacing on Windows.
        node_paths = {k:p.GetPath() for k,p in nodes.items()}
        del root,nodes,stage
        verify(file,doc,node_paths,bounds,spans)
        os.replace(file,OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
