"""Genera el activo PhysX y el banco a partir del USD ensamblado y el MJCF del proyecto."""
from functools import lru_cache
from pathlib import Path
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

HERE = Path(__file__).resolve().parent
CAD = HERE.parent / "cad"
sys.path.insert(0, str(CAD))
from cinematica_manos import REF, finger

ROOT = "/Humanoide"
ASSET = "humanoide_xavier_isaac.usd"
T = Sdf.ValueTypeNames
DIGITS = ("indice", "medio", "anular", "pulgar", "menique")
# Grados de CAD, N·m, grados/s, Kp en N·m/rad y Kd en N·m·s/rad.
# Los ejes sin motor definido usan un accionamiento virtual para ensayar poses.
PROFILES = {
    "cuello": (-40, 40, .98, 90, 8, .6, "ST3215"),
    "hombro": (-20, 65, 18, 45, 130, 10, "virtual"),
    "codo": (0, 90, 24, 420, 100, 6, "AK60-39"),
    "muneca": (-5, 25, 3, 45, 18, 1.1, "virtual"),
    "cadera": (-15, 35, 24, 420, 260, 16, "AK60-39 V3.0 KV80"),
    "apertura_cadera": (-8, 18, 24, 420, 220, 12, "AK60-39 V3.0 KV80"),
    "giro_cadera": (-20, 20, 24, 420, 140, 8, "AK60-39 V3.0 KV80"),
    "rodilla": (0, 80, 24, 420, 180, 10, "AK60-39"),
    "tobillo": (-12, 12, 18, 45, 100, 6, "virtual"),
}


def vec(a):
    return Gf.Vec3f(*map(float, a))


def quat(rotation):
    x, y, z, w = rotation.as_quat()
    return Gf.Quatf(float(w), vec((x, y, z)))


def prop(prim, name, kind, value):
    prim.CreateAttribute(name, kind, custom=False).Set(value)


def configure(stage):
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)


def collider(prim):
    UsdPhysics.CollisionAPI.Apply(prim)
    prim.AddAppliedSchema("PhysxCollisionAPI")
    prop(prim, "physxCollision:contactOffset", T.Float, .001)
    prop(prim, "physxCollision:restOffset", T.Float, 0.)
    UsdGeom.Imageable(prim).CreatePurposeAttr("guide")


def box(stage, path, center, size):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1)
    shape.AddTranslateOp().Set(Gf.Vec3d(*center))
    shape.AddScaleOp().Set(vec(size))
    collider(shape.GetPrim())
    return shape


MOTOR_KG, RADIO, LARGO = .750, .0395, .067
Z_GIRO = .875 + .096 + LARGO/2


def inercia_motor(eje):
    transversal = MOTOR_KG * (3*RADIO**2 + LARGO**2) / 12
    diagonal = np.full(3, transversal)
    diagonal["XYZ".index(eje)] = MOTOR_KG * RADIO**2 / 2
    return diagonal


def contacto_motor(stage, path, center, eje):
    forma = UsdGeom.Cylinder.Define(stage, path)
    forma.CreateRadiusAttr(RADIO)
    forma.CreateHeightAttr(LARGO)
    forma.CreateAxisAttr(eje)
    forma.AddTranslateOp().Set(Gf.Vec3d(*center))
    collider(forma.GetPrim())


class Robot:
    def __init__(self, stage):
        self.stage = stage
        self.root = UsdGeom.Xform.Define(stage, ROOT).GetPrim()
        stage.SetDefaultPrim(self.root)
        self.origins = {}
        self.paths = {}
        self.mapping = {(): "pelvis"}
        self.joints = []
        self.pad_ids = set()
        self.pads = []
        self.flexible = []
        self.root.AddAppliedSchema("IsaacRobotAPI")
        prop(self.root, "isaac:robotType", T.Token, "Humanoid")
        self.root.SetDocumentation("Humanoide Xavier. Base libre. Recorridos CAD y parámetros de dinámica provisionales.")
        UsdGeom.Scope.Define(stage, ROOT + "/Juntas")

    def body(self, name, origin, mass, center, inertia, axes=Gf.Quatf(1)):
        self.origins[name] = np.array(origin, dtype=float)
        path = ROOT + "/" + name
        self.paths[name] = path
        body = UsdGeom.Xform.Define(self.stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(*origin))
        prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(mass)
        api.CreateCenterOfMassAttr(vec(center))
        api.CreateDiagonalInertiaAttr(vec(inertia))
        api.CreatePrincipalAxesAttr(axes)
        UsdGeom.Scope.Define(self.stage, path + "/Visual")
        return prim

    def segment(self, name, center, size, mass, prefixes, *, contact=True):
        # Inercia de paralelepípedo uniforme de dimensiones en metros.
        x, y, z = np.square(size)
        body = self.body(name, center, mass, (0, 0, 0), mass / 12 * np.array([y+z, x+z, x+y]))
        if contact:
            box(self.stage, self.paths[name] + "/Contacto", (0, 0, 0), size)
        for prefix in prefixes:
            self.mapping[prefix] = name
        return body

    def joint(self, name, parent, child, pivot, axis=(1, 0, 0), *, ball=False, loop=False, profile=None):
        path = ROOT + "/Juntas/" + name
        schema = UsdPhysics.SphericalJoint if ball else UsdPhysics.RevoluteJoint
        joint = schema.Define(self.stage, path)
        joint.CreateBody0Rel().SetTargets([self.paths[parent]])
        joint.CreateBody1Rel().SetTargets([self.paths[child]])
        joint.CreateLocalPos0Attr(vec(pivot - self.origins[parent]))
        joint.CreateLocalPos1Attr(vec(pivot - self.origins[child]))
        orientation = Gf.Quatf(Gf.Rotation(Gf.Vec3d.XAxis(), Gf.Vec3d(*axis)).GetQuat())
        joint.CreateAxisAttr("X")
        joint.CreateLocalRot0Attr(orientation)
        joint.CreateLocalRot1Attr(orientation)
        if loop:
            joint.CreateExcludeFromArticulationAttr(True)
        if profile:
            low, high, torque, speed, kp, kd, motor = profile
            joint.CreateLowerLimitAttr(float(low))
            joint.CreateUpperLimitAttr(float(high))
            prim = joint.GetPrim()
            prim.SetCustomDataByKey("control:actuador", motor)
            prim.AddAppliedSchema("PhysxJointAPI")
            prop(prim, "physxJoint:maxJointVelocity", T.Float, float(speed))
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateMaxForceAttr(torque)
            drive.CreateStiffnessAttr(float(np.deg2rad(kp)))
            drive.CreateDampingAttr(float(np.deg2rad(kd)))
            drive.CreateTargetPositionAttr(0.)
            drive.CreateTargetVelocityAttr(0.)
            prim.AddAppliedSchema("PhysicsJointStateAPI:angular")
            prop(prim, "state:angular:physics:position", T.Float, 0.)
            prop(prim, "state:angular:physics:velocity", T.Float, 0.)
        self.joints.append(joint.GetPath())

    def trunk(self):
        base = self.segment("pelvis", (0, 0, 1.17), (.32, .165, .59), 6.8, [], contact=False)
        # Ambos estatores de giro van anclados al tronco. Ejes principales por simetría.
        masa = UsdPhysics.MassAPI(base)
        dz = Z_GIRO - 1.17
        total = 6.8 + 2*MOTOR_KG
        cz = 2*MOTOR_KG*dz/total
        tensor = np.array(masa.GetDiagonalInertiaAttr().Get()) + 6.8 * np.array([cz**2,cz**2,0])
        tensor += 2*(inercia_motor("Z") + MOTOR_KG*np.array([(dz-cz)**2, .112**2+(dz-cz)**2, .112**2]))
        masa.CreateMassAttr(total)
        masa.CreateCenterOfMassAttr(vec((0,0,cz)))
        masa.CreateDiagonalInertiaAttr(vec(tensor))
        for side, x in (("izq",-.112),("der",.112)):
            contacto_motor(self.stage,self.paths["pelvis"]+"/Motor_"+side,(x,0,dz),"Z")
        # Tres volúmenes siguen el cambio de sección pelvis-abdomen-tórax.
        for i, (z, size) in enumerate(((.92, (.255, .14, .14)), (1.10, (.19, .15, .23)),
                                       (1.345, (.32, .165, .26)))):
            box(self.stage, self.paths["pelvis"] + f"/Contacto{i}", (0, 0, z-1.17), size)
        UsdPhysics.ArticulationRootAPI.Apply(base)
        base.AddAppliedSchema("PhysxArticulationAPI")
        prop(base, "physxArticulation:solverPositionIterationCount", T.Int, 64)
        prop(base, "physxArticulation:solverVelocityIterationCount", T.Int, 4)
        prop(base, "physxArticulation:enabledSelfCollisions", T.Bool, True)
        self.segment("cabeza", (0, -.012, 1.63), (.13, .146, .19), .90, [(6,), (1, 15)])
        self.joint("cuello", "pelvis", "cabeza", (0, 0, 1.532), (0, 0, 1), profile=PROFILES["cuello"])

    def limb(self, sign, side, arm, leg, elbow, knee, wire, traction):
        def name(n):
            return f"{n}_{side}"
        x, h = sign * .236, sign * .112
        self.flexible.extend([(arm, 9), (16, wire, 3), (16, wire, 7),
                              (19, traction, 2), (19, traction, 3)])
        self.segment(name("brazo"), (x, -.003, 1.25), (.085, .09, .265), 1.45,
                     [(arm,), (elbow,), (19, traction, 4), (19, traction, 2), (19, traction, 3),
                      (16, wire, 12), (16, wire, 13), (16, wire, 3), (16, wire, 7)])
        self.segment(name("antebrazo"), (x, 0, .975), (.078, .076, .225), .72,
                     [(arm, k) for k in (1, 6, 7, 9)] + [(arm, 2, 3), (arm, 2, 4), (elbow, 3)]
                     + [(16, wire, k) for k in (4, 8, 9, 10, 11)])
        self.segment(name("mano"), (x, .003, .767), (.052, .12, .12), .34, [(arm, 8)], contact=False)
        self.pad_ids.add((arm, 8, 3))  # cubierta palmar
        self.segment(name("muslo"), (h, 0, .682), (.102, .102, .345), 2.05, [(leg,), (knee,)])
        self.segment(name("pantorrilla"), (h, -.006, .278), (.094, .094, .335), .95,
                     [(leg, k) for k in (2, 3)] + [(leg, 4, 3), (knee, 3)])
        self.segment(name("pie"), (h, .042, .029), (.119, .245, .058), .50, [(leg, 1)])
        pivot = (h, 0., .875)
        indice = 2 if sign < 0 else 4
        giro, apertura = name("bastidor_giro"), name("bastidor_apertura")
        for link, center, eje, grupo in (
            (giro,(h,-.024-LARGO/2,.875),"Y",indice),
            (apertura,(h-sign*(.040+LARGO/2),0.,.875),"X",indice+1),
        ):
            self.body(link,center,MOTOR_KG,(0,0,0),inercia_motor(eje))
            self.mapping[(20,grupo)] = link
            contacto_motor(self.stage,self.paths[link]+"/Motor",(0,0,0),eje)
        self.flexible.append((18,3 if sign < 0 else 4,7))
        for j,p,c,eje in (("giro_cadera","pelvis",giro,(0,0,-sign)),
                          ("apertura_cadera",giro,apertura,(0,-sign,0)),
                          ("cadera",apertura,name("muslo"),(1,0,0))):
            self.joint(name(j),p,c,pivot,eje,profile=PROFILES[j])
        # Las cajas envolventes del escudo y los componentes que aloja se solapan.
        # Excluir únicamente esos pares del mismo robot. Ambas piernas sí interactúan.
        filtro = UsdPhysics.FilteredPairsAPI.Apply(self.stage.GetPrimAtPath(self.paths["pelvis"]))
        for link in (apertura,name("muslo")):
            filtro.CreateFilteredPairsRel().AddTarget(self.paths[link])
        for j, p, c, pivot, axis in [
            ("hombro", "pelvis", name("brazo"), (x, 0, 1.4), 1),
            ("codo", name("brazo"), name("antebrazo"), (x, 0, 1.1), 1),
            ("muneca", name("antebrazo"), name("mano"), (x, 0, .845), 1),
            ("rodilla", name("muslo"), name("pantorrilla"), (h, 0, .48), -1),
            ("tobillo", name("pantorrilla"), name("pie"), (h, 0, .085), 1),
        ]:
            self.joint(name(j), p, c, pivot, (axis, 0, 0), profile=PROFILES[j])

    def digit(self, side, arm, digit_index, pose, groups):
        digit = DIGITS[digit_index]
        prefix = (arm, 8, digit_index + 6)
        palm = "mano_" + side
        key = digit + "_" + side
        names = [key + "_" + n for n in ("palanca1", "varilla1", "acople", "cardan", "distal",
                                         "proximal", "palanca2", "varilla2")]
        mount = np.eye(4)
        mount[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        mount[:3, 3] = pose[:3]
        rest, nodes = finger_data(digit == "pulgar")
        frames = [mount @ frame for frame in rest]
        for i, (frame, node, members) in enumerate(zip(frames, nodes, groups, strict=True)):
            it = node.find("inertial")
            a, b, c, xy, xz, yz = np.fromstring(it.get("fullinertia"), sep=" ")
            values, axes = np.linalg.eigh([[a,xy,xz],[xy,b,yz],[xz,yz,c]])
            axes[:, 0] *= np.linalg.det(axes)
            self.body(names[i], frame[:3, 3] * .001, float(it.get("mass")),
                      frame[:3, :3] @ np.fromstring(it.get("pos"), sep=" "), values,
                      quat(Rotation.from_matrix(frame[:3, :3] @ axes)))
            for part in members:
                self.mapping[prefix + (part,)] = names[i]
        self.pad_ids.update([prefix + (18,), prefix + (22,)])  # almohadillas CAD
        model = finger()
        sites = {}
        for idx, label, point in model.sites:
            transform = mount @ model.basis if idx < 0 else frames[idx]
            sites[label] = (transform[:3, :3] @ point + transform[:3, 3]) * .001

        def hinge_frame(label):
            point = sites[label]
            axis = sites[label + "_z"] - point
            return point, axis / np.linalg.norm(axis)

        # Árbol enraizado en la palma. Cada lazo añade una sola junta exterior al árbol.
        self.joint(key + "_base", palm, names[3], *hinge_frame("closing_1 (1)_1"))
        for parent, child, frame_index in ((3, 2, 3), (2, 4, 4), (4, 5, 5)):
            f = frames[frame_index]
            self.joint(key + f"_pasiva{child}", names[parent], names[child], f[:3,3]*.001, f[:3,2])
        for i, rod, number in ((0, 1, 1), (6, 7, 2)):
            # Coordenada de simulación relativa a la pose guardada en el USD.
            delta = (-33 if digit == "pulgar" else 57) * (1 if number == 1 else -1)
            profile = (min(0, delta), max(0, delta), .07, 90, 1.8, .035, "SCS0009")
            self.joint(key + f"_m{number}", palm, names[i], frames[i][:3,3]*.001, frames[i][:3,2],
                       profile=profile)
            self.joint(key + f"_rotula{number}", names[i], names[rod], frames[rod][:3,3]*.001, ball=True)
        self.joint(key + "_cierre1", names[1], names[2], frames[2][:3,3]*.001, ball=True, loop=True)
        self.joint(key + "_cierre2", names[7], names[2], sites["closing_ball1_2"], ball=True, loop=True)
        self.joint(key + "_cierre3", names[3], names[5], *hinge_frame("closing_3 (1)_1"), loop=True)
        # La palma y las dos almohadillas pertenecen al mismo mecanismo cerrado.
        for pair in ((palm, names[4]), (palm, names[5]), (names[4], names[5])):
            a, b = (self.stage.GetPrimAtPath(self.paths[n]) for n in pair)
            UsdPhysics.FilteredPairsAPI.Apply(a).CreateFilteredPairsRel().AddTarget(b.GetPath())

    def visuals(self, source):
        flat = source.Flatten()
        layer = self.stage.GetRootLayer()
        mat = ROOT + "/Materiales"
        Sdf.CopySpec(flat, mat, layer, mat)
        cache = UsdGeom.XformCache()
        for src in source.Traverse():
            if not src.IsA(UsdGeom.Mesh):
                continue
            parts = tuple(int(n) for n in src.GetAttribute("cad:ocurrencia").Get().split(".")[1:])
            prefix = parts
            while prefix not in self.mapping:
                prefix = prefix[:-1]
            name = self.mapping[prefix]
            path = self.paths[name] + "/Visual/p" + "_".join(map(str, parts))
            Sdf.CopySpec(flat, src.GetPath(), layer, path)
            mesh = UsdGeom.Mesh(self.stage.GetPrimAtPath(path))
            world = cache.GetLocalToWorldTransform(src)
            local = world * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*-self.origins[name]))
            mesh.MakeMatrixXform().Set(local)
            if any(parts[:len(p)] == p for p in self.flexible):
                mesh.CreatePurposeAttr("guide")
            if parts in self.pad_ids:
                # Envolvente convexa de la pieza exterior en coordenadas de su eslabón.
                shape = UsdGeom.Mesh.Define(self.stage, self.paths[name] + "/Contacto")
                for attr in ("points", "faceVertexCounts", "faceVertexIndices", "extent"):
                    shape.GetPrim().GetAttribute(attr).Set(src.GetAttribute(attr).Get())
                shape.AddTransformOp().Set(local)
                UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("convexHull")
                collider(shape.GetPrim())
                self.pads.append(parts)
        if set(self.pads) != self.pad_ids:
            raise ValueError("Faltan superficies de contacto de las manos")


@lru_cache(maxsize=2)
def finger_data(thumb):
    nodes = ET.parse(REF / "robot.xml").find("worldbody/body").findall("body")[:2]
    rest, _, _ = finger().solve((35, -35) if thumb else (-32, 32))
    return rest, [b for root in nodes for b in root.iter("body")]


def build_asset(directory):
    source = Usd.Stage.Open(str(CAD / "humanoide_xavier.usd"))
    if source.GetDefaultPrim().GetVariantSet("Vista").GetVariantSelection() != "Ensamblado":
        raise ValueError("Se requiere la variante CAD Ensamblado")
    rig = json.loads((CAD / "humanoide_xavier.params.js").read_text(encoding="utf8").split("const rig = ")[1].split(";")[0])
    stage = Usd.Stage.CreateInMemory()
    configure(stage)
    robot = Robot(stage)
    robot.trunk()
    for args in [(-1, "izq", 8, 7, 13, 11, 2, 1), (1, "der", 10, 9, 14, 12, 3, 2)]:
        robot.limb(*args)
        _, side, arm, *_ = args
        for i, pose in enumerate(rig["sides"][side]["digits"]):
            robot.digit(side, arm, i, pose, rig["groups"])
    robot.visuals(source)
    robot.root.CreateRelationship("isaac:physics:robotLinks", False).SetTargets(list(robot.paths.values()))
    robot.root.CreateRelationship("isaac:physics:robotJoints", False).SetTargets(robot.joints)
    stage.GetRootLayer().Export(str(directory / ASSET))


def build_bench(directory):
    stage = Usd.Stage.CreateInMemory()
    configure(stage)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    scene = UsdPhysics.Scene.Define(stage, "/World/Fisica")
    scene.CreateGravityDirectionAttr((0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    scene.GetPrim().AddAppliedSchema("PhysxSceneAPI")
    scene.GetPrim().CreateAttribute("physxScene:broadphaseType", T.Token, custom=False,
                                    variability=Sdf.VariabilityUniform).Set("MBP")
    prop(scene.GetPrim(), "physxScene:enableGPUDynamics", T.Bool, False)
    prop(scene.GetPrim(), "physxScene:timeStepsPerSecond", T.UInt, 240)
    prop(scene.GetPrim(), "physxScene:solverType", T.Token, "TGS")
    floor = box(stage, "/World/Suelo", (0, 0, -.035), (4, 4, .07))
    floor.CreatePurposeAttr("default")
    floor.CreateDisplayColorAttr([(.16, .18, .20)])
    dome = UsdLux.DomeLight.Define(stage, "/World/Luz")
    dome.CreateIntensityAttr(750)
    key = UsdLux.DistantLight.Define(stage, "/World/LuzPrincipal")
    key.CreateIntensityAttr(2000)
    key.AddRotateXYZOp().Set((35, -40, 0))
    # Author the reference after exporting so its relative path resolves from the bench.
    file = directory / "banco_xavier.usda"
    stage.GetRootLayer().Export(str(file))
    stage = Usd.Stage.Open(str(file))
    robot = UsdGeom.Xform.Define(stage, "/World/Humanoide")
    robot.GetPrim().GetReferences().AddReference("./" + ASSET)
    robot.AddTranslateOp().Set((0, 0, .12))
    fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Humanoide/Anclaje")
    fixed.CreateBody1Rel().SetTargets(["/World/Humanoide/pelvis"])
    fixed.CreateLocalPos0Attr((0, 0, 1.29))
    fixed.CreateLocalPos1Attr((0, 0, 0))
    # Una base fija lleva la raíz en su junta al mundo. El activo libre la lleva en la pelvis.
    pelvis = stage.GetPrimAtPath("/World/Humanoide/pelvis")
    pelvis.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    pelvis.RemoveAppliedSchema("PhysxArticulationAPI")
    root = fixed.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    root.AddAppliedSchema("PhysxArticulationAPI")
    for attr in ("solverPositionIterationCount", "solverVelocityIterationCount", "enabledSelfCollisions"):
        old = pelvis.GetAttribute("physxArticulation:" + attr)
        prop(root, old.GetName(), old.GetTypeName(), old.Get())
    stage.GetRootLayer().customLayerData = {"cameraSettings": {"Perspective": {
        "position": Gf.Vec3d(1.7, 2.9, 1.72), "target": Gf.Vec3d(0, 0, .98)}}}
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        directory = Path(temp)
        build_asset(directory)
        build_bench(directory)
        for filename in (ASSET, "banco_xavier.usda"):
            os.replace(directory / filename, HERE / filename)
    print(HERE / ASSET)
