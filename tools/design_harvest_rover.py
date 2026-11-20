"""Genera el anteproyecto de surco con OpenUSD del Python de Isaac Sim."""
import math
import os
import random
import sys
from pathlib import Path
from runpy import run_path

from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from rover_collision import configure_rover_collision


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dexman.robot.joints import ARM
from dexman.tasks.agriculture.mobile_state import HEAD_JOINTS, HEAD_LIMITS_DEG

DESIGN = ROOT / "assets/agriculture/robot/rover"
T = Sdf.ValueTypeNames
WORLD = "/Surco"
BASE_HEIGHT = .015


def schemas(prim, *names):
    op = Sdf.TokenListOp()
    op.prependedItems = list(names)
    prim.SetMetadata("apiSchemas", op)


def attr(prim, name, kind, value):
    prim.CreateAttribute(name, kind, custom=False).Set(value)


def move(prim, xyz):
    return UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*xyz))


def color(prim, rgb):
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def paint(geom, hex_color):
    color(geom, tuple((int(hex_color[i:i+2], 16)/255)**2.2 for i in (1, 3, 5)))


def meters(point_mm):
    return Gf.Vec3d(*(value/1000 for value in point_mm))


def friction(stage, path, static, dynamic, restitution=0.01):
    mat = UsdShade.Material.Define(stage, path)
    physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics.CreateStaticFrictionAttr(static)
    physics.CreateDynamicFrictionAttr(dynamic)
    physics.CreateRestitutionAttr(restitution)
    return mat


def bind(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")


def reference(stage, path, asset, xyz):
    node = UsdGeom.Xform.Define(stage, path)
    node.GetPrim().GetReferences().AddReference(os.path.relpath(asset, DESIGN).replace("\\", "/"))
    move(node, xyz)
    return node


def ground_z(x, y):
    # Caballones longitudinales y ondulación suave. Inicio llano para asentar las ruedas.
    irregular = .025 * math.sin(3*x + y) + .018 * math.sin(5*y - 1.7*x)
    ridge = .21 * math.exp(-((abs(y)-1.7)/.35)**2)
    return ridge + min(1., max(0., abs(x)-.8)) * irregular


def terrain(stage):
    mesh = UsdGeom.Mesh.Define(stage, WORLD + "/TierraFirme")
    points, faces = [], []
    ny = 53
    for i in range(121):
        for j in range(ny):
            x, y = -3 + i*.1, -2.6+j*.1
            points.append((x, y, ground_z(x, y)))
            if i and j:
                k = len(points)-1
                faces.extend([k-ny-1, k-1, k, k-ny])
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4]*(len(faces)//4))
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateSubdivisionSchemeAttr("none")
    color(mesh, (.30, .18, .085))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    bind(mesh.GetPrim(), friction(stage, WORLD + "/Materials/Firme", .75, .58))
    subsoil = UsdGeom.Cube.Define(stage, WORLD + "/Subsuelo")
    subsoil.CreateSizeAttr(1.)
    move(subsoil, (3, 0, -.23))
    subsoil.AddScaleOp().Set(Gf.Vec3f(12.4, 5.6, .34))
    UsdPhysics.CollisionAPI.Apply(subsoil.GetPrim())
    color(subsoil, (.23, .12, .06))

    # Capa granular sobre un tramo, con fondo rígido bajo los granos.
    system = stage.DefinePrim(WORLD + "/SueloGranular", "PhysxParticleSystem")
    system.CreateRelationship("simulationOwner", False).SetTargets([WORLD + "/Physics"])
    for name, value in (("restOffset", .012), ("solidRestOffset", .012),
                        ("contactOffset", .016), ("particleContactOffset", .016)):
        attr(system, name, T.Float, value)
    mat = UsdShade.Material.Define(stage, WORLD + "/Materials/TierraSuelta")
    schemas(mat.GetPrim(), "PhysxPBDMaterialAPI")
    attr(mat.GetPrim(), "physxPBDMaterial:friction", T.Float, .55)
    attr(system, "enableCCD", T.Bool, True)
    attr(system, "maxVelocity", T.Float, 2.5)
    attr(system, "solverPositionIterationCount", T.Int, 24)
    bind(system, mat)
    grains = UsdGeom.Points.Define(stage, WORLD + "/Terrones")
    prim = grains.GetPrim()
    schemas(prim, "PhysxParticleSetAPI")
    prim.CreateRelationship("physxParticle:particleSystem", False).SetTargets([system.GetPath()])
    attr(prim, "physxParticle:fluid", T.Bool, False)
    attr(prim, "physxParticle:selfCollision", T.Bool, True)
    rng = random.Random(41)
    centers = []
    for k in range(3):
        for i in range(48):
            for j in range(60):
                x = 1.4 + .025*i + rng.uniform(-.0004,.0004)
                y = -.75 + .025*j + rng.uniform(-.0004,.0004)
                centers.append((x, y, ground_z(x,y)+.018+.025*k))
    grains.CreatePointsAttr(centers)
    grains.CreateVelocitiesAttr([Gf.Vec3f(0.)] * len(centers))
    grains.CreateWidthsAttr([.024])
    grains.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    color(grains, (.35, .21, .09))
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(len(centers)*.009)


def rear_frame(stage, base_path, tubes):
    """Los tubos comparten el cuerpo rígido del chasis y tienen colisión exterior."""
    for tube in tubes:
        mesh = UsdGeom.Mesh.Define(stage, base_path + "/Refuerzos/" + tube.name)
        axes = tube.axes
        points, faces = [], []
        for end in (tube.start,tube.end):
            for width in (tube.width, tube.width-2*tube.wall):
                for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]:
                    points.append(tuple((end[j]+width/2*(a*axes[0][j]+b*axes[1][j]))/1000
                                        for j in range(3)))
        for i in range(4):
            j = (i+1)%4
            faces.extend([(i,j,j+8,i+8), (i+4,i+12,j+12,j+4),
                          (i,i+4,j+4,j), (i+8,j+8,j+12,i+12)])
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4]*len(faces))
        mesh.CreateFaceVertexIndicesAttr([idx for face in faces for idx in face])
        mesh.CreateSubdivisionSchemeAttr("none")
        color(mesh, (.12,.21,.18))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")


def add_wheels(stage, root, base_path, layout):
    rubber = friction(stage, root + "/Materials/Neumatico", .92, .73)
    geometry = {side: tuple(layout["wheel_meshes"](side)) for side in (1, -1)}
    for suffix, center_mm in layout["wheel_centers"]():
        center = meters(center_mm)
        path = root + "/Wheel_" + suffix
        wheel = UsdGeom.Xform.Define(stage, path)
        move(wheel, center + Gf.Vec3d(0, 0, BASE_HEIGHT))
        UsdPhysics.RigidBodyAPI.Apply(wheel.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(wheel.GetPrim())
        mass.CreateMassAttr(8.)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.))
        joint = UsdPhysics.RevoluteJoint.Define(stage, root + "/drive_" + suffix)
        joint.CreateBody0Rel().SetTargets([base_path])
        joint.CreateBody1Rel().SetTargets([path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(center))
        joint.CreateAxisAttr("Y")
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateDampingAttr(25.)
        drive.CreateMaxForceAttr(85.)
        drive.CreateTargetVelocityAttr(0.)
        for part in geometry[1 if center[1] > 0 else -1]:
            mesh = UsdGeom.Mesh.Define(stage, path + "/" + Tf.MakeValidIdentifier(part.name))
            mesh.CreatePointsAttr([Gf.Vec3f(meters(p)) for p in part.vertices])
            mesh.CreateFaceVertexCountsAttr([len(face) for face in part.faces])
            mesh.CreateFaceVertexIndicesAttr([i for face in part.faces for i in face])
            mesh.CreateSubdivisionSchemeAttr("none")
            paint(mesh, part.color)
            if part.collision:
                prim = mesh.GetPrim()
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
                prim.AddAppliedSchema("PhysxCollisionAPI")
                attr(prim, "physxCollision:contactOffset", T.Float, .003)
                bind(prim, rubber)


def add_camera_head(stage, root, base_path, layout):
    origins = {"HeadPan": meters(layout["HEAD_PAN_PIVOT"]), "HeadTilt": meters(layout["HEAD_TILT_PIVOT"])}
    # Masas nominales (kg). PhysX calcula las inercias a partir de los colisionadores del CAD.
    # Los drives de fuerza USD usan N·m/grado y N·m/(grado/s).
    parent, parent_origin = base_path, Gf.Vec3d(0.)
    axes = (("HeadPan", "Z", 1.2, 1.5, .12, 7., 55.),
            ("HeadTilt", "Y", 2.4, .85, .06, 5., 40.))
    for name, limits, settings in zip(HEAD_JOINTS, HEAD_LIMITS_DEG, axes, strict=True):
        link, axis, kg, stiffness, damping, torque, speed = settings
        path = root + "/" + link
        body = UsdGeom.Xform.Define(stage, path)
        origin = origins[link]
        move(body, origin + Gf.Vec3d(0, 0, BASE_HEIGHT))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(kg)
        joint = UsdPhysics.RevoluteJoint.Define(stage, root + "/" + name)
        joint.CreateBody0Rel().SetTargets([parent])
        joint.CreateBody1Rel().SetTargets([path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(origin-parent_origin))
        joint.CreateAxisAttr(axis)
        if axis == "Y":
            # El eje Y del marco articular apunta hacia la derecha: tilt positivo levanta la vista.
            reverse_y = Gf.Quatf(0., Gf.Vec3f(0, 0, 1))
            joint.CreateLocalRot0Attr(reverse_y)
            joint.CreateLocalRot1Attr(reverse_y)
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateLowerLimitAttr(limits[0])
        joint.CreateUpperLimitAttr(limits[1])
        joint.GetPrim().AddAppliedSchema("PhysxJointAPI")
        attr(joint.GetPrim(), "physxJoint:maxJointVelocity", T.Float, speed)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateMaxForceAttr(torque)
        drive.CreateTargetPositionAttr(0.)
        drive.CreateTargetVelocityAttr(0.)
        parent, parent_origin = path, origin

    ex, ey, ez = layout["EYE"]
    for side, sign in (("Left", 1), ("Right", -1)):
        eye = UsdGeom.Camera.Define(stage, root + "/HeadTilt/" + side + "Eye")
        location = meters((ex, sign*ey, ez)) - origins["HeadTilt"]
        axis = Gf.Vec3d(*layout["eye_axis"](sign))
        aim = Gf.Matrix4d().SetLookAt(location, location+axis, Gf.Vec3d(0,0,1))
        eye.AddTransformOp().Set(aim.GetInverse())
        eye.CreateFocalLengthAttr(12.)
        eye.CreateHorizontalApertureAttr(20.955)
        eye.CreateVerticalApertureAttr(20.955)
        eye.CreateClippingRangeAttr((.055, 18.))
        schemas(eye.GetPrim(), "OmniSensorAPI")
    return origins


def rover(stage):
    root = "/Rover"
    base_path = root + "/Base"
    assembly = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(assembly)
    UsdPhysics.ArticulationRootAPI.Apply(assembly)
    base = UsdGeom.Xform.Define(stage, base_path)
    move(base, (0, 0, BASE_HEIGHT))
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(base.GetPrim())
    layout = run_path(str(DESIGN/"layout.py"))
    tubes = layout["rear_frame"]()
    rear_frame(stage, base_path, tubes)
    liner = friction(stage, root + "/Materials/Revestimiento", .65, .48, restitution=.02)
    parts_mass = [(135., (-70., 0., 460.))]
    parts_mass.extend((tube.mass_kg, tube.center) for tube in tubes)
    for plate in layout["plates"]():
        mesh = UsdGeom.Mesh.Define(stage, base_path + "/" + plate.name)
        n = len(plate.outline)
        faces = [list(range(n)), list(reversed(range(n, 2*n)))]
        for i in range(n):
            j = (i+1) % n
            faces.append([i, i+n, j+n, j])
        mesh.CreatePointsAttr([Gf.Vec3f(*(v/1000 for v in point)) for point in plate.vertices()])
        mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
        mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
        mesh.CreateSubdivisionSchemeAttr("none")
        paint(mesh, plate.color)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        if plate.name.startswith(("Receptor_", "Lecho_")):
            bind(mesh.GetPrim(), liner)
        parts_mass.append(plate.added_mass_and_center())
    total_mass = sum(kg for kg, _ in parts_mass)
    com = [sum(kg*center[j] for kg, center in parts_mass)/(total_mass*1000) for j in range(3)]
    mass.CreateMassAttr(total_mass)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    add_wheels(stage, root, base_path, layout)
    head_origins = add_camera_head(stage, root, base_path, layout)
    for name, (kind, dimensions), mm, shade in layout["shapes"]():
        pos = meters(mm)
        link = layout["head_link"](name)
        owner = base_path
        if link:
            owner = root + "/" + link
            pos -= head_origins[link]
        path = owner + "/" + Tf.MakeValidIdentifier(name)
        size = [v/1000 for v in dimensions]
        if kind == "box":
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(1.)
        else:
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateAxisAttr(kind)
            geom.CreateRadiusAttr(size[0])
            geom.CreateHeightAttr(size[1])
        move(geom, pos)
        angles = layout["eye_housing_angles"](name)
        if angles:
            yaw, pitch = angles
            geom.AddRotateZOp().Set(yaw)
            geom.AddRotateYOp().Set(pitch)
        if kind == "box":
            geom.AddScaleOp().Set(Gf.Vec3f(*size))
        paint(geom, shade)
        if link or name.startswith(("Chasis", "Fondo", "Testero", "Placa", "Mastil", "Soporte_",
                                    "Cubierta", "Cuello_estator", "Carcasa_lidar", "Tapa_lidar")):
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    for side, sign in (("Left", 1), ("Right", -1)):
        sx, sy, sz = layout["SHOULDER"]
        pos = (sx/1000, sign*sy/1000, sz/1000)
        arm = reference(stage, root + "/" + side + "Arm",
                        ROOT / "assets/robots/ur10e_shadow/robot.usda", (pos[0],pos[1],pos[2]+BASE_HEIGHT))
        yaw_deg = sign*12.
        arm.AddRotateZOp().Set(yaw_deg)
        for joint_name, angle in zip(ARM, layout["ROVER_TRAVEL_DEG"][side], strict=True):
            joint = UsdPhysics.RevoluteJoint.Get(stage, arm.GetPath().AppendPath("Arm/joints/" + joint_name))
            # El estado inicial y su consigna coinciden al dar Play en el surco.
            joint.GetPrim().AddAppliedSchema("PhysicsJointStateAPI:angular")
            attr(joint.GetPrim(), "state:angular:physics:position", T.Float, angle)
            attr(joint.GetPrim(), "state:angular:physics:velocity", T.Float, 0.)
            UsdPhysics.DriveAPI.Get(joint.GetPrim(), "angular").CreateTargetPositionAttr(angle)
            if side == "Left" and joint_name in layout["LEFT_OPERATING_LIMITS_DEG"]:
                lower, upper = layout["LEFT_OPERATING_LIMITS_DEG"][joint_name]
                joint.CreateLowerLimitAttr(lower)
                joint.CreateUpperLimitAttr(upper)
        anchor = UsdPhysics.FixedJoint(stage.GetPrimAtPath(arm.GetPath().AppendPath("Arm/root_joint")))
        anchor.GetPrim().RemoveAPI(UsdPhysics.ArticulationRootAPI)
        anchor.CreateBody0Rel().SetTargets([base_path])
        anchor.CreateLocalPos0Attr(Gf.Vec3f(*pos))
        rot = Gf.Rotation(Gf.Vec3d(0,0,1), yaw_deg).GetQuat()
        anchor.CreateLocalRot0Attr(Gf.Quatf(rot))
    imu = stage.DefinePrim(base_path+"/IMU", "IsaacImuSensor")
    move(imu, (0,0,.36))
    lidar = stage.DefinePrim(base_path+"/Lidar", "OmniLidar")
    schemas(lidar, "OmniSensorGenericLidarCoreAPI")
    move(lidar, tuple(value/1000 for value in layout["LIDAR"]))
    configure_rover_collision(stage, assembly)


def hang_target_grapes(stage, plant, fruit_path, asset, sign):
    # Las vides de ambos lados presentan el racimo maduro hacia el pasillo.
    if sign < 0:
        plant.AddRotateZOp().Set(180.)
    anchor = plant.GetPath().AppendPath("Visual/Geometry/Cosecha/PedunculoObjetivo/Racimo")
    decorative = stage.GetPrimAtPath(anchor)
    if not decorative:
        raise ValueError(f"Falta el racimo de cosecha en {anchor}")
    pose = UsdGeom.XformCache().GetLocalToWorldTransform(decorative)
    stage.OverridePrim(anchor).SetActive(False)
    target = UsdGeom.Xform.Define(stage, fruit_path)
    target.GetPrim().GetReferences().AddReference(os.path.relpath(asset, DESIGN).replace("\\", "/"))
    parent = UsdGeom.XformCache().GetLocalToWorldTransform(target.GetPrim().GetParent())
    target.AddTransformOp().Set(pose * parent.GetInverse())


def build_field(robot_file):
    stage = Usd.Stage.CreateNew(str(DESIGN / "surco.usdc"))
    world = UsdGeom.Xform.Define(stage, WORLD)
    stage.SetDefaultPrim(world.GetPrim())
    stage.SetTimeCodesPerSecond(60)
    stage.SetEndTimeCode(36000)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    scene = UsdPhysics.Scene.Define(stage, WORLD + "/Physics")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
    scene.CreateGravityMagnitudeAttr(9.81)
    schemas(scene.GetPrim(), "PhysxSceneAPI")
    attr(scene.GetPrim(), "physxScene:enableGPUDynamics", T.Bool, True)
    terrain(stage)
    reference(stage, WORLD + "/Rover", robot_file, (0,0,0))
    props = ROOT / "assets/agriculture/props"
    for n, (plant, fruit) in enumerate([("orange_tree.usdc","orange"), ("banana_plant.usda","banana"),
                                        ("vineyard.usda","grapes"), ("berry_bush.usda","blueberry")]):
        for side, sign in (("Izq",1), ("Der",-1)):
            x = .2 + 2.15*n
            y = (2.55 if fruit == "banana" else 1.7)*sign
            z = ground_z(x, y)
            if fruit == "grapes":
                # La espaldera abarca cuatro metros sobre un caballón ondulado.
                z = min(ground_z(x + i*.1, y) for i in range(-21, 22))
            vine_or_tree = reference(stage, WORLD + f"/Plantas/{side}{n}", props/plant,
                                     (x, y, z))
            fruit_path = WORLD + f"/Frutos/{side}{n}"
            asset = props / (fruit + ".usda")
            if fruit == "grapes":
                hang_target_grapes(stage, vine_or_tree, fruit_path, asset, sign)
            else:
                reference(stage, fruit_path, asset,
                          (x, (1.30 if fruit == "banana" else 1.28)*sign, (.98, 1.38, 1.15, .72)[n]))
    sky = UsdLux.DomeLight.Define(stage, WORLD+"/Cielo")
    sky.CreateIntensityAttr(650.)
    sun = UsdLux.DistantLight.Define(stage, WORLD+"/Sol")
    sun.CreateIntensityAttr(2100.)
    sun.AddRotateXYZOp().Set((-35., -45., 20.))
    stage.GetRootLayer().Save()


def main():
    robot_file = DESIGN / "rover.usdc"
    assembly = Usd.Stage.CreateNew(str(robot_file))
    UsdGeom.SetStageUpAxis(assembly, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(assembly, 1.)
    rover(assembly)
    assembly.GetRootLayer().Save()
    build_field(robot_file)


if __name__ == "__main__":
    main()
