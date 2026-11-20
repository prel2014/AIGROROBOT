"""Compose a fixed-base arm/hand articulation using static OpenUSD only."""
import json
import math
from pathlib import Path

from usd_python import configure_usd

configure_usd()
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def world(prim):
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def flange_adapter(stage, mount):
    """Cover the 40 mm between the UR flange face and the Shadow forearm mesh."""
    root = mount.GetPath().AppendChild("FlangeAdapter")
    UsdGeom.Xform.Define(stage, root)

    def cylinder(name, center, radius, height, color, *, collides=False):
        part = UsdGeom.Cylinder.Define(stage, root.AppendChild(name))
        part.CreateAxisAttr(UsdGeom.Tokens.z)
        part.CreateRadiusAttr(radius)
        part.CreateHeightAttr(height)
        part.AddTranslateOp().Set(Gf.Vec3d(*center))
        part.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if collides:
            UsdPhysics.CollisionAPI.Apply(part.GetPrim())

    # All three pieces belong to the existing 100 g hand-mount rigid body.
    cylinder("WristFlange", (0., 0., .0035), .035, .009, (.34, .39, .43), collides=True)
    cylinder("Sleeve", (0., 0., .022), .031, .030, (.095, .13, .15), collides=True)
    cylinder("ForearmFlange", (0., .009, .038), .051, .008, (.34, .39, .43), collides=True)
    for n in range(4):
        angle = (n + .5) * math.pi / 2
        center = (.042 * math.cos(angle), .009 + .042 * math.sin(angle), .0335)
        cylinder(f"Screw{n+1}", center, .0034, .0025, (.06, .07, .08))


def build():
    output = ROOT / "assets/robots/ur10e_shadow/robot.usda"
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    robot = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    stage.SetDefaultPrim(robot)
    for name, asset in [("Arm", "UniversalRobots/ur10e/ur10e.usd"),
                        ("Hand", "ShadowRobot/ShadowHand/shadow_hand.usd")]:
        prim = stage.DefinePrim(f"/Robot/{name}", "Xform")
        prim.GetReferences().AddReference(f"../../vendor/Isaac/Robots/{asset}")
    wrist = stage.GetPrimAtPath("/Robot/Arm/wrist_3_link")
    mount = stage.GetPrimAtPath("/Robot/Hand/robot0_hand_mount")
    ee = UsdPhysics.Joint(stage.GetPrimAtPath("/Robot/Arm/joints/ee_joint"))
    flange = Gf.Matrix4d(1)
    flange.SetRotate(Gf.Quatd(ee.GetLocalRot0Attr().Get()))
    flange.SetTranslateOnly(Gf.Vec3d(ee.GetLocalPos0Attr().Get()))
    # The fixed joint aligns the coordinate frames. The sleeve spans the mesh gap.
    desired = flange * world(wrist)
    hand = UsdGeom.Xformable(stage.GetPrimAtPath("/Robot/Hand"))
    mount_to_root = world(mount) * world(hand.GetPrim()).GetInverse()
    hand.ClearXformOpOrder()
    hand.AddTransformOp().Set(mount_to_root.GetInverse() * desired)
    stage.GetPrimAtPath("/Robot/Arm/joints/ee_joint").SetActive(False)
    stage.GetPrimAtPath("/Robot/Hand/joints/rootJoint").SetActive(False)
    # The arm's fixed root is the only articulation root of the combined tree.
    for prim in list(Usd.PrimRange(stage.GetPrimAtPath("/Robot/Hand"))):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        for schema in prim.GetAppliedSchemas():
            if schema == "PhysxArticulationAPI":
                prim.RemoveAppliedSchema(schema)
    joint = UsdPhysics.FixedJoint.Define(stage, "/Robot/hand_attachment")
    joint.CreateBody0Rel().SetTargets([wrist.GetPath()])
    joint.CreateBody1Rel().SetTargets([mount.GetPath()])
    joint.CreateLocalPos0Attr().Set(ee.GetLocalPos0Attr().Get())
    joint.CreateLocalRot0Attr().Set(ee.GetLocalRot0Attr().Get())
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))
    joint.CreateCollisionEnabledAttr().Set(False)
    flange_adapter(stage, mount)
    # Ignore adjacent adapter bodies in contact generation.
    UsdPhysics.FilteredPairsAPI.Apply(wrist).CreateFilteredPairsRel().SetTargets([
        Sdf.Path("/Robot/Hand/robot0_hand_mount"), Sdf.Path("/Robot/Hand/robot0_forearm")])
    # A joint's source physics limits remain authoritative for the action decoder.
    movable = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(prim)
            if str(prim.GetPath()).startswith("/Robot/Hand/"):
                schemas = prim.GetMetadata("apiSchemas")
                for schema in schemas.GetAddedOrExplicitItems() if schemas else []:
                    if schema.startswith("PhysxTendonAxis"):
                        # These legacy tendons have zero stiffness/damping. This
                        # composition uses explicit software position synergies.
                        prim.RemoveAppliedSchema(schema)
                # The legacy Shadow USD has no angular drives. PhysX needs the
                # DriveAPI authored before articulation creation; writing gains
                # through the tensor API alone does not create the missing drives.
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateTypeAttr().Set("force")
                drive.CreateStiffnessAttr().Set(2.0)
                drive.CreateDampingAttr().Set(0.15)
                drive.CreateMaxForceAttr().Set(2.0)
                drive.CreateTargetPositionAttr().Set(0.0)
                drive.CreateTargetVelocityAttr().Set(0.0)
            movable.append({"name": prim.GetName(), "path": str(prim.GetPath()),
                "lower_rad": math.radians(j.GetLowerLimitAttr().Get()),
                "upper_rad": math.radians(j.GetUpperLimitAttr().Get())})
    assert len(movable) == 30, f"Expected 6 arm + 24 hand joints, got {len(movable)}"
    error = (world(mount).ExtractTranslation() - desired.ExtractTranslation()).GetLength()
    assert error < 1e-6, f"Mount translation mismatch: {error}"
    robot.SetCustomDataByKey("project:mountAssumption", "aligned frames with a 40 mm flange adapter on the hand mount")
    robot.SetCustomDataByKey("project:handControl", "angular drives with software finger synergies")
    stage.GetRootLayer().Save()
    output.write_text(output.read_text(encoding="utf-8").rstrip()+"\n", encoding="utf-8", newline="\n")
    (output.parent / "joint_manifest.json").write_text(json.dumps(movable, indent=2) + "\n")
    print(f"Built {output}; 30 joints; mount residual {error:.3g} m")


if __name__ == "__main__":
    build()
