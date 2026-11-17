"""Check downloaded bytes, all USD dependencies, joint topology, and mount frames offline."""
import ast
import hashlib
import json
from pathlib import Path

from usd_python import configure_usd

configure_usd()
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def validate():
    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    for relative, entry in manifest.items():
        path = ROOT / "assets/vendor" / relative
        assert path.stat().st_size == entry["bytes"], f"Size mismatch: {path}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"Hash mismatch: {path}"
    stage = Usd.Stage.Open(str(ROOT / "assets/robots/ur10e_shadow/robot.usda"))
    assert not stage.GetCompositionErrors(), stage.GetCompositionErrors()
    roots, movable, edges, bodies = [], [], [], set()
    hand_drives = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.add(prim.GetPath())
        if prim.IsA(UsdPhysics.RevoluteJoint):
            movable.append(prim.GetName())
            if str(prim.GetPath()).startswith("/Robot/Hand/"):
                assert prim.HasAPI(UsdPhysics.DriveAPI, "angular"), f"Missing hand motor: {prim.GetPath()}"
                hand_drives += 1
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            targets = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            assert targets[1], f"Dangling joint body1: {prim.GetPath()}"
            for group in targets:
                assert len(group) <= 1
                for target in group:
                    assert stage.GetPrimAtPath(target), f"Missing body: {target}"
            edges.append((targets[0][0] if targets[0] else None, targets[1][0]))
    assert roots == ["/Robot/Arm/root_joint"], roots
    assert len(movable) == len(set(movable)) == 30
    assert hand_drives == 24
    assert len(edges) == len(bodies), "Expected one fixed world edge + a tree of rigid bodies"
    connected = {None}
    while True:
        previous = len(connected)
        for first, second in edges:
            if first in connected:
                connected.add(second)
        if previous == len(connected):
            break
    assert bodies <= connected, f"Disconnected bodies: {bodies - connected}"
    mount = UsdPhysics.Joint(stage.GetPrimAtPath("/Robot/hand_attachment"))
    frames = []
    for body, pos, rot in [(mount.GetBody0Rel(), mount.GetLocalPos0Attr(), mount.GetLocalRot0Attr()),
                            (mount.GetBody1Rel(), mount.GetLocalPos1Attr(), mount.GetLocalRot1Attr())]:
        from pxr import Gf
        local = Gf.Matrix4d(1)
        local.SetRotate(Gf.Quatd(rot.Get()))
        local.SetTranslateOnly(Gf.Vec3d(pos.Get()))
        world = UsdGeom.Xformable(stage.GetPrimAtPath(body.GetTargets()[0])).ComputeLocalToWorldTransform(0)
        frames.append(local * world)
    frame_error = max(abs(frames[0][i][j] - frames[1][i][j]) for i in range(4) for j in range(4))
    assert frame_error < 1e-5, f"Mount frame mismatch: {frame_error}"
    files = [p for folder in ("src", "scripts", "tools", "tests") for p in (ROOT / folder).rglob("*.py")]
    for file in files:
        ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    output = ROOT / "docs/validation.json"
    result = json.loads(output.read_text()) if output.exists() else {}
    result.update({"asset_files_verified": len(manifest), "movable_joints": len(movable), "hand_angular_drives": hand_drives,
        "rigid_bodies": len(bodies), "articulation_roots": roots, "connected_tree": True,
        "mount_frame_max_error": frame_error, "python_files_parsed": len(files),
        "static_validation_only": True})
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    validate()
