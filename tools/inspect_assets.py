"""Static USD inspection; never starts simulation."""
import json
from pathlib import Path

from usd_python import configure_usd

configure_usd()
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def inspect(path):
    stage = Usd.Stage.Open(str(path))
    result = {"file": str(path.relative_to(ROOT)), "default_prim": str(stage.GetDefaultPrim().GetPath()),
              "joints": [], "bodies": [], "articulation_roots": []}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            result["articulation_roots"].append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            result["bodies"].append({"path": str(prim.GetPath()), "translation": list(matrix.ExtractTranslation())})
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            result["joints"].append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                "body0": [str(p) for p in joint.GetBody0Rel().GetTargets()],
                "body1": [str(p) for p in joint.GetBody1Rel().GetTargets()],
                "lower": prim.GetAttribute("physics:lowerLimit").Get(),
                "upper": prim.GetAttribute("physics:upperLimit").Get()})
    return result


if __name__ == "__main__":
    assets = ROOT / "assets/vendor/Isaac/Robots"
    report = [inspect(assets / p) for p in ["UniversalRobots/ur10e/ur10e.usd", "ShadowRobot/ShadowHand/shadow_hand.usd"]]
    output = ROOT / "assets/source_inspection.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
