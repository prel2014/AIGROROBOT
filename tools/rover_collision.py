"""Self-contact policy for the assembled mobile articulation."""
from pxr import Sdf, Usd, UsdPhysics


def configure_rover_collision(stage, assembly):
    """Enable arm/frame contacts at the rover root and preserve the source joint filters."""
    for child in Usd.PrimRange(assembly):
        # Inspect USD metadata even when generating offline without the PhysX plugin.
        schemas = child.GetMetadata("apiSchemas")
        if child != assembly and schemas and "PhysxArticulationAPI" in schemas.GetAppliedItems():
            child.RemoveAppliedSchema("PhysxArticulationAPI")

    assembly.AddAppliedSchema("PhysxArticulationAPI")
    assembly.CreateAttribute("physxArticulation:enabledSelfCollisions",
                             Sdf.ValueTypeNames.Bool, custom=False).Set(True)
    for side in ("Left", "Right"):
        path = assembly.GetPath().AppendPath(f"{side}Arm/Arm/root_joint")
        joint = UsdPhysics.FixedJoint.Get(stage, path)
        # Only the two bodies bolted together at this mount are adjacent.
        joint.CreateCollisionEnabledAttr(False)
