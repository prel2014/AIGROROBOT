"""Compute fruit clearance from the authored collider meshes and live rigid-body poses."""
from functools import cache

import numpy as np
import torch

from dexman.paths import ROOT


@cache
def fruit_clearance_points(fruit: str) -> np.ndarray:
    """Cache a fruit asset's enabled mesh collider vertices in root-link coordinates.

    The vertices give the authored geometry's support in any direction. PhysX
    cooking, particularly convex decomposition of the banana, can slightly
    change the contact surface relative to these source meshes.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    usd_path = ROOT / "assets" / "agriculture" / "props" / f"{fruit}.usda"
    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim()
    transforms = UsdGeom.XformCache()
    world_to_root = transforms.GetLocalToWorldTransform(root).GetInverse()
    vertices = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        collision = UsdPhysics.CollisionAPI(prim)
        if not collision or not collision.GetCollisionEnabledAttr().Get():
            continue
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Expected a mesh collider at {prim.GetPath()} in {usd_path}")
        collider_to_root = transforms.GetLocalToWorldTransform(prim) * world_to_root
        mesh_points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        vertices.extend(collider_to_root.Transform(point) for point in mesh_points)
    if not vertices:
        raise ValueError(f"No enabled mesh collider vertices in {usd_path}")
    points = np.asarray(vertices, dtype=np.float32)
    points.setflags(write=False)
    return points


def lowest_collider_height(
    position_w: torch.Tensor, quaternion_w: torch.Tensor, local_points: torch.Tensor
) -> torch.Tensor:
    """Return each collider's minimum world Z for root-link poses [N,3] and [N,4].

    Quaternions are unit XYZW. All N replicas share the local vertices [M,3].
    Projecting onto world up in the root frame avoids allocating [N,M,3].
    """
    x, y, z, w = quaternion_w.unbind(-1)
    up_in_link = torch.stack(
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x.square() + y.square())), dim=-1
    )
    return position_w[:, 2] + (up_in_link @ local_points.T).amin(dim=-1)
