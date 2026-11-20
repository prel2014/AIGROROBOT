"""Filtered arm impacts against fixtures, obstacles, and the opposite arm."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from dexman.paths import ROBOT_USD
from .interface import ARM_KEYS
from .randomization import OBSTACLE_RADIUS


@dataclass(frozen=True)
class ImpactMeasurement:
    """Peak normal force in newtons for each replica over the control interval."""

    environment_n: torch.Tensor
    inter_arm_n: torch.Tensor


@cache
def _collider_paths(usd_path: str, *, rigid_links: bool = False) -> tuple[str, ...]:
    """Resolve exact collider paths, or the rigid links that own those colliders."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    root_path = root.GetPath().pathString
    paths = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        collision = UsdPhysics.CollisionAPI(prim)
        if not collision or not collision.GetCollisionEnabledAttr().Get():
            continue
        if rigid_links:
            while prim and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim = prim.GetParent()
            if not prim:
                raise ValueError(f"Collider without a rigid link in {usd_path}")
        paths[prim.GetPath().pathString[len(root_path):]] = None
    return tuple(paths)


def configure_impact_sensors(scene_cfg, crop: str, *, history_length: int) -> None:
    """Add two grouped contact views after the crop has populated its scene."""
    from isaaclab.sensors import ContactSensorCfg
    from .scene import arm_path

    if history_length < 1:
        raise ValueError("Impact history must store at least one physics substep")

    fixtures = [scene_cfg.worktop, *(getattr(scene_cfg, f"leg_{i}") for i in range(4))]
    fixtures.extend(asset for name, asset in vars(scene_cfg).items() if name.startswith("pickup_support_"))
    if crop == "berries":
        fixtures.append(scene_cfg.planter)
    environment = [f"{item.prim_path}/geometry/mesh" for item in fixtures]
    plant = scene_cfg.plant
    # The orange branches cook to two entries per replica under one prim.
    if crop != "citrus":
        environment.extend(plant.prim_path + path for path in _collider_paths(plant.spawn.usd_path))
    environment.extend(getattr(scene_cfg, f"obstacle_{kind}").prim_path for kind in OBSTACLE_RADIUS)

    robot_links = _arm_links()
    names = "|".join(re.escape(path.rsplit("/", 1)[-1]) for path in robot_links)
    right_arm_path = arm_path(scene_cfg, "right_arm")
    opposing_links = [right_arm_path + link for link in robot_links]
    for arm_key in ARM_KEYS:
        partners = [*environment, *opposing_links] if arm_key == "left_arm" else environment.copy()
        # The installed PhysX backend resolves the leaves to distinct link patterns and
        # supplies one filter list per pattern to create_rigid_contact_view.
        setattr(scene_cfg, f"{arm_key}_impact", ContactSensorCfg(
            prim_path=f"{arm_path(scene_cfg, arm_key)}/(?:{names})", filter_prim_paths_expr=partners,
            history_length=history_length, update_period=0.0, max_contact_data_count_per_prim=0,
        ))


def _peak_by_partner(sensor, replicas: int, links: int) -> torch.Tensor:
    history = sensor.data.normal_force_matrix_w_history
    expected = (replicas, sensor.cfg.history_length, links, len(sensor.cfg.filter_prim_paths_expr), 3)
    forces = history.torch if history is not None else None
    if forces is None or forces.shape != expected:
        shape = tuple(forces.shape) if forces is not None else None
        raise RuntimeError(
            f"Impact force history {sensor.cfg.prim_path} has shape {shape}, expected {expected}"
        )
    return forces.norm(dim=-1).amax(dim=(1, 2))


@cache
def _arm_links() -> tuple[str, ...]:
    # PhysX exposes two entries for the Shadow forearm path, so it cannot be paired 1:1.
    return tuple(path for path in _collider_paths(str(ROBOT_USD), rigid_links=True)
                 if path != "/Hand/robot0_forearm")


def forbidden_impact(scene) -> ImpactMeasurement:
    """Reduce filtered normal-force history to a force peak per replica and category."""
    import torch

    links = len(_arm_links())
    left = _peak_by_partner(scene["left_arm_impact"], scene.num_envs, links)
    right = _peak_by_partner(scene["right_arm_impact"], scene.num_envs, links)
    environment_filters = right.shape[-1]
    environment = torch.maximum(left[:, :environment_filters].amax(-1), right.amax(-1))
    return ImpactMeasurement(environment, left[:, environment_filters:].amax(-1))
