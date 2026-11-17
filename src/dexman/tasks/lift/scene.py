"""Primitive tabletop task assets keep the scene light and self-contained."""
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import CuboidCfg, DistantLightCfg, PreviewSurfaceCfg, MassCfg
from isaaclab_physx.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg

from dexman.robot.config import robot_cfg


@configclass
class LiftSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = robot_cfg()
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=CuboidCfg(size=(2.2, 1.4, 0.10), collision_props=PhysxCollisionCfg(contact_offset=.001, rest_offset=0.),
                       visual_material=PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.55)),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=CuboidCfg(size=(0.05, 0.05, 0.05),
            rigid_props=PhysxRigidBodyCfg(max_depenetration_velocity=1.0),
            mass_props=MassCfg(mass=0.08), collision_props=PhysxCollisionCfg(contact_offset=.001, rest_offset=0.),
            physics_material=RigidBodyMaterialCfg(static_friction=.7, dynamic_friction=.5, restitution=0., friction_combine_mode="min"),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.08))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.625)),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=DistantLightCfg(intensity=2500.0))
