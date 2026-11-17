"""PhysX drives for the locally composed fixed-base articulation."""
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg, PhysxCollisionCfg

from dexman.paths import ROBOT_USD
from isaaclab_physx.sim.spawners.materials import RigidBodyMaterialCfg
from .joints import HOME


def robot_cfg():
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=str(ROBOT_USD),
            collision_props={"/.*": [PhysxCollisionCfg(contact_offset=.001, rest_offset=0.)]},
            physics_material=RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.6,
                                                  restitution=0., friction_combine_mode="min"),
            rigid_props={"/.*": [PhysxRigidBodyCfg(disable_gravity=False, max_depenetration_velocity=2.0)]},
            articulation_props={"/Arm/root_joint": [PhysxArticulationCfg(
                enabled_self_collisions=True, solver_position_iteration_count=16,
                solver_velocity_iteration_count=4)]},
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.6, 0.0, 0.6), joint_pos={**HOME, "robot0_.*": 0.0}),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "shoulder": ImplicitActuatorCfg(joint_names_expr=["shoulder_.*"],
                stiffness=1320.0, damping=72.66, joint_effort_limit=330.0),
            "elbow": ImplicitActuatorCfg(joint_names_expr=["elbow_joint"],
                stiffness=600.0, damping=34.64, joint_effort_limit=150.0),
            "wrist": ImplicitActuatorCfg(joint_names_expr=["wrist_.*"],
                stiffness=216.0, damping=29.39, joint_effort_limit=54.0),
            "hand": ImplicitActuatorCfg(joint_names_expr=["robot0_.*"],
                stiffness=2.0, damping=0.15, joint_effort_limit=2.0),
        },
    )
