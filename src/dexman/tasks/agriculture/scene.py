"""Agricultural preview scenes with a common pedestal and two controllable arms."""
from dataclasses import dataclass
import json
import math

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ImuCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import CuboidCfg, CylinderCfg, DistantLightCfg, DomeLightCfg, MassCfg, PreviewSurfaceCfg, SimulationCfg, UsdFileCfg
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.lights import SphereLightCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg
from isaaclab_physx.sim.spawners.materials import RigidBodyMaterialCfg

from dexman.paths import ROOT
from dexman.robot.config import robot_cfg
from dexman.robot.joints import ARM, FINGERTIPS
from dexman.sensors.camera import camera_cfg
from dexman.sensors.view import look_at_xyzw
from .interface import ARM_KEYS
from .randomization import FRUIT_PLACEMENTS, INACTIVE_Z, OBSTACLE_RADIUS, TABLE_Z


ASSETS = ROOT / "assets/agriculture"
ROVER_USD = ASSETS / "robot/rover/rover.usdc"
NS = "{ENV_REGEX_NS}"


@dataclass(frozen=True)
class CropScene:
    plant: str
    position: tuple[float, float, float]
    yaw_deg: float
    fruit: str
    instruction: str


CROPS = {
    "citrus": CropScene("orange_tree", (1.29, .42, 0.), -28., "orange",
        "Selecciona una naranja madura y colócala en la bandeja izquierda."),
    "banana": CropScene("banana_plant", (1.54, .48, 0.), 180., "banana",
        "Recoge el plátano con la derecha, pásalo a la izquierda y déjalo en la bandeja."),
    "vineyard": CropScene("vineyard", (1.14, .15, 0.), 90., "grapes",
        "Sujeta el racimo de uvas y evita los alambres de la espaldera."),
    "berries": CropScene("berry_bush", (1.12, .25, .60), 20., "blueberry",
        "Recoge un arándano con pulgar e índice sin desplazar los demás."),
}
FRUIT_MASS_KG = {"orange": .16, "banana": .12, "grapes": .20, "blueberry": .002}
PICKUP_SUPPORT_HEIGHT = .035
PICKUP_SUPPORTS = {
    "citrus": ((0., 0., .056, .056),),
    "banana": ((-.022, -.052, .042, .024), (-.038, .027, .042, .024)),
    "vineyard": ((.048, -.027, .080, .060),),
    "berries": ((0., 0., .010, .010),),
}
SUPPORTED_GRAPE_ROOT = (.252, -.092, .96849)
SUPPORTED_GRAPE_ROTATION = (.27575, .78403, -.06454, .55236)


def yaw(angle):
    half = math.radians(angle) / 2
    return (0., 0., math.sin(half), math.cos(half))


def fixed_box(path, size, position, color):
    return AssetBaseCfg(prim_path=path,
        spawn=CuboidCfg(size=size, collision_props=PhysxCollisionCfg(contact_offset=.001, rest_offset=0.),
                       visual_material=PreviewSurfaceCfg(diffuse_color=color)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position))


def visual_model(path, name, position, angle=0.):
    suffix = "usdc" if name == "orange_tree" else "usda"
    return AssetBaseCfg(prim_path=path, spawn=UsdFileCfg(usd_path=str(ASSETS / f"props/{name}.{suffix}")),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=yaw(angle)))


def arm(side):
    cfg = robot_cfg()
    cfg.prim_path = f"{NS}/Torso/{'LeftArm' if side > 0 else 'RightArm'}"
    cfg.init_state.pos = (-.77, side * .56, 1.08)
    cfg.init_state.rot = yaw(-side * 14.)
    seed = json.loads((ROOT / "configs/grasp_seed.json").read_text(encoding="utf-8"))
    cfg.init_state.joint_pos = {**dict(zip(ARM, seed["arm_q"], strict=True)), "robot0_.*": 0.}
    return cfg


def camera(path, position, target, resolution, focal_length):
    cfg = camera_cfg()
    cfg.prim_path = path
    cfg.offset.pos = position
    cfg.offset.rot = look_at_xyzw(position, target)
    cfg.width, cfg.height = resolution
    cfg.spawn.focal_length = focal_length
    cfg.spawn.clipping_range = (.055, 12.)
    return cfg


def eye(side):
    name = "LeftEye" if side > 0 else "RightEye"
    return camera(f"{NS}/Torso/Eyes/{name}", (-.487, side * .128, 1.48),
                  (.58, 0., 1.10), (640, 480), 15.)


def lens(eye_cfg):
    x, y, z, w = eye_cfg.offset.rot
    backward = (2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y))
    position = tuple(p + .017 * axis for p, axis in zip(eye_cfg.offset.pos, backward, strict=True))
    return AssetBaseCfg(prim_path=eye_cfg.prim_path + "Lens",
        spawn=CylinderCfg(radius=.039, height=.028,
                          visual_material=PreviewSurfaceCfg(diffuse_color=(.024, .09, .105), metallic=.45)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=eye_cfg.offset.rot))


def loose_fruit(name, index, position, angle=0.):
    return RigidObjectCfg(prim_path=f"{NS}/Produce/Fruit{index}",
        spawn=UsdFileCfg(usd_path=str(ASSETS / f"props/{name}.usda"),
            rigid_props=PhysxRigidBodyCfg(max_depenetration_velocity=.5),
            collision_props={"/.*": [PhysxCollisionCfg(contact_offset=.001, rest_offset=0.)]},
            mass_props=MassCfg(mass=FRUIT_MASS_KG[name]),
            physics_material=RigidBodyMaterialCfg(static_friction=.60, dynamic_friction=.48,
                                                   restitution=.03, friction_combine_mode="min")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=yaw(angle)))


@configclass
class AgriculturalBaseSceneCfg(InteractiveSceneCfg):
    ground = fixed_box("/World/Ground", (18., 18., .10), (0., 0., -.055), (.21, .28, .145))
    sunlight = AssetBaseCfg(prim_path="/World/Sun", spawn=DistantLightCfg(intensity=1550., angle=9.),
        init_state=AssetBaseCfg.InitialStateCfg(rot=look_at_xyzw((3., -4., 6.), (0., 0., 0.))))
    fill = AssetBaseCfg(prim_path="/World/Sky", spawn=DomeLightCfg(intensity=480., color=(.77, .85, 1.)))


@configclass
class AgriculturalSceneCfg(AgriculturalBaseSceneCfg):
    torso = AssetBaseCfg(prim_path=f"{NS}/Torso", spawn=UsdFileCfg(usd_path=str(ASSETS / "robot/torso.usda")))
    left_arm = arm(1)
    right_arm = arm(-1)
    left_eye = eye(1)
    right_eye = eye(-1)
    left_lens = lens(left_eye)
    right_lens = lens(right_eye)
    overview = camera(f"{NS}/Overview", (2.95, -3.85, 2.75), (.38, .10, 1.2), (1200, 800), 19.)


def rover_eye(name):
    # Bind the camera authored on the tilt body, retaining its measured optical transform and lens.
    cfg = camera_cfg()
    cfg.prim_path = f"{NS}/Rover/HeadTilt/{name}"
    cfg.spawn = None
    cfg.update_latest_camera_pose = True
    return cfg


@configclass
class AgriculturalRoverSceneCfg(AgriculturalBaseSceneCfg):
    rover = ArticulationCfg(prim_path=f"{NS}/Rover", spawn=UsdFileCfg(usd_path=str(ROVER_USD)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(-.68, 0., 0.)),
        actuators={"usd_drives": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)})
    imu = ImuCfg(prim_path=f"{NS}/Rover/Base/IMU")
    left_eye = rover_eye("LeftEye")
    right_eye = rover_eye("RightEye")


def arm_path(scene, key):
    if hasattr(scene, "rover"):
        side = "LeftArm" if key == "left_arm" else "RightArm"
        return f"{scene.rover.prim_path}/{side}"
    return getattr(scene, key).prim_path


@configclass
class AgriculturalPreviewCfg:
    scene = AgriculturalSceneCfg(num_envs=1, env_spacing=7., replicate_physics=False)
    sim = SimulationCfg(dt=1 / 120, render_interval=4, physics=PhysxCfg())


def make_preview(scenario, *, randomize=False, layout=None):
    cfg = AgriculturalPreviewCfg()
    scene = cfg.scene
    if layout is not None and layout.crop != scenario:
        raise ValueError("The preview layout must match the selected crop")
    populate_crop(scene, scenario, randomize=randomize or bool(layout and layout.obstacles),
                  raised_support=bool(layout and layout.raised_support))
    if layout is not None:
        scene.overview.offset.pos = layout.inspection_position
        scene.overview.offset.rot = look_at_xyzw(layout.inspection_position, layout.inspection_target)
    elif scenario == "banana":
        scene.overview.offset.pos = (2.88, -4.1, 2.84)
        scene.overview.offset.rot = look_at_xyzw(scene.overview.offset.pos, (.52, .15, 1.52))
    elif scenario == "vineyard":
        scene.overview.offset.pos = (1.75, -4.32, 3.28)
        scene.overview.offset.rot = look_at_xyzw(scene.overview.offset.pos, (.54, .15, 1.12))
    return cfg


def populate_crop(scene, scenario, *, randomize, raised_support=False):
    crop = CROPS[scenario]
    scene.plant = visual_model(f"{NS}/Crop/Plant", crop.plant, crop.position, crop.yaw_deg)
    scene.soil = fixed_box(f"{NS}/Crop/Soil", (1.8, 3.8, .05), (1.49, .22, -.028), (.24, .14, .083))
    if scenario == "berries":
        scene.planter = fixed_box(f"{NS}/Crop/Planter", (.83, 1.04, .60), (1.12, .25, .30), (.34, .20, .11))
        scene.berry_sample = visual_model(f"{NS}/Produce/BerrySample", "blueberries", (.59, .38, TABLE_Z + .01))
    wood = (.45, .29, .16)
    # The rover's folded left wrist occupies the rear overhang of the pedestal's table.
    # Bring the near edge back to the trays and keep both trays, fruit and supports on the surface.
    rover_bench = hasattr(scene, "rover")
    rear_edge, front_edge = (.035 if rover_bench else -.235), .895
    scene.worktop = fixed_box(f"{NS}/Bench/Top", (front_edge-rear_edge, 1.36, .065),
                              ((front_edge+rear_edge)/2, 0., TABLE_Z-.0325), wood)
    leg_x, leg_y = (.09, .642) if rover_bench else (-.16, .61)
    for index, (x, y) in enumerate(((leg_x, -leg_y), (leg_x, leg_y), (.82, -.61), (.82, .61))):
        setattr(scene, f"leg_{index}", fixed_box(f"{NS}/Bench/Leg{index}", (.065, .065, TABLE_Z-.06),
                                                  (x, y, (TABLE_Z-.06)/2), (.12, .17, .17)))
    for side, color in ((1, (.17, .34, .28)), (-1, (.23, .25, .33))):
        center = side*.46
        label = "Left" if side > 0 else "Right"
        path = f"{NS}/Bench/{label}Tray"
        setattr(scene, f"tray_{label}", fixed_box(path+"/Bottom", (.35, .29, .017), (.24, center, TABLE_Z+.009), color))
        for i, (dx, dy, size) in enumerate(((.175, 0., (.018,.29,.042)),(-.175, 0., (.018,.29,.042)),
                                            (0., .145, (.37,.018,.042)),(0., -.145, (.37,.018,.042)))):
            setattr(scene, f"tray_{label}_{i}", fixed_box(path+f"/Rim{i}", size, (.24+dx, center+dy, TABLE_Z+.026), color))
    for i, (x, y, angle) in enumerate(FRUIT_PLACEMENTS[scenario]):
        z = TABLE_Z + .006
        supported = raised_support and i == 0
        if supported and scenario == "vineyard":
            x, y, z = SUPPORTED_GRAPE_ROOT
        fruit = loose_fruit(crop.fruit, i, (x, y, z + PICKUP_SUPPORT_HEIGHT if supported else z), angle)
        if supported:
            if scenario == "vineyard":
                fruit.init_state.rot = SUPPORTED_GRAPE_ROTATION
            for j, (dx, dy, sx, sy) in enumerate(PICKUP_SUPPORTS[scenario]):
                setattr(scene, f"pickup_support_{j}", fixed_box(f"{NS}/Produce/Support{j}",
                    (sx, sy, PICKUP_SUPPORT_HEIGHT), (x + dx, y + dy, TABLE_Z + PICKUP_SUPPORT_HEIGHT / 2),
                    (.15, .22, .28)))
            if scenario == "citrus":
                # A low rim keeps the round fruit seated while the fingers approach above it.
                rim = ((-.028, 0., .004, .056), (.028, 0., .004, .056),
                       (0., -.028, .052, .004), (0., .028, .052, .004))
                for j, (dx, dy, sx, sy) in enumerate(rim):
                    setattr(scene, f"pickup_support_rim_{j}", fixed_box(f"{NS}/Produce/SupportRim{j}",
                        (sx, sy, .008), (x + dx, y + dy, TABLE_Z + PICKUP_SUPPORT_HEIGHT + .004),
                        (.15, .22, .28)))
        setattr(scene, f"fruit_{i}", fruit)
    if randomize:
        scene.sunlight.spawn.intensity = 220.
        scene.fill.spawn.intensity = 155.
        scene.local_light = AssetBaseCfg(prim_path=f"{NS}/Randomization/Light",
            spawn=SphereLightCfg(radius=.35, intensity=4700., enable_color_temperature=True),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., 2.8)))
        for kind in OBSTACLE_RADIUS:
            prop = RigidObjectCfg(prim_path=f"{NS}/Randomization/{kind}",
                spawn=UsdFileCfg(usd_path=str(ASSETS / f"props/{kind}.usda"),
                    rigid_props=UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                    mass_props=MassCfg(mass=.25),
                    collision_props={"/.*": [PhysxCollisionCfg(contact_offset=.001, rest_offset=0.)]}),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., INACTIVE_Z)))
            setattr(scene, f"obstacle_{kind}", prop)


def configure_open_support(scene, crop):
    """Provide finger clearance around loose fruit with narrow physical support points."""
    x, y, _ = scene.fruit_0.init_state.pos
    if crop == "citrus":
        points = ((0., 0., .042, .042),)
    elif crop == "banana":
        points = ((-.022, -.052, .022, .012), (-.038, .027, .022, .012))
    elif crop == "vineyard":
        points = ((.034, -.041, .014, .014), (.063, -.036, .014, .014), (.047, -.012, .014, .014))
    else:
        raise ValueError(f"Sin apoyo abierto para {crop}")
    for key in tuple(vars(scene)):
        if key.startswith("pickup_support_"):
            delattr(scene, key)
    for i, (dx, dy, sx, sy) in enumerate(points):
        setattr(scene, f"pickup_support_{i}", fixed_box(f"{NS}/Produce/Support{i}",
                (sx, sy, PICKUP_SUPPORT_HEIGHT), (x+dx, y+dy, TABLE_Z + PICKUP_SUPPORT_HEIGHT / 2),
                (.15, .22, .28)))
    if crop == "citrus":
        for i, (dx, dy, sx, sy) in enumerate(((-.021, 0., .004, .024), (.021, 0., .004, .024),
                                               (0., -.021, .024, .004), (0., .021, .024, .004))):
            setattr(scene, f"pickup_support_rim_{i}", fixed_box(f"{NS}/Produce/SupportRim{i}",
                    (sx, sy, .006), (x+dx, y+dy, TABLE_Z+PICKUP_SUPPORT_HEIGHT+.003), (.15, .22, .28)))


def enable_task_contacts(scene):
    """Measure fingertip contacts with the task fruit and physical obstacles."""
    from isaaclab.sensors import ContactSensorCfg

    target = f"{NS}/Produce/Fruit0"
    props = [f"{NS}/Randomization/{kind}" for kind in OBSTACLE_RADIUS]
    for arm_key in ARM_KEYS:
        actor = scene.rover if hasattr(scene, "rover") else getattr(scene, arm_key)
        actor.spawn.activate_contact_sensors = True
        for tip in FINGERTIPS:
            setattr(scene, f"{arm_key}_{tip}", ContactSensorCfg(
                prim_path=f"{arm_path(scene, arm_key)}/Hand/{tip}",
                filter_prim_paths_expr=[target, *props],
                track_contact_points=True, max_contact_data_count_per_prim=128))
    scene.fruit_0.spawn.activate_contact_sensors = True
    scene.fruit_obstacle_contact = ContactSensorCfg(prim_path=target, filter_prim_paths_expr=props)
