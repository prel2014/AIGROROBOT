"""Procedural, redistributable rigid assets and grounded, untrained task situations."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Situation:
    name: str
    label: str
    shape: str
    dimensions: tuple
    mass: float
    color: tuple
    table: tuple
    theme: str
    obstacles: tuple
    destination: tuple


# Obstacle records: center XY and box dimensions XYZ; all are physical colliders.
SITUATIONS = (
    Situation("clear_bench", "cubo rojo", "box", (.05,.05,.05), .08, (.85,.12,.08), (.35,.35,.38),
              "Banco despejado: agarre y sostén", (), (.15,0.)),
    Situation("color_sort", "caja azul", "box", (.045,.06,.055), .10, (.08,.25,.85), (.72,.69,.61),
              "Selección por color entre distractores", ((.14,.28,.10,.07,.20),), (0.,.15)),
    Situation("narrow_passage", "cilindro verde", "cylinder", (.025,.065), .09, (.12,.65,.22), (.22,.25,.28),
              "Pasillo con dos barreras", ((.29,.145,.11,.015,.24),(.48,.28,.10,.04,.18)), (0.,.25)),
    Situation("rolling_objects", "esfera amarilla", "sphere", (.028,), .06, (.95,.78,.05), (.60,.66,.70),
              "Empuje sobre la mesa sin elevar", ((.18,-.16,.08,.06,.13),), (.20,0.)),
    Situation("warehouse", "lata metálica", "cylinder", (.028,.07), .15, (.55,.60,.64), (.30,.25,.20),
              "Transporte entre cajas de almacén", ((.45,.16,.07,.09,.23),(.16,-.18,.12,.10,.16)), (.25,-.12)),
    Situation("capsule_detour", "cápsula naranja", "capsule", (.022,.035), .07, (.95,.36,.04), (.12,.20,.30),
              "Desvío hacia la derecha", ((.29,-.065,.12,.015,.24),), (0.,-.20)),
    Situation("diagonal_sort", "bloque de madera", "box", (.065,.035,.04), .11, (.58,.32,.12), (.75,.75,.75),
              "Colocación diagonal sin tocar los otros objetos", ((.43,-.05,.04,.08,.18),), (.15,-.25)),
    Situation("blocked_destination", "cubo violeta", "box", (.05,.05,.05), .08, (.55,.12,.72), (.35,.43,.35),
              "Destino ocupado: detenerse y solicitar otro destino", ((.29,.245,.12,.12,.22),), (0.,.20)),
)


def push_offset(situation):
    index = next(i for i,s in enumerate(SITUATIONS) if s.name == situation.name)
    return ((.05,0.,"5 cm hacia X positivo"),(0.,.10,"10 cm hacia Y positivo"),
            (-.15,0.,"15 cm hacia X negativo"),(0.,-.08,"8 cm hacia Y negativo"))[index % 4]


def instructions(situation):
    label = situation.label
    article = "la" if situation.shape in ("sphere","capsule") or label.startswith(("caja","lata")) else "el"
    dx, dy = situation.destination
    direction = ("delante" if dx>0 else "detrás") if dx else ("la izquierda" if dy>0 else "la derecha")
    distance = round(100 * max(abs(dx), abs(dy)))
    avoid = " Si ves obstáculos, esquívalos."
    displacement = (f"{round(dx*100)} cm en X y {round(dy*100)} cm en Y" if dx and dy
                    else f"{distance} cm hacia {direction}")
    push_direction = push_offset(situation)[2]
    return [f"Selecciona {article} {label}, eleva ese objeto 15 cm y sostenlo medio segundo.{avoid}",
            f"Lleva {article} {label} {displacement}, baja el objeto y suéltalo sobre la mesa.{avoid}",
            f"Empuja {article} {label} {push_direction} sin levantar ese objeto ni mover los demás.{avoid}"]


def scenario_catalog():
    return [{**asdict(s), "instructions": instructions(s), "trained": False} for s in SITUATIONS]


def apply_situation(cfg, name):
    """Apply before launching simulation. One target and two movable distractors."""
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.sim import CuboidCfg, SphereCfg, CylinderCfg, CapsuleCfg, MassCfg, PreviewSurfaceCfg
    from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg
    from isaaclab_physx.sim.spawners.materials import RigidBodyMaterialCfg
    from dexman.robot.joints import FINGERTIPS
    s = next(item for item in SITUATIONS if item.name == name)
    common = dict(rigid_props=PhysxRigidBodyCfg(max_depenetration_velocity=.5),
                  mass_props=MassCfg(mass=s.mass), collision_props=PhysxCollisionCfg(contact_offset=.001,rest_offset=0.),
                  physics_material=RigidBodyMaterialCfg(static_friction=.7,dynamic_friction=.5,restitution=0.,friction_combine_mode="min"),
                  visual_material=PreviewSurfaceCfg(diffuse_color=s.color))
    if s.shape == "box":
        spawn, half = CuboidCfg(size=s.dimensions, **common), s.dimensions[2]/2
    elif s.shape == "sphere":
        spawn, half = SphereCfg(radius=s.dimensions[0], **common), s.dimensions[0]
    else:
        radius, height = s.dimensions
        spawn = (CylinderCfg if s.shape == "cylinder" else CapsuleCfg)(radius=radius,height=height,**common)
        half = height/2 + (radius if s.shape == "capsule" else 0)
    cfg.scene.object.spawn = spawn
    cfg.scene.object.spawn.activate_contact_sensors = True
    x, y, _ = cfg.scene.object.init_state.pos
    cfg.object_initial_height = cfg.table_height + half
    cfg.scene.object.init_state.pos = (x,y,cfg.object_initial_height)
    cfg.asset_shape, cfg.asset_dimensions, cfg.scenario_name = s.shape, s.dimensions, s.name
    cfg.object_label = s.label
    cfg.instruction = instructions(s)[0]
    cfg.scene.table.spawn.visual_material = PreviewSurfaceCfg(diffuse_color=s.table)
    cfg.scene.light.spawn.intensity = 600. if name in ("warehouse","capsule_detour") else 900.
    cfg.scene.env_spacing = 5.
    static = PhysxCollisionCfg(contact_offset=.001,rest_offset=0.)
    cfg.scene.floor = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Floor",
        spawn=CuboidCfg(size=(4.,4.,.05),collision_props=static,
            visual_material=PreviewSurfaceCfg(diffuse_color=tuple(v*.55 for v in s.table))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.,0.,-.025)))
    cfg.scene.backdrop = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Backdrop",
        spawn=CuboidCfg(size=(.05,3.5,2.1),collision_props=static,
            visual_material=PreviewSurfaceCfg(diffuse_color=s.table)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.6,0.,1.05)))
    for index,(lx,ly) in enumerate(((-.9,-.5),(-.9,.5),(.9,-.5),(.9,.5))):
        setattr(cfg.scene,f"table_leg_{index}",AssetBaseCfg(prim_path="{ENV_REGEX_NS}/TableLeg"+str(index),
            spawn=CuboidCfg(size=(.08,.08,.50),collision_props=static,
                visual_material=PreviewSurfaceCfg(diffuse_color=(.16,.18,.20))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(lx,ly,.25))))
    for index, (ox,oy,sx,sy,sz) in enumerate(s.obstacles):
        # Kinematic rigid bodies also support GPU pair-filtered contact reporting;
        # bare static colliders do not support that filter in PhysX tensors.
        setattr(cfg.scene, f"obstacle_{index}", RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Obstacle"+str(index),
            spawn=CuboidCfg(size=(sx,sy,sz), collision_props=PhysxCollisionCfg(contact_offset=.001,rest_offset=0.),
                            rigid_props=UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                            mass_props=MassCfg(mass=1.),
                            visual_material=PreviewSurfaceCfg(diffuse_color=(.38,.42,.48))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(ox,oy,cfg.table_height+sz/2))))
    for index, (position,color) in enumerate((((.05,.35,.625),(.05,.75,.8)),((.65,-.30,.625),(.85,.2,.55)))):
        setattr(cfg.scene, f"distractor_{index}", RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Distractor"+str(index),
            spawn=CuboidCfg(size=(.045,.045,.05),rigid_props=PhysxRigidBodyCfg(),mass_props=MassCfg(mass=.09),
                collision_props=PhysxCollisionCfg(contact_offset=.001,rest_offset=0.),
                visual_material=PreviewSurfaceCfg(diffuse_color=color)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=position)))
    filters = ["{ENV_REGEX_NS}/Object"] + ["{ENV_REGEX_NS}/Obstacle"+str(i) for i in range(len(s.obstacles))]
    filters += ["{ENV_REGEX_NS}/Distractor0","{ENV_REGEX_NS}/Distractor1"]
    for tip in FINGERTIPS:
        getattr(cfg.scene,tip).filter_prim_paths_expr = filters
    cfg.scene.target_obstacle_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object", filter_prim_paths_expr=filters[1:], update_period=0.)
    return s


def add_scenario_argument(parser):
    parser.add_argument("--scenario", choices=["red_cube"] + [s.name for s in SITUATIONS],
                        default="red_cube", help="Physical scene; must match the checkpoint contract")


def configure_scenario(cfg, name):
    if name != "red_cube":
        apply_situation(cfg, name)
