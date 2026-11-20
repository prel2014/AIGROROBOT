"""Film three independent, scripted orange pickups and basket deliveries in Isaac Sim."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


TRIALS = (("Frontal",1,(.090,0.,.06)), ("Lateral",0,(-.035,.135,.110)),
          ("Diagonal",1,(.055,-.080,.10)))
PICKUP_XY = (.16,-.04)
# Centre of the authored Orange02 on row 2, tree 3, in naranjal.usdc.
FIELD_FRUIT = (7.94930, .05068, 1.39921)
FIELD_GROUND = .272
BRANCH_REACH = .32
FRUIT_ROOT_Z = FIELD_FRUIT[2] - FIELD_GROUND - .0381 - .154
STEM_BREAK_N = 4.
STEM_STIFFNESS = 500.
STEM_DAMPING = 4.0
OFFSETS = [(-.006,-.006,.005),(-.010,-.005,.012),(-.004,-.011,.012),
           (-.013,-.010,.006),(-.012,.004,.008),(-.016,-.003,.003)]




def config(trials, probe):
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim import UsdFileCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxCollisionCfg
    from dexman.robot.config import robot_cfg
    from dexman.tasks.agriculture.batch import make_batch_config
    from dexman.tasks.agriculture.scene import camera, fixed_box, yaw, NS
    from dexman.tasks.agriculture.field_layout import FIELDS
    cfg = make_batch_config(len(trials),crop="citrus",rover=True,contacts=True,
                            rgbd=not probe,rgbd_size=384 if len(trials)<=3 else 224,
                            nominal=True,raised_support=True,physics_substeps=4)
    scene = cfg.scenes.citrus
    for name in tuple(vars(scene)):
        if name.startswith(("pickup_support_","tray_","leg_","fruit_")) and name not in {"fruit_0","fruit_obstacle_contact"}:
            delattr(scene,name)
    scene.worktop = scene.plant = scene.soil = scene.ground = None
    scene.env_spacing = 120.
    scene.orchard = AssetBaseCfg(prim_path=f"{NS}/Parcela",spawn=UsdFileCfg(
        usd_path=str(ROOT/"assets/agriculture/parcelas/naranjal_cosecha.usda")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(PICKUP_XY[0]+BRANCH_REACH-FIELD_FRUIT[1],
            PICKUP_XY[1]+FIELD_FRUIT[0],-FIELD_GROUND),rot=yaw(-90.)))
    # Load the dynamic fruit and its visible stem before binding the fruit sensor.
    scene.plant = AssetBaseCfg(prim_path=f"{NS}/Produce", spawn=UsdFileCfg(
        usd_path=str(ROOT/"assets/agriculture/props/naranja_en_rama.usda")))
    scene.fruit_0.spawn = None
    scene.rover.init_state.pos = (-.68, 0., .075)
    # Physical chocks on both sides of each tyre leave the chassis on its wheels.
    for axle in (-.55,.55):
        for y in (-.51,.51):
            for direction in (-1,1):
                name = f"Calzo_{round(100*axle)+55}_{round(100*y)+51}_{direction+1}"
                x = scene.rover.init_state.pos[0]+axle+.21*direction
                ground = FIELDS["citrus"].height(FIELD_FRUIT[0]-(y-PICKUP_XY[1]),
                                                FIELD_FRUIT[1]+x-PICKUP_XY[0]-BRANCH_REACH)-FIELD_GROUND
                setattr(scene,name,fixed_box(f"{NS}/Parking/{name}",(.07,.12,.072),
                        (x,y,ground+.032),(.25,.21,.08)))
    scene.rover.spawn.articulation_props = {"": [PhysxArticulationCfg(
        enabled_self_collisions=True,solver_position_iteration_count=16,solver_velocity_iteration_count=4)]}
    scene.rover.spawn.collision_props = {"/(Left|Right)Arm/Hand/.*":[
        PhysxCollisionCfg(contact_offset=.001,rest_offset=0.)]}
    other = scene.rover.actuators["usd_drives"]
    other.joint_names_expr = ["drive_.*","head_.*"]
    calibrated = robot_cfg().actuators
    scene.rover.spawn.physics_material = robot_cfg().spawn.physics_material
    calibrated["elbow"].joint_names_expr = ["elbow_joint.*"]
    scene.rover.actuators = {**calibrated,"wheels_and_head":other}
    scene.fruit_0.init_state.pos = (*PICKUP_XY,FRUIT_ROOT_Z)
    scene.fruit_0.init_state.rot = yaw(12.)
    if not probe:
        overview, detail = ((768,512),(432,512)) if len(trials)<=3 else ((384,256),(216,256))
        scene.witness = camera(f"{NS}/Testigo",(-.90,-2.25,1.88),(-.25,-.04,.91),overview,17.)
        scene.witness.data_types = ["rgb"]
        scene.witness.spawn.clipping_range = (.05,65.)
        scene.detail = camera(f"{NS}/Detalle",(-.30,.62,1.94),(-.01,-.04,.90),detail,24.)
        scene.detail.data_types = ["rgb"]
        scene.detail.spawn.clipping_range = (.05,3.5)
    scene.basket_contact = ContactSensorCfg(prim_path=f"{NS}/Produce/Fruit0",
                                            filter_prim_paths_expr=[f"{NS}/Rover/Base"])
    return cfg


def matrix(pose):
    m = np.eye(4)
    m[:3,:3] = Rotation.from_quat(pose[3:]).as_matrix()
    m[:3,3] = pose[:3]
    return m


def make_path(env, index, side, approach, reference, fruit, offset, probe):
    from agriculture_grasp_geometry import _Kinematics, _solve_arm
    from dexman.tasks.agriculture.scene import arm
    group, = env.batch.groups
    owner = ("LeftArm","RightArm")[side]
    path = group.scene.env_prim_paths[index]+"/Rover/"+owner
    kin = _Kinematics(group.scene.stage,path)
    robot = group.rover.robot.data
    base_id = group.rover.bodies[(owner,"base_link")]
    base = matrix(robot.body_link_pose_w.torch[index,base_id].cpu().numpy())
    origin = group.origins[index].cpu().numpy()
    base[:3,3] -= origin
    chassis = matrix(robot.root_link_pose_w.torch[index].cpu().numpy())
    chassis[:3,3] -= origin
    seed_base = arm(-1).init_state
    old_base = matrix([*seed_base.pos,*seed_base.rot])
    old_fruit = np.array([.30,-.12,.9355])
    a = old_fruit[:2]-old_base[:2,3]
    b = fruit[:2]-base[:2,3]
    turn = Rotation.from_rotvec([0,0,np.arctan2(b[1],b[0])-np.arctan2(a[1],a[0])]).as_matrix()
    limits = env.batch.joint_limits[index,side*30:(side+1)*30].cpu().numpy()
    low, high = limits.T
    def solve(q,position,axes):
        result,error = _solve_arm(kin,base,q,low,high,position,axes)
        if error>.006:
            # The Shadow wrist adds two degrees of freedom close to the left shoulder limit.
            result,error = _solve_arm(kin,base,result,low,high,position,axes,moving=range(8))
        if error>.014:
            raise ValueError(f"{owner} unreachable at {position.round(3)}, IK residual {error:.4f}")
        return result
    aligned = []
    for q in reference:
        old = kin.fk(kin.arm,q,old_base)
        aligned.append(solve(q,fruit+turn@(old[:3,3]-old_fruit+offset),turn@old[:3,:3]))
    pickup = kin.fk(kin.arm,aligned[0],base)
    beginning = solve(aligned[0],pickup[:3,3]+approach,pickup[:3,:3])
    knots = [(beginning,"aproximacion",1),(aligned[0],"aproximacion",30),
             (aligned[1],"aproximacion",36),(aligned[2],"cierre",52),(aligned[2],"cierre",24)]
    knots.extend((q,"elevacion",8) for q in aligned[3:])
    knots.append((aligned[-1],"sosten",70))
    if probe:
        return knots
    top = kin.fk(kin.arm,aligned[-1],base)
    closed = kin.fk(kin.arm,aligned[2],base)
    palm_offset = closed[:3,3]-fruit
    pivot = -closed[:3,:3].T@palm_offset
    def carry(q, palm_goal):
        from scipy.optimize import least_squares
        # The fingers keep their physical preload. Let the wrist tilt while the
        # estimated centre of the grasp follows the delivery path.
        centre_goal = palm_goal+top[:3,:3]@pivot
        def residual(angles):
            pose = kin.fk(kin.arm,np.r_[angles,q[8:]],base)
            dv = pose[:3,3]+pose[:3,:3]@pivot-centre_goal
            dr = Rotation.from_matrix(top[:3,:3].T@pose[:3,:3]).as_rotvec()
            return np.r_[dv,.022*dr,.002*(angles-q[:8])]
        result = least_squares(residual,np.clip(q[:8],low[:8],high[:8]),bounds=(low[:8],high[:8]))
        if np.linalg.norm(result.fun[:3])>.012:
            raise ValueError(f"{owner} cannot carry fruit to {centre_goal}: {result.fun[:3]}")
        return np.r_[result.x,q[8:]]
    sign = 1 if side==0 else -1
    release = (chassis@np.array([.59,sign*.13,.81,1.]))[:3]+palm_offset
    # First clear the canopy, then descend over the basket.
    over = release.copy()
    over[2] = top[2,3]+.025
    current = aligned[-1].copy()
    for start,end in ((top[:3,3],over),(over,release)):
        for t in np.linspace(.125,1.,8):
            current = carry(current,(1-t)*start+t*end)
            knots.append((current,"traslado",8))
    opened = current.copy()
    opened[8:] = 0.
    knots.append((opened,"apertura",32))
    away = carry(opened,release+np.array([-.04,sign*.03,.17]))
    knots.extend(((away,"retirada",42),(away,"retirada",75)))
    return knots


class Film:
    def __init__(self, folder, trials):
        self.folder = folder
        self.trials = trials
        self.frames = {f"robot_{i:03d}{suffix}": folder/f"fotogramas_{i:03d}{suffix}"
                       for i in range(len(trials)) for suffix in ("","_ojos")}
        self.index = 0
        for directory in self.frames.values():
            directory.mkdir()

    def frame(self, batch, camera, detail, phase, lift, held):
        from isaaclab.sensors import Camera
        from PIL import Image,ImageDraw
        eyes_rgb, eyes_depth = (v.cpu().numpy() for v in batch.rgbd())
        Camera.update_batch([camera,detail],0.)
        rgb = camera.data.output["rgb"].torch[...,:3].cpu().numpy()
        overhead = detail.data.output["rgb"].torch[...,:3].cpu().numpy()
        h, w = rgb.shape[1:3]
        for i,(name,side,_,_) in enumerate(self.trials):
            if rgb[i].std()<2 or overhead[i].std()<2:
                raise RuntimeError(f"Sin imagen RTX en robot {i}")
            canvas = Image.new("RGB",(w+overhead.shape[2],h+44),(19,29,24))
            canvas.paste(Image.fromarray(rgb[i]),(0,0))
            canvas.paste(Image.fromarray(overhead[i]),(w,0))
            label = f"Robot {i:03d} | {name} / {('izquierda','derecha')[side]} | {phase} | {100*lift[i]:+.1f} cm | " + ("agarre" if held[i] else "sin agarre")
            ImageDraw.Draw(canvas).text((10,h+10),label,fill="white",font_size=18 if w>=768 else 15)
            name = f"robot_{i:03d}"
            frame_path = f"{self.index:05d}.jpg"
            canvas.save(self.frames[name]/frame_path,quality=93)
            self.eyes(eyes_rgb[i],eyes_depth[i],i,phase).save(self.frames[name+"_ojos"]/frame_path,quality=93)
        self.index += 1

    @staticmethod
    def eyes(rgb, depth, index, phase):
        from PIL import Image, ImageDraw

        h, w = rgb.shape[1:3]
        valid = np.isfinite(depth) & (depth > 0.)
        if any(view.std() < 2 for view in rgb) or not valid.any(axis=(1,2)).all():
            raise RuntimeError(f"Falta RGB o profundidad en un ojo del robot {index}")
        gray = np.zeros(depth.shape, dtype=np.uint8)
        gray[valid] = (255 * np.clip((2.5-depth[valid])/(2.5-.055),0,1)).astype(np.uint8)
        canvas = Image.new("RGB",(2*w,2*h+82),(19,29,24))
        labels = ImageDraw.Draw(canvas)
        labels.text((8,4),f"Robot {index:03d} | {phase}",fill="white",font_size=16)
        for eye, side in enumerate(("izquierdo","derecho")):
            x = eye*w
            labels.text((x+8,27),f"RGB | ojo {side}",fill="white",font_size=15)
            canvas.paste(Image.fromarray(rgb[eye]),(x,48))
            labels.text((x+8,h+52),"Profundidad | 0,055 a 2,5 m",fill="white",font_size=14)
            canvas.paste(Image.fromarray(gray[eye]),(x,h+82))
        return canvas

    def encode(self,ffmpeg):
        for name,directory in self.frames.items():
            dest = self.folder/f"{name}.mp4"
            subprocess.run([ffmpeg,"-hide_banner","-loglevel","error","-framerate","15","-i",
                str(directory/"%05d.jpg"),"-an","-c:v","libx264","-threads","2","-crf","19",
                "-pix_fmt","yuv420p","-movflags","+faststart",str(dest)],check=True)
            if dest.stat().st_size<1024:
                raise RuntimeError(f"Video vacío: {dest}")
            for j in range(self.index):
                (directory/f"{j:05d}.jpg").unlink()
            directory.rmdir()


class FruitStem:
    """A damped elastic attachment at the calyx, released once its load exceeds the threshold."""

    def __init__(self, group):
        import torch
        self.fruit = group.scene["fruit_0"]
        self.point = group.origins.new_tensor((0.,0.,.071)).expand_as(group.origins)
        self.neutral = group.origins + group.origins.new_tensor((*PICKUP_XY,FRUIT_ROOT_Z+.071))
        self.preload = group.origins.new_tensor((0.,0.,.16*9.81))
        self.attached = torch.ones(len(group.origins),dtype=torch.bool,device=group.origins.device)

    def apply(self):
        import torch
        from isaaclab.utils.math import quat_apply
        data = self.fruit.data
        point = data.root_link_pos_w.torch + quat_apply(data.root_link_quat_w.torch,self.point)
        arm = point - data.root_com_pos_w.torch
        vel = data.root_com_vel_w.torch
        speed = vel[:,:3] + torch.cross(vel[:,3:],arm,dim=-1)
        force = self.preload + STEM_STIFFNESS*(self.neutral-point) - STEM_DAMPING*speed
        self.attached &= force.norm(dim=-1) < STEM_BREAK_N
        force = torch.where(self.attached[:,None],force,0.)
        self.fruit.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force[:,None,:],torques=torch.cross(arm,force,dim=-1)[:,None,:],is_global=True)


def encoder_path(value):
    """Resolve the binary before launching Kit, including a Scoop console shim on Windows."""
    executable = shutil.which(value) if value else None
    if not executable:
        raise ValueError("Indica la ruta de FFmpeg con --ffmpeg")
    path = Path(executable)
    shim = path.with_suffix(".shim")
    if shim.is_file():
        for line in shim.read_text(encoding="utf-8-sig").splitlines():
            key,_,target = line.partition("=")
            if key.strip()=="path":
                native = Path(target.strip().strip('"'))
                if native.is_file():
                    path = native
                break
    return str(path)


def aim_active_eye(group, sides, target_world):
    """Aim the eye on the working side using the two measured head angles."""
    import torch
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    rover = group.rover
    data = rover.robot.data
    rotation = data.body_link_quat_w.torch[:,rover.tilt_body]
    centre = data.body_link_pos_w.torch[:,rover.tilt_body] + quat_apply(rotation,rover.eye_offsets[sides])
    axis = quat_apply(rotation,rover.eye_axes[sides])
    base = data.root_link_quat_w.torch
    def bearings(vector):
        x,y,z = quat_apply_inverse(base,vector).unbind(-1)
        return torch.stack((torch.atan2(y,x),torch.atan2(z,torch.hypot(x,y))),dim=-1)
    error = bearings(target_world-centre) - bearings(axis)
    return data.joint_pos.torch[:,rover.head] + torch.atan2(error.sin(),error.cos())


def run(env, out, ffmpeg, trials, probe):
    import torch
    from isaaclab.utils.math import quat_apply_inverse
    from dexman.tasks.agriculture.contact_quality import steady_contacts
    from dexman.tasks.agriculture.rover_basket_task import BasketTrial
    from dexman.tasks.agriculture.mobile_state import VERSION as mobile_contract
    from dexman.experiments import sha256
    batch = env.batch
    group, = batch.groups
    stem = FruitStem(group)
    robot = group.rover.robot
    seed_file = ROOT/"assets/agriculture/grasps/citrus.json"
    reference = np.array(json.loads(seed_file.read_text())["waypoints_rad"])
    base_home = env.home.clone()
    # Only stage the initial arm posture. The orange stays dynamic throughout.
    for tick in range(36):
        batch.step(base_home,before_physics_step=stem.apply)
        if tick in (0,5,35):
            print("Naranja suspendida XYZ", tick, env._measure()[0][:,:3].cpu().numpy().round(3).tolist(),flush=True)
    start = env._measure()
    fruit = start[0][:,:3].cpu().numpy()
    support = start[-1].clone()
    if np.max(np.linalg.norm(fruit-np.array([*PICKUP_XY,FRUIT_ROOT_Z]),axis=1))>.012:
        raise RuntimeError(f"La fruta perdió la rama durante la puesta en escena: {fruit.round(3).tolist()}")
    paths = [make_path(env,i,side,approach,reference,fruit[i],offset,probe)
             for i,(_,side,approach,offset) in enumerate(trials)]
    commands = []
    for n in range(len(paths[0])):
        goals = base_home.clone()
        for i,(_,side,_,_) in enumerate(trials):
            goals[i,30*side:30*(side+1)] = goals.new_tensor(paths[i][n][0])
        commands.append(goals)
    for i,(_,side,_,_) in enumerate(trials):
        ids = group.joint_ids[side]
        initial = commands[0][i:i+1,side*30:(side+1)*30]
        env_ids = torch.tensor([i],device=env.device)
        robot.write_joint_position_to_sim_index(position=initial,joint_ids=ids,env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(initial),joint_ids=ids,env_ids=env_ids)
    for _ in range(36):
        batch.step(commands[0],before_physics_step=stem.apply)
    start = env._measure()
    fruit_drift = (start[0][:,:3] - start[0].new_tensor((*PICKUP_XY,FRUIT_ROOT_Z))).norm(dim=-1)
    ready = stem.attached & (fruit_drift <= .01)
    print("Fruta aún en la rama al comenzar",ready.tolist(),flush=True)
    chassis_start = robot.data.root_link_pos_w.torch.clone()
    tests = [BasketTrial() for _ in trials]
    ids = torch.arange(len(trials),device=env.device)
    sides = torch.tensor([s for _,s,_,_ in trials],device=env.device)
    film = None if probe else Film(out,trials)
    previous, step = commands[0],0
    history = []
    try:
        for k, goal in enumerate(commands):
            _,phase,ticks = paths[0][k]
            for frame in range(1,ticks+1):
                t = frame/ticks
                record = film is not None and step%2==0
                gaze = aim_active_eye(group,sides,stem.fruit.data.root_com_pos_w.torch)
                batch.step(previous.lerp(goal,t),head_targets=gaze,render=record,before_physics_step=stem.apply)
                m = env._measure()
                pos = robot.data.root_link_pos_w.torch
                basket_coords = quat_apply_inverse(robot.data.root_link_quat_w.torch,m[8]+group.origins-pos)
                force = group.scene["basket_contact"].data.normal_force_matrix_w.torch.flatten(1).norm(dim=-1)
                held = m[6][ids,sides]
                touching = m[5].any((1,2))
                steady = steady_contacts(m[11][ids,sides],m[5][ids,sides])
                gaps = (m[4]-m[8][:,None,None]).norm(dim=-1).amin((1,2))
                values = torch.column_stack((m[-1]-support,held,steady,m[1][:,:3].norm(dim=-1),basket_coords,
                    force,gaps,touching,(pos-chassis_start).norm(dim=-1),m[7],stem.attached)).cpu().numpy()
                for i,row in enumerate(values):
                    lift,h,st,speed,x,y,z,normal,gap,touch,drift,impact,attached = map(float,row)
                    tests[i].observe(phase=phase,lift=lift,held=bool(h),steady=bool(st) and not attached,speed=speed,
                        basket_position=(x,y,z),basket_contact=normal>.04,finger_clearance=gap,
                        touching=bool(touch),drift=drift,impact=impact,dt=env.dt)
                if record:
                    film.frame(batch,group.scene["witness"],group.scene["detail"],phase,values[:,0],values[:,1])
                history.append((step,phase,values.tolist()))
                step+=1
            previous = goal
            print(f"{phase} {step} lift/grip {values[:,:2].round(3).tolist()}",flush=True)
            if phase=="cierre" and k==4:
                print("palm relative to fruit", (m[3][ids,sides]-m[0][:,:3]).cpu().numpy().round(3).tolist(),flush=True)
                print("fuerzas por dedo N",m[10][ids,sides].cpu().numpy().round(3).tolist(),flush=True)
            if phase=="sosten":
                print("fingertip slip m/s",m[11][ids,sides].cpu().numpy().round(3).tolist(),flush=True)
                print("object velocity",m[1].cpu().numpy().round(3).tolist(),flush=True)
    except Exception:
        print(f"Fotogramas del ensayo interrumpido en {out}",flush=True)
        raise
    report = dict(seed=batch.seed, simulation="Isaac Sim 6.2 / PhysX",parking="four chocked wheels",
                  state_contract=mobile_contract,
                  scene="naranjal.usdc", target="Hileras/Fila2/Planta03/Fruit/Orange02",
                  stem_break_n=STEM_BREAK_N,
                  stem_stiffness_n_m=STEM_STIFFNESS,stem_damping_ns_m=STEM_DAMPING,
                  control_hz=1/env.dt,physics_hz=1/batch.sim.get_physics_dt(),
                  rover_sha256=env.rover_asset_sha256, reference_sha256=sha256(seed_file),
                  arm_sha256=sha256(ROOT/"assets/robots/ur10e_shadow/robot.usda"),
                  field_sha256=sha256(ROOT/"assets/agriculture/parcelas/naranjal.usdc"),
                  field_task_sha256=sha256(ROOT/"assets/agriculture/parcelas/naranjal_cosecha.usda"),
                  fruit_task_sha256=sha256(ROOT/"assets/agriculture/props/naranja_en_rama.usda"),
                  instruction="Sujeta la naranja, elévala 15 cm durante medio segundo y déjala en la canasta.",
                  columns=["lift_m","held","steady","speed_m_s","basket_x","basket_y","basket_z","basket_force_n",
                           "finger_gap_m","touching","base_drift_m","forbidden_force_n","stem_attached"],
                  robots=[dict(robot=i,trajectory=name,hand=("izquierda","derecha")[side],offset_m=offset,
                               detached=not bool(stem.attached[i]), ready=bool(ready[i]),
                               **tests[i].result(env.dt,valid_start=bool(ready[i])))
                          for i,(name,side,_,offset) in enumerate(trials)])
    (out/"mediciones.json").write_text(json.dumps(history),encoding="utf-8")
    (out/"resultado.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report["robots"]),flush=True)
    if film:
        film.encode(ffmpeg)


if __name__=="__main__":
    from isaaclab.app import add_launcher_args, launch_simulation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=512)
    parser.add_argument("--trajectory", choices=[t[0] for t in TRIALS])
    parser.add_argument("--ffmpeg",default="ffmpeg")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--probe",action="store_true",help="Barrido geométrico hasta el sostén, sin video")
    modes.add_argument("--sweep",action="store_true",help="Graba las tres trayectorias con seis posiciones de agarre cada una")
    parser.add_argument("--offsets",type=json.loads,default=[[-.013,-.010,.002],[-.006,-.006,.002],[-.013,-.010,-.002]],
                        help="Desplazamientos XYZ en metros por trayectoria")
    add_launcher_args(parser)
    args = parser.parse_args()
    if np.asarray(args.offsets).shape!=(len(TRIALS),3) or not np.isfinite(args.offsets).all():
        parser.error("--offsets requiere tres vectores XYZ finitos")
    if not args.probe:
        args.ffmpeg = encoder_path(args.ffmpeg)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    trials = [(*trial,delta) for i,trial in enumerate(TRIALS)
              for delta in (OFFSETS if args.probe or args.sweep else [args.offsets[i]])]
    if args.trajectory:
        trials = [trial for trial in trials if trial[0] == args.trajectory]
    cfg = config(trials,args.probe)
    with launch_simulation(cfg,args):
        import torch
        from isaaclab.sim import SimulationContext
        from dexman.tasks.agriculture.environment import AgriculturalPPOEnv
        with torch.inference_mode():
            run(AgriculturalPPOEnv(SimulationContext(cfg.sim),cfg,seed=args.seed),args.output,args.ffmpeg,trials,args.probe)
