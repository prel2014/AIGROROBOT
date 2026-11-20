"""Record continuous, physically driven field approaches and harvest attempts, one pair per rover."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from rover_orange_trial import Film, ROOT, aim_active_eye, encoder_path, make_path, matrix
from dexman.tasks.agriculture.field_layout import FIELDS
from dexman.tasks.agriculture.harvest_route import ATTACHMENT_M, SITES, arm_occlusion, binocular_visibility, layout

SOURCE_DIGESTS = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (
    Path(__file__),ROOT/"tools/rover_orange_trial.py",ROOT/"src/dexman/tasks/agriculture/harvest_route.py",
    ROOT/"tools/field_contact_trace.py",ROOT/"src/dexman/tasks/agriculture/field_lift_control.py",
    ROOT/"src/dexman/tasks/agriculture/batch.py",
    ROOT/"src/dexman/tasks/agriculture/contact_quality.py",ROOT/"src/dexman/tasks/agriculture/rover_basket_task.py")}


VERSIONS = (("Acceso corto",1,(.02,-.13,.04),(-.008,-.006,.002)),
            ("Acceso lateral",0,(.02,.15,.065),(-.005,-.006,.003)))

# Both right-hand profiles use the surveyed pickup on the aisle side of the plant.
RIGHT_PROFILES = ("derecha_despejada", "derecha_calibrada")
BANANA_GRASPS = ("referencia_derecha", *RIGHT_PROFILES, "lateral")


def scene_files(out,crop,site,geometry,orientation):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from dexman.tasks.agriculture.scene import CROPS
    assets = ROOT/"assets/agriculture"
    parcel = assets/"parcelas"/(FIELDS[crop].name+".usdc")
    field = Usd.Stage.CreateNew(str(out/"parcela.usda"))
    root = UsdGeom.Xform.Define(field,"/Parcela").GetPrim()
    field.SetDefaultPrim(root)
    root.GetReferences().AddReference(os.path.relpath(parcel,out).replace(os.sep,"/"))
    for name in ("Rover","Physics","Sol","Cielo","Vistas","Cubresuelo","Linderos","Terrones","SueloGranular"):
        if root.GetChild(name):
            field.OverridePrim("/Parcela/"+name).SetActive(False)
    foliage = {"vineyard":"vineyard/cepa.usdc","berries":"arandano/mata.usdc"}
    if crop in foliage:
        from field_plant_instances import compose_plants
        compose_plants(field,out,assets/"models"/foliage[crop])
    field.GetRootLayer().Save()
    fruit = Usd.Stage.CreateNew(str(out/"fruto.usda"))
    root = UsdGeom.Xform.Define(fruit,"/Produce").GetPrim()
    fruit.SetDefaultPrim(root)
    dynamic = UsdGeom.Xform.Define(fruit,"/Produce/Fruit0")
    model = assets/"props"/(CROPS[crop].fruit+".usda")
    prim = dynamic.GetPrim()
    prim.GetReferences().AddReference(os.path.relpath(model,out).replace(os.sep,"/"))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(site.mass)
    prim.AddAppliedSchema("PhysxContactReportAPI")
    prim.CreateAttribute("physxContactReport:threshold",Sdf.ValueTypeNames.Float).Set(0.)
    skin = UsdShade.Material.Define(fruit,"/Produce/Piel")
    material = UsdPhysics.MaterialAPI.Apply(skin.GetPrim())
    material.CreateStaticFrictionAttr(.60)
    material.CreateDynamicFrictionAttr(.48)
    material.CreateRestitutionAttr(.03)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(skin,materialPurpose="physics")
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            p.AddAppliedSchema("PhysxCollisionAPI")
            p.CreateAttribute("physxCollision:contactOffset",Sdf.ValueTypeNames.Float).Set(.0006)
            p.CreateAttribute("physxCollision:restOffset",Sdf.ValueTypeNames.Float).Set(0.)
    dynamic.AddTranslateOp().Set(Gf.Vec3d(*geometry["fruit"]))
    dynamic.AddOrientOp().Set(Gf.Quatf(float(orientation[3]),Gf.Vec3f(*orientation[:3])))
    start = geometry["anchor"]
    end = geometry["branch_end"]
    branch = UsdGeom.BasisCurves.Define(fruit,"/Produce/Pedunculo")
    branch.CreateTypeAttr("linear")
    branch.CreateCurveVertexCountsAttr([4])
    branch.CreatePointsAttr([Gf.Vec3f(*p) for p in [start,.58*start+.42*end+[0.,0.,.025],
                                                      end+[0.,0.,.035],end]])
    width = .004 if crop=="berries" else .012
    branch.CreateWidthsAttr([width,width*.78,width*.45,width*.28])
    branch.SetWidthsInterpolation("vertex")
    branch.CreateDisplayColorAttr([(.12,.19,.042)])
    fruit.GetRootLayer().Save()
    return parcel,model


def configure(out,crop,versions,survey=False,substeps=4,verify_eyes=False):
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim import UsdFileCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxCollisionCfg
    from dexman.robot.config import robot_cfg
    from dexman.tasks.agriculture.batch import make_batch_config
    from dexman.tasks.agriculture.scene import camera, yaw, NS
    g = layout(crop)
    cfg = make_batch_config(len(versions),crop=crop,rover=True,contacts=True,rgbd=not survey,
                            rgbd_size=512 if crop=="banana" and verify_eyes and len(versions)<=2 else 320,
                            nominal=True,raised_support=True,physics_substeps=substeps)
    cfg.sim.enable_scene_query_support = True
    scene = getattr(cfg.scenes,crop)
    orient = scene.fruit_0.init_state.rot
    g["branch_end"] = g["fruit"]+[0.,0.,SITES[crop].local_tip_z]
    g["attachment"] = np.array(ATTACHMENT_M[crop])
    g["fruit"] = g["branch_end"]-Rotation.from_quat(orient).apply(g["attachment"])
    for name in tuple(vars(scene)):
        if name.startswith(("pickup_support_","tray_","leg_","fruit_")) and name not in {"fruit_0","fruit_obstacle_contact"}:
            delattr(scene,name)
    for name in ("worktop","soil","ground","berry_sample","planter"):
        if hasattr(scene,name):
            setattr(scene,name,None)
    scene.env_spacing = 120.
    scene.orchard = AssetBaseCfg(prim_path=NS+"/Parcela",spawn=UsdFileCfg(usd_path=str(out/"parcela.usda")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(g["translation"]),rot=yaw(g["rotation"])))
    scene.plant = AssetBaseCfg(prim_path=NS+"/Produce",spawn=UsdFileCfg(usd_path=str(out/"fruto.usda")))
    scene.fruit_0.spawn = None
    scene.fruit_0.init_state.pos = tuple(g["fruit"])
    route = g["route"]
    scene.rover.init_state.pos = (*route[0],g["start_z"])
    bearing = route[-1]-route[0]
    scene.rover.init_state.rot = yaw(np.degrees(np.arctan2(bearing[1],bearing[0]))+1.5)
    scene.rover.spawn.articulation_props = {"":[PhysxArticulationCfg(enabled_self_collisions=True,
        solver_position_iteration_count=16,solver_velocity_iteration_count=4)]}
    scene.rover.spawn.collision_props = {"/(Left|Right)Arm/Hand/.*":[PhysxCollisionCfg(contact_offset=.001,rest_offset=0.)]}
    other = scene.rover.actuators["usd_drives"]
    other.joint_names_expr = ["drive_.*","head_.*"]
    calibrated = robot_cfg().actuators
    calibrated["elbow"].joint_names_expr = ["elbow_joint.*"]
    scene.rover.actuators = {**calibrated,"wheels_and_head":other}
    scene.rover.spawn.physics_material = robot_cfg().spawn.physics_material
    scene.fill.spawn.texture_file = str(ROOT/"assets/agriculture/parcelas/textures/cielo.jpg")
    scene.fill.spawn.intensity = 400.
    scene.sunlight.spawn.intensity = 1750.
    scene.witness = camera(NS+"/Testigo",(-1.5,-2.5,1.9),(-.2,0.,.9),(704,448),19.)
    scene.detail = camera(NS+"/Detalle",(.5,.95,1.7),(.05,-.015,1.),(400,448),22.)
    for cam in (scene.witness,scene.detail):
        cam.data_types = ["rgb"]
        cam.spawn.clipping_range = (.05,75.)
    if survey:
        scene.witness = scene.detail = None
    elif verify_eyes:
        for cam in (scene.left_eye,scene.right_eye):
            cam.data_types = [*cam.data_types,"instance_id_segmentation_fast"]
            cam.renderer_cfg.colorize_instance_id_segmentation = False
    scene.basket_contact = ContactSensorCfg(prim_path=NS+"/Produce/Fruit0",filter_prim_paths_expr=[NS+"/Rover/Base"])
    return cfg,g,orient


class Stem:
    def __init__(self,group,site,g,orient):
        import torch
        self.fruit = group.scene["fruit_0"]
        origin = group.origins
        self.point = origin.new_tensor(g["attachment"]).expand_as(origin)
        self.anchor = origin+origin.new_tensor(g["branch_end"])
        self.orientation = origin.new_tensor(orient).expand(len(origin),4)
        from isaaclab.utils.math import quat_apply,quat_apply_inverse
        data = self.fruit.data
        local_com = quat_apply_inverse(data.root_link_quat_w.torch,data.root_com_pos_w.torch-data.root_link_pos_w.torch)
        self.rest_com = self.anchor+quat_apply(self.orientation,local_com-self.point)
        self.preload = origin.new_tensor([0.,0.,site.mass*9.81])
        self.k = site.mass*740.
        self.d = 2*.52*np.sqrt(site.mass*self.k)
        self.limit = site.break_n
        inertia = .4*site.mass*site.radius**2
        self.bending = 144.*inertia
        self.bend_damping = 2*.7*12.*inertia
        self.attached = torch.ones(len(origin),device=origin.device,dtype=torch.bool)

    def apply(self):
        import torch
        from isaaclab.utils.math import quat_error_magnitude,quat_mul,quat_conjugate
        d = self.fruit.data
        vel = d.root_com_vel_w.torch
        # Diagonal COM stiffness and angular compliance avoid coupling a lightweight
        # tip's angular velocity into an unstable translational force.
        f = self.preload+self.k*(self.rest_com-d.root_com_pos_w.torch)-self.d*vel[:,:3]
        # A real peduncle also resists bending. At rest its bending moment balances
        # the fruit's off-centre weight, particularly important for a curved banana.
        delta = quat_mul(self.orientation,quat_conjugate(d.root_link_quat_w.torch))
        axis = torch.nn.functional.normalize(delta[:,:3],dim=-1)*torch.sign(delta[:,3:])
        bend = self.bending*axis*quat_error_magnitude(self.orientation,d.root_link_quat_w.torch)[:,None]
        moment = bend-self.bend_damping*vel[:,3:]
        self.attached &= f.norm(dim=-1)<self.limit
        f = torch.where(self.attached[:,None],f,0.)
        moment = torch.where(self.attached[:,None],moment,0.)
        self.fruit.permanent_wrench_composer.set_forces_and_torques_index(forces=f[:,None],torques=moment[:,None],is_global=True)


def sightline_solver(eyes,target,chassis,side):
    def clearance(kin,base,q,focus):
        directions = focus-eyes
        frame,links = base,[]
        for index,joint in enumerate(kin.arm,1):
            frame = kin.fk((joint,),q,frame)
            if index>=2:
                links.append(frame[:3,3])
        joints = np.stack(links)
        samples = np.concatenate((joints,(joints[1:]+joints[:-1])*.5))
        # Include the Shadow forearm, wrist and palm mounting collar after UR10e joint six.
        radii = np.array([.18]*5+[.12,.12,.087,.055])
        clearance_radius = np.r_[radii,(radii[:-1]+radii[1:])*.5]
        rel = samples[:,None]-eyes
        along = (rel*directions).sum(-1)/np.square(directions).sum(-1).clip(1e-6)
        gap = np.linalg.norm(rel-along[:,:,None].clip(0.,.92)*directions,axis=-1)
        near_lens = np.linalg.norm(rel,axis=-1)
        local = (joints[1:4]-chassis[:3,3])@chassis[:3,:3]
        lateral = local[:,1]*(1 if side==0 else -1)
        high_inboard = np.maximum(0.,local[:,2]-1.16)*np.clip((.85-lateral)/.40,0.,1.)
        over_basket = (abs(local[:,1])<.49)&(local[:,0]>-.25)&(local[:,0]<.79)
        basket = np.maximum(0.,.86-local[:,2])*over_basket
        return np.r_[.84*np.maximum(0.,clearance_radius[:,None]-gap).ravel(),1.4*np.maximum(0.,.26-near_lens).ravel(),
                     .32*high_inboard,1.5*basket]
    def solve(kin,base,seed,low,high,pos,axes,focus=None):
        chain = kin.arm
        focus = target if focus is None else focus
        def residual(values,clearance_weight=1.):
            q = np.r_[values,seed[8:]]
            hand = kin.fk(chain,q,base)
            return np.r_[hand[:3,3]-pos,.24*Rotation.from_matrix(axes.T@hand[:3,:3]).as_rotvec(),
                         clearance_weight*clearance(kin,base,q,focus),.0015*(values-seed[:8])]
        initial = np.clip(seed[:8],low[:8],high[:8])
        res = least_squares(residual,initial,bounds=(low[:8],high[:8]),max_nfev=48)
        if np.linalg.norm(res.fun[:3])>.006:
            res = least_squares(residual,res.x,bounds=(low[:8],high[:8]),args=(.08,),max_nfev=60)
        return np.r_[res.x,seed[8:]],np.linalg.norm(res.fun[:3])
    return solve,clearance


def tuck(group,index,side,q,limits):
    """Put the idle palm outboard and below the lenses, with the elbow clear of the eye baseline."""
    from agriculture_grasp_geometry import _Kinematics
    owner = ("LeftArm","RightArm")[side]
    kin = _Kinematics(group.scene.stage,group.scene.env_prim_paths[index]+"/Rover/"+owner)
    data = group.rover.robot.data
    base = matrix(data.body_link_pose_w.torch[index,group.rover.bodies[owner,"base_link"]].cpu().numpy())
    chassis = matrix(data.root_link_pose_w.torch[index].cpu().numpy())
    sign = 1 if side==0 else -1
    goal = (chassis@np.array([.08,sign*.72,.55,1.]))[:3]
    def cost(v):
        p = np.r_[v,q[6:]]
        links = np.stack([kin.fk(kin.arm[:i],p,base)[:3,3] for i in (3,4,5)])
        local = (links-chassis[:3,3])@chassis[:3,:3]
        return np.r_[kin.fk(kin.arm,p,base)[:3,3]-goal,
                     .75*np.maximum(0.,.51-sign*local[:,1]),
                     1.1*np.maximum(0.,local[:,2]-1.20),.014*(v-q[:6])]
    low,high = limits[:6].T
    res = least_squares(cost,np.clip(q[:6],low,high),bounds=(low,high))
    return np.r_[res.x,q[6:]]


def visibility(group,depth,center,radius):
    """Check actual metric depth around the projected fruit. Foreground can include foliage or fingers."""
    from isaaclab.utils.math import quat_apply_inverse
    result = []
    for side,key in enumerate(("left_eye","right_eye")):
        cam = group.scene[key].data
        coords = quat_apply_inverse(cam.quat_w_ros.torch,center-cam.pos_w.torch).cpu().numpy()
        intrinsics = cam.intrinsic_matrices.torch.cpu().numpy()
        views = []
        for i,((x,y,z),k) in enumerate(zip(coords,intrinsics)):
            frame = depth[i,side]
            if not (np.isfinite(frame)&(frame>0.)).any():
                views.append("sin_profundidad")
                continue
            if ((frame>.01)&(frame<.24)).mean()>.55:
                views.append("visor_cubierto")
                continue
            u,v = (k[0,0]*x/max(.001,z)+k[0,2],k[1,1]*y/max(.001,z)+k[1,2])
            if z<.055 or not (2<=u<frame.shape[1]-2 and 2<=v<frame.shape[0]-2):
                views.append("fuera")
                continue
            r = max(1,min(16,round(.52*radius*k[0,0]/z)))
            iy,ix = round(v),round(u)
            patch = frame[max(0,iy-r):iy+r+1,max(0,ix-r):ix+r+1]
            near = np.isfinite(patch)&(patch>.01)&(patch<z-radius-.025)
            views.append("primer_plano" if near.mean()>.65 else "linea_libre")
        result.append(views)
    return list(map(list,zip(*result)))


def first_blockers(group,centers,query):
    """Ray-test the optical centres against actual PhysX bodies, including both arms."""
    targets = centers.cpu().numpy()
    rows = [[] for _ in targets]
    for eye in ("left_eye","right_eye"):
        origins = group.scene[eye].data.pos_w.torch.cpu().numpy()
        for i,(origin,target) in enumerate(zip(origins,targets)):
            vector = target-origin
            distance = np.linalg.norm(vector)
            direction = vector/max(.001,distance)
            hit = query.raycast_closest(tuple(map(float,origin+.056*direction)),tuple(map(float,direction)),
                                        max(.001,float(distance-.056)))
            path = str(hit.get("collision",""))
            if "/Hand/" in path:
                label = "mano"
            elif "/LeftArm/" in path or "/RightArm/" in path:
                label = "brazo"
            elif "/Fruit0" in path:
                label = "fruto"
            elif path:
                label = "otra_superficie"
            else:
                label = "sin_impacto"
            rows[i].append(label)
    return rows


def eye_pixels(group,centers=None):
    """Count visible target and arm pixels using the native renderer's mesh identifiers."""
    channel = "instance_id_segmentation_fast"
    views,blockers = [],[]
    for eye in ("left_eye","right_eye"):
        data = group.scene[eye].data
        labels = data.info[channel].get("idToLabels",{})
        frames = data.output[channel].torch[...,0].cpu().numpy()
        fruit_ids,arm_ids = [],[]
        for key,path in labels.items():
            name = str(path)
            if "/Produce/Fruit0" in name:
                fruit_ids.append(int(key))
            elif ("/LeftArm/" in name or "/RightArm/" in name) and (
                "/Hand/" not in name or "/robot0_forearm" in name or "/robot0_wrist" in name
            ):
                arm_ids.append(int(key))
        views.append([[int(np.isin(frame,fruit_ids).sum()),int(np.isin(frame,arm_ids).sum())] for frame in frames])
        if centers is not None:
            from isaaclab.utils.math import quat_apply_inverse
            coords = quat_apply_inverse(data.quat_w_ros.torch,centers-data.pos_w.torch).cpu().numpy()
            projected = []
            for frame,(x,y,z),k in zip(frames,coords,data.intrinsic_matrices.torch.cpu().numpy(),strict=True):
                u,v = round(k[0,0]*x/max(.001,z)+k[0,2]),round(k[1,1]*y/max(.001,z)+k[1,2])
                if z<=.055 or not (0<=u<frame.shape[1] and 0<=v<frame.shape[0]):
                    projected.append("fuera")
                    continue
                idx = int(frame[v,u])
                path = str(labels.get(str(idx),labels.get(idx,"")))
                label = ("fruto" if idx in fruit_ids else "brazo" if idx in arm_ids else "mano" if "/Hand/" in path
                         else "chasis" if "/Rover/" in path else "entorno")
                projected.append(label)
            blockers.append(projected)
    pixels = list(map(list,zip(*views)))
    return (pixels,list(map(list,zip(*blockers)))) if centers is not None else pixels


def record(env,crop,out,versions,g,files,orient,ffmpeg, *, survey=False,support_check=False,extended_lift=False,physics_only=False,
           controlled=False,force_servo=False,surface_audit=False,lift_reserve=None,height_servo=False):
    import torch
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from dexman.experiments import sha256
    from dexman.tasks.agriculture.contact_quality import antipodal_power_grasp, steady_contacts
    from dexman.tasks.agriculture.mobile_state import VERSION
    from dexman.tasks.agriculture.rover_basket_task import BasketTrial,over_basket
    site = SITES[crop]
    batch = env.batch
    group, = batch.groups
    rover = group.rover
    d = rover.robot.data
    origin = group.origins
    n = len(versions)
    ids = torch.arange(n,device=env.device)
    sides = ids.new_tensor([v[1] for v in versions])
    stem = Stem(group,site,g,orient)
    film = None if survey or physics_only else Film(out,versions,record_obstructions=True)
    contacts = []
    pairs = []
    nominal = env.home.clone()
    stowed = nominal.clone()
    for i in range(n):
        for side in (0,1):
            part = slice(30*side,30*(side+1))
            stowed[i,part] = origin.new_tensor(tuck(group,i,side,nominal[i,part].cpu().numpy(),
                                                        batch.joint_limits[i,part].cpu().numpy()))
    # Only motors move the chassis and both arms, including their transition into the transport pose.
    for t in range(54):
        target_world = stem.fruit.data.root_com_pos_w.torch
        look = .5*(aim_active_eye(group,torch.zeros_like(sides),target_world)+aim_active_eye(group,torch.ones_like(sides),target_world))
        batch.step(nominal.lerp(stowed,min(1.,(t+1)/42)),head_targets=look,before_physics_step=stem.apply)
        if support_check and t in (0,1,2,5,12,30,53):
            print("pedunculo",t,stem.attached.tolist(),(stem.fruit.data.root_link_pos_w.torch-origin).cpu().numpy().round(3).tolist(),flush=True)
    if support_check:
        if not stem.attached.all():
            raise RuntimeError("Desprendimiento antes del recorrido")
        return
    report_rows = []
    view_rows = []
    ray_rows = []
    pixel_rows = []
    target_rows = []
    import omni.physx
    query = omni.physx.get_physx_scene_query_interface()
    routes = []
    step = 0
    # Store the actual articulated poses of physics-only candidates so individual
    # replicas can later be rendered without rerunning a different contact solve.
    recorded_states = []
    recorded_phases = []
    executed_phases = []
    def save_state(phase,stops=None):
        if controlled and step%2==0:
            state = torch.cat((d.root_link_pose_w.torch[:,:3]-origin,d.root_link_quat_w.torch,
                               d.joint_pos.torch,stem.fruit.data.root_link_pos_w.torch-origin,
                               stem.fruit.data.root_link_quat_w.torch),dim=-1)
            recorded_states.append(state.cpu().numpy().copy())
            recorded_phases.append(phase)
            executed_phases.append([phase if stops is None or stops[i] is None else "detenido" for i in range(n)])
    reference_z = env._measure()[-1].clone()
    from field_contact_trace import ContactTrace
    trace = ContactTrace(group,stem.fruit,sides,env.fruit_collider_points[group.crop]) if surface_audit else None
    start = (d.root_link_pos_w.torch-origin).cpu().numpy().copy()
    zeros = origin.new_zeros((n,4))
    route = g["route"]
    # Two real approach arcs converge to the same physical harvesting station.
    for name,side,_,_ in versions:
        curve = route.copy()
        curve[:,1] += (.045 if side==0 else -.04)*np.sin(np.linspace(0,np.pi,len(curve)))**2
        routes.append(curve)
    indices = np.zeros(n,dtype=int)
    reached = np.zeros(n,dtype=bool)
    path_lengths = np.zeros(n)
    prev_xy = start[:,:2]
    def render_frame(phase,lift,held,labels=None):
        witness,detail = group.scene["witness"],group.scene["detail"]
        depth = film.frame(batch,witness,detail,phase if labels is None else labels,lift,held)
        view_rows.append([step,phase,visibility(group,depth,stem.fruit.data.root_com_pos_w.torch,site.radius)])
        ray_rows.append([step,phase,first_blockers(group,stem.fruit.data.root_com_pos_w.torch,query)])
        if controlled:
            count,front = eye_pixels(group,stem.fruit.data.root_com_pos_w.torch)
            pixel_rows.append([step,phase,count])
            target_rows.append([step,phase,front])
    def cameras(nav):
        pos = d.root_link_pos_w.torch
        fruit = stem.fruit.data.root_com_pos_w.torch
        def delta(xyz):
            return origin.new_tensor(xyz)
        focus = pos+delta([.26,0.,.46])
        group.scene["witness"].set_world_poses_from_view(pos+delta([-1.72,-1.45,2.95]),focus)
        if nav:
            group.scene["detail"].set_world_poses_from_view(pos+delta([1.92,.50,1.82]),focus)
        else:
            # Independent exterior close-up remains on the hand and the basket if the fruit falls.
            focal = fruit if controlled else origin+origin.new_tensor(g["fruit"])
            group.scene["detail"].set_world_poses_from_view(focal+delta([.38,-.70,.43]),focal+delta([-.06,0.,-.04]))
    def stereo_gaze(target=None):
        if target is None:
            target = stem.fruit.data.root_com_pos_w.torch
        return .5*(aim_active_eye(group,torch.zeros_like(sides),target)+aim_active_eye(group,torch.ones_like(sides),target))
    for tick in range(600):
        local = (d.root_link_pos_w.torch-origin).cpu().numpy()
        heading = Rotation.from_quat(d.root_link_quat_w.torch.cpu().numpy()).as_euler("xyz")[:,2]
        speed = []
        for i,curve in enumerate(routes):
            xy = local[i,:2]
            nearest = indices[i]+np.argmin(np.linalg.norm(curve[indices[i]:]-xy,axis=1))
            indices[i] = nearest
            vec = curve[min(len(curve)-1,nearest+12)]-xy
            distance = np.linalg.norm(curve[-1]-xy)
            err = np.arctan2(vec[1],vec[0])-heading[i]
            err = np.arctan2(np.sin(err),np.cos(err))
            forward_gap = curve[-1,0]-xy[0]
            # Brake at the longitudinal mark even on a cross slope. Steering toward
            # a point behind a four-wheel rover would drive past the fruit.
            reached[i] |= forward_gap<.045
            if reached[i]:
                v,w = 0.,0.
            else:
                v = min(.55,max(.095,1.8*distance))*max(.25,np.cos(err))
                # Keep all tyres rolling while approaching the parking mark.
                w = np.clip(1.10*err,-.94*v,.94*v)
            speed.append([(v-.59*w)/.222,(v+.59*w)/.222]*2)
        rover.robot.set_joint_velocity_target_index(target=origin.new_tensor(speed),joint_ids=rover.wheels)
        record_now = film is not None and step%2==0
        if record_now:
            cameras(True)
        gaze = stereo_gaze()
        batch.step(stowed,head_targets=gaze,render=record_now,before_physics_step=stem.apply)
        if record_now:
            render_frame("recorrido del surco",np.zeros(n),np.zeros(n))
        save_state("recorrido")
        path_lengths += np.linalg.norm(local[:,:2]-prev_xy,axis=1)
        prev_xy = local[:,:2].copy()
        report_rows.append([step,"recorrido",local.tolist()])
        step += 1
        if tick%90==0:
            print("recorrido",tick,local[:,:2].round(3).tolist(),heading.round(2).tolist(),
                  d.joint_vel.torch[:,rover.wheels].cpu().numpy().round(1).tolist(),flush=True)
        if tick>60 and reached.all():
            break
    rover.robot.set_joint_velocity_target_index(target=zeros,joint_ids=rover.wheels)
    if survey:
        print("Error final de estacionamiento (m)",np.linalg.norm(local[:,:2]-route[-1],axis=1),
              "fruto en planta",stem.attached.tolist(),flush=True)
        return
    for tick in range(24):
        rec = film is not None and step%2==0
        if rec:
            cameras(False)
        batch.step(stowed,head_targets=stereo_gaze(),
                   render=rec,before_physics_step=stem.apply)
        if rec:
            render_frame("estacionamiento",np.zeros(n),np.zeros(n))
        save_state("estacionamiento")
        step+=1
    parked = d.root_link_pos_w.torch.clone()
    park_local = (parked-origin).cpu().numpy()
    print("estacionado",park_local.round(3).tolist(),"distancia",path_lengths.round(2).tolist(),flush=True)
    m = env._measure()
    ready = stem.attached.clone()
    seedfile = ROOT/"assets/agriculture/grasps"/(crop+".json")
    sources = [*files,seedfile]
    reference = np.array(json.loads(seedfile.read_text())["waypoints_rad"])
    right_seed = None
    if any(name in RIGHT_PROFILES for name,_,_,_ in versions):
        from dexman.robot.joints import JOINT_NAMES
        seed_asset = seedfile.parent/"banana_rover_right.json"
        calibration = json.loads(seed_asset.read_text(encoding="utf-8"))
        right_seed = np.asarray(calibration["closed_rad"],dtype=float)
        if tuple(calibration["joints"])!=JOINT_NAMES[:8] or right_seed.shape!=(8,) or not np.isfinite(right_seed).all():
            raise ValueError("Calibración de muñeca incompatible con el rover")
        if tuple(calibration["hold"]["finger_joints"])!=JOINT_NAMES[8:]:
            raise ValueError("Orden de dedos incompatible con la calibración de sostén")
        sources.append(seed_asset)
    if extended_lift:
        # Follow each calibrated finger's direction of closure; the physical joint limits
        # and actuator effort limits stay in effect. Allow for compliance during lifting.
        reference[2:,8:] += .12*(reference[2,8:]-reference[1,8:])
    paths,errors = [],[[] for _ in versions]
    for i,(name,side,approach,offset) in enumerate(versions):
        tilt = d.body_link_pose_w.torch[i,rover.tilt_body]
        eyes = (tilt[:3]-origin[i]+quat_apply(tilt[3:].expand(2,4),rover.eye_offsets)).cpu().numpy()
        fruit = np.array(g["fruit"])
        chassis = matrix(d.root_link_pose_w.torch[i].cpu().numpy())
        chassis[:3,3] -= origin[i].cpu().numpy()
        solver,clearance = sightline_solver(eyes,fruit,chassis,side)
        paths.append(make_path(env,i,side,approach,reference,fruit,offset,False,
                     reference_fruit=site.reference,sightline_solver=solver,residuals=errors[i],clearance_cost=clearance,
                     keep_fruit_heading=crop in ("banana","vineyard"),lift_margin_m=lift_reserve if lift_reserve is not None else (.075 if extended_lift else 0.),
                     controlled_delivery=controlled,pickup_seed=right_seed if name in RIGHT_PROFILES else None,
                     basket_target=calibration["basket_targets_m"][name] if controlled and name in RIGHT_PROFILES else None))
    print("IK max residuals m",[round(max(e),3) for e in errors],flush=True)
    tests = [BasketTrial(require_continuous_grip=controlled) for _ in versions]
    surface_tests = [BasketTrial(require_continuous_grip=True) for _ in versions]
    from dexman.robot.joints import FINGERTIPS
    substeps = []
    tip_sensors = [[group.scene[f"{key}_{tip}"] for tip in FINGERTIPS] for key in ("left_arm","right_arm")]
    palm_ids = [group.arm_body_ids(("left_arm","right_arm")[side],["robot0_palm"])[0] for _,side,_,_ in versions]
    previous = stowed
    from dexman.tasks.agriculture.field_lift_control import LiftHeightControl
    height_control = LiftHeightControl(paths,versions) if height_servo else None
    calibrated_rows = [i for i,(name,_,_,_) in enumerate(versions) if name=="derecha_calibrada"]
    if calibrated_rows:
        height_control.feedforward[calibrated_rows] = calibration["hold"]["wrist_path_fraction"]
        loaded_targets = origin.new_tensor(calibration["hold"]["motor_targets_rad"])
        carry_fingers = loaded_targets.expand(n,-1).clone()
    finger_deflection = torch.zeros_like(stowed)
    relief = np.zeros(n)
    stopped = [None]*n
    last_command = stowed.clone()
    for k in range(len(paths[0])):
        _,phase,ticks = paths[0][k]
        if force_servo and phase=="sosten":
            # Give the compliant wrist time to settle at its final motor
            # target before judging the hold or starting the delivery.
            ticks = 210+round(2.5/env.dt)
        if k==0:
            ticks = 68
        target = stowed.clone()
        for i,(_,side,_,_) in enumerate(versions):
            target[i,side*30:(side+1)*30] = origin.new_tensor(paths[i][k][0])
            if force_servo and phase=="sosten":
                fingers = slice(side*30+8,(side+1)*30)
                finger_deflection[i,fingers] = target[i,fingers]-d.joint_pos.torch[i,group.joint_ids[side][8:]]
        for tick in range(ticks):
            rec = film is not None and step%2==0
            if rec:
                cameras(False)
            u = (tick+1)/ticks
            # During the delivery observe the moving hand even if the fruit was lost.
            gaze_target = None
            if not controlled and phase in ("traslado","apertura","retirada"):
                palms = torch.stack([d.body_link_pos_w.torch[:,group.arm_body_ids(key,["robot0_palm"])[0]]
                                     for key in ("left_arm","right_arm")],dim=1)
                gaze_target = palms[ids,sides]
            gaze = stereo_gaze(gaze_target)
            if controlled and phase in ("sosten","traslado","apertura","retirada"):
                error_body = quat_apply_inverse(d.root_link_quat_w.torch,parked-d.root_link_pos_w.torch)
                drive = (3.*error_body[:,0]/.222).clamp(-.6,.6)
                rover.robot.actuators.target_command.set_velocity_index(value=drive[:,None].expand(-1,4).contiguous(),joint_ids=rover.wheels)
            command = previous.lerp(target,u*u*(3-2*u))
            if height_control and phase in ("elevacion","sosten","traslado"):
                height_control.apply(command,phase,data[:,0],env.dt)
            if force_servo and phase in ("sosten","traslado","apertura"):
                if phase=="sosten" and tick<210:
                    for i,test in enumerate(surface_tests if surface_audit else tests):
                        if not test.retained:
                            # Settle first, then gradually reduce compression of the
                            # measured motor springs. Lock it once a real hold is proven.
                            relief[i] = min(.90 if height_servo else .80,max(0.,(tick-15)/145.))
                release_factor = 1.-u if phase=="apertura" else 1.
                command -= origin.new_tensor(relief)[:,None]*finger_deflection*release_factor
                for i in calibrated_rows:
                    # Interpolate real motor targets recorded during a stable
                    # right-hand grasp. The fruit remains a free rigid body.
                    fingers = slice(38,60)
                    if phase=="sosten":
                        u_load = min(1.,tick*env.dt/3.5)
                        command[i,fingers] = target[i,fingers].lerp(loaded_targets,u_load*u_load*(3.-2.*u_load))
                    elif phase=="traslado":
                        command[i,fingers] = carry_fingers[i]
                    else:
                        command[i,fingers] = carry_fingers[i].lerp(target[i,fingers],u*u*(3.-2.*u))
                if phase=="sosten":
                    for i,test in enumerate(surface_tests if surface_audit else tests):
                        if test.retained or tick>=210:
                            command[i] = last_command[i]
                            if height_control:
                                side = height_control.sides[i]
                                height_control.bias[i] = command[i,side*30:side*30+8].cpu().numpy()-height_control.paths[i][-1]
                    for i in calibrated_rows:
                        carry_fingers[i] = command[i,38:60]
            if force_servo and phase in ("traslado","apertura","retirada"):
                for i,test in enumerate(surface_tests if surface_audit else tests):
                    if stopped[i] is None:
                        if phase=="traslado" and not test.retained:
                            stopped[i] = "sostén sin validar"
                        elif phase in ("traslado","apertura") and test.lost_before_release:
                            stopped[i] = "pérdida de oposición"
                        elif phase=="apertura" and tick==0 and not over_basket(data[i,4:7]):
                            stopped[i] = "fruto fuera de la boca de la canasta"
                    if stopped[i] is not None:
                        command[i] = last_command[i]
            tracing = trace is not None and phase in ("cierre","elevacion","sosten","traslado")
            if tracing:
                trace.begin()
            batch.step(command,head_targets=gaze,render=rec,before_physics_step=stem.apply,
                       after_physics_step=trace.sample if tracing else None)
            last_command = command
            if tracing:
                measured_substeps = torch.stack(trace.samples).cpu().numpy()
                substeps.append([step,phase,np.where(np.isfinite(measured_substeps),measured_substeps,None).tolist()])
            m = env._measure()
            held = m[6][ids,sides]
            steady = steady_contacts(m[11][ids,sides],m[5][ids,sides])
            # Per-digit contact evidence and fruit centre relative to the actual palm.
            palm = d.body_link_pose_w.torch[ids,palm_ids]
            relative = quat_apply_inverse(palm[:,3:],m[8]+origin-palm[:,:3])
            samples = torch.cat((m[10][ids,sides],m[11][ids,sides],relative),dim=1).cpu().tolist()
            contacts.append([step,phase,[[v if np.isfinite(v) else None for v in row] for row in samples]])
            if controlled and crop=="banana":
                measurements = []
                for sensors in tip_sensors:
                    f = torch.stack([sensor.data.normal_force_matrix_w.torch[:,0,0] for sensor in sensors],dim=1)
                    p = torch.stack([sensor.data.contact_pos_w.torch[:,0,0] for sensor in sensors],dim=1)
                    measurements.append(torch.cat((f,p-stem.fruit.data.root_com_pos_w.torch[:,None]),dim=-1))
                vectors = torch.stack(measurements,dim=1)[ids,sides]
                held = antipodal_power_grasp(vectors[:,:,:3],vectors[:,:,3:],m[5][ids,sides],.60)
                if phase in ("cierre","elevacion","sosten","traslado"):
                    raw = vectors.cpu().numpy()
                    pairs.append([step,phase,np.where(np.isfinite(raw),raw,None).tolist()])
            lift = m[-1]-reference_z
            pos = d.root_link_pos_w.torch
            basket = quat_apply_inverse(d.root_link_quat_w.torch,m[8]+origin-pos)
            normal = group.scene["basket_contact"].data.normal_force_matrix_w.torch.flatten(1).norm(dim=-1)
            data = torch.column_stack((lift,held,steady,m[1][:,:3].norm(dim=1),basket,normal,
                (m[4]-m[8][:,None,None]).norm(dim=-1).amin((1,2)),m[5].any((1,2)),(pos-parked).norm(dim=1),m[7],stem.attached)).cpu().numpy()
            for i,(height,h,s,vel,x,y,z,force,gap,touch,drift,impact,attached) in enumerate(data.tolist()):
                executed_phase = phase if stopped[i] is None else "detenido"
                tests[i].observe(phase=executed_phase,lift=height,held=bool(h),steady=bool(s) and not attached,speed=vel,
                    basket_position=(x,y,z),basket_contact=force>.04,finger_clearance=gap,touching=bool(touch),
                    drift=drift,impact=impact,dt=env.dt)
                if surface_audit:
                    if tracing:
                        samples_i = measured_substeps[:,i]
                        h = h and samples_i[:,0].all()
                        s = s and (samples_i[:,1]<.08).all()
                        vel = max(vel,float(samples_i[:,2].max()))
                        height = float(samples_i[:,3].min())
                    surface_tests[i].observe(phase=executed_phase,lift=height,held=bool(h),steady=bool(s) and not attached,
                        speed=vel,basket_position=(x,y,z),basket_contact=force>.04,finger_clearance=gap,
                        touching=bool(touch),drift=drift,impact=impact,dt=env.dt)
            if rec:
                render_frame(phase,data[:,0],data[:,1],[phase if reason is None else "detenido" for reason in stopped])
            save_state(phase,stopped)
            report_rows.append([step,phase,data.tolist()])
            step+=1
            active_tests = surface_tests if surface_audit else tests
            if force_servo and phase=="sosten" and all(test.retained or not bool(ready[i]) or data[i,0]<.08
                                                      for i,test in enumerate(active_tests)) and tick>=30:
                break
        previous = target
        if k+1==len(paths[0]) or paths[0][k+1][1]!=phase:
            print(phase,step,"altura / sostén / traslado con agarre / en canasta",
                  [(round(float(row[0]),3),round(test.longest_hold*env.dt,2),
                    f"{test.held_delivery_ticks}/{test.delivery_ticks}",round(test.basket_ticks*env.dt,2))
                   for row,test in zip(data,tests,strict=True)],flush=True)
    robots = []
    for i,(name,side,_,_) in enumerate(versions):
        distance = float(np.linalg.norm(park_local[i,:2]-route[-1]))
        valid_run = bool(ready[i]) and bool(reached[i]) and distance<.38 and stopped[i] is None
        observed = [v[i] for _,phase,v in view_rows if phase in ("aproximacion","cierre")]
        free = [round(sum(row[j]=="linea_libre" for row in observed)/len(observed),3) if observed else None for j in (0,1)]
        checked = [v[i] for _,phase,v in ray_rows if phase in ("aproximacion","cierre")]
        arm_block,ray_coverage = arm_occlusion(checked)
        robots.append(dict(robot=i,version=name,hand=("izquierda","derecha")[side],travel_m=round(path_lengths[i],3),
            motor_spring_relief=round(float(relief[i]),3),motion_stop_reason=stopped[i],
            parking_error_m=round(distance,3),started_attached=bool(ready[i]),detached=not bool(stem.attached[i]),
            max_ik_residual_m=round(max(errors[i]),4),free_depth_fraction_left_right=free,
            arm_block_fraction_of_identified_hits=arm_block,ray_hit_fraction_left_right=ray_coverage,
            **tests[i].result(env.dt,valid_start=valid_run)))
        if surface_audit:
            robots[-1]["surface_audit"] = surface_tests[i].result(env.dt,valid_start=valid_run)
            robots[-1]["success"] &= robots[-1]["surface_audit"]["success"]
        if pixel_rows:
            robots[-1]["binocular_visibility"] = binocular_visibility(
                [(step,phase,rows[i]) for step,phase,rows in pixel_rows],
                [(step,phase,rows[i]) for step,phase,rows in view_rows])
    def write(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2 if name=="resultado.json" else None),encoding="utf-8")
    write("resultado.json",dict(crop=crop,scene=files[0].name,seed=batch.seed,simulation="Isaac Sim 6.2 / PhysX",
         profile="Traslado continuo" if controlled else ("Elevación ampliada" if extended_lift else "Trayectoria base"),
         contract=VERSION,control_hz=1/env.dt,physics_hz=1/batch.sim.get_physics_dt(),frames=film.index if film else 0,fps=15,
         columns=dict(recorrido=["base_x_m","base_y_m","base_z_m"],manipulacion=["lift_m","held","steady","speed_m_s",
             "basket_x_m","basket_y_m","basket_z_m","basket_force_n","finger_gap_m","touching","base_drift_m","forbidden_force_n","stem_attached"]),
         instruction=f"Recorre el surco y estaciónate frente a la planta. Toma {site.fruit}. Eleva la fruta 15 cm, sostenla medio segundo y déjala en la canasta.",
         field_seed=FIELDS[crop].seed,target_plant="Hileras/Fila2/Planta03",stem_break_n=site.break_n,
         grasp_evidence="tres dedos y par antipodal Coulomb mu=0.60" if controlled and crop=="banana" else "oposición centrada",
         arguments=sys.argv[1:],source_sha256={**SOURCE_DIGESTS,**{p.name:sha256(p) for p in sources}},robots=robots))
    write("mediciones.json",report_rows)
    if substeps:
        write("subpasos.json",dict(columns=["antipodal","surface_slip_max_m_s","fruit_com_speed_m_s","lowest_lift_m"],samples=substeps))
    if recorded_states:
        np.savez_compressed(out/"poses_fisicas.npz",poses=np.stack(recorded_states),
                            joints=np.array(rover.robot.joint_names),phases=np.array(recorded_phases),
                            executed_phases=np.array(executed_phases),
                            step=np.arange(len(recorded_states))*2)
    write("contactos.json",dict(columns=["normal_n_5","slip_m_s_5","fruit_in_palm_m_xyz"],samples=contacts))
    write("pares.json",dict(columns=["force_on_finger_w_n_xyz","point_from_fruit_com_w_m_xyz"],samples=pairs))
    write("visibilidad.json",view_rows)
    write("rayos.json",ray_rows)
    if pixel_rows:
        write("pixeles.json",dict(columns=["fruit_px","arm_px"],samples=pixel_rows))
        write("superficie_frente_al_fruto.json",target_rows)
    print(json.dumps(robots),flush=True)
    if film:
        film.encode(ffmpeg)


def resolve_grasps(args):
    """Resolve the physical demonstration and its control settings before starting Kit."""
    versions = VERSIONS
    if args.crop=="banana" and args.banana_grasps is None:
        args.banana_grasps = ("derecha_calibrada","lateral")
    if args.banana_grasps:
        if args.crop!="banana":
            raise ValueError("--banana-grasps requiere --crop banana")
        versions = [(name,*VERSIONS[1 if name=="lateral" else 0][1:]) for name in args.banana_grasps]
        if any(name in RIGHT_PROFILES for name in args.banana_grasps):
            args.controlled_delivery = args.extended_lift = True
            args.height_servo = args.force_servo = args.surface_audit = True
    if args.force_servo and (args.crop!="banana" or not args.controlled_delivery):
        raise ValueError("--force-servo requiere --crop banana --controlled-delivery")
    if "derecha_calibrada" in (args.banana_grasps or ()) and args.lift_reserve is not None:
        raise ValueError("La postura de retención derecha_calibrada se midió con la reserva estándar de 7,5 cm")
    return versions


if __name__=="__main__":
    from isaaclab.app import add_launcher_args,launch_simulation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop",required=True,choices=SITES)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=751)
    parser.add_argument("--ffmpeg",default="ffmpeg")
    parser.add_argument("--survey",action="store_true",help="Mide el recorrido sin grabar ni mover los brazos a la planta")
    parser.add_argument("--physics-only",action="store_true",help="Evalúa toda la trayectoria física sin renderizar")
    parser.add_argument("--controlled-delivery",action="store_true",help="Estacionamiento activo y traslado continuo hacia el interior de la canasta")
    parser.add_argument("--physics-substeps",type=int,default=4,help="Pasos PhysX por consigna a 30 Hz")
    parser.add_argument("--height-servo",action="store_true",help="Regula la altura del punto inferior del fruto a 15,7 cm durante el sostén")
    parser.add_argument("--force-servo",action="store_true",help="Ajusta la compresión de los dedos hasta medir medio segundo estable")
    parser.add_argument("--surface-audit",action="store_true",help="Mide el desplazamiento tangencial de puntos materiales en cada subpaso PhysX")
    parser.add_argument("--lift-reserve",type=float,help="Reserva de altura de muñeca en metros para compensar la flexión medida")
    parser.add_argument("--banana-grasps",nargs="+",choices=BANANA_GRASPS,help="Agarres laterales de plátano a evaluar en paralelo")
    parser.add_argument("--support-check",action="store_true",help="Comprueba el pedúnculo en reposo antes de grabar")
    parser.add_argument("--extended-lift",action="store_true",help="Añade 7,5 cm al recorrido de la muñeca y 12 % de cierre al agarre calibrado")
    add_launcher_args(parser)
    args = parser.parse_args()
    out = args.output.resolve()
    ffmpeg = None if args.physics_only or args.survey or args.support_check else encoder_path(args.ffmpeg)
    try:
        versions = resolve_grasps(args)
    except ValueError as error:
        parser.error(str(error))
    out.mkdir(parents=True,exist_ok=False)
    survey = args.survey or args.support_check
    cfg,g,orient = configure(out,args.crop,versions,survey or args.physics_only,args.physics_substeps,
                             verify_eyes=args.controlled_delivery)
    with launch_simulation(cfg,args):
        import torch
        from isaaclab.sim import SimulationContext
        from dexman.tasks.agriculture.environment import AgriculturalPPOEnv
        files = scene_files(out,args.crop,SITES[args.crop],g,orient)
        with torch.inference_mode():
            env = AgriculturalPPOEnv(SimulationContext(cfg.sim),cfg,seed=args.seed)
            record(env,args.crop,out,versions,g,files,orient,ffmpeg,survey=survey,support_check=args.support_check,
                   extended_lift=args.extended_lift,physics_only=args.physics_only,controlled=args.controlled_delivery,
                   force_servo=args.force_servo,surface_audit=args.surface_audit,lift_reserve=args.lift_reserve,
                   height_servo=args.height_servo)
