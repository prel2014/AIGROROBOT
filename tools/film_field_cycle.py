"""Record continuous, physically driven field approaches and harvest attempts, one pair per rover."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from rover_orange_trial import Film, ROOT, aim_active_eye, encoder_path, make_path, matrix
from dexman.tasks.agriculture.field_layout import FIELDS
from dexman.tasks.agriculture.harvest_route import ATTACHMENT_M, SITES, arm_occlusion, layout


VERSIONS = (("Acceso corto",1,(.02,-.13,.04),(-.008,-.006,.002)),
            ("Acceso lateral",0,(.02,.15,.065),(-.005,-.006,.003)))


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


def configure(out,crop,versions,survey=False):
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim import UsdFileCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxCollisionCfg
    from dexman.robot.config import robot_cfg
    from dexman.tasks.agriculture.batch import make_batch_config
    from dexman.tasks.agriculture.scene import camera, yaw, NS
    g = layout(crop)
    cfg = make_batch_config(len(versions),crop=crop,rover=True,contacts=True,rgbd=not survey,
                            rgbd_size=320,nominal=True,raised_support=True,physics_substeps=4)
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
    initial_correction = 5. if crop=="berries" else 1.5
    scene.rover.init_state.rot = yaw(np.degrees(np.arctan2(bearing[1],bearing[0]))+initial_correction)
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
    def solve(kin,base,seed,low,high,pos,axes):
        chain = kin.arm
        def residual(values,clearance_weight=1.):
            q = np.r_[values,seed[8:]]
            hand = kin.fk(chain,q,base)
            return np.r_[hand[:3,3]-pos,.24*Rotation.from_matrix(axes.T@hand[:3,:3]).as_rotvec(),
                         clearance_weight*clearance(kin,base,q,target),.0015*(values-seed[:8])]
        res = least_squares(residual,np.clip(seed[:8],low[:8],high[:8]),bounds=(low[:8],high[:8]),max_nfev=48)
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


def record(env,crop,out,versions,g,files,orient,ffmpeg,survey=False,support_check=False,extended_lift=False):
    import torch
    from isaaclab.utils.math import quat_apply_inverse
    from dexman.experiments import sha256
    from dexman.tasks.agriculture.contact_quality import steady_contacts
    from dexman.tasks.agriculture.mobile_state import VERSION
    from dexman.tasks.agriculture.rover_basket_task import BasketTrial
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
    film = None if survey else Film(out,versions,record_obstructions=True)
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
    import omni.physx
    query = omni.physx.get_physx_scene_query_interface()
    routes = []
    step = 0
    reference_z = env._measure()[-1].clone()
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
    def render_frame(phase,lift,held):
        witness,detail = group.scene["witness"],group.scene["detail"]
        depth = film.frame(batch,witness,detail,phase,lift,held)
        view_rows.append([step,phase,visibility(group,depth,stem.fruit.data.root_com_pos_w.torch,site.radius)])
        ray_rows.append([step,phase,first_blockers(group,stem.fruit.data.root_com_pos_w.torch,query)])
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
            focal = origin+origin.new_tensor(g["fruit"])
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
        print("Error final de estacionamiento (m)",np.linalg.norm(local[:,:2]-route[-1],axis=1),flush=True)
        return
    for tick in range(24):
        rec = step%2==0
        if rec:
            cameras(False)
        batch.step(stowed,head_targets=stereo_gaze(),
                   render=rec,before_physics_step=stem.apply)
        if rec:
            render_frame("estacionamiento",np.zeros(n),np.zeros(n))
        step+=1
    parked = d.root_link_pos_w.torch.clone()
    park_local = (parked-origin).cpu().numpy()
    print("estacionado",park_local.round(3).tolist(),"distancia",path_lengths.round(2).tolist(),flush=True)
    m = env._measure()
    ready = stem.attached.clone()
    seedfile = ROOT/"assets/agriculture/grasps"/(crop+".json")
    reference = np.array(json.loads(seedfile.read_text())["waypoints_rad"])
    if extended_lift:
        # Follow each calibrated finger's direction of closure; the physical joint limits
        # and actuator effort limits stay in effect. Allow for compliance during lifting.
        reference[2:,8:] += .12*(reference[2,8:]-reference[1,8:])
    paths,errors = [],[[] for _ in versions]
    for i,(_,side,approach,offset) in enumerate(versions):
        eyes = np.stack([(group.scene[k].data.pos_w.torch[i]-origin[i]).cpu().numpy() for k in ("left_eye","right_eye")])
        fruit = np.array(g["fruit"])
        chassis = matrix(d.root_link_pose_w.torch[i].cpu().numpy())
        chassis[:3,3] -= origin[i].cpu().numpy()
        solver,clearance = sightline_solver(eyes,fruit,chassis,side)
        paths.append(make_path(env,i,side,approach,reference,fruit,offset,False,
                     reference_fruit=site.reference,sightline_solver=solver,residuals=errors[i],clearance_cost=clearance,
                     keep_fruit_heading=crop in ("banana","vineyard"),lift_margin_m=.075 if extended_lift else 0.))
    print("IK max residuals m",[round(max(e),3) for e in errors],flush=True)
    tests = [BasketTrial() for _ in versions]
    previous = stowed
    for k in range(len(paths[0])):
        _,phase,ticks = paths[0][k]
        if k==0:
            ticks = 68
        target = stowed.clone()
        for i,(_,side,_,_) in enumerate(versions):
            target[i,side*30:(side+1)*30] = origin.new_tensor(paths[i][k][0])
        for tick in range(ticks):
            rec = step%2==0
            if rec:
                cameras(False)
            u = (tick+1)/ticks
            # During the delivery observe the moving hand even if the fruit was lost.
            gaze_target = None
            if phase in ("traslado","apertura","retirada"):
                palms = torch.stack([d.body_link_pos_w.torch[:,group.arm_body_ids(key,["robot0_palm"])[0]]
                                     for key in ("left_arm","right_arm")],dim=1)
                gaze_target = palms[ids,sides]
            gaze = stereo_gaze(gaze_target)
            batch.step(previous.lerp(target,u*u*(3-2*u)),head_targets=gaze,render=rec,before_physics_step=stem.apply)
            m = env._measure()
            held = m[6][ids,sides]
            steady = steady_contacts(m[11][ids,sides],m[5][ids,sides])
            lift = m[-1]-reference_z
            pos = d.root_link_pos_w.torch
            basket = quat_apply_inverse(d.root_link_quat_w.torch,m[8]+origin-pos)
            normal = group.scene["basket_contact"].data.normal_force_matrix_w.torch.flatten(1).norm(dim=-1)
            data = torch.column_stack((lift,held,steady,m[1][:,:3].norm(dim=1),basket,normal,
                (m[4]-m[8][:,None,None]).norm(dim=-1).amin((1,2)),m[5].any((1,2)),(pos-parked).norm(dim=1),m[7],stem.attached)).cpu().numpy()
            for i,(height,h,s,vel,x,y,z,force,gap,touch,drift,impact,attached) in enumerate(data.tolist()):
                tests[i].observe(phase=phase,lift=height,held=bool(h),steady=bool(s) and not attached,speed=vel,
                    basket_position=(x,y,z),basket_contact=force>.04,finger_clearance=gap,touching=bool(touch),
                    drift=drift,impact=impact,dt=env.dt)
            if rec:
                render_frame(phase,data[:,0],data[:,1])
            report_rows.append([step,phase,data.tolist()])
            step+=1
        previous = target
        print(phase,step,flush=True)
    robots = []
    for i,(name,side,_,_) in enumerate(versions):
        distance = float(np.linalg.norm(park_local[i,:2]-route[-1]))
        observed = [v[i] for _,phase,v in view_rows if phase in ("aproximacion","cierre")]
        free = [round(sum(row[j]=="linea_libre" for row in observed)/len(observed),3) for j in (0,1)]
        checked = [v[i] for _,phase,v in ray_rows if phase in ("aproximacion","cierre")]
        arm_block,ray_coverage = arm_occlusion(checked)
        robots.append(dict(robot=i,version=name,hand=("izquierda","derecha")[side],travel_m=round(path_lengths[i],3),
            parking_error_m=round(distance,3),started_attached=bool(ready[i]),detached=not bool(stem.attached[i]),
            max_ik_residual_m=round(max(errors[i]),4),free_depth_fraction_left_right=free,
            arm_block_fraction_of_identified_hits=arm_block,ray_hit_fraction_left_right=ray_coverage,
            **tests[i].result(env.dt,valid_start=bool(ready[i]) and bool(reached[i]) and distance<.38)))
    def write(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2 if name=="resultado.json" else None),encoding="utf-8")
    write("resultado.json",dict(crop=crop,scene=files[0].name,seed=batch.seed,simulation="Isaac Sim 6.2 / PhysX",
         profile="Elevación ampliada" if extended_lift else "Trayectoria base",
         contract=VERSION,control_hz=1/env.dt,physics_hz=1/batch.sim.get_physics_dt(),frames=film.index,fps=15,
         columns=dict(recorrido=["base_x_m","base_y_m","base_z_m"],manipulacion=["lift_m","held","steady","speed_m_s",
             "basket_x_m","basket_y_m","basket_z_m","basket_force_n","finger_gap_m","touching","base_drift_m","forbidden_force_n","stem_attached"]),
         instruction=f"Recorre el surco, estaciónate frente a la planta, toma {site.fruit}, elévala 15 cm, sostenla medio segundo y déjala en la canasta.",
         field_seed=FIELDS[crop].seed,target_plant="Hileras/Fila2/Planta03",stem_break_n=site.break_n,
         source_sha256={p.name:sha256(p) for p in (*files,seedfile,Path(__file__),
             ROOT/"tools/rover_orange_trial.py",ROOT/"src/dexman/tasks/agriculture/harvest_route.py")},robots=robots))
    write("mediciones.json",report_rows)
    write("visibilidad.json",view_rows)
    write("rayos.json",ray_rows)
    print(json.dumps(robots),flush=True)
    film.encode(ffmpeg)


if __name__=="__main__":
    from isaaclab.app import add_launcher_args,launch_simulation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop",required=True,choices=SITES)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=751)
    parser.add_argument("--ffmpeg",default="ffmpeg")
    parser.add_argument("--survey",action="store_true",help="Mide el recorrido sin grabar ni mover los brazos a la planta")
    parser.add_argument("--support-check",action="store_true",help="Comprueba el pedúnculo en reposo antes de grabar")
    parser.add_argument("--extended-lift",action="store_true",help="Añade 7,5 cm al recorrido de la muñeca y 12 % de cierre al agarre calibrado")
    add_launcher_args(parser)
    args = parser.parse_args()
    out = args.output.resolve()
    ffmpeg = encoder_path(args.ffmpeg)
    out.mkdir(parents=True,exist_ok=False)
    cfg,g,orient = configure(out,args.crop,VERSIONS,args.survey or args.support_check)
    with launch_simulation(cfg,args):
        import torch
        from isaaclab.sim import SimulationContext
        from dexman.tasks.agriculture.environment import AgriculturalPPOEnv
        files = scene_files(out,args.crop,SITES[args.crop],g,orient)
        with torch.inference_mode():
            env = AgriculturalPPOEnv(SimulationContext(cfg.sim),cfg,seed=args.seed)
            record(env,args.crop,out,VERSIONS,g,files,orient,ffmpeg,args.survey or args.support_check,args.support_check,args.extended_lift)
