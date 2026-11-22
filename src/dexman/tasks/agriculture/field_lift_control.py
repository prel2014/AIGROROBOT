"""Height feedback along the wrist path of a physically loaded hand."""
import numpy as np


class PrecisionPalmControl:
    """Align the measured fruit root with its reference in the closing hand."""
    def __init__(self, group, palm_ids, sides, reference):
        import torch
        self.robot = group.rover.robot
        self.rows = torch.arange(len(sides), device=sides.device)
        self.palms = sides.new_tensor(palm_ids)
        self.joints = sides.new_tensor([ids[:8] for ids in group.joint_ids])[sides]
        self.columns = 30*sides[:,None]+self.rows.new_tensor(range(8))
        self.bias = group.origins.new_zeros((len(sides),8))
        self.targets = {key: group.origins.new_tensor(value) for key,value in reference.items()}
        self.eye = torch.eye(6,device=sides.device)*.03**2

    def correct(self, twist, active):
        import torch
        jac = self.robot.data.body_link_jacobian_w.torch[self.rows,self.palms]
        base_dofs = jac.shape[-1]-self.robot.num_joints
        jac = jac.gather(2,(self.joints+base_dofs)[:,None,:].expand(-1,6,-1))
        system = jac@jac.transpose(1,2)+self.eye
        dq = (jac.transpose(1,2)@torch.linalg.solve(system,twist[:,:,None]))[:,:,0]
        self.bias = (self.bias+active[:,None]*dq.clamp(-.007,.007)).clamp(-.18,.18)

    def apply(self, command, phase, progress, align, fruit, attached, held):
        import torch
        from isaaclab.utils.math import quat_apply
        if align:
            data = self.robot.data
            palm = data.body_link_pose_w.torch[self.rows,self.palms]
            target = self.targets["pregrasp"].lerp(self.targets["closed"],progress)
            desired = palm[:,:3]+quat_apply(palm[:,3:],target.expand_as(palm[:,:3]))
            error = fruit-desired
            # Bound each correction to a millimetre so the approach can settle on the plant.
            increment = .20*error/error.norm(dim=-1,keepdim=True).clamp_min(.005)*.005
            twist = torch.cat((increment,torch.zeros_like(increment)),dim=-1)
            active = attached & ~held & (error.norm(dim=-1)<.12)
            self.correct(twist,active)
        command[self.rows[:,None],self.columns] += self.bias


class FieldWristControl(PrecisionPalmControl):
    """Follow the surveyed field wrist relative to the unloaded stem during approach and closure."""
    def __init__(self,group,palms,sides,profile):
        super().__init__(group,palms,sides,profile["waypoints"])
        offset = profile.get("approach_clearance",{}).get("local_palm_offset_m",[0.,0.,0.])
        self.clearance_offset = group.origins.new_tensor(offset)

    def follow(self,command,knot,fraction,rest_com,attached,held):
        import torch
        from isaaclab.utils.math import quat_apply, quat_box_minus
        samples = self.targets.get(str(knot))
        if samples is not None:
            t = fraction*(len(samples)-1)
            j = min(int(t),len(samples)-2)
            goal = samples[j].lerp(samples[j+1],t-j)
            palm = self.robot.data.body_link_pose_w.torch[self.rows,self.palms]
            delta = rest_com+goal[:3]-palm[:,:3]
            rotation = torch.nn.functional.normalize(goal[3:],dim=0).expand_as(palm[:,3:])
            padding = 1. if knot in (1,2) else 1.-fraction if knot==3 else 0.
            delta += padding*quat_apply(rotation,self.clearance_offset.expand_as(delta))
            angle = quat_box_minus(rotation,palm[:,3:])
            linear = .22*delta/delta.norm(dim=-1,keepdim=True).clamp_min(.006)*.006
            angular = .20*angle/angle.norm(dim=-1,keepdim=True).clamp_min(.045)*.045
            self.correct(torch.cat((linear,angular),dim=-1),attached & ~held & (delta.norm(dim=-1)<.12))
        command[self.rows[:,None],self.columns] += self.bias

    def unload_stem(self,command,stem,loads):
        """Back the wrist toward the unloaded fruit while the fingers build normal force."""
        import torch
        data = stem.fruit.data
        effort = stem.k*(stem.rest_com-data.root_com_pos_w.torch)-stem.d*data.root_com_vel_w.torch[:,:3]
        total = effort+stem.preload
        # Keep a margin to the rupture load until the intended pulling phase.
        excess = (total.norm(dim=-1)-.62*stem.limit).clamp(min=0.)/stem.k
        motion = torch.nn.functional.normalize(effort,dim=-1)*(excess*.40).clamp(max=.0014)[:,None]
        active = stem.attached & ((loads>.20).sum(-1)>=2) & (excess>0.)
        before = self.bias
        self.correct(torch.cat((motion,torch.zeros_like(motion)),dim=-1),active)
        command[self.rows[:,None],self.columns] += self.bias-before


class TripodPreload:
    """Track calibrated loads on the index, middle finger and thumb."""
    def __init__(self,group,sides,profile):
        import torch
        from dexman.robot.joints import FINGERTIPS, JOINT_NAMES
        self.robot = group.rover.robot
        self.rows = torch.arange(len(sides),device=sides.device)
        self.sides = sides
        indices = (0,1,4)
        fingers = (("FF",(3,2,1,0)),("MF",(3,2,1,0)),("TH",(4,2,1,0)))
        local = sides.new_tensor([[JOINT_NAMES.index(f"robot0_{finger}J{joint}") for joint in joints]
                                  for finger,joints in fingers])
        self.columns = 30*sides[:,None]+local.reshape(1,-1)
        self.joints = sides.new_tensor(group.joint_ids)[sides][:,local]
        keys = ("left_arm","right_arm")
        tips = [FINGERTIPS[j] for j in indices]
        self.links = sides.new_tensor([group.arm_body_ids(key,tips) for key in keys])[sides]
        self.palms = sides.new_tensor([group.arm_body_ids(key,["robot0_palm"])[0] for key in keys])[sides]
        self.fruit = group.scene["fruit_0"]
        self.middle_offset = group.origins.new_tensor(profile["middle_pad_from_com_in_palm_m"])
        # Centreline endpoints of the distal collision capsules in each link frame, metres.
        self.pad_axes = group.origins.new_tensor([[0.,0.,.024],[0.,0.,.024],[0.,0.,.026]])
        self.sensors = [[group.scene[f"{key}_{tip}"] for tip in tips] for key in keys]
        self.bias = group.origins.new_zeros((len(sides),local.numel()))
        self.last_load = group.origins.new_zeros((len(sides),len(tips)))
        # Mean normal forces in the last ten closure ticks of the reference orange.
        self.target = group.origins.new_tensor(profile.get("target_normal_force_n",[3.19,2.80,4.38]))

    def apply(self,command,phase,knot,fraction,loads,latches,*,update=None):
        import torch
        from isaaclab.utils.math import quat_apply
        if phase not in ("cierre","traccion","recuperacion","elevacion","sosten","traslado","apertura") or knot<3:
            return
        if phase!="apertura" and update is not False:
            pairs = []
            for sensors in self.sensors:
                force = torch.stack([s.data.normal_force_matrix_w.torch[:,0,0] for s in sensors],dim=1)
                point = torch.stack([s.data.contact_pos_w.torch[:,0,0] for s in sensors],dim=1)
                pairs.append(torch.cat((force,point),dim=-1))
            force,point = torch.stack(pairs,dim=1)[self.rows,self.sides].split(3,dim=-1)
            actual = loads[:,[0,1,4]]
            active = (actual>.04) & point.isfinite().all(-1)
            data = self.robot.data
            tip = data.body_link_pose_w.torch[self.rows[:,None],self.links]
            center = self.fruit.data.root_com_pos_w.torch
            axes = quat_apply(tip[:,:,3:],self.pad_axes.expand_as(tip[:,:,:3]))
            u = ((center[:,None]-tip[:,:,:3])*axes).sum(-1)/axes.square().sum(-1)
            pad = tip[:,:,:3]+axes*u.clamp(0.,1.)[:,:,None]
            normal = torch.nn.functional.normalize(torch.where(active[:,:,None],force,pad-center[:,None]),dim=-1)
            offset = torch.where(active[:,:,None],point,pad)-tip[:,:,:3]
            jac = data.body_link_jacobian_w.torch[self.rows[:,None],self.links]
            free = jac.shape[-1]-self.robot.num_joints
            jac = jac.gather(3,(self.joints+free)[:,:,None,:].expand(-1,-1,6,-1)).transpose(2,3)
            linear = jac[:,:,:,:3]+torch.cross(jac[:,:,:,3:],offset[:,:,None,:].expand_as(jac[:,:,:,3:]),dim=-1)
            gradient = -(linear*normal[:,:,None,:]).sum(-1)
            error = self.target-actual
            pulling = phase in ("traccion","recuperacion")
            carrying = phase in ("elevacion","sosten","traslado")
            if pulling or carrying:
                # The peduncle stops carrying load within one 120-Hz substep.
                # Preload the middle finger before breakage and counter falling
                # pad forces ahead of the measured mechanical response time.
                reserve = [0.,.75,.35] if pulling else [0.,.60,.20]
                error += actual.new_tensor(reserve) + 1.5*(self.last_load-actual).clamp(0.,1.6)
            self.last_load = actual.clone()
            rate = .00035 if pulling or carrying else .00013
            travel = .0008 if pulling or carrying else .00024
            step = torch.where(error.abs()>.25,error*rate,0.).clamp(-travel,travel)
            engaged = (actual[:,0]>.35)&(actual[:,2]>.35)
            verified = torch.tensor([q is not None for q in latches],device=actual.device)
            engaged |= verified
            increment = (step*engaged[:,None])[:,:,None]*gradient/(gradient.square().sum(-1,keepdim=True)+.008**2)
            if phase=="cierre" or pulling or carrying:
                # A missing middle-finger contact has no sensor normal. Guide its
                # distal capsule to the site measured in the loaded reference hand.
                palm = data.body_link_pose_w.torch[self.rows,self.palms,3:]
                goal = center+quat_apply(palm,self.middle_offset.expand_as(center))
                delta_pad = goal-pad[:,1]
                shift = .35*delta_pad/delta_pad.norm(dim=-1,keepdim=True).clamp_min(.003)*.003
                jac_pad = linear[:,1].transpose(1,2)
                metric = jac_pad@jac_pad.transpose(1,2)+torch.eye(3,device=center.device)*.007**2
                motion = (jac_pad.transpose(1,2)@torch.linalg.solve(metric,shift[:,:,None]))[:,:,0]
                search = engaged & (actual[:,1]<(3.0 if carrying else 2.5))
                increment[:,1] += search[:,None]*motion.clamp(-.022,.022)
            delta = .014 if pulling or carrying else .005
            increment[:,[0,2]] = increment[:,[0,2]].clamp(-delta,delta)
            if update is not None:
                increment *= update[:,None,None]
            self.bias = (self.bias+increment.flatten(1).clamp(-.025,.025)).clamp(-.23,.23)
        command[self.rows[:,None],self.columns] += self.bias*(1.-fraction if phase=="apertura" else 1.)


class LiftHeightControl:
    def __init__(self,paths,versions,target_height=.157):
        self.target_height = target_height
        self.sides = [side for _,side,_,_ in versions]
        self.paths = []
        for path in paths:
            closed = next(q for q,phase,_ in reversed(path) if phase=="cierre")
            self.paths.append(np.stack([closed[:8],*(q[:8] for q,phase,_ in path if phase=="elevacion")]))
        self.progress = np.zeros(len(paths))
        self.hold_goal = self.progress.copy()
        self.feedforward = np.full(len(paths),np.nan)
        self.rise_time = 0.
        self.hold_heights = []
        self.bias = np.zeros((len(paths),8))
        self.transfer_ticks = 0

    def apply(self,command,phase,heights,dt):
        if phase in ("elevacion","sosten"):
            if phase=="elevacion":
                # Approach from below. Reversing on instantaneous measurements
                # excites the compliant arm and the banana swings in the fingers.
                rate = np.clip(8.*(self.target_height+.008-heights),0.,.45)
                self.progress = np.minimum(1.,self.progress+dt*rate)
                self.rise_time += dt
                u = min(1.,self.rise_time/2.4)
                mask = np.isfinite(self.feedforward)
                self.progress[mask] = self.feedforward[mask]*u*u*(3.-2.*u)
                self.hold_goal = self.progress.copy()
            else:
                self.hold_heights.append(heights.copy())
                settle_ticks = max(2,round(.8/dt))
                if len(self.hold_heights)>=settle_ticks:
                    settled = np.median(self.hold_heights[-max(1,settle_ticks//3):],axis=0)
                    error = self.target_height-settled
                    correction = np.where((error>.004)|(error<-.011),np.clip(error/.22,-.04,.04),0.)
                    # The recorded elbow branch already has a stationary hold.
                    # Lift further only if the actual loaded fruit settled too low.
                    correction[np.isfinite(self.feedforward)&(settled>=.152)] = 0.
                    self.hold_goal = np.clip(self.progress+correction,0.,1.)
                    self.hold_heights.clear()
                self.progress += np.clip(self.hold_goal-self.progress,-.085*dt,.085*dt)
            for i,(side,path) in enumerate(zip(self.sides,self.paths,strict=True)):
                t = self.progress[i]*(len(path)-1)
                j = min(int(t),len(path)-2)
                q = path[j]+(t-j)*(path[j+1]-path[j])
                command[i,30*side:30*side+8] = command.new_tensor(q)
                self.bias[i] = q-path[-1]
        elif phase=="traslado":
            self.transfer_ticks += 1
            t = min(1.,self.transfer_ticks*dt/1.8)
            for i,side in enumerate(self.sides):
                command[i,30*side:30*side+8] += command.new_tensor(self.bias[i])*(1.-t*t*(3.-2.*t))
