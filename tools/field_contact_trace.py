"""Read actual fingertip and fruit poses after each PhysX solve in a field trial."""
import torch

from dexman.robot.joints import FINGERTIPS
from dexman.tasks.agriculture.contact_quality import antipodal_power_grasp, measure_contact_quality, surface_travel_speed
from dexman.tasks.agriculture.objectives import OBJECTIVES
from dexman.tasks.agriculture.fruit_clearance import lowest_collider_height
from dexman.tasks.agriculture.harvest_route import SITES


class ContactTrace:
    def __init__(self, group, fruit, sides, collider_points, *, inspect_velocity=False, stem_attached=None):
        self.group, self.fruit = group, fruit
        self.data = group.rover.robot.data
        self.rows = torch.arange(len(sides),device=sides.device)
        self.sides = sides
        self.precision = OBJECTIVES[group.crop].precision
        self.stem_attached = stem_attached
        keys = ("left_arm","right_arm")
        self.links = sides.new_tensor([group.arm_body_ids(key,FINGERTIPS) for key in keys])[sides]
        self.sensors = [[group.scene[f"{key}_{tip}"] for tip in FINGERTIPS] for key in keys]
        self.collider_points = collider_points
        self.reference_z = self.height()
        self.inspect_velocity = inspect_velocity
        self.velocity_checks = []
        self.cluster_contacts = []

    def height(self):
        data = self.fruit.data
        return lowest_collider_height(data.root_link_pos_w.torch,data.root_link_quat_w.torch,self.collider_points)

    def poses(self):
        tip = self.data.body_link_pose_w.torch[self.rows[:,None],self.links].clone()
        fruit = self.fruit.data.root_link_pose_w.torch[:,None].clone()
        # Cancel the shared simulation origin before rotating millimetre displacements.
        tip[...,:3] -= self.group.origins[:,None]
        fruit[...,:3] -= self.group.origins[:,None]
        return tip,fruit

    def begin(self):
        self.last = self.poses()
        self.samples = []

    def sample(self,dt):
        current = self.poses()
        pairs = []
        for sensors in self.sensors:
            p = torch.stack([s.data.contact_pos_w.torch[:,0,0] for s in sensors],dim=1)
            f = torch.stack([s.data.normal_force_matrix_w.torch[:,0,0] for s in sensors],dim=1)
            pairs.append(torch.cat((p-self.group.origins[:,None],f),dim=-1))
        points,forces = torch.stack(pairs,dim=1)[self.rows,self.sides].split(3,dim=-1)
        loaded = forces.norm(dim=-1)>=(.003 if self.precision else .04)
        touching = loaded & points.isfinite().all(-1)
        if self.precision:
            center = self.fruit.data.root_com_pos_w.torch-self.group.origins
            grip = measure_contact_quality(forces,points,center,precision=True).held
        elif self.group.crop=="vineyard":
            from dexman.tasks.agriculture.cluster_grasp import cluster_grip
            center = self.fruit.data.root_com_pos_w.torch-self.group.origins
            observations = torch.cat((forces,points-center[:,None,:]),dim=-1).cpu().numpy()
            self.cluster_contacts.append(observations)
            enclosing = measure_contact_quality(forces,points,center).held
            valid = [cluster_grip(row[:,:3],row[:,3:],SITES["vineyard"].mass,.60) for row in observations]
            grip = enclosing & touching.new_tensor(valid)
        else:
            grip = antipodal_power_grasp(forces,points,touching,.60)
        slip = surface_travel_speed(points,forces,self.last[0],current[0],self.last[1],current[1],dt)
        peak = torch.where(loaded,torch.nan_to_num(slip,nan=torch.inf),0.).amax(-1)
        from isaaclab.utils.math import quat_apply, quat_apply_inverse
        fc = self.fruit.data.root_com_pos_w.torch-self.group.origins
        old,now = self.last[1][:,0],current[1][:,0]
        com_offset = quat_apply_inverse(now[:,3:],fc-now[:,:3])
        old_com = old[:,:3]+quat_apply(old[:,3:],com_offset)
        speed = (fc-old_com).norm(dim=-1)/dt
        native_speed = self.fruit.data.root_com_vel_w.torch[:,:3].norm(dim=-1)
        dz = self.height()-self.reference_z
        if self.inspect_velocity and len(self.velocity_checks)<20 and (grip & (dz>.15)).any():
            from dexman.tasks.agriculture.contact_quality import contact_slip_speed
            from isaaclab.utils.math import quat_mul, quat_conjugate, quat_apply, quat_apply_inverse
            com = self.data.body_com_pos_w.torch[self.rows[:,None],self.links]-self.group.origins[:,None]
            vel = self.data.body_com_vel_w.torch[self.rows[:,None],self.links]
            fc = self.fruit.data.root_com_pos_w.torch-self.group.origins
            fv = self.fruit.data.root_com_vel_w.torch
            def measured(old, pose, center):
                local = quat_apply_inverse(pose[...,3:],center-pose[...,:3])
                previous = old[...,:3]+quat_apply(old[...,3:],local)
                dq = quat_mul(pose[...,3:],quat_conjugate(old[...,3:]))
                angle = 2*torch.atan2(dq[...,:3].norm(dim=-1,keepdim=True),dq[...,3:].abs())
                omega = torch.nn.functional.normalize(dq[...,:3],dim=-1)*angle*dq[...,3:].sign()/dt
                return torch.cat(((center-previous)/dt,omega),dim=-1)
            native = contact_slip_speed(points,forces,com,vel,fc,fv)
            tv = measured(self.last[0],current[0],com)
            ov = measured(self.last[1],current[1],fc[:,None])[:,0]
            self.velocity_checks.append(dict(lift=dz.tolist(),thumb_native=vel[:,-1].tolist(),
                thumb_pose_derivative=tv[:,-1].tolist(),fruit_native=fv.tolist(),fruit_pose_derivative=ov.tolist(),
                thumb_distance_to_contact_m=(points[:,-1]-com[:,-1]).norm(dim=-1).tolist(),
                thumb_slip_velocity_m_s=native[:,-1].tolist(),thumb_slip_displacement_m_s=slip[:,-1].tolist()))
        values = (grip,peak,speed,dz,native_speed)
        if self.stem_attached is not None:
            values += (self.stem_attached,)
        self.samples.append(torch.stack(values,dim=-1))
        self.last = current
