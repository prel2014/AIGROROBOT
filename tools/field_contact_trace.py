"""Read actual fingertip and fruit poses after each PhysX solve in a field trial."""
import torch

from dexman.robot.joints import FINGERTIPS
from dexman.tasks.agriculture.contact_quality import antipodal_power_grasp, surface_travel_speed
from dexman.tasks.agriculture.fruit_clearance import lowest_collider_height


class ContactTrace:
    def __init__(self, group, fruit, sides, collider_points):
        self.group, self.fruit = group, fruit
        self.data = group.rover.robot.data
        self.rows = torch.arange(len(sides),device=sides.device)
        self.sides = sides
        keys = ("left_arm","right_arm")
        self.links = sides.new_tensor([group.arm_body_ids(key,FINGERTIPS) for key in keys])[sides]
        self.sensors = [[group.scene[f"{key}_{tip}"] for tip in FINGERTIPS] for key in keys]
        self.collider_points = collider_points
        self.reference_z = self.height()

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
        loaded = forces.norm(dim=-1)>=.04
        touching = loaded & points.isfinite().all(-1)
        grip = antipodal_power_grasp(forces,points,touching,.60)
        slip = surface_travel_speed(points,forces,self.last[0],current[0],self.last[1],current[1],dt)
        peak = torch.where(loaded,torch.nan_to_num(slip,nan=torch.inf),0.).amax(-1)
        speed = self.fruit.data.root_com_vel_w.torch[:,:3].norm(dim=-1)
        dz = self.height()-self.reference_z
        self.samples.append(torch.stack((grip,peak,speed,dz),dim=-1))
        self.last = current
