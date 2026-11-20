"""Batched PhysX binding for the complete rover articulation and its two optical frames."""
from math import radians

from dexman.robot.joints import JOINT_NAMES
from dexman.sensors.mobile import rover_indices
from .interface import ARM_KEYS
from .mobile_state import HEAD_JOINTS, HEAD_LIMITS_DEG, WHEEL_JOINTS, mobile_state


class RoverBinding:
    def __init__(self, scene, origins):
        import torch
        from isaaclab.utils.math import combine_frame_transforms
        from pxr import Gf, Usd, UsdGeom

        self.robot = robot = scene["rover"]
        self.imu = scene["imu"]
        # The public order is the native PhysX order (no ordering override on this articulation).
        root = robot.root_view.dof_paths[0][0].split("/Rover/")[0] + "/Rover"
        paths = robot.root_view.dof_paths[0]
        joints = rover_indices(paths, root)
        self.bodies = rover_indices(robot.root_view.link_paths[0], root)
        self.owners = dict(zip(ARM_KEYS, ("LeftArm", "RightArm"), strict=True))
        self.joints = tuple([joints[(owner, name)] for name in JOINT_NAMES] for owner in self.owners.values())
        self.arm_joints = self.joints[0] + self.joints[1]
        self.head = [joints[("", name)] for name in HEAD_JOINTS]
        self.head_limits = robot.data.joint_pos_limits.torch[:, self.head].clone()
        expected_limits = torch.deg2rad(self.head_limits.new_tensor(HEAD_LIMITS_DEG))
        if not torch.allclose(self.head_limits, expected_limits.expand_as(self.head_limits), atol=1e-5):
            raise ValueError("Los topes del cabezal del USD no coinciden con el contrato de acciones")
        self.wheels = [joints[("", name)] for name in WHEEL_JOINTS]
        self.tilt_body = self.bodies[("", "HeadTilt")]
        # The spawn transform belongs to /Rover. PhysX resets /Rover/Base, whose
        # authored frame sits above it. A live pose here already includes Kit startup settling.
        source = Usd.Stage.Open(robot.cfg.spawn.usd_path)
        base = UsdGeom.Xformable(source.GetDefaultPrim().GetChild("Base")).GetLocalTransformation()
        rot = base.ExtractRotationQuat()
        spawn = robot.data.default_root_pose.torch
        pos, quat = combine_frame_transforms(spawn[:, :3] + origins, spawn[:, 3:],
            spawn.new_tensor(tuple(base.ExtractTranslation())).expand_as(spawn[:, :3]),
            spawn.new_tensor((*rot.GetImaginary(), rot.GetReal())).expand_as(spawn[:, 3:]))
        self.root_pose = torch.cat((pos, quat), dim=-1)
        self.rest = robot.data.default_joint_pos.torch.clone()
        # Each mounted UR10e has its own transport pose. Preserve the actual asset's joint state.
        for i, path in enumerate(paths):
            attr = scene.stage.GetPrimAtPath(path).GetAttribute("state:angular:physics:position")
            angle = attr.Get() if attr else None
            if angle is not None:
                self.rest[:, i] = radians(angle)
        offsets, axes = [], []
        for eye in ("LeftEye", "RightEye"):
            optical = UsdGeom.Camera.Get(scene.stage, f"{root}/HeadTilt/{eye}")
            transform = optical.GetLocalTransformation()
            offsets.append(tuple(transform.ExtractTranslation()))
            axes.append(tuple(transform.TransformDir(Gf.Vec3d(0., 0., -1.)).GetNormalized()))
        self.eye_offsets = self.rest.new_tensor(offsets)
        self.eye_axes = self.rest.new_tensor(axes)

    def body_ids(self, side, names):
        return [self.bodies[(self.owners[side], name)] for name in names]

    def reset(self, ids):
        pose, q = self.root_pose[ids], self.rest[ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=pose.new_zeros((len(ids), 6)), env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=q.new_zeros(q.shape), env_ids=ids)
        self.robot.set_joint_position_target_index(target=q, env_ids=ids)

    def state(self, arms):
        data, imu = self.robot.data, self.imu.data
        return mobile_state(arms, data.joint_vel.torch[:, self.wheels], imu.lin_acc_b.torch,
                            imu.ang_vel_b.torch, data.joint_pos.torch[:, self.head])

    def gaze_alignment(self, target_world):
        from isaaclab.utils.math import quat_apply
        from torch.nn.functional import cosine_similarity

        data = self.robot.data
        position = data.body_link_pos_w.torch[:, self.tilt_body, None]
        rotation = data.body_link_quat_w.torch[:, self.tilt_body, None]
        origins = position + quat_apply(rotation, self.eye_offsets)
        return cosine_similarity(target_world[:, None] - origins, quat_apply(rotation, self.eye_axes), dim=-1)

    def compensate_arms(self):
        # Floating-base PhysX generalised forces start with six chassis components.
        forces = self.robot.data.gravity_compensation_forces.torch[:, -self.robot.num_joints:]
        self.robot.actuators.target_command.set_effort_index(value=forces[:, self.arm_joints], joint_ids=self.arm_joints)
