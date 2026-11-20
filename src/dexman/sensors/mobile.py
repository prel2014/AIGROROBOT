"""Adaptadores de lectura de la maqueta móvil para Isaac Sim 6.2 tras iniciar Kit."""
import numpy as np

from dexman.robot.joints import JOINT_NAMES
from dexman.tasks.agriculture.mobile_state import HEAD_JOINTS, HEAD_LIMITS_DEG, WHEEL_JOINTS, mobile_state


class RoverSensors:
    def __init__(self, base_path, resolution=(224, 224)):
        from isaacsim.sensors.experimental.physics import IMUSensor
        from isaacsim.sensors.experimental.rtx import CameraSensor, Lidar, LidarSensor, RtxCamera

        self.root = base_path.removesuffix("/Base")
        self.eyes = tuple(
            CameraSensor(RtxCamera(f"{self.root}/HeadTilt/{eye}"), resolution=resolution,
                         annotators=["rgb", "distance_to_image_plane"])
            for eye in ("LeftEye", "RightEye")
        )
        self.lidar = LidarSensor(Lidar(f"{self.root}/Base/Lidar"), annotators=["generic-model-output"])
        self.imu = IMUSensor(f"{self.root}/Base/IMU")


class RoverArticulation:
    """Encoders y drives de un rover activo. Los dos UR10e comparten nombres de junta."""

    def __init__(self, root):
        from isaacsim.core.experimental.prims import Articulation

        self.view = Articulation(root)
        paths, = self.view.dof_paths
        by_link = {}
        for i, path in enumerate(paths):
            local = path.removeprefix(root + "/")
            side, _, tail = local.partition("/")
            key = (side, tail.rsplit("/", 1)[-1]) if tail else ("", local)
            if key in by_link:
                raise ValueError(f"Nombre articular ambiguo en {root}: {path}")
            by_link[key] = i
        self.arms = [by_link[(side, name)] for side in ("LeftArm", "RightArm") for name in JOINT_NAMES]
        self.wheels = [by_link[("", name)] for name in WHEEL_JOINTS]
        self.head = [by_link[("", name)] for name in HEAD_JOINTS]

    def set_head(self, angles_deg):
        angles = np.asarray(angles_deg, dtype=np.float32)
        limits = np.asarray(HEAD_LIMITS_DEG)
        if angles.shape != (2,) or not np.all((angles >= limits[:, 0]) & (angles <= limits[:, 1])):
            raise ValueError(f"Consigna pan/tilt fuera de sus límites en grados: {angles}")
        self.view.set_dof_position_targets(np.deg2rad(angles)[None], dof_indices=self.head)

    def state(self, imu_frame):
        import torch

        if imu_frame["time"] <= 0:
            raise ValueError("La IMU todavía no tiene una muestra física")
        q = torch.from_numpy(self.view.get_dof_positions().numpy())
        dq = torch.from_numpy(self.view.get_dof_velocities().numpy())
        accel = torch.as_tensor(imu_frame["linear_acceleration"], dtype=q.dtype).reshape(1, 3)
        gyro = torch.as_tensor(imu_frame["angular_velocity"], dtype=q.dtype).reshape(1, 3)
        return mobile_state(torch.cat((q[:, self.arms], dq[:, self.arms]), dim=1),
                            dq[:, self.wheels], accel, gyro, q[:, self.head])

    def arm_limits(self):
        lower, upper = self.view.get_dof_limits(dof_indices=self.arms)
        return np.stack((lower.numpy()[0], upper.numpy()[0]), axis=1).tolist()
