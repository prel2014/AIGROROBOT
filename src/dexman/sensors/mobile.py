"""Adaptadores de lectura de la maqueta móvil para Isaac Sim 6.2 tras iniciar Kit."""
from dexman.tasks.agriculture.mobile_state import WHEEL_JOINTS


class RoverSensors:
    def __init__(self, base_path, resolution=(224, 224)):
        from isaacsim.sensors.experimental.physics import IMUSensor
        from isaacsim.sensors.experimental.rtx import CameraSensor, Lidar, LidarSensor, RtxCamera

        self.eyes = tuple(
            CameraSensor(RtxCamera(f"{base_path}/{eye}"), resolution=resolution,
                         annotators=["rgb", "distance_to_image_plane"])
            for eye in ("LeftEye", "RightEye")
        )
        self.lidar = LidarSensor(Lidar(f"{base_path}/Lidar"), annotators=["generic-model-output"])
        self.imu = IMUSensor(f"{base_path}/IMU")


def measured_wheel_velocity(articulation):
    """Velocidad física de las ruedas FL, FR, RL, RR en rad/s."""
    ids, names = articulation.find_joints(list(WHEEL_JOINTS), preserve_order=True)
    if tuple(names) != WHEEL_JOINTS:
        raise ValueError(f"El rover carece de los cuatro encoders esperados: {names}")
    return articulation.data.joint_vel.torch[:, ids]
