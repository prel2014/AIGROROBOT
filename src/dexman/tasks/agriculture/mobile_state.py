"""Contrato móvil. Los ángulos y las velocidades usan radianes y segundos."""
from .interface import JOINT_ORDER, STATE_DIM


VERSION = "agriculture-mobile-v2"
WHEEL_JOINTS = ("drive_frontal_izq", "drive_frontal_der", "drive_trasero_izq", "drive_trasero_der")
HEAD_JOINTS = ("head_pan_joint", "head_tilt_joint")
HEAD_LIMITS_DEG = ((-40., 40.), (-22., 38.))
STATE_ORDER = (tuple(f"q_rad/{name}" for name in JOINT_ORDER)
               + tuple(f"dq_rad_s/{name}" for name in JOINT_ORDER)
               + tuple(f"dq_rad_s/{name}" for name in WHEEL_JOINTS)
               + tuple(f"imu_accel_m_s2/{axis}" for axis in "xyz")
               + tuple(f"imu_gyro_rad_s/{axis}" for axis in "xyz")
               + tuple(f"q_rad/{name}" for name in HEAD_JOINTS))
MOBILE_STATE_DIM = len(STATE_ORDER)
STATE_LAYOUT = "q_left,q_right,dq_left,dq_right,wheel_dq,imu_acceleration,imu_angular_velocity,head_pan,head_tilt"


def mobile_state(angular_state, wheel_velocity, acceleration, angular_velocity, head_position):
    """Medidas por réplica. Pan positivo a la izquierda y tilt positivo hacia arriba."""
    import torch

    n = angular_state.shape[0]
    values = (angular_state, wheel_velocity, acceleration, angular_velocity, head_position)
    for value, width in zip(values, (STATE_DIM, len(WHEEL_JOINTS), 3, 3, len(HEAD_JOINTS)), strict=True):
        if value.shape != (n, width):
            raise ValueError(f"Observación móvil incompatible: {value.shape}, se esperaba {(n, width)}")
    result = torch.cat(values, dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("El estado móvil requiere medidas finitas de cada sensor")
    return result
