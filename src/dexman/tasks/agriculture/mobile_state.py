"""Contrato móvil. Los ángulos y las velocidades usan radianes y segundos."""
from .interface import STATE_DIM


VERSION = "agriculture-mobile-v1"
WHEEL_JOINTS = ("drive_frontal_izq", "drive_frontal_der", "drive_trasero_izq", "drive_trasero_der")
MOBILE_STATE_DIM = STATE_DIM + len(WHEEL_JOINTS) + 6


def mobile_state(angular_state, wheel_velocity, acceleration, angular_velocity):
    """Une [q, dq], cuatro encoders, acelerómetro y giroscopio por réplica."""
    import torch

    n = angular_state.shape[0]
    values = (angular_state, wheel_velocity, acceleration, angular_velocity)
    for value, width in zip(values, (STATE_DIM, 4, 3, 3), strict=True):
        if value.shape != (n, width):
            raise ValueError(f"Observación móvil incompatible: {value.shape}, se esperaba {(n, width)}")
    return torch.cat(values, dim=-1)
