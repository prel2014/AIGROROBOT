"""Simulator-independent dimensions, order and units for agricultural VLA."""
import numpy as np

from dexman.tasks.agriculture.interface import ACTION_DIM, EYE_KEYS, JOINT_ORDER, STATE_DIM
from dexman.tasks.agriculture import mobile_state as mobile


DEMO_SCHEMA = "dexman.agriculture.demo.v1"
MOBILE_SCHEMA = "dexman.agriculture.mobile.v2"
OUTPUT_SPACE = "applied_joint_target_rad"
POLICY_VERSION = "agriculture_rgbd_bimanual_v2"


def state_field(schema):
    if schema == MOBILE_SCHEMA:
        return "mobile_state", mobile.MOBILE_STATE_DIM
    if schema == DEMO_SCHEMA:
        return "angular_state", STATE_DIM
    raise ValueError(f"Esquema agrícola desconocido: {schema}")


def architecture_config(schema=DEMO_SCHEMA):
    _, state_dim = state_field(schema)
    moving = schema == MOBILE_SCHEMA
    return {"name": "agriculture_rgbd_mobile_v2" if moving else POLICY_VERSION,
            "vision_tokens": 100, "vision_dim": 512, "text_dim": 384,
            "depth_grid": 7, "width": 256, "layers": 6, "attention_heads": 8,
            "memory_dim": 256, "horizon": 8, "state_dim": state_dim,
            "action_dim": ACTION_DIM}


def mobile_metadata():
    return {"schema": MOBILE_SCHEMA, "state_version": mobile.VERSION, "state_order": list(mobile.STATE_ORDER),
            "state_layout": mobile.STATE_LAYOUT, "joint_order": list(JOINT_ORDER), "eye_order": list(EYE_KEYS),
            "depth_unit": "m", "joint_command_space": OUTPUT_SPACE}


def bimanual_contract(metadata):
    if metadata.get("schema") == MOBILE_SCHEMA:
        expected = mobile_metadata()
        if metadata.get("sample_kind") != "robot_observation":
            expected["sample_hz"] = 30
    else:
        expected = {"schema": DEMO_SCHEMA, "joint_order": list(JOINT_ORDER), "eye_order": list(EYE_KEYS),
                    "state_layout": "q_left,q_right,dq_left,dq_right", "depth_unit": "m",
                    "joint_command_space": OUTPUT_SPACE, "sample_hz": 30}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("El perfil agrícola de cámaras, estado, unidades o comandos no coincide con la captura")
    limits = np.asarray(metadata.get("joint_limits_rad"), dtype=np.float32)
    if (limits.shape != (ACTION_DIM, 2) or not np.isfinite(limits).all()
            or not (limits[:, 1] > limits[:, 0]).all()):
        raise ValueError("La captura agrícola requiere límites articulares físicos finitos [60,2], mínimo < máximo")
    return {**expected, "joint_limits_rad": limits.tolist()}
