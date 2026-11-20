"""Simulator-independent dimensions, order and units for agricultural VLA."""
import numpy as np

from dexman.tasks.agriculture.interface import ACTION_DIM, EYE_KEYS, JOINT_ORDER, STATE_DIM
from dexman.tasks.agriculture import mobile_state as mobile


DEMO_SCHEMA = "dexman.agriculture.demo.v1"
LEGACY_MOBILE_SCHEMA = "dexman.agriculture.mobile.v2"
MOBILE_SCHEMA = "dexman.agriculture.mobile.v3"
MOBILE_SCHEMAS = (LEGACY_MOBILE_SCHEMA, MOBILE_SCHEMA)
OUTPUT_SPACE = "applied_joint_target_rad"
POLICY_VERSION = "agriculture_rgbd_bimanual_v2"


def state_field(schema):
    if schema in MOBILE_SCHEMAS:
        return "mobile_state", mobile.MOBILE_STATE_DIM
    if schema == DEMO_SCHEMA:
        return "angular_state", STATE_DIM
    raise ValueError(f"Esquema agrícola desconocido: {schema}")


def action_order(schema):
    if schema == MOBILE_SCHEMA:
        return JOINT_ORDER + mobile.HEAD_JOINTS
    if schema in (DEMO_SCHEMA, LEGACY_MOBILE_SCHEMA):
        return JOINT_ORDER
    raise ValueError(f"Esquema agrícola desconocido: {schema}")


def architecture_config(schema=DEMO_SCHEMA):
    _, state_dim = state_field(schema)
    name = {DEMO_SCHEMA: POLICY_VERSION, LEGACY_MOBILE_SCHEMA: "agriculture_rgbd_mobile_v2",
            MOBILE_SCHEMA: "agriculture_rgbd_mobile_v3"}[schema]
    return {"name": name,
            "vision_tokens": 100, "vision_dim": 512, "text_dim": 384,
            "depth_grid": 7, "width": 256, "layers": 6, "attention_heads": 8,
            "memory_dim": 256, "horizon": 8, "state_dim": state_dim,
            "action_dim": len(action_order(schema))}


def mobile_metadata(schema=MOBILE_SCHEMA):
    if schema not in MOBILE_SCHEMAS:
        raise ValueError(f"Esquema móvil desconocido: {schema}")
    return {"schema": schema, "state_version": mobile.VERSION, "state_order": list(mobile.STATE_ORDER),
            "state_layout": mobile.STATE_LAYOUT, "joint_order": list(action_order(schema)), "eye_order": list(EYE_KEYS),
            "depth_unit": "m", "joint_command_space": OUTPUT_SPACE}


def bimanual_contract(metadata):
    schema = metadata.get("schema")
    if schema in MOBILE_SCHEMAS:
        expected = mobile_metadata(schema)
        if metadata.get("sample_kind") != "robot_observation":
            expected["sample_hz"] = 30
    else:
        expected = {"schema": DEMO_SCHEMA, "joint_order": list(JOINT_ORDER), "eye_order": list(EYE_KEYS),
                    "state_layout": "q_left,q_right,dq_left,dq_right", "depth_unit": "m",
                    "joint_command_space": OUTPUT_SPACE, "sample_hz": 30}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("El perfil agrícola de cámaras, estado, unidades o comandos no coincide con la captura")
    limits = np.asarray(metadata.get("joint_limits_rad"), dtype=np.float32)
    count = len(action_order(schema))
    if (limits.shape != (count, 2) or not np.isfinite(limits).all()
            or not (limits[:, 1] > limits[:, 0]).all()):
        raise ValueError(f"La captura agrícola requiere límites articulares físicos finitos [{count},2], mínimo < máximo")
    if schema == MOBILE_SCHEMA and not np.allclose(limits[ACTION_DIM:], np.deg2rad(mobile.HEAD_LIMITS_DEG), atol=1e-5):
        raise ValueError("Los límites físicos de giro y elevación no corresponden al cabezal del rover")
    return {**expected, "joint_limits_rad": limits.tolist()}


def matches_robot_contract(observed, trained, *, observation_only=False):
    """Compare validated schemas and allow roundoff between native and Isaac Lab limit readers."""
    actual, expected = dict(observed), dict(trained)
    actual_limits = np.asarray(actual.pop("joint_limits_rad"))
    expected_limits = np.asarray(expected.pop("joint_limits_rad"))
    if observation_only:
        expected.pop("sample_hz", None)
        # A v2 checkpoint consumes the same encoders and commands only the first 60 joints.
        if actual["schema"] == MOBILE_SCHEMA and expected["schema"] == LEGACY_MOBILE_SCHEMA:
            actual["schema"] = LEGACY_MOBILE_SCHEMA
            actual["joint_order"] = actual["joint_order"][:ACTION_DIM]
            actual_limits = actual_limits[:ACTION_DIM]
    return (actual == expected and actual_limits.shape == expected_limits.shape
            and np.allclose(actual_limits, expected_limits, atol=1e-6, rtol=0.))
