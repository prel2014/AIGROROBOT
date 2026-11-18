"""Simulator-independent dimensions, order and units for agricultural VLA."""
import numpy as np

from dexman.tasks.agriculture.interface import ACTION_DIM, EYE_KEYS, JOINT_ORDER, STATE_DIM


DEMO_SCHEMA = "dexman.agriculture.demo.v1"
OUTPUT_SPACE = "applied_joint_target_rad"
POLICY_VERSION = "agriculture_rgbd_bimanual_v2"


def architecture_config():
    return {"name": POLICY_VERSION, "vision_tokens": 100, "vision_dim": 512, "text_dim": 384,
            "depth_grid": 7, "width": 256, "layers": 6, "attention_heads": 8,
            "memory_dim": 256, "horizon": 8, "state_dim": STATE_DIM, "action_dim": ACTION_DIM}


def bimanual_contract(metadata):
    expected = {"schema": DEMO_SCHEMA, "joint_order": list(JOINT_ORDER), "eye_order": list(EYE_KEYS),
                "state_layout": "q_left,q_right,dq_left,dq_right", "depth_unit": "m",
                "joint_command_space": OUTPUT_SPACE, "sample_hz": 30}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Se requiere una captura agrícola sincronizada con órdenes articulares aplicadas a 30 Hz")
    limits = np.asarray(metadata.get("joint_limits_rad"), dtype=np.float32)
    if (limits.shape != (ACTION_DIM, 2) or not np.isfinite(limits).all()
            or not (limits[:, 1] > limits[:, 0]).all()):
        raise ValueError("La captura agrícola requiere límites articulares físicos [60,2]. Repite agriculture_collect.")
    return {**expected, "joint_limits_rad": limits.tolist()}
