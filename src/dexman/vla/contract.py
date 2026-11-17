"""Versioned, simulator-independent contract for visual joint-angle policies."""
import json

import numpy as np

from dexman.paths import JOINT_MANIFEST
from dexman.robot.control import action_limits
from dexman.robot.joints import ACTION_NAMES, JOINT_NAMES

POLICY_VERSION = 4
OUTPUT_SPACE = "absolute_joint_position_rad"
TARGET_SEMANTICS = "desired_joint_position_before_smoothing"
VLA_TASK = "Dexman-UR10e-Shadow-Lift-VLA-v2"
TEACHER_TASK = "Dexman-UR10e-Shadow-Lift-RGBD-v1"
LIFT_OBJECTIVE = {"reward_version": 3, "success_criterion": "minimum_lift_and_hold_v1",
                  "minimum_lift_required_m": .15, "success_lift_overshoot_m": .025,
                  "hold_seconds": .5}


def _validate_lift_objective(contract):
    for name, expected in LIFT_OBJECTIVE.items():
        actual = contract.get(name)
        if isinstance(expected, str):
            valid = actual == expected
        elif isinstance(expected, int):
            valid = type(actual) is int and actual == expected
        else:
            valid = (type(actual) in (int, float) and np.isfinite(actual)
                     and abs(actual - expected) <= 1e-8)
        if not valid:
            raise ValueError(f"Minimum-lift objective mismatch: {name}; recollect demonstrations "
                             "and retrain the student for at least 15 cm held for 0.5 s")


def default_architecture_config(horizon=8):
    """Self-contained v4 topology and frozen-encoder feature semantics."""
    return {
        "name": "clip_spatial_recurrent_transformer", "version": POLICY_VERSION,
        "width": 256, "depth": 6, "heads": 8, "feedforward_dim": 1024, "dropout": 0.1,
        "vision_tokens": 50, "embedding_dim": 512, "state_dim": 60, "memory_dim": 256,
        "action_dim": 30, "horizon": horizon, "encoder_model": "openai/clip-vit-base-patch32",
        "encoder_features": "clip_vit_cls_patches_post_layernorm_projected_l2_v1",
        "attention": "context_isolated_action_queries_memory_write_v1",
        "memory_update": "gated_tanh_residual_v1",
    }


def validate_architecture_config(config, *, horizon=None):
    """Reject incomplete, unknown or incompatible checkpoint architectures.

    Width, depth, attention heads, feed-forward width and dropout are explicit
    tuning parameters. Observation/action dimensions and feature semantics are
    fixed by the v4 interface; they must never be silently inferred from weights.
    """
    expected = default_architecture_config()
    if not isinstance(config, dict) or set(config) != set(expected):
        raise ValueError("VLA v4 requires a complete config['architecture']; legacy policies require new training")
    tunable = {"width", "depth", "heads", "feedforward_dim", "dropout", "horizon"}
    for key in set(expected) - tunable:
        if type(config[key]) is not type(expected[key]) or config[key] != expected[key]:
            raise ValueError(f"Unsupported VLA architecture {key}: {config[key]!r}")
    for key in ("width", "depth", "heads", "feedforward_dim", "horizon"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Architecture {key} must be a positive integer")
    dropout = config["dropout"]
    if (type(dropout) not in (float, int) or not np.isfinite(dropout) or not 0 <= dropout < 1
            or config["width"] % config["heads"]):
        raise ValueError("Architecture requires 0 <= dropout < 1 and width divisible by heads")
    if horizon is not None and (type(horizon) is not int or config["horizon"] != horizon):
        raise ValueError("Checkpoint horizon differs from its architecture")
    return dict(config)


def check_environment_contract(contract, cfg):
    """Require identical control semantics before running a student in simulation."""
    validate_contract(contract)
    expected = action_limits(json.loads(JOINT_MANIFEST.read_text(encoding="utf-8")),
                             cfg.arm_half_range, cfg.arm_center)
    if (contract["task"] != TEACHER_TASK
            or contract["reward_version"] != cfg.reward_version
            or contract["success_criterion"] != cfg.success_criterion
            or abs(contract["minimum_lift_required_m"] - cfg.lift_height) > 1e-8
            or abs(contract["success_lift_overshoot_m"] - cfg.success_lift_overshoot_m) > 1e-8
            or abs(contract["hold_seconds"] - cfg.success_hold_steps * cfg.sim.dt * cfg.decimation) > 1e-8
            or not np.allclose(contract["joint_limits_rad"], expected, atol=1e-6, rtol=0)
            or not np.allclose(contract["action_center_q"], cfg.action_center, atol=1e-6, rtol=0)
            or not np.allclose(contract["arm_center_q"], cfg.arm_center, atol=1e-6, rtol=0)
            or abs(contract["action_smoothing"] - cfg.action_smoothing) > 1e-8):
        raise ValueError("Student checkpoint and environment control contracts differ")
    from dexman.sensors.view import camera_signature, LEGACY_POSITION, LEGACY_ROTATION_XYZW
    actual_camera = camera_signature(cfg.scene.camera)
    expected_camera = contract.get("camera")
    if expected_camera is None:
        if (actual_camera["position"] != list(LEGACY_POSITION)
                or actual_camera["rotation_xyzw"] != list(LEGACY_ROTATION_XYZW)):
            raise ValueError("Legacy student has no camera contract; use legacy playback or recollect with the MVP view")
    elif expected_camera != actual_camera:
        raise ValueError("Student checkpoint and environment camera views differ")


def validate_contract(contract):
    """Fail before loading weights or launching physics if joint semantics differ."""
    if not isinstance(contract, dict):
        raise ValueError("Expected a robot observation/action contract dictionary")
    _validate_lift_objective(contract)
    if contract.get("output_space") != OUTPUT_SPACE or contract.get("target_semantics") != TARGET_SEMANTICS:
        raise ValueError("Expected absolute joint angles in radians, before target smoothing")
    if contract.get("joint_names") != list(JOINT_NAMES):
        raise ValueError("Joint names/order must match the canonical 30-joint JOINT_NAMES")
    limits = np.asarray(contract.get("joint_limits_rad"), dtype=np.float64)
    if limits.shape != (30, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Expected 30 finite, strictly increasing joint limits in radians")
    smoothing = contract.get("action_smoothing")
    if not isinstance(smoothing, (int, float)) or not 0 < smoothing <= 1:
        raise ValueError("action_smoothing must be in (0,1]")
    if "camera" in contract:
        camera = contract["camera"]
        position = np.asarray(camera.get("position"), dtype=float)
        rotation = np.asarray(camera.get("rotation_xyzw"), dtype=float)
        if (position.shape != (3,) or rotation.shape != (4,) or not np.isfinite(position).all()
                or not np.isfinite(rotation).all() or abs(np.linalg.norm(rotation) - 1) > 1e-5
                or camera.get("convention") not in ("opengl", "ros", "world")
                or any(not isinstance(camera.get(key), int) or camera[key] < 1 for key in ("width", "height"))
                or any(not isinstance(camera.get(key), (int, float)) or not np.isfinite(camera[key])
                       or camera[key] <= 0 for key in ("focal_length", "horizontal_aperture"))):
            raise ValueError("Invalid camera observation contract")
    return contract


def contract_from_metadata(metadata):
    """Read recordings certified for the current minimum-height and hold objective."""
    _validate_lift_objective(metadata)
    if metadata.get("schema_version") not in ("1.0", "1.1"):
        raise ValueError("Unsupported demonstration schema")
    required = ("task", "joint_names", "action_names", "action_encoding", "arm_half_range_rad",
                "arm_center_q", "action_center_q")
    if any(key not in metadata for key in required):
        raise ValueError("Missing robot/action metadata; recollect demonstrations with collect_demos")
    if list(metadata["joint_names"]) != list(JOINT_NAMES):
        raise ValueError("Demonstration joint names/order differ from JOINT_NAMES")
    if list(metadata["action_names"]) != list(ACTION_NAMES):
        raise ValueError("Demonstration teacher action order differs from ACTION_NAMES")
    if metadata["action_encoding"] != "normalized_absolute_position_synergy_centered_v1":
        raise ValueError("Only corrected v1 teacher actions can be migrated to joint-angle targets")
    if metadata["task"] != TEACHER_TASK:
        raise ValueError("Expected corrected RGB-D v1 teacher demonstrations")
    limits = action_limits(json.loads(JOINT_MANIFEST.read_text(encoding="utf-8")),
                           metadata["arm_half_range_rad"], metadata["arm_center_q"])
    # New recordings are self-describing; do not silently substitute another robot.
    recorded_limits = metadata.get("joint_limits_rad", limits)
    recorded = np.asarray(recorded_limits, dtype=np.float64)
    if recorded.shape != (30, 2) or not np.allclose(recorded, limits, atol=1e-6, rtol=0):
        raise ValueError("Recorded joint limits differ from the current robot manifest")
    if metadata.get("schema_version") == "1.1":
        if metadata.get("target_semantics") != TARGET_SEMANTICS:
            raise ValueError("Recording does not identify pre-smoothing joint commands")
        if "action_smoothing" not in metadata or "joint_limits_rad" not in metadata:
            raise ValueError("Schema 1.1 requires smoothing and joint limits")
    center = np.asarray(metadata["action_center_q"], dtype=np.float64)
    bounds = np.asarray(limits)
    if center.shape != (30,) or not np.isfinite(center).all() or np.any(
            (center < bounds[:, 0]) | (center > bounds[:, 1])):
        raise ValueError("Invalid teacher action center")
    contract = {
        "task": metadata["task"], "joint_names": list(JOINT_NAMES),
        "output_space": OUTPUT_SPACE, "target_semantics": TARGET_SEMANTICS,
        "joint_limits_rad": recorded_limits,
        "arm_half_range_rad": metadata["arm_half_range_rad"],
        "arm_center_q": metadata["arm_center_q"], "action_center_q": metadata["action_center_q"],
        "action_encoding": metadata["action_encoding"],
        "action_smoothing": metadata.get("action_smoothing", 0.05),
        **{name: metadata[name] for name in LIFT_OBJECTIVE},
    }
    if "camera" in metadata:
        contract["camera"] = metadata["camera"]
    return validate_contract(contract)
