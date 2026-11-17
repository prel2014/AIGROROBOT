"""One teacher-to-joint command contract shared by demonstrations and corrections."""
import json

from dexman.experiments import sha256
from dexman.paths import JOINT_MANIFEST, ROBOT_USD
from dexman.robot.control import action_limits
from dexman.robot.joints import ACTION_NAMES, FINGERTIPS, JOINT_NAMES
from dexman.vla.contract import TARGET_SEMANTICS, TEACHER_TASK
from dexman.sensors.view import camera_signature


def teacher_metadata(checkpoint, cfg, sample_kind="teacher_demonstration"):
    return {"schema_version": "1.1", "action_names": list(ACTION_NAMES), "joint_names": list(JOINT_NAMES),
        "action_encoding": "normalized_absolute_position_synergy_centered_v1", "quaternion_order": "xyzw",
        "depth_invalid": 0., "seed": cfg.seed, "task": TEACHER_TASK, "sample_kind": sample_kind,
        "arm_half_range_rad": cfg.arm_half_range, "reward_version": cfg.reward_version,
        "success_criterion": cfg.success_criterion, "minimum_lift_required_m": cfg.lift_height,
        "minimum_contact_force_n": cfg.minimum_contact_force_n,
        "scenario_name": cfg.scenario_name, "asset_shape": cfg.asset_shape,
        "asset_dimensions": list(cfg.asset_dimensions),
        "success_lift_overshoot_m": cfg.success_lift_overshoot_m,
        "hold_seconds": cfg.success_hold_steps * cfg.sim.dt * cfg.decimation,
        "action_center_q": list(cfg.action_center), "arm_center_q": list(cfg.arm_center),
        "joint_limits_rad": action_limits(json.loads(JOINT_MANIFEST.read_text(encoding="utf-8")),
                                           cfg.arm_half_range, cfg.arm_center),
        "target_semantics": TARGET_SEMANTICS, "action_smoothing": cfg.action_smoothing,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "robot_sha256": sha256(ROBOT_USD), "joint_manifest_sha256": sha256(JOINT_MANIFEST),
        "control_hz": 1 / (cfg.sim.dt * cfg.decimation), "target_lift_m": cfg.lift_height,
        "camera": camera_signature(cfg.scene.camera),
        "camera_secondary": camera_signature(cfg.scene.camera_secondary),
        "fingertip_contact_names": list(FINGERTIPS),
        "reset": "calibrated_pregrasp", "independent_finger_supervision": False}
