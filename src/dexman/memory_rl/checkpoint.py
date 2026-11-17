"""Versioned policy contract and safe checkpoint loading."""
import copy
import torch
from dexman.robot.joints import JOINT_NAMES
from dexman.robot.commands import JointCommands
from dexman.tasks.manipulation.catalog import catalog
from .config import actor_config

VERSION = "multitask_rgb_text_q_gru_ppo_v1"


def checkpoint_contract(*, joint_limits, joint_center, smoothing, step_dt, episode_length_s,
                        object_xy_range, feature_dimension, training):
    """Build metadata from values rather than depending on an Isaac environment object."""
    return {"version": VERSION, "joint_names": list(JOINT_NAMES), "units": "radians",
            "output": "30_absolute_joint_angles_before_smoothing", "tasks": catalog(),
            "joint_limits": torch.as_tensor(joint_limits).cpu().tolist(), "joint_center": list(joint_center),
            "smoothing": smoothing, "step_dt": step_dt,
            "episode_length_s": episode_length_s, "object_xy_range": object_xy_range,
            "actor_inputs": ["rgb", "instruction", "qpos"], "feature_dimension": feature_dimension,
            "training_config": copy.deepcopy(training)}


def load_checkpoint(path):
    saved = torch.load(path, weights_only=True, map_location="cpu")
    contract = copy.deepcopy(saved.get("infos", {})) if isinstance(saved, dict) else {}
    if (not isinstance(contract, dict) or contract.get("version") != VERSION
            or contract.get("joint_names") != list(JOINT_NAMES) or contract.get("tasks") != catalog()
            or contract.get("units") != "radians"
            or contract.get("actor_inputs") != ["rgb", "instruction", "qpos"]
            or contract.get("output") != "30_absolute_joint_angles_before_smoothing"):
        raise ValueError("Expected a compatible multitask recurrent PPO checkpoint")
    JointCommands(contract["joint_limits"], contract["joint_center"])
    # Never dynamically instantiate classes named by a checkpoint.
    actor = contract.get("training_config", {}).get("actor", {})
    # This refactor changes only the import path, not the architecture or weight names.
    if actor.get("class_name") == "dexman.memory_rl.policy:MemoryActor":
        actor["class_name"] = "dexman.agents.recurrent:MemoryActor"
    if actor != actor_config():
        raise ValueError("Unsupported recurrent actor configuration")
    return saved, contract
