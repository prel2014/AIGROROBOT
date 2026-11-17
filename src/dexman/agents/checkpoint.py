"""Validate teacher architecture and checkpoint state before launching a simulator."""
import importlib.metadata
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from dexman.agents.models import RSL_RL_VERSION, build_teacher_model, teacher_architecture


CHECKPOINT_VERSION = 4
LEGACY_ERROR = ("This checkpoint is an MLP/legacy teacher or has no Transformer architecture contract. "
                "The new state Transformer architecture requires training from scratch; "
                "do not use --resume or --checkpoint for the first Transformer run.")


def validate_teacher_runtime():
    installed = importlib.metadata.version("rsl-rl-lib")
    if installed != RSL_RL_VERSION:
        raise ValueError(f"Teacher requires rsl-rl-lib {RSL_RL_VERSION}; installed {installed}")


def _validate_training_schedule(schedule):
    fields = {"version", "remaining", "total", "actor_lr", "critic_lr", "kl_limit",
              "min_actor_lr", "accepted", "rejected"}
    if (not isinstance(schedule, dict) or set(schedule) != fields
            or type(schedule["version"]) is not int or schedule["version"] != 1):
        raise ValueError("Teacher checkpoint requires the complete version 1 training schedule")
    for name in ("remaining", "total", "accepted", "rejected"):
        if type(schedule[name]) is not int or schedule[name] < 0:
            raise ValueError(f"Invalid teacher training schedule counter: {name}")
    if schedule["remaining"] > schedule["total"] or (
            schedule["remaining"] and (schedule["accepted"] or schedule["rejected"])):
        raise ValueError("Teacher warmup remaining/total and actor update counters are inconsistent")
    for name in ("actor_lr", "critic_lr", "kl_limit", "min_actor_lr"):
        value = schedule[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid teacher training schedule value: {name}")
    if schedule["min_actor_lr"] > schedule["actor_lr"]:
        raise ValueError("Teacher minimum actor learning rate exceeds its current learning rate")
    return schedule


def _validate_optimizer(optimizer, parameters_by_role, schedule):
    if not isinstance(optimizer, dict) or not {"state", "param_groups"}.issubset(optimizer):
        raise ValueError("Invalid teacher optimizer state")
    groups, states = optimizer["param_groups"], optimizer["state"]
    if not isinstance(groups, list) or len(groups) != 2 or not isinstance(states, dict):
        raise ValueError("Teacher optimizer requires separate actor and critic parameter groups")
    all_identifiers = set()
    for role, group in zip(("actor", "critic"), groups, strict=True):
        parameters = parameters_by_role[role]
        identifiers = group.get("params") if isinstance(group, dict) else None
        if (not isinstance(group, dict) or group.get("name") != role or not isinstance(identifiers, list)
                or len(identifiers) != len(parameters)
                or any(type(identifier) is not int or identifier < 0 for identifier in identifiers)
                or len(set(identifiers)) != len(identifiers) or all_identifiers.intersection(identifiers)):
            raise ValueError(f"Teacher {role} optimizer group does not match its model parameters")
        all_identifiers.update(identifiers)
        expected_lr = schedule["critic_lr"] if role == "critic" else (
            0. if schedule["remaining"] else schedule["actor_lr"])
        rate = group.get("lr")
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate != expected_lr:
            raise ValueError(f"Teacher {role} optimizer learning rate differs from the saved training schedule")
        for identifier, parameter in zip(identifiers, parameters, strict=True):
            # Adam initializes moments lazily. Warmup legitimately has no actor state;
            # rejected first actor proposals can also restore that absent state.
            state = states.get(identifier, {})
            if not isinstance(state, dict):
                raise ValueError(f"Invalid teacher {role} Adam parameter state")
            if state and not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
                raise ValueError(f"Incomplete teacher {role} Adam moments")
            for name, value in state.items():
                if name == "step":
                    if isinstance(value, torch.Tensor):
                        if value.numel() != 1 or not torch.isfinite(value).all():
                            raise ValueError(f"Invalid teacher {role} Adam step")
                        value = value.item()
                    if (type(value) not in (int, float) or not math.isfinite(value)
                            or value < 0 or not float(value).is_integer()):
                        raise ValueError(f"Invalid teacher {role} Adam step")
                elif (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                      or value.shape != parameter.shape or not torch.isfinite(value).all()):
                    raise ValueError(f"Invalid teacher {role} optimizer tensor: {name}")
    if any(type(identifier) is not int or identifier not in all_identifiers for identifier in states):
        raise ValueError("Teacher optimizer state references an unknown model parameter")


def load_teacher_checkpoint(checkpoint):
    """Load on CPU and verify model shapes, normalizers and full resume metadata."""
    validate_teacher_runtime()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError(LEGACY_ERROR)
    infos = saved.get("infos")
    info = infos.get("dexman_teacher") if isinstance(infos, dict) else None
    architecture = info.get("architecture") if isinstance(info, dict) else None
    if not isinstance(architecture, dict) or architecture.get("type") != "structured_state_transformer":
        raise ValueError(LEGACY_ERROR)
    if info.get("checkpoint_version") != CHECKPOINT_VERSION or info["architecture"] != teacher_architecture():
        raise ValueError("Teacher Transformer checkpoint schema/architecture differs from this implementation")
    required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter"}
    metadata = {"contract", "learning_state", "training_config", "training_schedule",
                "torch_rng", "python_rng", "numpy_rng", "cuda_rng"}
    if not required.issubset(saved) or not metadata.issubset(info):
        raise ValueError("Incomplete Transformer teacher checkpoint; full optimizer/curriculum/RNG state is required")
    if not isinstance(saved["iter"], int) or saved["iter"] < 0:
        raise ValueError("Invalid teacher iteration")
    if not isinstance(info["training_config"], dict):
        raise ValueError("Invalid teacher training configuration")
    for role in ("actor", "critic"):
        if info["training_config"].get(role) != teacher_architecture()[role]:
            raise ValueError(f"Teacher {role} constructor does not match the architecture contract")
    schedule = _validate_training_schedule(info["training_schedule"])
    # Preserve caller RNG: model construction is validation, not a new training initialization.
    with torch.random.fork_rng(devices=[]):
        parameters_by_role = {}
        for role in ("actor", "critic"):
            model = build_teacher_model(role)
            parameters_by_role[role] = list(model.parameters())
            try:
                model.load_state_dict(saved[f"{role}_state_dict"], strict=True)
            except (RuntimeError, TypeError) as error:
                raise ValueError(f"Invalid Transformer teacher {role} state: {error}") from error
            for name, value in model.state_dict().items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite teacher {role} parameter/statistic: {name}")
            if (model.obs_normalizer._std < 0).any() or (model.obs_normalizer._var < 0).any():
                raise ValueError(f"Invalid teacher {role} observation normalization")
        _validate_optimizer(saved["optimizer_state_dict"], parameters_by_role, schedule)
    learning = info["learning_state"]
    if not isinstance(learning, dict) or not {"curriculum", "episodes", "successes"}.issubset(learning):
        raise ValueError("Incomplete teacher curriculum/episode state")
    if not 0 <= learning["successes"] <= learning["episodes"]:
        raise ValueError("Invalid teacher episode counters")
    try:
        torch.Generator(device="cpu").set_state(info["torch_rng"])
        random.Random().setstate(info["python_rng"])
        rng = info["numpy_rng"]
        np.random.RandomState().set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:]))
        if not isinstance(info["cuda_rng"], list) or any(
                not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1
                for value in info["cuda_rng"]):
            raise ValueError("Invalid CUDA RNG encoding")
    except (TypeError, ValueError, RuntimeError, IndexError) as error:
        raise ValueError(f"Invalid teacher RNG state: {error}") from error
    return saved


def validate_teacher(checkpoint, cfg):
    saved = load_teacher_checkpoint(checkpoint)
    from dexman.agents.teacher import teacher_contract
    contract = teacher_contract(cfg)
    if saved["infos"]["dexman_teacher"]["contract"] != contract:
        raise ValueError("Teacher control/physics/reward contract differs from this environment")
    from dexman.tasks.lift.curriculum import LiftCurriculum
    curriculum = LiftCurriculum(tuple(cfg.curriculum_heights), cfg.curriculum_window,
                                cfg.curriculum_success_threshold, cfg.curriculum_initial_level)
    curriculum.load_state_dict(saved["infos"]["dexman_teacher"]["learning_state"]["curriculum"])
    root = Path(checkpoint).resolve().parent
    manifest = json.loads((root / "run.json").read_text())
    if manifest.get("task") != "Dexman-UR10e-Shadow-Lift-v1" or manifest.get("validation_only"):
        raise ValueError("Use a corrected v1 training checkpoint, not a legacy or startup-test checkpoint")
    if "control_contract" in manifest:
        if manifest["control_contract"] != contract:
            raise ValueError("Teacher control/physics/reward contract differs from this environment")
    if manifest.get("teacher_architecture") != teacher_architecture():
        raise ValueError("Run manifest is missing or mismatches the Transformer teacher architecture")
    node = yaml.compose((root / "params/env.yaml").read_text(encoding="utf-8"))
    fields = {key.value: value for key, value in node.value}
    for name in ("arm_half_range", "arm_center", "action_center"):
        field = fields.get(name)
        if field is None or field.tag.endswith(":null"):
            raise ValueError(f"Checkpoint has no corrected control contract ({name}); retrain")
        expected = getattr(cfg, name)
        actual = [float(x.value) for x in field.value] if isinstance(expected, (list, tuple)) else [float(field.value)]
        expected = list(expected) if isinstance(expected, (list, tuple)) else [expected]
        if len(actual) != len(expected) or any(abs(a - b) > 1e-6 for a, b in zip(actual, expected)):
            raise ValueError(f"Checkpoint uses a different {name}; use a matching configuration or retrain")
