"""Validate recorded episodes before allocating a model or GPU simulator."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from dexman.experiments import sha256
from dexman.robot.control import decode
from dexman.robot.joints import FINGERTIPS
from dexman.vla.bimanual_contract import DEMO_SCHEMA
from dexman.vla.contract import contract_from_metadata


def inspect_episode(path):
    path = Path(path)
    with h5py.File(path, "r") as episode:
        metadata = json.loads(episode.attrs["metadata"])
        if metadata.get("schema") == DEMO_SCHEMA:
            from dexman.vla.bimanual_data import inspect_bimanual_episode

            return inspect_bimanual_episode(path, episode, metadata)
        contract = contract_from_metadata(metadata)
        length = len(episode["action"])
        if not length or episode["action"].shape != (length, 26):
            raise ValueError("Expected nonempty teacher actions [T,26]")
        if not str(episode.attrs.get("instruction", "")).strip():
            raise ValueError("Missing instruction")
        for key in ("qpos", "qvel"):
            if episode[key].shape != (length, 30):
                raise ValueError(f"Expected {key} [T,30]")
        rgb = episode["rgb"]
        if rgb.ndim != 4 or rgb.shape[0] != length or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB [T,H,W,3]")
        if "rgb_secondary" not in episode or episode["rgb_secondary"].shape != rgb.shape or episode["rgb_secondary"].dtype != np.uint8:
            raise ValueError("Expected synchronized uint8 secondary RGB with identical shape")
        for key in ("depth_m", "depth_secondary_m"):
            if key not in episode or episode[key].shape != (*rgb.shape[:3], 1):
                raise ValueError(f"Expected aligned metric depth [T,H,W,1]: {key}")
            for start in range(0, length, 256):
                values = episode[key][start:start+256]
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f"Invalid metric depth: {key}")
        forces = episode.get("fingertip_normal_force_w_n")
        if forces is not None or "fingertip_contact_names" in metadata:
            if (metadata.get("fingertip_contact_names") != list(FINGERTIPS)
                    or forces is None or forces.shape != (length, len(FINGERTIPS), 3)
                    or not np.isfinite(forces[:]).all()):
                raise ValueError("Expected finite fingertip force vectors [T,5,3] in FINGERTIPS order")
        if "done" not in episode or episode["done"].shape != (length,):
            raise ValueError("Missing aligned episode termination flags")
        done = episode["done"][:].astype(bool)
        if not done[-1] or done[:-1].any():
            raise ValueError("A complete episode must terminate exactly at its last frame")
        if metadata["schema_version"] != "1.0" and episode.get("joint_command", None) is None:
            raise ValueError("Missing pre-smoothing joint_command")
        limits = torch.tensor(contract["joint_limits_rad"], dtype=torch.float32)
        center = torch.tensor(contract["action_center_q"], dtype=torch.float32)
        for start in range(0, length, 256):
            stop = start + 256
            actions = episode["action"][start:stop]
            if not np.isfinite(actions).all() or (np.abs(actions) > 1.000001).any():
                raise ValueError("Invalid normalized teacher action")
            for key in ("qpos", "qvel"):
                if not np.isfinite(episode[key][start:stop]).all():
                    raise ValueError(f"Nonfinite {key}")
            if "joint_command" in episode:
                command = episode["joint_command"][start:stop]
                expected = decode(torch.as_tensor(actions), limits, center).numpy()
                if command.shape != expected.shape or not np.isfinite(command).all() or not np.allclose(
                        command, expected, atol=1e-5, rtol=0):
                    raise ValueError("joint_command differs from the teacher label before smoothing")
        if "timestamp_s" in episode:
            times = episode["timestamp_s"][:]
            if times.shape != (length,) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
                raise ValueError("Timestamps must be finite and strictly increasing")
        kind = metadata.get("sample_kind", "teacher_demonstration")
        if kind not in ("teacher_demonstration", "teacher_correction"):
            raise ValueError("Unknown sample_kind")
        return {"sha256": sha256(path), "samples": length, "success": bool(episode.attrs["success"]),
                "sample_kind": kind, "seed": metadata.get("seed"),
                "teacher_sha256": metadata.get("checkpoint_sha256"), "contract": contract}


def audit_dataset(root):
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Dataset root must be a directory")
    episodes, errors = [], []
    for path in sorted(root.rglob("*.h5")):
        try:
            episodes.append({"path": path.relative_to(root).as_posix(), **inspect_episode(path)})
        except (ValueError, KeyError, OSError, TypeError) as error:
            errors.append({"path": path.relative_to(root).as_posix(), "error": str(error)})
    contracts = {json.dumps(row["contract"], sort_keys=True) for row in episodes}
    if len(contracts) > 1:
        errors.append({"path": ".", "error": "Mixed robot/action contracts"})
    return {"root": str(root), "episodes": episodes, "errors": errors,
            "partial_files": len(list(root.rglob("*.h5.partial"))),
            "successes": sum(row["success"] for row in episodes),
            "corrections": sum(row["sample_kind"] == "teacher_correction" for row in episodes),
            "valid": not errors and bool(episodes)}
