"""Contiguous observation sequences and padded action chunks within each episode."""
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dexman.robot.control import decode
from .contract import contract_from_metadata


def dataset_contract(files):
    """Prevent mixing joint orders, limits, target semantics or teacher decoders."""
    contracts = set()
    for path in files:
        with h5py.File(path) as episode:
            metadata = json.loads(episode.attrs["metadata"])
        contract = contract_from_metadata(metadata)
        contracts.add(json.dumps(contract, sort_keys=True))
    if len(contracts) != 1:
        raise ValueError("Mixed robot/action contracts: use demonstrations from one task version only")
    return json.loads(contracts.pop())


def split_episodes(root, seed=42, validation_fraction=0.2):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"No existe la carpeta del dataset: {root}. "
                         "Primero genera demostraciones con -Stage collect; "
                         "despues usa su carpeta con -Dataset.")
    completed = sorted(root.rglob("*.h5"))
    files = [path for path in completed if is_success(path)]
    if len(files) < 2:
        partials = sum(1 for _ in root.rglob("*.h5.partial"))
        raise ValueError(f"Dataset insuficiente en {root}: {len(files)} episodios exitosos, "
                         f"{len(completed)} archivos .h5 completos y {partials} parciales. "
                         "Se necesitan al menos 2 episodios completos y exitosos para train/validation. "
                         "Genera demostraciones con -Stage collect; los parciales y los fallos no se usan.")
    random.Random(seed).shuffle(files)
    count = min(len(files) - 1, max(1, round(len(files) * validation_fraction)))
    return files[count:], files[:count]


def is_success(path):
    with h5py.File(path) as episode:
        metadata = json.loads(episode.attrs["metadata"])
        if metadata.get("schema_version") not in ("1.0", "1.1"):
            raise ValueError(f"Unknown dataset schema: {path}")
        return bool(episode.attrs["success"])


class Demonstrations(Dataset):
    def __init__(self, files, horizon=8, sequence_length=16):
        if horizon < 1 or sequence_length < 2:
            raise ValueError("horizon must be positive and sequence_length must be at least two")
        self.sequence_length = sequence_length
        self.contract = dataset_contract(files)
        self.limits = torch.tensor(self.contract["joint_limits_rad"], dtype=torch.float32)
        self.center = torch.tensor(self.contract["action_center_q"], dtype=torch.float32)
        self.horizon, self.index = horizon, []
        for path in files:
            with h5py.File(path) as episode:
                if episode["action"].shape[1] != 26 or episode["qpos"].shape[1] != 30:
                    raise ValueError(f"Robot dimension mismatch: {path}")
                length = len(episode["action"])
                if not length or any(len(episode[k]) != length for k in ("qpos", "qvel", "rgb")):
                    raise ValueError(f"Empty or misaligned episode: {path}")
                if episode["qvel"].shape[1] != 30:
                    raise ValueError(f"Velocity dimension mismatch: {path}")
                metadata = json.loads(episode.attrs["metadata"])
                if metadata["schema_version"] == "1.1" and (
                        "joint_command" not in episode or episode["joint_command"].shape != (length, 30)):
                    raise ValueError(f"Schema 1.1 requires joint_command [T,30]: {path}")
                if not str(episode.attrs["instruction"]).strip():
                    raise ValueError(f"An instruction is required: {path}")
                self.index.extend((path, start) for start in range(0, length, sequence_length))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        path, start = self.index[index]
        with h5py.File(path) as episode:
            correction = json.loads(episode.attrs["metadata"]).get("sample_kind") == "teacher_correction"
            stop = min(start + self.sequence_length, len(episode["action"]))
            target_stop = stop if correction else stop + self.horizon - 1
            actions = episode["action"][start:target_stop].astype(np.float32)
            if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.000001):
                raise ValueError(f"Invalid normalized teacher actions: {path}")
            decoded = decode(torch.from_numpy(actions), self.limits, self.center).numpy()
            # The applied `joint_target` is filtered. It is deliberately NOT the label.
            targets = episode["joint_command"][start:target_stop] if "joint_command" in episode else decoded
            if not np.isfinite(targets).all() or not np.allclose(targets, decoded, atol=1e-5, rtol=0):
                raise ValueError(f"Joint commands must match decoded pre-smoothing teacher actions: {path}")
            valid = stop - start
            steps = np.arange(self.sequence_length)[:, None]
            future = np.arange(self.horizon)
            offsets = steps + future
            mask = (steps < valid) & (offsets < len(targets))
            if correction:
                mask &= future == 0
            chunks = targets[np.minimum(offsets, len(targets) - 1)].astype(np.float32)
            state = np.concatenate((episode["qpos"][start:stop], episode["qvel"][start:stop]), axis=-1)
            if not np.isfinite(state).all():
                raise ValueError(f"Non-finite robot state: {path}")
            state = np.pad(state.astype(np.float32), ((0, self.sequence_length - valid), (0, 0)))
            images = [Image.fromarray(frame) for frame in episode["rgb"][start:stop]]
            return {"image": images, "text": episode.attrs["instruction"], "state": state,
                    "joint_angles": chunks, "mask": mask}
