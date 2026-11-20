"""Agricultural HDF5 contract and action chunks for the bimanual VLA."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dexman.experiments import sha256
from dexman.tasks.agriculture.interface import EYE_KEYS
from dexman.tasks.agriculture.objectives import SUBTASKS_BY_CROP
from .bimanual_contract import MOBILE_SCHEMA, action_order, bimanual_contract, state_field
from .bimanual_language import instruction_prompt, tokenize_instructions


def episode_instructions(episode, phases, override=None):
    instruction = episode.attrs["instruction"] if override is None else override
    crop = episode.attrs["crop"]
    versioned = episode.attrs.get("subtasks_by_phase")
    subtasks = (json.loads(versioned) if versioned is not None else
                {str(key): value for key, value in SUBTASKS_BY_CROP[crop].items()})
    return [instruction_prompt(instruction, subtasks[str(int(phase))]) for phase in phases]


def _episode_shape(episode, schema):
    rgb = episode["rgb"]
    length = len(rgb)
    if not length or rgb.shape != (length, len(EYE_KEYS), 224, 224, 3) or rgb.dtype != np.uint8:
        raise ValueError("RGB agrícola requiere [T,2,224,224,3] uint8 y dos vistas sincronizadas")
    key, width = state_field(schema)
    shapes = {"depth_m": rgb.shape[:-1], "depth_valid": rgb.shape[:-1], key: (length, width),
              "joint_command": (length, len(action_order(schema))), "phase": (length,), "done": (length,),
              "time_s": (length,)}
    if schema == MOBILE_SCHEMA:
        shapes.update(camera_pos_env_m=(length, len(EYE_KEYS), 3), camera_quat_ros_xyzw=(length, len(EYE_KEYS), 4))
        intrinsics = np.asarray(episode.attrs.get("camera_intrinsics"))
        if (intrinsics.shape != (len(EYE_KEYS), 3, 3) or not np.isfinite(intrinsics).all()
                or (intrinsics[:, (0, 1), (0, 1)] <= 0).any()):
            raise ValueError("Se requieren los intrínsecos de ambos ojos con focales positivas")
    for key, shape in shapes.items():
        if episode[key].shape != shape:
            raise ValueError(f"Campo agrícola {key} fuera de contrato: {episode[key].shape}, se espera {shape}")
    if not np.issubdtype(episode["phase"].dtype, np.integer):
        raise ValueError("Las fases deben ser enteras")
    phases, done = episode["phase"][:], episode["done"][:]
    if done.dtype != np.bool_ or not done[-1] or done[:-1].any():
        raise ValueError("Un episodio agrícola completo termina exactamente en el último paso")
    if (np.diff(phases.astype(np.int64)) < 0).any():
        raise ValueError("Un episodio agrícola conserva el orden de las fases")
    episode_instructions(episode, phases)
    return length


def inspect_bimanual_episode(path, episode, metadata):
    contract = bimanual_contract(metadata)
    length = _episode_shape(episode, contract["schema"])
    limits = np.asarray(contract["joint_limits_rad"])
    times = episode["time_s"][:]
    if (metadata.get("sample_hz") != 30 or not np.isfinite(times).all()
            or not np.allclose(np.diff(times), 1 / 30, atol=2e-5)):
        raise ValueError("Los timestamps agrícolas deben avanzar a 30 Hz")
    key, _ = state_field(contract["schema"])
    stereo_baseline = None
    for start in range(0, length, 32):
        window = slice(start, start + 32)
        depth, valid = episode["depth_m"][window], episode["depth_valid"][window]
        if (valid.dtype != np.bool_ or not np.isfinite(depth).all() or (depth < 0).any()
                or not np.array_equal(valid, depth > 0)):
            raise ValueError("Inconsistencia entre profundidad en metros y máscara válida")
        command, state = episode["joint_command"][window], episode[key][window]
        if (not np.isfinite(command).all() or not np.isfinite(state).all()
                or (command < limits[:, 0] - 1e-6).any() or (command > limits[:, 1] + 1e-6).any()):
            raise ValueError("Estado o comando agrícola no finito o fuera de límites físicos")
        if contract["schema"] == MOBILE_SCHEMA:
            position = episode["camera_pos_env_m"][window]
            orientation = episode["camera_quat_ros_xyzw"][window]
            if not np.isfinite(position).all() or not np.allclose(np.linalg.norm(orientation, axis=-1), 1., atol=1e-3):
                raise ValueError("Las poses de ambos ojos deben ser finitas con cuaterniones unitarios")
            separation = np.linalg.norm(position[:, 0] - position[:, 1], axis=-1)
            if stereo_baseline is None:
                stereo_baseline = separation[0]
            if stereo_baseline <= 0. or not np.allclose(separation, stereo_baseline, rtol=0., atol=.003):
                raise ValueError("La distancia entre los ojos cambió dentro del episodio. Revisa la sincronización RGB-D.")
    reset = None
    if metadata.get("seed") is not None and all(key in episode.attrs for key in ("replica_id", "episode_id")):
        reset = [str(episode.attrs["crop"]), int(metadata["seed"]), int(episode.attrs["replica_id"]),
                 int(episode.attrs["episode_id"])]
    return {"sha256": sha256(path), "samples": length, "success": bool(episode.attrs["success"]),
            "scene_reset": reset, "exercise": metadata.get("exercise", "default"),
            "sample_kind": "teacher_demonstration", "seed": metadata.get("seed"),
            "teacher_sha256": metadata.get("teacher", {}).get("sha256"), "contract": contract,
            "crop": str(episode.attrs["crop"])}


def dataset_contract(files):
    contracts = set()
    for path in files:
        with h5py.File(path, "r") as episode:
            contracts.add(json.dumps(bimanual_contract(json.loads(episode.attrs["metadata"])), sort_keys=True))
    if len(contracts) != 1:
        raise ValueError("Se requiere un solo contrato agrícola de cámaras, juntas y comandos")
    return json.loads(contracts.pop())


class BimanualDemonstrations(Dataset):
    def __init__(self, files, horizon=8, sequence_length=8, *, prompts="guided"):
        if horizon < 1 or sequence_length < 2:
            raise ValueError("Se requiere un horizonte positivo y al menos dos pasos para entrenar memoria")
        self.horizon, self.sequence_length = horizon, sequence_length
        if prompts not in ("guided", "global", "both"):
            raise ValueError("Modo de órdenes agrícolas desconocido")
        self.prompts = prompts
        self.index = []
        contract = None
        for path in map(Path, files):
            with h5py.File(path, "r") as episode:
                if not bool(episode.attrs["success"]):
                    raise ValueError(f"El VLA agrícola solo aprende de episodios exitosos: {path}")
                metadata = json.loads(episode.attrs["metadata"])
                episode_contract = bimanual_contract(metadata)
                if contract is not None and episode_contract != contract:
                    raise ValueError("No se pueden mezclar perfiles de estado o límites articulares en un lote")
                contract = episode_contract
                schema = contract["schema"]
                length = _episode_shape(episode, schema)
            key, _ = state_field(schema)
            self.index.extend((path, start, length, key) for start in range(0, length, sequence_length))

    def __len__(self):
        return len(self.index) * (2 if self.prompts == "both" else 1)

    def __getitem__(self, index):
        guided = self.prompts == "guided"
        if self.prompts == "both":
            index, guided = divmod(index, 2)
        path, start, length, key = self.index[index]
        stop = min(start + self.sequence_length, length)
        target_stop = min(stop + self.horizon - 1, length)
        valid = stop - start
        with h5py.File(path, "r") as episode:
            phases = episode["phase"][start:target_stop]
            target = episode["joint_command"][start:target_stop].astype(np.float32)
            offsets = np.arange(self.sequence_length)[:, None] + np.arange(self.horizon)
            clipped = np.minimum(offsets, len(target) - 1)
            start_phases = phases[np.minimum(np.arange(self.sequence_length), len(phases) - 1)]
            mask = ((np.arange(self.sequence_length)[:, None] < valid) & (offsets < len(target))
                    & (start_phases[:, None] == phases[clipped]))
            state = episode[key][start:stop].astype(np.float32)
            depth = episode["depth_m"][start:stop].astype(np.float32)
            depth = np.where(episode["depth_valid"][start:stop], depth, 0.)
            return {"rgb": episode["rgb"][start:stop], "depth": depth,
                    "text": (episode_instructions(episode, phases[:valid]) if guided else
                             [instruction_prompt(episode.attrs["instruction"])] * valid),
                    "state": np.pad(state, ((0, self.sequence_length - valid), (0, 0))),
                    "joint_angles": target[clipped], "mask": mask}


def collate(samples, processor, tokenizer):
    images = [view for sample in samples for pair in sample["rgb"] for view in pair]
    prompts = [prompt for sample in samples for prompt in sample["text"]]
    unique = list(dict.fromkeys(prompts))
    index = {prompt: i for i, prompt in enumerate(unique)}
    return {"pixels": processor(images=images, return_tensors="pt")["pixel_values"],
            "text": tokenize_instructions(tokenizer, unique),
            "text_index": torch.tensor([index[prompt] for prompt in prompts]),
            "depth": torch.from_numpy(np.concatenate([sample["depth"] for sample in samples])),
            **{key: torch.from_numpy(np.stack([sample[key] for sample in samples]))
               for key in ("state", "joint_angles", "mask")}}
