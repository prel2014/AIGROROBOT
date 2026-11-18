"""Predict joint angles and updated memory from a recorded observation."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from dexman.experiments import sha256
from dexman.vla.inference import BimanualVLAPredictor, load_predictor
from dexman.vla.contract import contract_from_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--memory-in", help="Previous VLA prediction .npz to continue its memory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="logs/vla_prediction.npz")
    args = parser.parse_args()
    if args.instruction is not None and not args.instruction.strip():
        parser.error("instruction must contain text")
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; choose a new filename")
    context = {}
    try:
        predictor = load_predictor(args.checkpoint, args.device)
        with h5py.File(args.episode) as episode:
            metadata = json.loads(episode.attrs["metadata"])
            instruction = args.instruction or episode.attrs["instruction"]
            if isinstance(predictor, BimanualVLAPredictor):
                from dexman.vla.bimanual_contract import bimanual_contract
                from dexman.vla.bimanual_data import episode_instructions

                if bimanual_contract(metadata) != predictor.robot_contract:
                    raise ValueError("El episodio y el checkpoint difieren en su contrato agrícola")
                if not 0 <= args.frame < len(episode["rgb"]):
                    raise ValueError("El fotograma no pertenece al episodio agrícola")
                context = {"episode": str(Path(args.episode).resolve()), "checkpoint_sha256": sha256(args.checkpoint),
                           "instruction": instruction, "frame": args.frame}
                memory = _read_memory(args.memory_in, context)
                frame = args.frame
                prompt, = episode_instructions(episode, [episode["phase"][frame]], override=instruction)
                depth = np.where(episode["depth_valid"][frame], episode["depth_m"][frame], 0.)
                joint_angles, memory = predictor.predict(episode["rgb"][frame], depth, prompt,
                                                          episode["angular_state"][frame], memory)
            else:
                if contract_from_metadata(metadata) != predictor.robot_contract:
                    raise ValueError("episode and checkpoint robot/action contracts differ")
                if not 0 <= args.frame < len(episode["action"]):
                    raise ValueError("frame outside episode")
                memory = _read_memory(args.memory_in)
                joint_angles, memory = predictor.predict(
                    np.stack((episode["rgb"][args.frame], episode["rgb_secondary"][args.frame])), instruction,
                    episode["qpos"][args.frame], episode["qvel"][args.frame], memory)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez(stream, joint_angles=joint_angles, memory=memory, **context)
    print(f"Saved joint angles {joint_angles.shape} in radians and memory {memory.shape} to {output}")


def _read_memory(path, context=None):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as prior:
        if context is not None:
            expected = {**context, "frame": context["frame"] - 1}
            if any(key not in prior or prior[key].ndim != 0 or prior[key].item() != value
                   for key, value in expected.items()):
                raise ValueError("La memoria agrícola debe provenir del fotograma anterior, misma orden, episodio y checkpoint")
        return prior["memory"]
