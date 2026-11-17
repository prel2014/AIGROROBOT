"""Predict joint angles and updated memory from a recorded observation."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from dexman.vla.inference import VLAPredictor
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
    memory = None
    if args.memory_in:
        try:
            with np.load(args.memory_in, allow_pickle=False) as prior:
                memory = prior["memory"]
        except (OSError, ValueError, TypeError, KeyError) as error:
            parser.error(f"Cannot read VLA memory from {args.memory_in}: {error}")
    predictor = VLAPredictor(args.checkpoint, args.device)
    with h5py.File(args.episode) as episode:
        if contract_from_metadata(json.loads(episode.attrs["metadata"])) != predictor.robot_contract:
            parser.error("episode and checkpoint robot/action contracts differ")
        if not 0 <= args.frame < len(episode["action"]):
            parser.error("frame outside episode")
        joint_angles, memory = predictor.predict(
            np.stack((episode["rgb"][args.frame], episode["rgb_secondary"][args.frame])), args.instruction or episode.attrs["instruction"],
            episode["qpos"][args.frame], episode["qvel"][args.frame], memory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez(stream, joint_angles=joint_angles, memory=memory)
    print(f"Saved joint angles {joint_angles.shape} in radians and memory {memory.shape} to {output}")
