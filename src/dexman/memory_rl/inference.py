"""Stateful RGB + angles + text inference without simulator or optimizer imports."""
from pathlib import Path
import torch
from tensordict import TensorDict
from dexman.agents.recurrent import make_actor
from dexman.robot.commands import JointCommands
from dexman.tasks.manipulation.catalog import resolve_instruction
from .checkpoint import load_checkpoint
from .observations import VisualFeatures


class MemoryPredictor:
    """Public API: RGB + q[30] + text -> target angles[30], with memory per replica."""
    def __init__(self, checkpoint, num_envs=1, device="cpu"):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        saved, self.contract = load_checkpoint(checkpoint)
        self.device, self.num_envs = device, num_envs
        self.commands = JointCommands(self.contract["joint_limits"], self.contract["joint_center"], device)
        self.encoder = VisualFeatures(Path(checkpoint).parent / "encoder", self.commands, device, local_only=True)
        if self.encoder.dimension != self.contract["feature_dimension"]:
            raise ValueError("Encoder dimension differs from checkpoint")
        self.actor = make_actor(self.encoder.dimension, self.contract["training_config"]["actor"], device)
        self.actor.load_state_dict(saved["actor_state_dict"])
        self.actor.eval()
        self.instructions = [None] * num_envs

    @torch.inference_mode()
    def reset(self, dones=None):
        if dones is None:
            dones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        if dones.shape != (self.num_envs,):
            raise ValueError("Expected one reset flag per replica")
        self.actor.reset(dones)
        for index in dones.nonzero().flatten().tolist():
            self.instructions[index] = None

    @torch.inference_mode()
    def predict_batch(self, images, qpos, instructions):
        if len(images) != self.num_envs or len(instructions) != self.num_envs:
            raise ValueError("The batch must retain one slot per robot; reset a slot before reusing it")
        for text in instructions:
            resolve_instruction(text)
        changed = torch.tensor([old != new for old, new in zip(self.instructions, instructions)], device=self.device)
        self.reset(changed)
        features = self.encoder(images, instructions, qpos)
        self.instructions = list(instructions)
        normalized = self.actor(TensorDict({"policy": features}, batch_size=[self.num_envs]))
        return self.commands.angles(normalized).cpu().numpy()
