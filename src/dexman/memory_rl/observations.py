"""Compose reusable visual/language features with normalized joint positions."""
import torch
from dexman.perception.clip import load_frozen_clip, clip_embeddings, prepare_inputs


class VisualFeatures:
    """Actor input is exclusively RGB + instruction + current q; the GRU supplies history."""
    def __init__(self, source, commands, device="cpu", local_only=False):
        from transformers import CLIPProcessor
        self.device, self.commands = device, commands
        self.processor = CLIPProcessor.from_pretrained(source, local_files_only=local_only)
        self.backbone = load_frozen_clip(source, local_only=local_only).to(device)
        self.dimension = 2 * self.backbone.config.projection_dim + 30

    def save(self, directory):
        self.processor.save_pretrained(directory)
        self.backbone.save_pretrained(directory)

    @torch.no_grad()
    def __call__(self, images, instructions, qpos):
        qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        if qpos.shape != (len(images), 30) or not torch.isfinite(qpos).all():
            raise ValueError("Expected finite qpos [N,30] in JOINT_NAMES order")
        inputs = prepare_inputs(self.processor, images, instructions, self.device)
        return torch.cat((clip_embeddings(self.backbone, inputs), self.commands.normalize(qpos)), -1)
