"""RGB, text, joint state and memory to absolute joint-angle chunks and updated memory."""
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPProcessor

from dexman.perception.clip import prepare_multiview_inputs

from .contract import POLICY_VERSION, validate_architecture_config, validate_contract
from .model import ActionChunkPolicy


class VLAPredictor:
    def __init__(self, checkpoint, device="cpu", *, saved=None):
        checkpoint = Path(checkpoint)
        if saved is None:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or saved.get("policy_version") != POLICY_VERSION:
            raise ValueError("Expected a VLA v5 spatial Transformer checkpoint; legacy policies require new training")
        config = saved.get("config")
        if (not isinstance(config, dict) or type(config.get("horizon")) is not int
                or config["horizon"] < 1 or "robot_contract" not in config):
            raise ValueError("VLA v5 checkpoint is missing its horizon or robot contract")
        architecture = validate_architecture_config(config.get("architecture"), horizon=config["horizon"])
        self.robot_contract = validate_contract(config["robot_contract"])
        normalization = {}
        for key in ("state_mean", "state_std"):
            value = saved.get(key)
            if (not isinstance(value, torch.Tensor) or value.shape != (60,)
                    or not value.is_floating_point() or not torch.isfinite(value).all()
                    or (key == "state_std" and not (value > 0).all())):
                raise ValueError(f"VLA v5 checkpoint requires finite {key} [60] and positive state_std")
            normalization[key] = value
        self.device = device
        processor = checkpoint.parent / "processor"
        backbone = checkpoint.parent / "backbone"
        if not backbone.is_dir() or not processor.is_dir():
            raise ValueError("Missing local frozen backbone; keep backbone/ and processor/ beside the checkpoint")
        self.processor = CLIPProcessor.from_pretrained(processor, local_files_only=True)
        self.model = ActionChunkPolicy(str(backbone), config["horizon"],
            joint_limits=self.robot_contract["joint_limits_rad"], local_only=True, architecture_config=architecture)
        try:
            self.model.head.load_state_dict(saved["head"], strict=True)
        except (KeyError, RuntimeError, TypeError) as error:
            raise ValueError("Checkpoint weights differ from the declared VLA v5 architecture; new training is required") from error
        self.model.state_mean.copy_(normalization["state_mean"])
        self.model.state_std.copy_(normalization["state_std"])
        self.model.to(device).eval()

    def predict(self, image, instruction, qpos, qvel, memory=None):
        """Return angles [horizon,30] and updated memory [256] for one robot."""
        batch_memory = None if memory is None else np.asarray(memory)[None]
        angles, updated = self.predict_batch(
            [image], [instruction], np.asarray(qpos)[None], np.asarray(qvel)[None], batch_memory)
        return angles[0], updated[0]

    @torch.inference_mode()
    def predict_batch(self, images, instructions, qpos, qvel, memory=None):
        """Return angles [N,horizon,30] and updated memory [N,256] for N robots."""
        qpos, qvel = np.asarray(qpos), np.asarray(qvel)
        count = len(images)
        if count < 1 or qpos.shape != (count, 30) or qvel.shape != (count, 30) or len(instructions) != count:
            raise ValueError("Expected N images/instructions and qpos/qvel [N,30] in JOINT_NAMES order")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("Non-finite robot state")
        memory_shape = (count, self.model.memory_dim)
        memory = np.zeros(memory_shape, dtype=np.float32) if memory is None else np.asarray(memory, dtype=np.float32)
        if memory.shape != memory_shape or not np.isfinite(memory).all():
            raise ValueError(f"Expected finite memory with shape {memory_shape}")
        inputs = prepare_multiview_inputs(self.processor, images, instructions, self.device)
        state = torch.as_tensor(np.concatenate((qpos, qvel), axis=-1), dtype=torch.float32, device=self.device)
        memory_tensor = torch.as_tensor(memory, device=self.device)
        encoded = self.model.encode(**inputs)
        angles, updated = self.model(encoded["vision_tokens"], encoded["text_embedding"], state, memory_tensor)
        angles, updated = angles.cpu().numpy(), updated.cpu().numpy()
        if not np.isfinite(angles).all() or not np.isfinite(updated).all():
            raise ValueError("Non-finite policy output")
        return angles, updated


class BimanualVLAPredictor:
    def __init__(self, checkpoint, device="cpu", *, saved=None):
        from transformers import AutoTokenizer, CLIPImageProcessor

        from dexman.perception.bimanual import load_frozen_bimanual

        from .bimanual import BimanualActionPolicy
        from .bimanual_contract import POLICY_VERSION, architecture_config, bimanual_contract

        checkpoint = Path(checkpoint)
        if saved is None:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = saved.get("config") if isinstance(saved, dict) else None
        if (not isinstance(config, dict) or saved.get("policy_version") != POLICY_VERSION
                or config.get("horizon") != BimanualActionPolicy.horizon):
            raise ValueError("Se requiere un checkpoint VLA agrícola RGB-D bimanual compatible")
        self.robot_contract = bimanual_contract(config["robot_contract"])
        schema = self.robot_contract["schema"]
        if config.get("architecture") != architecture_config(schema):
            raise ValueError("La dimensión o el perfil de estado del checkpoint no coinciden con el robot")
        self.sources = config["model"], config["text_model"]
        assets = {name: checkpoint.parent / name for name in ("processor", "backbone", "tokenizer", "text_encoder")}
        if not all(path.is_dir() for path in assets.values()):
            raise ValueError("Mantén processor/, backbone/, tokenizer/ y text_encoder/ junto al checkpoint agrícola")
        self.processor = CLIPImageProcessor.from_pretrained(assets["processor"], local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(assets["tokenizer"], local_files_only=True)
        encoders = load_frozen_bimanual(assets["backbone"], assets["text_encoder"], local_only=True)
        self.model = BimanualActionPolicy(self.robot_contract["joint_limits_rad"], *encoders, schema=schema)
        declared = self.model.head.joint_limits.detach().clone()
        self.model.head.load_state_dict(saved["head"], strict=True)
        head = self.model.head
        if (not torch.equal(head.joint_limits, declared) or not torch.equal(head.joint_offset, declared.mean(-1))
                or not torch.equal(head.joint_scale, declared.diff(dim=-1).squeeze(-1) * .5)):
            raise ValueError("Los límites articulares guardados no coinciden con el contrato agrícola")
        self.model.to(device).eval()

    def predict(self, rgb, depth, instruction, state, memory=None, *, subtask=None):
        """Return eight target vectors in robot_contract joint order and the next memory."""
        memory = np.zeros(self.model.memory_dim, dtype=np.float32) if memory is None else np.asarray(memory)
        angles, updated = self.model.predict(
            self.processor, self.tokenizer, np.asarray(rgb)[None], np.asarray(depth)[None], [instruction],
            np.asarray(state)[None], memory[None], subtasks=[subtask])
        if not torch.isfinite(angles).all() or not torch.isfinite(updated).all():
            raise ValueError("El VLA agrícola devolvió ángulos o memoria no finitos")
        return angles[0].cpu().numpy(), updated[0].cpu().numpy()


def load_predictor(checkpoint, device="cpu"):
    from .bimanual_contract import POLICY_VERSION as BIMANUAL_VERSION

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    kind = BimanualVLAPredictor if isinstance(saved, dict) and saved.get("policy_version") == BIMANUAL_VERSION else VLAPredictor
    return kind(checkpoint, device, saved=saved)


class ChunkController:
    """Keep the recurrent memory and cached actions separate for each robot replica."""
    def __init__(self, predictor, num_envs, execute_steps=1):
        if num_envs < 1 or not 1 <= execute_steps <= predictor.model.horizon:
            raise ValueError("num_envs > 0 and 1 <= execute_steps <= checkpoint horizon required")
        self.predictor, self.execute_steps = predictor, execute_steps
        self.cursor = np.full(num_envs, execute_steps, dtype=np.int64)
        self.chunks = np.zeros((num_envs, predictor.model.horizon, 30), dtype=np.float32)
        self.memory = np.zeros((num_envs, predictor.model.memory_dim), dtype=np.float32)
        self.instructions = [None] * num_envs

    def reset(self, dones):
        dones = np.asarray(dones, dtype=bool)
        if dones.shape != self.cursor.shape:
            raise ValueError("Expected one done flag per robot")
        self.cursor[dones] = self.execute_steps
        self.chunks[dones] = 0
        self.memory[dones] = 0
        for index in np.flatnonzero(dones):
            self.instructions[index] = None

    def act(self, images, instructions, qpos, qvel):
        count = len(self.cursor)
        if len(images) != count or len(instructions) != count:
            raise ValueError("Expected one RGB image and instruction per robot")
        changed = np.asarray([old != current for old, current in zip(self.instructions, instructions)])
        input_memory = np.where(changed[:, None], 0, self.memory)
        proposed, memory = self.predictor.predict_batch(images, instructions, qpos, qvel, input_memory)
        self.memory[:] = memory
        self.instructions = list(instructions)
        replan = changed | (self.cursor >= self.execute_steps)
        self.chunks[replan] = proposed[replan]
        self.cursor[replan] = 0
        actions = self.chunks[np.arange(count), self.cursor].copy()
        self.cursor += 1
        return actions
