"""Inference adapters with matching observations and independent episode resets."""
import numpy as np
import torch


class TeacherPolicy:
    def __init__(self, checkpoint, device):
        from dexman.agents.teacher import load_teacher_actor
        self.actor = load_teacher_actor(checkpoint, device)

    def act(self, raw, observations):
        from tensordict import TensorDict
        return self.actor(TensorDict(observations, batch_size=[raw.num_envs])).clamp(-1, 1)

    def reset(self, dones):
        self.actor.reset(torch.as_tensor(dones, device=next(self.actor.parameters()).device))


class VisualPolicy:
    def __init__(self, checkpoint, device, num_envs, execute_steps=1):
        from dexman.vla.inference import ChunkController, VLAPredictor
        self.predictor = VLAPredictor(checkpoint, device)
        self.controller = ChunkController(self.predictor, num_envs, execute_steps)

    def act(self, raw, observations):
        images = torch.stack([raw.scene[name].data.output["rgb"].torch[..., :3]
                              for name in ("camera", "camera_secondary")], dim=1).to(torch.uint8).cpu().numpy()
        qpos = raw.robot.data.joint_pos.torch[:, raw.joint_ids].cpu().numpy()
        qvel = raw.robot.data.joint_vel.torch[:, raw.joint_ids].cpu().numpy()
        targets = self.controller.act(images, [raw.cfg.instruction] * raw.num_envs, qpos, qvel)
        return torch.as_tensor(targets, device=raw.device, dtype=torch.float32)

    def reset(self, dones):
        self.controller.reset(np.asarray(dones, dtype=bool))
