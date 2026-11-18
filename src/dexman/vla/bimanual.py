"""RGB-D and language policy for the fixed torso with two articulated arms."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dexman.tasks.agriculture.interface import ACTION_DIM, ARM_KEYS, EYE_KEYS, JOINTS_PER_ARM, STATE_DIM
from .bimanual_contract import POLICY_VERSION, architecture_config
from .bimanual_language import instruction_prompt, tokenize_instructions
from .model import TransformerBlock


class BimanualTransformer(nn.Module):
    """Fuse two eyes, separate arm states, an instruction and recurrent memory."""

    architecture = POLICY_VERSION
    embedding_dim = 512
    text_dim = 384
    patches_per_eye = 50
    depth_grid = 7
    memory_dim = 256
    horizon = 8
    state_dim = STATE_DIM
    action_dim = ACTION_DIM

    def __init__(self, joint_limits):
        super().__init__()
        limits = torch.as_tensor(joint_limits, dtype=torch.float32)
        if (limits.shape != (self.action_dim, 2) or not torch.isfinite(limits).all()
                or not (limits[:, 1] > limits[:, 0]).all()):
            raise ValueError("Expected finite bimanual joint_limits [60,2], lower < upper")
        self.register_buffer("joint_limits", limits.detach().clone())
        self.register_buffer("joint_offset", self.joint_limits.mean(-1))
        self.register_buffer("joint_scale", self.joint_limits.diff(dim=-1).squeeze(-1) * 0.5)

        width = 256
        self.vision_tokens = len(EYE_KEYS) * self.patches_per_eye
        self.depth_tokens = len(EYE_KEYS) * self.depth_grid**2
        self.context_tokens = self.vision_tokens + self.depth_tokens + len(ARM_KEYS) + 2
        self.rgb_projection = nn.Linear(self.embedding_dim, width)
        self.depth_projection = nn.Linear(2, width)
        self.instruction_projection = nn.Linear(self.text_dim, width)
        self.arm_projection = nn.Linear(2 * JOINTS_PER_ARM, width)
        self.memory_projection = nn.Linear(self.memory_dim, width)
        self.rgb_positions = nn.Parameter(torch.empty(self.vision_tokens, width))
        self.depth_positions = nn.Parameter(torch.empty(self.depth_tokens, width))
        self.arm_identity = nn.Parameter(torch.empty(len(ARM_KEYS), width))
        self.action_queries = nn.Parameter(torch.empty(self.horizon, len(ARM_KEYS), width))
        self.memory_query = nn.Parameter(torch.empty(1, width))
        for parameter in (self.rgb_positions, self.depth_positions, self.arm_identity,
                          self.action_queries, self.memory_query):
            nn.init.normal_(parameter, std=.02)
        self.blocks = nn.ModuleList(TransformerBlock(width, 8, 1024, .1) for _ in range(6))
        self.output_norm = nn.LayerNorm(width)
        self.arm_output = nn.Linear(width, JOINTS_PER_ARM)
        self.memory_candidate = nn.Linear(width, self.memory_dim)
        self.memory_gate = nn.Linear(width, self.memory_dim)
        nn.init.constant_(self.memory_gate.bias, -1.)

        count = self.context_tokens + self.horizon * len(ARM_KEYS) + 1
        allowed = torch.zeros(count, count, dtype=torch.bool)
        allowed[:, :self.context_tokens] = True
        allowed[self.context_tokens:-1, self.context_tokens:-1] = True
        allowed[-1, -1] = True
        self.register_buffer("attention_mask", allowed, persistent=False)

    def _project_depth(self, depth):
        # Pool valid metric depths separately from the fraction of valid returns.
        # Empty cells remain observable through their validity channel.
        grid = (self.depth_grid, self.depth_grid)
        valid = torch.isfinite(depth) & (depth > 0.)
        pixels = torch.where(valid, depth.float().clamp_max(12.), 0.)
        coverage = F.adaptive_avg_pool2d(valid.float(), grid)
        distance = F.adaptive_avg_pool2d(pixels, grid) / coverage.clamp_min(1e-6)
        features = torch.stack((torch.log1p(distance), coverage), dim=-1).flatten(1, 3)
        return self.depth_projection(features.to(self.depth_projection.weight)) + self.depth_positions

    def forward(self, vision_tokens, text_embedding, depth, state, memory):
        """Return synchronized angles [B,8,60] in radians and next memory [B,256]."""
        batch = state.shape[0]
        if (vision_tokens.shape != (batch, self.vision_tokens, self.embedding_dim)
                or text_embedding.shape != (batch, self.text_dim)
                or depth.ndim != 4 or depth.shape[:2] != (batch, len(EYE_KEYS))
                or min(depth.shape[2:]) < self.depth_grid
                or state.shape != (batch, self.state_dim) or memory.shape != (batch, self.memory_dim)):
            raise ValueError("Expected RGB tokens [B,100,512], text [B,384], two depth maps, "
                             "angular state [B,120] and memory [B,256]")
        q, dq = state.split(self.action_dim, dim=-1)
        q = (q - self.joint_offset) / self.joint_scale
        arms = torch.cat((q.reshape(batch, len(ARM_KEYS), JOINTS_PER_ARM),
                          torch.asinh(dq).reshape(batch, len(ARM_KEYS), JOINTS_PER_ARM)), dim=-1)
        context = (
            self.rgb_projection(vision_tokens) + self.rgb_positions,
            self._project_depth(depth),
            self.instruction_projection(text_embedding).unsqueeze(1),
            self.arm_projection(arms) + self.arm_identity,
            self.memory_projection(memory).unsqueeze(1),
        )
        queries = (self.action_queries + self.arm_identity).flatten(0, 1)
        tokens = torch.cat((*context, queries.expand(batch, -1, -1),
                            self.memory_query.expand(batch, -1, -1)), dim=1)
        for block in self.blocks:
            tokens = block(tokens, self.attention_mask)
        output = self.output_norm(tokens[:, self.context_tokens:])
        normalized = self.arm_output(output[:, :-1]).reshape(batch, self.horizon, self.action_dim).tanh()
        angles = self.joint_offset + normalized * self.joint_scale
        angles = angles.clamp(self.joint_limits[:, 0], self.joint_limits[:, 1])
        write = output[:, -1]
        gate = self.memory_gate(write).sigmoid()
        next_memory = (1. - gate) * memory.clamp(-1., 1.) + gate * self.memory_candidate(write).tanh()
        return angles, next_memory


class BimanualActionPolicy(nn.Module):
    """Frozen CLIP vision and multilingual E5 encoders with a trainable action head."""

    memory_dim = BimanualTransformer.memory_dim
    horizon = BimanualTransformer.horizon

    def __init__(self, joint_limits, vision_encoder, text_encoder):
        super().__init__()
        self.head = BimanualTransformer(joint_limits)
        self.backbone = vision_encoder.requires_grad_(False).eval()
        self.text_encoder = text_encoder.requires_grad_(False).eval()

    @property
    def architecture_config(self):
        return architecture_config()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.text_encoder.eval()
        return self

    @torch.no_grad()
    def encode_vision(self, pixel_values):
        if len(pixel_values) % len(EYE_KEYS):
            raise ValueError("Expected two synchronized RGB views per replica")
        return self.backbone(pixel_values).reshape(-1, self.head.vision_tokens, self.head.embedding_dim)

    @torch.no_grad()
    def encode_text(self, inputs):
        return self.text_encoder(inputs)

    def forward(self, vision_tokens, text_embedding, depth, state, memory):
        return self.head(vision_tokens, text_embedding, depth, state, memory)

    @torch.inference_mode()
    def predict(self, processor, tokenizer, rgb, depth, instructions, state, memory, *, subtasks=None):
        """Apply the policy to both eyes with an optional active subtask per replica."""
        if isinstance(instructions, str) or not len(instructions):
            raise ValueError("Expected one instruction per replica")
        if subtasks is None:
            subtasks = [None] * len(instructions)
        if isinstance(subtasks, str) or len(subtasks) != len(instructions):
            raise ValueError("Expected one optional subtask per instruction")
        prompts = [instruction_prompt(text, subtask) for text, subtask in zip(instructions, subtasks, strict=True)]
        parameter = self.head.rgb_projection.weight
        rgb = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else rgb
        rgb = np.asarray(rgb)
        if (rgb.ndim != 5 or rgb.shape[:2] != (len(prompts), len(EYE_KEYS)) or rgb.shape[-1] != 3
                or rgb.dtype != np.uint8 or min(rgb.shape[2:4]) < 1):
            raise ValueError("Expected uint8 RGB [B,2,H,W,3]")
        if tuple(depth.shape) != rgb.shape[:-1]:
            raise ValueError("La profundidad [B,2,H,W] debe coincidir con el RGB de cada ojo")
        pixels = processor(images=rgb.reshape(-1, *rgb.shape[2:]), return_tensors="pt")["pixel_values"]
        text_inputs = tokenize_instructions(tokenizer, prompts)
        tensors = [torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype)
                   for value in (depth, state, memory)]
        if not torch.isfinite(tensors[1]).all() or not torch.isfinite(tensors[2]).all():
            raise ValueError("Joint state and incoming memory must be finite")
        vision = self.encode_vision(pixels.to(parameter))
        text = self.encode_text({key: value.to(parameter.device) for key, value in text_inputs.items()})
        return self(vision.to(parameter), text.to(parameter), *tensors)
