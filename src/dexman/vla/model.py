"""Frozen spatial CLIP and a masked recurrent Transformer for joint-angle chunks."""
import torch
from torch import nn
from torch.nn import functional as F

from .contract import default_architecture_config, validate_architecture_config


class TransformerBlock(nn.Module):
    def __init__(self, width, heads, feedforward_dim, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.residual_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, feedforward_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward_dim, width), nn.Dropout(dropout))

    def forward(self, tokens, attention_mask):
        batch, count, width = tokens.shape
        qkv = self.qkv(self.attention_norm(tokens))
        qkv = qkv.reshape(batch, count, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).reshape(batch, count, width)
        tokens = tokens + self.residual_dropout(self.attention_output(attended))
        return tokens + self.feed_forward(self.feed_forward_norm(tokens))


class ActionTransformer(nn.Module):
    memory_dim = 256

    def __init__(self, architecture_config):
        super().__init__()
        config = validate_architecture_config(architecture_config)
        width = config["width"]
        self.horizon, self.vision_tokens = config["horizon"], config["vision_tokens"]
        self.embedding_dim = config["embedding_dim"]
        self.context_tokens = self.vision_tokens + 3
        self.image_projection = nn.Linear(self.embedding_dim, width)
        self.text_projection = nn.Linear(self.embedding_dim, width)
        self.state_projection = nn.Linear(config["state_dim"], width)
        self.memory_projection = nn.Linear(self.memory_dim, width)
        self.visual_positions = nn.Parameter(torch.empty(self.vision_tokens, width))
        # Vision, language, proprioception, memory input, action readout, memory write.
        self.token_types = nn.Parameter(torch.empty(6, width))
        self.action_queries = nn.Parameter(torch.empty(self.horizon, width))
        self.memory_query = nn.Parameter(torch.empty(1, width))
        for parameter in (self.visual_positions, self.token_types, self.action_queries, self.memory_query):
            nn.init.normal_(parameter, std=0.02)
        self.blocks = nn.ModuleList(TransformerBlock(
            width, config["heads"], config["feedforward_dim"], config["dropout"])
            for _ in range(config["depth"]))
        self.output_norm = nn.LayerNorm(width)
        self.joint_output = nn.Linear(width, config["action_dim"])
        self.memory_candidate = nn.Linear(width, self.memory_dim)
        self.memory_gate = nn.Linear(width, self.memory_dim)
        nn.init.constant_(self.memory_gate.bias, -1.0)
        # SDPA boolean masks use True for permitted edges. Context never reads
        # output queries; action queries read context and their action-query set;
        # memory writes read context and themselves, with no route from actions.
        count = self.context_tokens + self.horizon + 1
        allowed = torch.zeros(count, count, dtype=torch.bool)
        allowed[:, :self.context_tokens] = True
        allowed[self.context_tokens:-1, self.context_tokens:-1] = True
        allowed[-1, -1] = True
        self.register_buffer("attention_mask", allowed, persistent=False)

    def forward(self, vision_tokens, text_embedding, state, memory):
        batch = state.shape[0]
        if (vision_tokens.shape != (batch, self.vision_tokens, self.embedding_dim)
                or text_embedding.shape != (batch, self.embedding_dim)
                or state.shape != (batch, 60) or memory.shape != (batch, self.memory_dim)):
            raise ValueError("Expected vision [B,100,512], text [B,512], state [B,60], memory [B,256]")
        vision = self.image_projection(vision_tokens) + self.visual_positions + self.token_types[0]
        text = self.text_projection(text_embedding) + self.token_types[1]
        proprioception = self.state_projection(state) + self.token_types[2]
        remembered = self.memory_projection(memory) + self.token_types[3]
        context = torch.cat((vision, torch.stack((text, proprioception, remembered), dim=1)), dim=1)
        actions = (self.action_queries + self.token_types[4]).expand(batch, -1, -1)
        write = (self.memory_query + self.token_types[5]).expand(batch, -1, -1)
        tokens = torch.cat((context, actions, write), dim=1)
        for block in self.blocks:
            tokens = block(tokens, self.attention_mask)
        outputs = self.output_norm(tokens)
        normalized_angles = self.joint_output(outputs[:, self.context_tokens:-1]).tanh()
        write_output = outputs[:, -1]
        candidate = self.memory_candidate(write_output).tanh()
        gate = self.memory_gate(write_output).sigmoid()
        # Bounded state and a learned retention path keep repeated updates stable.
        next_memory = (1.0 - gate) * memory.tanh() + gate * candidate
        return normalized_angles, next_memory


class ActionChunkPolicy(nn.Module):
    memory_dim = ActionTransformer.memory_dim

    def __init__(self, model_name="openai/clip-vit-base-patch32", horizon=8, *, joint_limits,
                 local_only=False, architecture_config=None):
        from dexman.perception.clip import load_frozen_clip

        super().__init__()
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self._architecture_config = validate_architecture_config(
            default_architecture_config(horizon) if architecture_config is None else architecture_config,
            horizon=horizon)
        limits = torch.as_tensor(joint_limits, dtype=torch.float32)
        if limits.shape != (30, 2) or not torch.isfinite(limits).all() or not (limits[:, 1] > limits[:, 0]).all():
            raise ValueError("Expected finite joint_limits [30,2], lower < upper")
        self.horizon = horizon
        self.register_buffer("joint_limits", limits.clone())
        self.register_buffer("joint_offset", limits.mean(-1))
        self.register_buffer("joint_scale", (limits[:, 1] - limits[:, 0]) * 0.5)
        self.register_buffer("state_mean", torch.zeros(60))
        self.register_buffer("state_std", torch.ones(60))
        self.backbone = load_frozen_clip(model_name, local_only=local_only)
        self._validate_backbone()
        self.head = ActionTransformer(self._architecture_config)

    @property
    def architecture_config(self):
        return dict(self._architecture_config)

    def _validate_backbone(self):
        vision = self.backbone.config.vision_config
        text = self.backbone.config.text_config
        if (self.backbone.config.projection_dim != 512 or vision.image_size != 224 or vision.patch_size != 32
                or vision.hidden_size != 768 or vision.num_hidden_layers != 12 or vision.num_attention_heads != 12
                or text.hidden_size != 512 or text.num_hidden_layers != 12 or text.num_attention_heads != 8):
            raise ValueError("VLA v5 requires frozen CLIP ViT-B/32 with 50 projected 512-dimensional vision tokens")

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, **inputs):
        from dexman.perception.clip import clip_spatial_features

        features = clip_spatial_features(self.backbone, inputs)
        count = features["vision_tokens"].shape[0]
        if count % 2:
            raise ValueError("CLIP input must contain synchronized camera pairs")
        return {"vision_tokens": features["vision_tokens"].reshape(count//2, 100, 512),
                "text_embedding": features["text_embedding"].reshape(count//2, 2, 512)[:, 0]}

    def forward(self, vision_tokens, text_embedding, state, memory):
        """Return [B,horizon,30] absolute radians and bounded [B,256] next memory."""
        normalized = (state - self.state_mean) / self.state_std.clamp_min(1e-3)
        normalized_angles, next_memory = self.head(vision_tokens, text_embedding, normalized, memory)
        angles = self.joint_offset + normalized_angles * self.joint_scale
        angles = torch.maximum(torch.minimum(angles, self.joint_limits[:, 1]), self.joint_limits[:, 0])
        return angles, next_memory


def masked_loss(prediction, target, mask, scale=None):
    """Range-normalized joint-angle MSE; padded future steps never contribute."""
    delta = prediction - target
    if scale is not None:
        delta = delta / scale.clamp_min(1e-6)
    error = delta.square().mean(-1)
    return (error * mask).sum() / mask.sum().clamp_min(1)
