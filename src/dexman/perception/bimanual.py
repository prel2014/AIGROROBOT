"""Hugging Face adapters for frozen RGB and multilingual instruction encoders."""
import torch
from torch import nn
from torch.nn import functional as F


VISION_MODEL = "openai/clip-vit-base-patch32"
TEXT_MODEL = "intfloat/multilingual-e5-small"


class FrozenClipVision(nn.Module):
    def __init__(self, model):
        super().__init__()
        config = model.config
        if config.projection_dim != 512 or (config.image_size, config.patch_size) != (224, 32):
            raise ValueError("Expected CLIP ViT-B/32 with 50 vision tokens per eye and projection dimension 512")
        self.model = model.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, pixel_values):
        vision = self.model.vision_model
        hidden = vision(pixel_values=pixel_values, return_dict=True).last_hidden_state
        return F.normalize(self.model.visual_projection(vision.post_layernorm(hidden)), dim=-1)

    def save_pretrained(self, directory):
        self.model.save_pretrained(directory)


class FrozenMultilingualText(nn.Module):
    def __init__(self, model):
        super().__init__()
        if model.config.hidden_size != 384:
            raise ValueError("Expected a multilingual E5 text encoder with 384 features")
        self.model = model.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, inputs):
        hidden = self.model(**inputs, return_dict=True).last_hidden_state
        valid = inputs["attention_mask"].unsqueeze(-1).to(hidden)
        return F.normalize((hidden * valid).sum(1) / valid.sum(1).clamp_min(1), dim=-1)

    def save_pretrained(self, directory):
        self.model.save_pretrained(directory)


def load_frozen_bimanual(clip_source, text_source, *, local_only=False):
    from transformers import AutoModel, CLIPVisionModelWithProjection

    return (FrozenClipVision(CLIPVisionModelWithProjection.from_pretrained(clip_source, local_files_only=local_only)),
            FrozenMultilingualText(AutoModel.from_pretrained(text_source, local_files_only=local_only)))
