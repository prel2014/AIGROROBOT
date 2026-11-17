"""Shared frozen CLIP loading, feature extraction and RGB/text validation."""
import numpy as np
import torch
from torch.nn import functional as F
from transformers import CLIPModel


def load_frozen_clip(source, *, local_only=False):
    """The caller owns device placement and persistence of the pretrained weights."""
    return CLIPModel.from_pretrained(source, local_files_only=local_only).requires_grad_(False).eval()


@torch.no_grad()
def clip_embeddings(backbone, inputs):
    if "pixel_values" not in inputs or "input_ids" not in inputs:
        raise ValueError("Both image pixels and tokenized instruction text are required")
    backbone.eval()
    features = backbone(**inputs, return_dict=True)
    return torch.cat((features.image_embeds, features.text_embeds), dim=-1)


@torch.no_grad()
def clip_spatial_features(backbone, inputs):
    """Return normalized projected CLS + patches and pooled instruction features.

    Keep the ordered 7x7 patch grid as well as CLS. Every token uses the final
    vision normalization and projection used by CLIP's pooled image path. Text
    retains CLIP's pooled EOS representation and learned projection.
    """
    if "pixel_values" not in inputs or "input_ids" not in inputs:
        raise ValueError("Both image pixels and tokenized instruction text are required")
    backbone.eval()
    features = backbone(**inputs, return_dict=True)
    vision_hidden = backbone.vision_model.post_layernorm(features.vision_model_output.last_hidden_state)
    vision_tokens = F.normalize(backbone.visual_projection(vision_hidden), dim=-1)
    return {"vision_tokens": vision_tokens, "text_embedding": features.text_embeds}


def prepare_inputs(processor, images, instructions, device):
    """Validate public RGB/text inputs and place processed tensors on the policy device."""
    if not len(images) or len(instructions) != len(images):
        raise ValueError("One image and text instruction per robot are required")
    if any(not isinstance(text, str) or not text.strip() for text in instructions):
        raise ValueError("A non-empty text instruction is required for each robot")
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    for image in images:
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 1:
            raise ValueError("Images must be uint8 RGB [height,width,3]")
    inputs = processor(images=list(images), text=list(instructions), padding=True,
                       truncation=True, return_tensors="pt")
    return {key: value.to(device) for key, value in inputs.items()}
