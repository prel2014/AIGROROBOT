"""Recurrent behavior cloning from synchronized agricultural RGB-D demonstrations."""
import torch

from .model import masked_loss


def epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            state, target, mask = (batch[key].to(device) for key in ("state", "joint_angles", "mask"))
            real = mask[..., 0]
            pixels = batch["pixels"]
            vision = torch.cat([model.encode_vision(chunk.to(device)) for chunk in pixels.split(8)])
            text = model.encode_text({key: value.to(device) for key, value in batch["text"].items()})
            text = text[batch["text_index"].to(device)]
            depth = batch["depth"].to(device)
            # One flat row per real observation; no image/text encoding of padding.
            rows = real.reshape(-1).long().cumsum(0).sub(1).clamp_min(0).reshape_as(real)
            memory = state.new_zeros((len(state), model.memory_dim))
            predictions = []
            for step in range(state.size(1)):
                index = rows[:, step]
                angles, updated = model(vision[index], text[index], depth[index], state[:, step], memory)
                memory = torch.where(real[:, step, None], updated, memory)
                predictions.append(angles)
            loss = masked_loss(torch.stack(predictions, dim=1), target, mask, scale=model.head.joint_scale)
            if not torch.isfinite(loss):
                raise ValueError("La pérdida VLA agrícola es no finita. Revisa datos y pesos.")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.)
                optimizer.step()
            weight = int(mask.sum().item())
            total += loss.item() * weight
            count += weight
    if not count:
        raise ValueError("La partición agrícola no contiene observaciones válidas")
    return total / count
