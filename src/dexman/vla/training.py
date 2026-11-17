"""Batch preparation and train/evaluation loops independent of simulation."""
import numpy as np
import torch


def collate(samples, processor):
    images = [image for sample in samples for image in sample["image"]]
    text = [sample["text"] for sample in samples for _ in sample["image"]]
    result = dict(processor(images=images, text=text, padding=True, truncation=True, return_tensors="pt"))
    for key in ("state", "joint_angles", "mask"):
        result[key] = torch.from_numpy(np.stack([sample[key] for sample in samples]))
    return result


def state_statistics(files):
    import h5py
    total = np.zeros(60, dtype=np.float64)
    squares = total.copy()
    count = 0
    for path in files:
        with h5py.File(path) as episode:
            states = np.concatenate((episode["qpos"][:], episode["qvel"][:]), axis=-1).astype(np.float64)
        if not np.isfinite(states).all():
            raise ValueError(f"Non-finite joint state in {path}")
        total += states.sum(0)
        squares += (states ** 2).sum(0)
        count += len(states)
    if count == 0:
        raise ValueError("Training episodes contain no states")
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 1e-6))
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def epoch(model, loader, device, optimizer=None):
    from .model import masked_loss
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            state = batch.pop("state").to(device)
            target = batch.pop("joint_angles").to(device)
            mask = batch.pop("mask").to(device)
            robots, steps = state.shape[:2]
            # CLIP is frozen. Encode real frames in bounded batches before unrolling the trainable memory.
            encoded_batches = [
                model.encode(**{key: value[start:start + robots].to(device) for key, value in batch.items()})
                for start in range(0, len(batch["pixel_values"]), robots)
            ]
            features = {}
            for key in ("vision_tokens", "text_embedding"):
                encoded = torch.cat([item[key] for item in encoded_batches], dim=0)
                padded = encoded.new_zeros((robots, steps, *encoded.shape[1:]))
                padded[mask[..., 0]] = encoded
                features[key] = padded
            memory = state.new_zeros((robots, model.memory_dim))
            predictions = []
            for step in range(steps):
                angles, updated = model(features["vision_tokens"][:, step], features["text_embedding"][:, step],
                                        state[:, step], memory)
                # Padded frames do not advance a replica's recurrent state.
                memory = torch.where(mask[:, step, :1], updated, memory)
                predictions.append(angles)
            loss = masked_loss(torch.stack(predictions, dim=1), target, mask, scale=model.joint_scale)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss: inspect dataset and robot state")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
                optimizer.step()
            weight = int(mask.sum().item())
            total += loss.item() * weight
            count += weight
    if not count:
        raise ValueError("Empty data loader")
    return total / count
