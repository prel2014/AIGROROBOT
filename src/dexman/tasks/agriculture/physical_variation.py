"""Deterministic per-episode variation of a target fruit's physical properties."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FruitPhysics:
    mass_scale: float = 1.0
    friction_scale: float = 1.0

    def __post_init__(self):
        if any(not math.isfinite(scale) or scale <= 0 for scale in (self.mass_scale, self.friction_scale)):
            raise ValueError("Fruit mass and friction scales must be positive and finite")


def _sample_offset(rng, std, bound):
    value = rng.normal(0.0, std)
    while abs(value) > bound:
        value = rng.normal(0.0, std)
    return float(value)


def sample_fruit_physics(seed, replica_id, episode_id, *, strength=1.0):
    """Sample truncated normals independently of reset order and lighting events.

    Full strength allows ±20% mass and ±15% friction around the nominal material.
    These are initial training ranges, scaled toward nominal as strength approaches zero.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Physical variation strength must be in [0, 1]")
    if strength == 0.0:
        return FruitPhysics()
    rng = np.random.default_rng(np.random.SeedSequence([seed, replica_id, episode_id, 2]))
    mass = 1.0 + strength * _sample_offset(rng, 0.075, 0.20)
    # Bounded material buckets avoid creating a new PhysX material for every episode.
    friction = round(1.0 + strength * _sample_offset(rng, 0.06, 0.15), 2)
    return FruitPhysics(mass, friction)


class FruitPhysicsRandomizer:
    """Apply scales relative to the same nominal fruit data on every reset."""

    def __init__(self, fruit):
        import warp as wp

        self._fruit = fruit
        self._nominal_mass = fruit.data.body_mass.torch.clone()
        self._nominal_inertia = fruit.data.body_inertia.torch.clone()
        self._nominal_materials = wp.to_torch(fruit.root_view.get_material_properties()).cpu().clone()
        self._materials = self._nominal_materials.clone()

    def apply(self, local_ids, samples):
        """Update only the selected replica rows, in the order of the samples."""
        import torch
        import warp as wp

        ids = torch.as_tensor(local_ids, device="cpu", dtype=torch.long).reshape(-1)
        if len(samples) != len(ids):
            raise ValueError("Expected one fruit sample per replica ID")
        if not ids.numel():
            return
        if (ids < 0).any() or (ids >= len(self._nominal_mass)).any() or ids.unique().numel() != ids.numel():
            raise ValueError("Fruit variation requires unique replica IDs in bounds")

        cpu_ids = ids.to(torch.int32)
        device_ids = ids.to(self._nominal_mass.device)
        mass_scale = self._nominal_mass.new_tensor([sample.mass_scale for sample in samples])[:, None]
        masses = self._nominal_mass[device_ids] * mass_scale
        # Density changes at a fixed shape and center of mass scale all inertia entries equally.
        inertias = self._nominal_inertia[device_ids] * mass_scale[..., None]
        materials = self._nominal_materials[ids]
        friction_scale = materials.new_tensor([sample.friction_scale for sample in samples])[:, None, None]
        materials[..., :2] *= friction_scale
        materials[..., 1] = torch.minimum(materials[..., 0], materials[..., 1])
        self._materials[ids] = materials

        self._fruit.set_masses_index(masses=masses, env_ids=device_ids)
        self._fruit.set_inertias_index(inertias=inertias, env_ids=device_ids)
        self._fruit.root_view.set_material_properties(
            wp.from_torch(self._materials, dtype=wp.float32), wp.from_torch(cpu_ids, dtype=wp.int32))
