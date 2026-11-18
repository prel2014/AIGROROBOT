"""Physical layout recipes shared by scene adapters and inspection tools."""
from dataclasses import dataclass, replace

from .randomization import BENCH_COLORS, EpisodeVariation, ObstaclePlacement, sample_episode


@dataclass(frozen=True)
class AgriculturalLayout:
    crop: str
    raised_support: bool
    objective: str
    inspection_position: tuple[float, float, float]
    inspection_target: tuple[float, float, float]
    obstacles: tuple[ObstaclePlacement, ...] = ()


LAYOUTS = {
    "berry-pinch": AgriculturalLayout(
        "berries", True,
        "Pinzar y levantar el arándano objetivo sin desplazar las bayas vecinas.",
        (.76, -.48, 1.24), (.34, -.078, .942),
    ),
    "orange-packing": AgriculturalLayout(
        "citrus", True,
        "Levantar la naranja de su cuna y depositarla estable en la bandeja izquierda.",
        (2.28, -2.96, 2.23), (.31, .16, .98),
    ),
    "banana-handover": AgriculturalLayout(
        "banana", True,
        "Tomar el plátano con la derecha, pasarlo a la izquierda y depositarlo en la bandeja.",
        (2.42, -2.84, 2.53), (.26, .00, 1.07),
    ),
    "grape-occlusion": AgriculturalLayout(
        "vineyard", False,
        "Identificar y levantar el racimo suelto entre obstáculos sin golpear la mesa ni los útiles.",
        (.74, -2.75, 2.40), (.34, .10, .99),
        (ObstaclePlacement("harvest_bucket", (.10, .185), 0.),
         ObstaclePlacement("pruning_shears", (.64, .32), -33.)),
    ),
}


def layout_variation(layout, seed, episode, *, randomize=False):
    """Keep a named layout's obstacle geometry when varying lights and table appearance."""
    if randomize:
        sampled = sample_episode(layout.crop, seed, 0, episode)
        return replace(sampled, obstacles=layout.obstacles)
    if layout.obstacles:
        return EpisodeVariation(4700., 5700., (0., 0., 2.8), BENCH_COLORS[0], layout.obstacles)
    return None
