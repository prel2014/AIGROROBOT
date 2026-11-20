"""Reproducible lighting and obstacle layouts keyed by replica and episode."""
from dataclasses import dataclass, replace
import math

import numpy as np


TABLE_Z = .90
GROUND_Z = -.005
OBSTACLE_LAYOUT_VERSION = 2
FRUIT_PLACEMENTS = {
    "citrus": ((.30, -.12, 12.), (.48, .10, 44.), (.59, -.07, 105.)),
    "banana": ((.30, -.115, 15.), (.56, .13, 145.)),
    "vineyard": ((.34, .08, 0.), (.65, -.16, 48.)),
    "berries": ((.32, -.08, 0.), (.35, -.04, 28.), (.38, -.09, 65.), (.35, -.125, 105.)),
}
FRUIT_RADIUS = {"citrus": .044, "banana": .126, "vineyard": .055, "berries": .016}
# Conservative XY radii in metres. Rotating a prop keeps its footprint in this circle.
OBSTACLE_RADIUS = {
    "branch": .114, "nursery_pot": .062, "irrigation_hose": .104,
    "field_rock": .099, "flat_rock": .115, "pruning_shears": .123,
    "brick": .102, "hand_trowel": .142, "hand_cultivator": .137,
    "harvest_bucket": .132, "field_hoe": .345,
}
GROUND_OBSTACLES = frozenset(("harvest_bucket", "field_hoe"))
INACTIVE_Z = -2.
BENCH_COLORS = ((.45, .29, .16), (.27, .22, .18), (.46, .40, .31), (.41, .34, .30))


@dataclass(frozen=True)
class ObstaclePlacement:
    kind: str
    xy: tuple[float, float]
    yaw_deg: float
    surface: str = "bench"


@dataclass(frozen=True)
class EpisodeVariation:
    light_intensity: float
    color_temperature_k: float
    light_position_m: tuple[float, float, float]
    bench_color: tuple[float, float, float]
    obstacles: tuple[ObstaclePlacement, ...]


def _clear(scenario, xy, radius, obstacles, surface="bench"):
    x, y = xy
    occupied = [(prop.xy, OBSTACLE_RADIUS[prop.kind]) for prop in obstacles if prop.surface == surface]
    if any(math.dist(xy, other) < radius + size + .025 for other, size in occupied):
        return False
    if surface == "ground":
        # Both aisles are beyond the legs and pedestal and end before the crop bed.
        return -.46+radius <= x <= .59-radius and .72+radius <= abs(y) <= 1.91-radius
    # Leave a central strip and keep each prop inside the accessible tabletop.
    if abs(y) < .07 + radius or not .02 + radius <= x <= .82 - radius or abs(y) + radius > .61:
        return False
    for side in (-1, 1):
        dx = max(abs(x - .24) - .19, 0.)
        dy = max(abs(y - side*.46) - .16, 0.)
        if math.hypot(dx, dy) < radius + .018:
            return False
    produce = [((fx, fy), FRUIT_RADIUS[scenario]) for fx, fy, _ in FRUIT_PLACEMENTS[scenario]]
    if scenario == "berries":
        produce.append(((.59, .38), .078))  # bowl of berries
    return all(math.dist(xy, other) >= radius + size + .025 for other, size in produce)


def _sample_prop(scenario, rng, kinds, obstacles, available):
    for kind in kinds:
        surface = "ground" if kind in GROUND_OBSTACLES else "bench"
        for _ in range(64):
            if surface == "ground":
                xy = (float(rng.uniform(-.34, .49)), float(rng.choice((-1, 1))*rng.uniform(.94, 1.65)))
            else:
                xy = tuple(float(v) for v in rng.uniform((.08, -.54), (.78, .54)))
            if _clear(scenario, xy, OBSTACLE_RADIUS[kind], obstacles, surface) and available(kind, xy, surface):
                return ObstaclePlacement(str(kind), xy, float(rng.uniform(-180., 180.)), surface)
    return None


def sample_episode(scenario, seed, replica_id, episode_id):
    """Sample in scene coordinates, independently of the order replicas reset."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, replica_id, episode_id]))
    intensity = float(rng.uniform(3000., 7400.))
    temperature = float(rng.uniform(3900., 7800.))
    position = tuple(float(v) for v in rng.uniform((-1., -1.55, 2.45), (1.1, 1.55, 3.15)))
    color = BENCH_COLORS[int(rng.integers(len(BENCH_COLORS)))]
    count = int(rng.choice((0, 1, 2, 3), p=(.12, .22, .32, .34)))
    obstacles = []
    for kind in rng.permutation(tuple(OBSTACLE_RADIUS)):
        if len(obstacles) == count:
            break
        prop = _sample_prop(scenario, rng, (kind,), obstacles, lambda *_: True)
        if prop is not None:
            obstacles.append(prop)
    return EpisodeVariation(intensity, temperature, position, color, tuple(obstacles))


class EpisodeDynamics:
    """Reproducible lighting and independent obstacle events in simulation time."""

    def __init__(self, seed, replica_id, episode_id, layout, start_tick):
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, replica_id, episode_id, 1]))
        self.initial = self.layout = layout
        self.start_tick = start_tick
        self.light_periods = self.rng.uniform((7., 9.), (16., 21.))
        self.next_event_s = self.rng.uniform(1.5, 4.5)
        self.added = self.removed = 0

    def update_light(self, seconds):
        light_wave, color_wave = np.sin(2. * np.pi * seconds / self.light_periods)
        self.layout = replace(
            self.layout, light_intensity=self.initial.light_intensity * (1. + .24 * light_wave),
            color_temperature_k=float(np.clip(self.initial.color_temperature_k + 520. * color_wave, 3900., 7800.)))

    def update_obstacles(self, scenario, seconds, arm_points, fruit_points):
        """Return the changed obstacle kind, or defer an unsafe event."""
        self.next_event_s = seconds + self.rng.uniform(1.5, 4.5)
        if not np.isfinite(arm_points).all() or not np.isfinite(fruit_points).all():
            return None
        obstacles = self.layout.obstacles

        def available(kind, xy, surface):
            radius = OBSTACLE_RADIUS[kind]
            z = TABLE_Z if surface == "bench" else GROUND_Z
            near_arm_height = (arm_points[:, 2] >= z - .32) & (arm_points[:, 2] <= z + .67)
            arm_distance = np.linalg.norm(arm_points[near_arm_height, :2] - xy, axis=-1)
            fruit_distance = np.linalg.norm(fruit_points[:, :2] - xy, axis=-1)
            return (arm_distance > radius + .38).all() and (fruit_distance > radius + FRUIT_RADIUS[scenario] + .07).all()

        if obstacles and (len(obstacles) == 3 or self.rng.random() < .5):
            for index in self.rng.permutation(len(obstacles)):
                prop = obstacles[index]
                if available(prop.kind, prop.xy, prop.surface):
                    self.layout = replace(self.layout, obstacles=obstacles[:index] + obstacles[index + 1:])
                    self.removed += 1
                    return prop.kind
            return None
        active = {prop.kind for prop in obstacles}
        candidates = self.rng.permutation([kind for kind in OBSTACLE_RADIUS if kind not in active])
        prop = _sample_prop(scenario, self.rng, candidates, obstacles, available)
        if prop is not None:
            self.layout = replace(self.layout, obstacles=(*obstacles, prop))
            self.added += 1
            return prop.kind
        return None


def write_obstacles(scene, kind, states, world_origins):
    """Set one obstacle's pose per selected replica, or park it beneath the ground."""
    import torch

    env_ids = torch.as_tensor(list(states), device=scene.device, dtype=torch.long)
    values = []
    for prop in states.values():
        if prop is None:
            values.append((0., 0., INACTIVE_Z, 0., 0., 0., 1.))
        else:
            half = math.radians(prop.yaw_deg) / 2
            height = TABLE_Z if prop.surface == "bench" else GROUND_Z
            values.append((*prop.xy, height+.001, 0., 0., math.sin(half), math.cos(half)))
    obj = scene[f"obstacle_{kind}"]
    poses = obj.data.default_root_pose.torch.new_tensor(values)
    poses[:, :3] += world_origins[env_ids]
    obj.write_root_pose_to_sim_index(root_pose=poses, env_ids=env_ids)


def write_lighting(scene, layouts):
    from pxr import UsdLux

    for index, layout in layouts.items():
        light = scene.stage.GetPrimAtPath(scene.env_prim_paths[index] + "/Randomization/Light")
        api = UsdLux.LightAPI(light)
        api.GetIntensityAttr().Set(layout.light_intensity)
        api.GetColorTemperatureAttr().Set(layout.color_temperature_k)


def apply_variations(scene, layouts, *, world_origins=None):
    """Apply indexed layouts after reset, before taking the first observation."""
    from pxr import Gf, UsdShade

    if not layouts:
        return
    origins = scene.env_origins if world_origins is None else world_origins
    placed = {index: {prop.kind: prop for prop in layout.obstacles} for index, layout in layouts.items()}
    for kind in OBSTACLE_RADIUS:
        write_obstacles(scene, kind, {index: props.get(kind) for index, props in placed.items()}, origins)
    write_lighting(scene, layouts)
    for index, layout in layouts.items():
        env_path = scene.env_prim_paths[index]
        light = scene.stage.GetPrimAtPath(env_path + "/Randomization/Light")
        light.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*layout.light_position_m))
        if "orchard" in scene.keys():
            continue
        shader = UsdShade.Shader.Get(scene.stage, env_path + "/Bench/Top/geometry/material/Shader")
        if not shader:
            raise RuntimeError(f"Missing local bench shader in replica {index}")
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(*layout.bench_color))
