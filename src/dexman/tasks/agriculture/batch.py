"""Parallel crop groups with indexed episode resets and bimanual joint control."""
from dataclasses import dataclass, replace
from math import ceil, sqrt
from types import SimpleNamespace
from typing import TYPE_CHECKING

from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg

from dexman.robot.joints import JOINT_NAMES
from .interface import ACTION_DIM, ARM_KEYS, EYE_KEYS, JOINTS_PER_ARM
from .physical_variation import FruitPhysicsRandomizer, sample_fruit_physics
from .randomization import TABLE_Z, EpisodeDynamics, apply_variations, sample_episode, write_lighting, write_obstacles
from .scene import AgriculturalSceneCfg, CROPS, NS, PICKUP_SUPPORT_HEIGHT, eye, fixed_box, populate_crop

if TYPE_CHECKING:
    import torch
    from isaaclab.scene import InteractiveScene
    from .grasp_curriculum import GraspCalibration


DEFAULT_REPLICAS = 64
ENV_SPACING = 7.
CONTROL_DECIMATION = 4


@dataclass
class AgriculturalBatchCfg:
    sim: SimulationCfg
    scenes: SimpleNamespace
    monitor: SimpleNamespace | None = None
    nominal: bool = False
    physics_variation: float = 0.
    raised_support: bool = False
    grasp: "GraspCalibration | None" = None


def make_batch_config(num_envs=DEFAULT_REPLICAS, *, rgbd=False, rgbd_size=224, contacts=False, monitor=False,
                      nominal=False, physics_substeps=CONTROL_DECIMATION, physics_variation=0., raised_support=False,
                      grasp=None):
    crops = (grasp.crop,) if grasp is not None else tuple(CROPS)
    if type(num_envs) is not int or num_envs < len(crops):
        raise ValueError(f"Parallel agriculture requires at least {len(crops)} replicas")
    if not 0. <= physics_variation <= 1.:
        raise ValueError("Physical variation strength must be in [0, 1]")
    if type(physics_substeps) is not int or physics_substeps < 1:
        raise ValueError("Physics substeps must be a positive integer")
    raised_support = raised_support or grasp is not None
    base, remaining = divmod(num_envs, len(crops))
    scenes = {}
    for index, crop in enumerate(crops):
        scene = AgriculturalSceneCfg(
            num_envs=base + (index < remaining), env_spacing=ENV_SPACING,
            replicate_physics=False, filter_collisions=False)
        scene.clone_cfg.clone_template = f"/World/Agriculture/{crop}/env_{{}}"
        scene.ground = fixed_box(f"{NS}/Ground", (6.95, 6.95, .10),
                                 (0., 0., -.055), (.21, .28, .145))
        scene.overview = None
        if rgbd:
            for key in EYE_KEYS:
                sensor = getattr(scene, key)
                sensor.width = sensor.height = rgbd_size
        else:
            scene.left_eye = scene.right_eye = None
        populate_crop(scene, crop, randomize=True, raised_support=raised_support)
        if grasp is not None:
            active = getattr(scene, ARM_KEYS[grasp.side])
            active.init_state.joint_pos = dict(zip(JOINT_NAMES, grasp.initial_joints, strict=True))
            scene.fruit_0.init_state.pos = grasp.initial_fruit_pose[:3]
            scene.fruit_0.init_state.rot = grasp.initial_fruit_pose[3:]
        if contacts:
            from .scene import enable_task_contacts
            enable_task_contacts(scene)
        if index:
            scene.sunlight = scene.fill = None
        scenes[crop] = scene
    monitor_eyes = None
    if monitor:
        monitor_eyes = SimpleNamespace(
            left=eye(1).replace(prim_path="/World/AgricultureMonitor/LeftEye", width=224, height=224, update_period=0.),
            right=eye(-1).replace(prim_path="/World/AgricultureMonitor/RightEye", width=224, height=224, update_period=0.),
        )
    return AgriculturalBatchCfg(
        sim=SimulationCfg(dt=1 / (30 * physics_substeps), render_interval=physics_substeps, physics=PhysxCfg()),
        # The Isaac Lab launcher discovers camera configurations through object attributes.
        scenes=SimpleNamespace(**scenes), monitor=monitor_eyes, nominal=nominal,
        physics_variation=physics_variation, raised_support=raised_support, grasp=grasp)


@dataclass
class CropBatch:
    crop: str
    scene: "InteractiveScene"
    start: int
    origins: "torch.Tensor"
    joint_ids: tuple[list[int], ...] = ()

    @property
    def stop(self):
        return self.start + self.scene.num_envs


class AgriculturalBatch:
    """Global replica IDs are contiguous by crop, with left then right joint ordering."""

    def __init__(self, sim, cfg, *, seed=42):
        from isaaclab.cloner import filter_collisions
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim.utils import standardize_xform_ops
        from .impact_sensors import configure_impact_sensors

        self.sim, self.seed = sim, seed
        self.nominal = cfg.nominal
        self.pickup_surface_z = TABLE_Z + (PICKUP_SUPPORT_HEIGHT if cfg.raised_support else 0.)
        self.physics_variation = cfg.physics_variation
        self.control_decimation = cfg.sim.render_interval
        self.control_dt = self.control_decimation * sim.get_physics_dt()
        self.groups = []
        scenes = vars(cfg.scenes)
        group_spacing = ENV_SPACING * (ceil(sqrt(max(scene.num_envs for scene in scenes.values()))) + 1)
        start = 0
        for index, (crop, scene_cfg) in enumerate(scenes.items()):
            if hasattr(scene_cfg, "fruit_obstacle_contact"):
                configure_impact_sensors(scene_cfg, crop, history_length=self.control_decimation)
            scene = InteractiveScene(scene_cfg)
            offset = (group_spacing * (index % 2), group_spacing * (index // 2), 0.)
            namespace = scene.stage.DefinePrim(scene.env_ns, "Xform")
            standardize_xform_ops(namespace, translation=offset)
            origins = scene.env_origins + scene.env_origins.new_tensor(offset)
            # USD clones are already authored. Release this plan before
            # creating the next crop and let views discover prims by namespace.
            sim.set_clone_plan(None)
            group = CropBatch(crop, scene, start, origins)
            self.groups.append(group)
            start = group.stop
        first_scene = self.groups[0].scene
        filter_collisions(first_scene.stage, first_scene.physics_scene_path, "/World/AgricultureCollisions",
                          [path for group in self.groups for path in group.scene.env_prim_paths])
        self.num_envs = start
        self.monitor_eyes = ()
        if cfg.monitor is not None:
            from isaaclab.sensors import Camera

            self.monitor_eyes = (Camera(cfg.monitor.left), Camera(cfg.monitor.right))
        self.control_tick = 0
        self.episode_ids = [-1] * self.num_envs
        self.dynamics = [None] * self.num_envs
        self.fruit_randomizers = {}
        self.instructions = tuple(CROPS[group.crop].instruction + " Si ves obstáculos, esquívalos."
                                  for group in self.groups for _ in range(group.scene.num_envs))

    def initialize(self):
        """Initialize PhysX views once, then reset all replicas to episode zero."""
        import torch

        self.sim.reset()
        limits = []
        for group in self.groups:
            joint_ids, arm_limits = [], []
            for key in ARM_KEYS:
                arm = group.scene[key]
                indices, names = arm.find_joints(list(JOINT_NAMES), preserve_order=True)
                if tuple(names) != JOINT_NAMES or len(arm.joint_names) != JOINTS_PER_ARM:
                    raise RuntimeError(f"Unexpected joint order for {group.crop}/{key}: {names}")
                joint_ids.append(indices)
                arm_limits.append(arm.data.joint_pos_limits.torch[:, indices])
            group.joint_ids = tuple(joint_ids)
            limits.append(torch.cat(arm_limits, dim=1))
        self.joint_limits = torch.cat(limits)
        if self.physics_variation:
            self.fruit_randomizers = {
                group.crop: FruitPhysicsRandomizer(group.scene["fruit_0"]) for group in self.groups}
        self.reset()

    def reset(self, replica_ids=None):
        """Restore selected arms and fruit, then sample their next episode layouts."""
        import torch

        if isinstance(replica_ids, torch.Tensor):
            replica_ids = replica_ids.tolist()
        selected = set(range(self.num_envs) if replica_ids is None else replica_ids)
        if any(not 0 <= index < self.num_envs for index in selected):
            raise IndexError("Replica ID is outside the agricultural batch")
        for group in self.groups:
            local = [index - group.start for index in sorted(selected) if group.start <= index < group.stop]
            if not local:
                continue
            scene = group.scene
            ids = torch.as_tensor(local, dtype=torch.long, device=scene.device)
            scene.reset(ids)
            for key in ARM_KEYS:
                arm = scene[key]
                positions = arm.data.default_joint_pos.torch[ids]
                velocities = arm.data.default_joint_vel.torch[ids]
                arm.write_joint_position_to_sim_index(position=positions, env_ids=ids)
                arm.write_joint_velocity_to_sim_index(velocity=velocities, env_ids=ids)
                arm.set_joint_position_target_index(target=positions, env_ids=ids)
            for key, fruit in scene.rigid_objects.items():
                if key.startswith("fruit_"):
                    pose = fruit.data.default_root_pose.torch[ids]
                    pose[:, :3] += group.origins[ids]
                    fruit.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
                    fruit.write_root_velocity_to_sim_index(root_velocity=pose.new_zeros((len(ids), 6)), env_ids=ids)
            layouts = {}
            for index in local:
                replica_id = group.start + index
                self.episode_ids[replica_id] += 1
                episode = self.episode_ids[replica_id]
                layout = sample_episode(group.crop, self.seed, replica_id, episode)
                layouts[index] = replace(layout, obstacles=()) if self.nominal else layout
                self.dynamics[replica_id] = EpisodeDynamics(
                    self.seed, replica_id, episode, layouts[index], self.control_tick)
            if group.crop in self.fruit_randomizers:
                samples = [sample_fruit_physics(self.seed, group.start + index,
                                                self.episode_ids[group.start + index],
                                                strength=self.physics_variation) for index in local]
                self.fruit_randomizers[group.crop].apply(local, samples)
            apply_variations(scene, layouts, world_origins=group.origins)

    def _advance_variations(self):
        import torch

        tick_seconds = self.control_dt
        for group in self.groups:
            dynamics = self.dynamics[group.start:group.stop]
            seconds = [(self.control_tick - episode.start_tick) * tick_seconds for episode in dynamics]
            if self.control_tick % 3 == 0:
                for episode, elapsed in zip(dynamics, seconds, strict=True):
                    episode.update_light(elapsed)
                write_lighting(group.scene, {i: episode.layout for i, episode in enumerate(dynamics)})
            due = [i for i, (episode, elapsed) in enumerate(zip(dynamics, seconds, strict=True))
                   if elapsed >= episode.next_event_s]
            if not due:
                continue
            origins = group.origins[due, None, :]
            arms = torch.cat([group.scene[key].data.body_pos_w.torch[due] for key in ARM_KEYS], dim=1)
            fruits = torch.stack([obj.data.root_pos_w.torch[due]
                                  for name, obj in group.scene.rigid_objects.items() if name.startswith("fruit_")], dim=1)
            arm_points = (arms - origins).cpu().numpy()
            fruit_points = (fruits - origins).cpu().numpy()
            changes = {}
            for index, links, produce in zip(due, arm_points, fruit_points, strict=True):
                episode = dynamics[index]
                kind = episode.update_obstacles(group.crop, seconds[index], links, produce)
                if kind is not None:
                    prop = next((item for item in episode.layout.obstacles if item.kind == kind), None)
                    changes.setdefault(kind, {})[index] = prop
            for kind, poses in changes.items():
                write_obstacles(group.scene, kind, poses, group.origins)

    def angular_state(self):
        """Read q[60] then dq[60], each ordered left arm followed by right arm."""
        import torch

        states = []
        for group in self.groups:
            arms = [group.scene[key] for key in ARM_KEYS]
            positions = [arm.data.joint_pos.torch[:, ids] for arm, ids in zip(arms, group.joint_ids, strict=True)]
            velocities = [arm.data.joint_vel.torch[:, ids] for arm, ids in zip(arms, group.joint_ids, strict=True)]
            states.append(torch.cat((*positions, *velocities), dim=-1))
        return torch.cat(states)

    def rgbd(self):
        """Read synchronized eye pairs in RGB uint8 [B,2,H,W,3] and metric depth [B,2,H,W]."""
        import torch

        if any(key not in self.groups[0].scene.sensors for key in EYE_KEYS):
            raise RuntimeError("Batch was initialized without RGB-D sensors")
        rgbs, depths = [], []
        for group in self.groups:
            outputs = [group.scene[key].data.output for key in EYE_KEYS]
            rgbs.append(torch.stack([output["rgb"].torch[..., :3] for output in outputs], dim=1))
            depths.append(torch.stack([output["distance_to_image_plane"].torch[..., 0] for output in outputs], dim=1))
        return torch.cat(rgbs), torch.cat(depths)

    def step(self, angles, *, render=False):
        """Apply one [B,60] target and advance all crops by one control tick."""
        import torch

        if angles.shape != (self.num_envs, ACTION_DIM) or not torch.isfinite(angles).all():
            raise ValueError(f"Expected finite joint targets [{self.num_envs},{ACTION_DIM}]")
        self.control_tick += 1
        if not self.nominal:
            self._advance_variations()
        angles = angles.to(self.joint_limits).clamp(self.joint_limits[..., 0], self.joint_limits[..., 1])
        for group in self.groups:
            targets = angles[group.start:group.stop].split(JOINTS_PER_ARM, dim=-1)
            for key, ids, target in zip(ARM_KEYS, group.joint_ids, targets, strict=True):
                group.scene[key].set_joint_position_target_index(target=target, joint_ids=ids)
        dt = self.sim.get_physics_dt()
        for index in range(self.control_decimation):
            for group in self.groups:
                for key in ARM_KEYS:
                    arm = group.scene[key]
                    arm.actuators.target_command.set_effort_index(value=arm.data.gravity_compensation_forces.torch)
                group.scene.write_data_to_sim()
            self.sim.step(render=render and index == self.control_decimation - 1)
            for group in self.groups:
                group.scene.update(dt)
