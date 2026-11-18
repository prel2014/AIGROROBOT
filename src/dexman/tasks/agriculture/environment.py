"""RSL-RL VecEnv over the four agricultural crop groups in one PhysX context."""
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from torch.nn.functional import one_hot

from dexman.robot.joints import FINGERTIPS, PALM
from dexman.sensors.grasp import opposed_contact_grasp
from .batch import AgriculturalBatch, CONTROL_DECIMATION
from .interface import ACTION_DIM, ARM_KEYS, JOINTS_PER_ARM
from .objectives import (EPISODE_SECONDS, LIFT, MAX_DISTRACTORS, OBJECTIVES, OBS_DIM, PHASE_TEXT,
                         PLACE, REACH, SETTLE, TRANSFER, TRAY_CENTER)
from .randomization import GROUND_Z, OBSTACLE_RADIUS, TABLE_Z


class AgriculturalPPOEnv(VecEnv):
    def __init__(self, sim, cfg, *, seed=42, capture=False, render=False):
        self.cfg = cfg
        self.batch = AgriculturalBatch(sim, cfg, seed=seed)
        self.batch.initialize()
        self.device, self.num_envs, self.num_actions = sim.device, self.batch.num_envs, ACTION_DIM
        self.capture, self.render = capture, capture or render
        self.dt = sim.get_physics_dt() * CONTROL_DECIMATION
        self.max_episode_length = round(EPISODE_SECONDS / self.dt)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.phase = torch.zeros_like(self.episode_length_buf)
        self.hold = torch.zeros_like(self.phase)
        self.overlap_steps = torch.zeros_like(self.phase)
        self.handover_seen = torch.zeros_like(self.phase, dtype=torch.bool)

        self.fruits, self.body_ids, starting_poses, velocity_limits = [], [], [], []
        crops = []
        for group in self.batch.groups:
            scene = group.scene
            produce = [obj for name, obj in scene.rigid_objects.items() if name.startswith("fruit_")]
            self.fruits.append(produce)
            starting_poses.append(self._pad_distractors(
                [fruit.data.default_root_pose.torch[:, :3] for fruit in produce[1:]], group.scene.num_envs))
            crops.extend([list(OBJECTIVES).index(group.crop)] * group.scene.num_envs)
            keys = (PALM, *FINGERTIPS)
            self.body_ids.append([scene[key].find_bodies(list(keys), preserve_order=True)[0] for key in ARM_KEYS])
            velocity_limits.append(torch.cat([scene[key].data.joint_vel_limits.torch[:, ids]
                                              for key, ids in zip(ARM_KEYS, group.joint_ids, strict=True)], dim=-1))
        self.crop_ids = torch.tensor(crops, device=self.device)
        self.crop_encoding = one_hot(self.crop_ids, len(OBJECTIVES))
        specs = [list(OBJECTIVES.values())[i] for i in crops]
        self.lift_height = torch.tensor([spec.lift for spec in specs], device=self.device)
        self.deposit = torch.tensor([spec.deposit for spec in specs], device=self.device)
        self.handover = torch.tensor([spec.handover for spec in specs], device=self.device)
        self.precision = torch.tensor([spec.precision for spec in specs], device=self.device)
        self.instructions = tuple(spec.instruction + " Si ves obstáculos, esquívalos." for spec in specs)
        self.starting_distractors = torch.cat(starting_poses)
        self.tray = self.lift_height.new_tensor(TRAY_CENTER)
        self.home = self.batch.angular_state()[:, :ACTION_DIM].clone()
        physical_low, physical_high = self.batch.joint_limits.unbind(-1)
        # Explore around the calibrated starting posture. Fingers retain their physical range.
        span = self.home.new_tensor(([1.5] * 6 + [10.] * (JOINTS_PER_ARM - 6)) * len(ARM_KEYS))
        self.low = torch.maximum(physical_low, self.home - span)
        self.high = torch.minimum(physical_high, self.home + span)
        self.home = self.home.clamp(self.low, self.high)
        self.targets = self.home.clone()
        self.step_limit = torch.cat(velocity_limits).clamp_max(4.) * self.dt
        self.previous_potential = torch.zeros_like(self.lift_height)
        self.potential_valid = torch.zeros_like(self.deposit)
        self._update_observations()
        if capture:
            self.batch.sim.render()

    def _pad_distractors(self, positions, count):
        padding = [torch.zeros((count, 3), device=self.device)] * (MAX_DISTRACTORS - len(positions))
        return torch.stack((*positions, *padding), dim=1)

    def _measure(self):
        poses, velocities, others, palms, tips, active, grasp, collisions = ([] for _ in range(8))
        for group, produce, body_ids in zip(self.batch.groups, self.fruits, self.body_ids, strict=True):
            scene, origin = group.scene, group.origins
            target = produce[0]
            pos = target.data.root_pos_w.torch
            poses.append(torch.cat((pos - origin, target.data.root_quat_w.torch), dim=-1))
            velocities.append(target.data.root_vel_w.torch)
            others.append(self._pad_distractors(
                [fruit.data.root_pos_w.torch - origin for fruit in produce[1:]], scene.num_envs))
            links, contacts, gripped = [], [], []
            forbidden = scene["fruit_obstacle_contact"].data.normal_force_matrix_w.torch[:, 0].norm(dim=-1)
            max_collision = forbidden.amax(-1)
            for side, ids in zip(ARM_KEYS, body_ids, strict=True):
                links.append(scene[side].data.body_pos_w.torch[:, ids] - origin[:, None, :])
                sensors = [scene[f"{side}_{tip}"].data for tip in FINGERTIPS]
                forces = torch.stack([sensor.normal_force_matrix_w.torch[:, 0] for sensor in sensors], dim=1)
                points = torch.stack([sensor.contact_pos_w.torch[:, 0, 0] for sensor in sensors], dim=1)
                max_collision = torch.maximum(max_collision, forces[:, :, 1:].norm(dim=-1).amax((1, 2)))
                if OBJECTIVES[group.crop].precision:
                    held, _, _ = opposed_contact_grasp(forces[:, [0, 4], 0], points[:, [0, 4]], pos, .003,
                                                        minimum_separation=.003, required_fingers=1)
                    touching = forces[:, :, 0].norm(dim=-1) >= .003
                else:
                    held, touching, _ = opposed_contact_grasp(forces[:, :, 0], points, pos, .04)
                contacts.append(touching)
                gripped.append(held)
            links = torch.stack(links, dim=1)
            palms.append(links[:, :, 0])
            tips.append(links[:, :, 1:])
            active.append(torch.stack(contacts, dim=1))
            grasp.append(torch.stack(gripped, dim=1))
            collisions.append(max_collision)
        return tuple(torch.cat(field) for field in (poses, velocities, others, palms, tips, active, grasp, collisions))

    def _obstacles(self):
        rows = []
        for episode in self.batch.dynamics:
            active = {prop.kind: prop for prop in episode.layout.obstacles}
            row = []
            for kind, radius in OBSTACLE_RADIUS.items():
                prop = active.get(kind)
                if prop is None:
                    row.append((0., 0., 0., 0., 0.))
                else:
                    z = TABLE_Z if prop.surface == "bench" else GROUND_Z
                    row.append((*prop.xy, z, radius, 1.))
            rows.append(row)
        return self.tray.new_tensor(rows)

    def _update_observations(self, measurement=None, angular_state=None):
        if measurement is None:
            measurement = self._measure()
        pose, velocity, others, palm, tip, contacts, held, _ = measurement
        pos = pose[:, :3]
        self.angular_state = self.batch.angular_state() if angular_state is None else angular_state
        q, dq = self.angular_state.split(ACTION_DIM, dim=-1)
        half = (self.high - self.low).clamp_min(.01) * .5
        joint_features = torch.stack(((q - self.home) / half, torch.asinh(dq), (self.targets - self.home) / half),
                                     dim=-1).flatten(1)
        arm_features = torch.cat((palm - pos[:, None], (tip - pos[:, None, None]).flatten(2),
                                  contacts, held.unsqueeze(-1)), dim=-1)
        lift_goal = torch.stack((pos[:, 0], pos[:, 1], TABLE_Z + .006 + self.lift_height), dim=-1)
        goal = torch.where((self.phase >= PLACE)[:, None], self.tray, lift_goal)
        task_features = torch.cat((pose, velocity, goal - pos, others.flatten(1),
                                   self.crop_encoding, one_hot(self.phase, len(PHASE_TEXT)),
                                   (self.hold / 15.)[:, None], (self.episode_length_buf / self.max_episode_length)[:, None],
                                   self.handover_seen[:, None]), dim=-1)
        self.obstacle_features = self._obstacles()
        policy = torch.cat((joint_features, arm_features.flatten(1), task_features,
                            self.obstacle_features.flatten(1)), dim=-1)
        if policy.shape != (self.num_envs, OBS_DIM):
            raise RuntimeError(f"Agricultural observation contract changed: {policy.shape}")
        self.observation = TensorDict({"policy": torch.nan_to_num(policy, nan=0., posinf=0., neginf=0.)},
                                      batch_size=[self.num_envs])

    def get_observations(self):
        return self.observation

    def _reset(self, ids):
        self.batch.reset(ids)
        self.episode_length_buf[ids] = 0
        self.phase[ids] = REACH
        self.hold[ids] = 0
        self.overlap_steps[ids] = 0
        self.handover_seen[ids] = False
        self.potential_valid[ids] = False
        self.targets[ids] = self.home[ids]
        if self.capture:
            self.batch.sim.render()

    def _score(self, measurements, angular_state, actions, invalid, movement):
        pose, velocity, others, palm, tips, contacts, held, collision = measurements
        pos = pose[:, :3]
        left, right = held.unbind(-1)
        grasp = torch.where(self.handover, right, held.any(-1))
        lift = pos[:, 2] - (TABLE_Z + .006)
        displacement = (others - self.starting_distractors).norm(dim=-1).amax(-1)
        delta = (pos - self.tray).abs()
        in_tray = (delta[:, 0] < .095) & (delta[:, 1] < .076) & (delta[:, 2] < .018)
        quiet = velocity[:, :3].norm(dim=-1) < .065
        released = ~contacts.any((1, 2))
        impeded = collision > 1.0
        outside = (pos[:, 0] < -.26) | (pos[:, 0] > .94) | (pos[:, 1].abs() > .72)
        dropped = (pos[:, 2] < TABLE_Z - .065) | (outside & (pos[:, 2] < TABLE_Z + .012))
        displaced_off_table = ((others[..., 0] < -.26) | (others[..., 0] > .94)
                               | (others[..., 1].abs() > .72)) & (others[..., 2] < TABLE_Z + .012)
        distractor_dropped = (others[..., 2] < TABLE_Z - .065) | displaced_off_table
        dropped |= (distractor_dropped & (self.starting_distractors[..., 2] > 0.)).any(-1)
        q, dq = angular_state.split(ACTION_DIM, dim=-1)
        low, high = self.batch.joint_limits.unbind(-1)
        impossible = ((q < low - .05) | (q > high + .05) | ~torch.isfinite(q) | ~torch.isfinite(dq)).any(-1)
        invalid |= impossible | ~torch.isfinite(pose).all(-1) | ~torch.isfinite(velocity).all(-1)
        fail = dropped | invalid | (self.precision & (displacement > .018))

        # Dense approach/transport shaping is bounded. Milestones require real contact.
        palm_distance = (palm - pos[:, None]).norm(dim=-1)
        tip_distance = (tips - pos[:, None, None]).norm(dim=-1)
        approach = (.55 * (1. - torch.tanh(palm_distance / .25))
                    + .45 * (1. - torch.tanh(tip_distance.amin(-1) / .09)))
        potential = torch.where(self.handover, approach[:, 1], approach.amax(-1))
        potential = torch.where(self.phase == LIFT, .6 * grasp + .4 * (lift / self.lift_height).clamp(0., 1.), potential)
        potential = torch.where(self.phase == TRANSFER, approach[:, 0] + .35 * left, potential)
        potential = torch.where(self.phase >= PLACE, 1. - torch.tanh(delta.norm(dim=-1) / .23), potential)
        progress = torch.where(self.potential_valid, potential - self.previous_potential, 0.)

        overlapped = (self.phase == TRANSFER) & left & right & (lift > .04)
        self.overlap_steps.copy_(torch.where(overlapped & ~impeded & ~fail, self.overlap_steps + 1, 0))
        self.handover_seen |= self.overlap_steps >= 2
        transfer_done = (self.phase == TRANSFER) & self.handover_seen & left & ~contacts[:, 1].any(-1) & (lift > .04)
        criteria = ((self.phase == REACH) & grasp
                    | (self.phase == LIFT) & grasp & (lift >= self.lift_height)
                    | transfer_done
                    | (self.phase == PLACE) & in_tray & quiet & released
                    | (self.phase == SETTLE) & in_tray & quiet & released)
        self.hold.copy_(torch.where(criteria & ~impeded & ~fail, self.hold + 1, 0))
        direct_success = (self.phase == LIFT) & ~self.deposit
        cutoff = torch.where(direct_success, 15, torch.where(self.phase == SETTLE, 9, 3))
        reached = (self.hold >= cutoff) & ~fail
        success = reached & (direct_success | (self.phase == SETTLE))
        advance = reached & ~success
        following = torch.where((self.phase == LIFT) & ~self.handover, PLACE, self.phase + 1)
        self.phase.copy_(torch.where(advance, following, self.phase))
        self.hold.masked_fill_(advance, 0)
        self.previous_potential.copy_(torch.nan_to_num(potential))
        self.potential_valid.copy_(~advance & ~fail)

        reward = (2. * progress.clamp(-.4, .4) + .018 * potential - .01
                  + 2. * advance + 18. * success - 8. * fail - .4 * impeded
                  - .025 * actions.square().mean(-1) - .04 * movement.abs().mean(-1)
                  - .20 * displacement.clamp_max(.1))
        return torch.nan_to_num(reward, nan=-8.), success, fail, dropped, impeded, lift

    def step(self, actions):
        if actions.shape != self.targets.shape:
            raise ValueError(f"Expected bimanual actions {tuple(self.targets.shape)}")
        invalid = ~torch.isfinite(actions).all(-1)
        actions = torch.where(invalid[:, None], 0., actions).clamp(-1., 1.)
        bound = torch.where(actions >= 0., self.high - self.home, self.home - self.low)
        desired = self.home + bound * actions
        movement = (.35 * (desired - self.targets)).clamp(-self.step_limit, self.step_limit)
        movement = torch.where(invalid[:, None], 0., movement)
        command = self.targets + movement
        self.targets.copy_(command)
        self.batch.step(command, render=self.render)
        self.episode_length_buf += 1
        measurements = self._measure()
        angular_state = self.batch.angular_state()
        rewards, success, fail, dropped, impeded, lift = self._score(
            measurements, angular_state, actions, invalid, movement)
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~(success | fail)
        done = success | fail | timeout
        log = {"Metrics/fruit_lift_cm": 100. * torch.nan_to_num(lift).mean(),
               "Metrics/obstacle_contact_fraction": impeded.float().mean(),
               "Metrics/drop_fraction": dropped.float().mean()}
        if done.any():
            log["Metrics/success_rate"] = success.float().sum() / done.sum()
        for index, crop in enumerate(OBJECTIVES):
            selected = self.crop_ids == index
            log[f"Progress/{crop}_phase"] = self.phase[selected].float().mean()
            if (done & selected).any():
                log[f"Success/{crop}"] = (success & selected).sum() / (done & selected).sum()
        extras = {"time_outs": timeout, "log": log, "success": success, "dropped": dropped,
                  "invalid_pose": invalid, "obstacle_contact": impeded, "joint_command": command,
                  "target_pose": measurements[0], "fingertip_contacts": measurements[5]}
        ids = done.nonzero(as_tuple=True)[0]
        if ids.numel():
            self._reset(ids)
            self._update_observations()
        else:
            self._update_observations(measurements, angular_state)
        return self.observation, rewards, done.long(), extras
