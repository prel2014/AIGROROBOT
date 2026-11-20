"""RSL-RL VecEnv over the four agricultural crop groups in one PhysX context."""
import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from torch.nn.functional import one_hot

from dexman.experiments import sha256
from dexman.robot.joints import FINGERTIPS, PALM
from .actuation import JointCommandController, NOMINAL_RESPONSE_TIME_S
from .batch import AgriculturalBatch
from .contact_quality import contact_slip_speed, grasp_progress_masks, measure_contact_quality
from .fruit_clearance import fruit_clearance_points, lowest_collider_height
from .grasp_curriculum import GRASP_ACTION_DIM, GRASP_EPISODE_SECONDS, GraspMotion
from .impact_sensors import forbidden_impact
from .interface import ACTION_DIM, ARM_KEYS, JOINTS_PER_ARM, STATE_DIM
from .mobile_state import HEAD_LIMITS_DEG
from .objectives import (EPISODE_SECONDS, LIFT, MAX_DISTRACTORS, OBJECTIVES, OBS_DIM, PHASE_TEXT,
                         PLACE, ROVER_ACTION_DIM, ROVER_OBS_DIM, TRANSFER, exercise_text)
from .randomization import GROUND_Z, OBSTACLE_RADIUS, TABLE_Z
from .scene import CROPS
from .task_rules import AgriculturalTaskRules


class AgriculturalPPOEnv(VecEnv):
    def __init__(self, sim, cfg, *, seed=42, capture=False, render=False):
        self.cfg = cfg
        if cfg.rover:
            scene_cfg = next(iter(vars(cfg.scenes).values()))
            self.rover_asset_sha256 = sha256(scene_cfg.rover.spawn.usd_path)
        self.batch = AgriculturalBatch(sim, cfg, seed=seed)
        self.batch.initialize()
        self.device, self.num_envs = sim.device, self.batch.num_envs
        self.num_actions = ROVER_ACTION_DIM if cfg.rover else GRASP_ACTION_DIM if cfg.grasp else ACTION_DIM
        self.capture, self.render = capture, capture or render
        self.on_step = None
        self.on_rollout_step = None
        self.dt = self.batch.control_dt
        seconds = GRASP_EPISODE_SECONDS if cfg.grasp and cfg.exercise != "return" else EPISODE_SECONDS
        self.max_episode_length = round(seconds / self.dt)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.fruits, self.body_ids, starting_poses, velocity_limits = [], [], [], []
        self.fruit_collider_points = {crop: torch.tensor(fruit_clearance_points(spec.fruit), device=self.device)
                                      for crop, spec in CROPS.items()}
        crops = []
        for group in self.batch.groups:
            scene = group.scene
            produce = [obj for name, obj in scene.rigid_objects.items() if name.startswith("fruit_")]
            self.fruits.append(produce)
            starting_poses.append(self._pad_distractors(
                [fruit.data.default_root_pose.torch[:, :3] for fruit in produce[1:]], group.scene.num_envs))
            crops.extend([list(OBJECTIVES).index(group.crop)] * group.scene.num_envs)
            keys = (PALM, *FINGERTIPS)
            self.body_ids.append([group.arm_body_ids(key, keys) for key in ARM_KEYS])
            velocity_limits.append(torch.cat([group.arm(key).data.joint_vel_limits.torch[:, ids]
                                              for key, ids in zip(ARM_KEYS, group.joint_ids, strict=True)], dim=-1))
        self.crop_ids = torch.tensor(crops, device=self.device)
        self.crop_encoding = one_hot(self.crop_ids, len(OBJECTIVES))
        specs = [list(OBJECTIVES.values())[i] for i in crops]
        self.lift_height = torch.tensor([spec.lift for spec in specs], device=self.device)
        self.deposit = torch.tensor([spec.deposit for spec in specs], device=self.device)
        self.handover = torch.tensor([spec.handover for spec in specs], device=self.device)
        self.precision = torch.tensor([spec.precision for spec in specs], device=self.device)
        self.instructions = tuple(exercise_text(list(OBJECTIVES)[index], cfg.exercise)[0]
                                  + " Si ves obstáculos, esquívalos." for index in crops)
        self.rules = AgriculturalTaskRules(self.lift_height, self.deposit, self.handover, self.precision,
                                           torch.cat(starting_poses), self.batch.joint_limits, self.dt,
                                           reward_lift=cfg.grasp is not None, exercise=cfg.exercise,
                                           pickup=cfg.grasp.initial_fruit_pose[:3] if cfg.exercise else None)
        self.tray = self.rules.tray
        self.home = self.batch.angular_state()[:, :ACTION_DIM].clone()
        self.action_limits = self.batch.joint_limits
        if cfg.rover:
            self.head_limits = self.home.new_tensor(HEAD_LIMITS_DEG).deg2rad()
            self.head_targets = self.home.new_zeros((self.num_envs, len(HEAD_LIMITS_DEG)))
            self.action_limits = torch.cat((self.action_limits, self.head_limits.expand(self.num_envs, -1, -1)), dim=1)
        physical_low, physical_high = self.batch.joint_limits.unbind(-1)
        # Explore around the calibrated starting posture. Fingers retain their physical range.
        span = self.home.new_tensor(([1.5] * 6 + [10.] * (JOINTS_PER_ARM - 6)) * len(ARM_KEYS))
        self.low = torch.maximum(physical_low, self.home - span)
        self.high = torch.minimum(physical_high, self.home + span)
        self.home = self.home.clamp(self.low, self.high)
        self.controller = JointCommandController(
            self.home, self.low, self.high, torch.cat(velocity_limits).clamp_max(4.),
            dt=self.dt, max_delay_steps=1, seed=seed)
        self.targets = self.controller.targets
        self.grasp_motion = GraspMotion(cfg.grasp, self.home, self.dt) if cfg.grasp else None
        self._configure_controller(range(self.num_envs))
        self._update_observations()
        if capture and not cfg.rover:
            self.batch.sim.render()

    @property
    def phase(self):
        return self.rules.phase

    def _pad_distractors(self, positions, count):
        padding = [torch.zeros((count, 3), device=self.device)] * (MAX_DISTRACTORS - len(positions))
        return torch.stack((*positions, *padding), dim=1)

    def _measure(self):
        poses, velocities, others, palms, tips, active, grasp, collisions = ([] for _ in range(8))
        centers, opposition, normal_force, slip_speed, clearance = [], [], [], [], []
        for group, produce, body_ids in zip(self.batch.groups, self.fruits, self.body_ids, strict=True):
            scene, origin = group.scene, group.origins
            target = produce[0]
            pos = target.data.root_pos_w.torch
            center = target.data.root_com_pos_w.torch
            min_z = lowest_collider_height(target.data.root_link_pos_w.torch, target.data.root_link_quat_w.torch,
                                           self.fruit_collider_points[group.crop])
            clearance.append(min_z - origin[:, 2] - self.batch.pickup_surface_z)
            centers.append(center - origin)
            com_velocity = target.data.root_com_vel_w.torch
            poses.append(torch.cat((pos - origin, target.data.root_quat_w.torch), dim=-1))
            velocities.append(target.data.root_vel_w.torch)
            others.append(self._pad_distractors(
                [fruit.data.root_pos_w.torch - origin for fruit in produce[1:]], scene.num_envs))
            links, contacts, gripped, opposed, normals, slips = ([] for _ in range(6))
            forbidden = scene["fruit_obstacle_contact"].data.normal_force_matrix_w.torch[:, 0].norm(dim=-1)
            impact = forbidden_impact(scene)
            max_collision = torch.maximum(forbidden.amax(-1), torch.maximum(impact.environment_n, impact.inter_arm_n))
            for side, ids in zip(ARM_KEYS, body_ids, strict=True):
                arm_data = group.arm(side).data
                links.append(arm_data.body_pos_w.torch[:, ids] - origin[:, None, :])
                sensors = [scene[f"{side}_{tip}"].data for tip in FINGERTIPS]
                forces = torch.stack([sensor.normal_force_matrix_w.torch[:, 0] for sensor in sensors], dim=1)
                points = torch.stack([sensor.contact_pos_w.torch[:, 0, 0] for sensor in sensors], dim=1)
                max_collision = torch.maximum(max_collision, forces[:, :, 1:].norm(dim=-1).amax((1, 2)))
                precision = OBJECTIVES[group.crop].precision
                quality = measure_contact_quality(forces[:, :, 0], points, center, precision=precision)
                contacts.append(quality.touching)
                gripped.append(quality.held)
                opposed.append(quality.opposed)
                normals.append(quality.normal_force_n)
                slips.append(contact_slip_speed(points, forces[:, :, 0], arm_data.body_com_pos_w.torch[:, ids[1:]],
                                                arm_data.body_com_vel_w.torch[:, ids[1:]], center, com_velocity,
                                                minimum_force=.003 if precision else .04))
            links = torch.stack(links, dim=1)
            palms.append(links[:, :, 0])
            tips.append(links[:, :, 1:])
            active.append(torch.stack(contacts, dim=1))
            grasp.append(torch.stack(gripped, dim=1))
            collisions.append(max_collision)
            opposition.append(torch.stack(opposed, dim=1))
            normal_force.append(torch.stack(normals, dim=1))
            slip_speed.append(torch.stack(slips, dim=1))
        return tuple(torch.cat(field) for field in (
            poses, velocities, others, palms, tips, active, grasp, collisions,
            centers, opposition, normal_force, slip_speed, clearance))

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

    def _update_observations(self, measurement=None, angular_state=None, gaze_alignment=None):
        if measurement is None:
            measurement = self._measure()
        pose, velocity, others, palm, tip, contacts, held, _, center, *_ = measurement
        pos = pose[:, :3]
        self.angular_state = self.batch.angular_state() if angular_state is None else angular_state
        q, dq = self.angular_state.split(ACTION_DIM, dim=-1)
        half = (self.high - self.low).clamp_min(.01) * .5
        pending = self.controller.pending_targets[:, 0]
        joint_features = torch.stack(((q - self.home) / half, torch.asinh(dq),
                                      (self.targets - self.home) / half, (pending - self.home) / half),
                                     dim=-1).flatten(1)
        arm_features = torch.cat((palm - center[:, None], (tip - center[:, None, None]).flatten(2),
                                  contacts, held.unsqueeze(-1)), dim=-1)
        lift_goal = pos + torch.stack((torch.zeros_like(self.lift_height), torch.zeros_like(self.lift_height),
                                      self.lift_height - measurement[-1]), dim=-1)
        goal = torch.where((self.phase >= PLACE)[:, None], self.tray, lift_goal)
        task_features = torch.cat((pose, velocity, goal - pos, others.flatten(1),
                                   self.crop_encoding, one_hot(self.phase, len(PHASE_TEXT)),
                                   (self.rules.hold / 15.)[:, None], (self.episode_length_buf / self.max_episode_length)[:, None],
                                   self.rules.handover_seen[:, None], self.controller.delay_steps[:, None],
                                   self.controller.response_time_s[:, :1] / NOMINAL_RESPONSE_TIME_S), dim=-1)
        self.obstacle_features = self._obstacles()
        policy = torch.cat((joint_features, arm_features.flatten(1), task_features,
                            self.obstacle_features.flatten(1)), dim=-1)
        self.vla_state = self.angular_state
        if self.cfg.rover:
            self.vla_state = self.batch.mobile_state(self.angular_state)
            if gaze_alignment is None:
                gaze_alignment = self.batch.gaze_alignment(center)
            policy = torch.cat((policy, self.vla_state[:, STATE_DIM:], self.head_targets, gaze_alignment), dim=-1)
        if policy.shape != (self.num_envs, ROVER_OBS_DIM if self.cfg.rover else OBS_DIM):
            raise RuntimeError(f"Agricultural observation contract changed: {policy.shape}")
        self.observation = TensorDict({"policy": torch.nan_to_num(policy, nan=0., posinf=0., neginf=0.)},
                                      batch_size=[self.num_envs])

    def get_observations(self):
        return self.observation

    def _configure_controller(self, replica_ids):
        strength = self.batch.physics_variation
        if not strength:
            return
        ids = list(replica_ids)
        response, delay = [], []
        for index in ids:
            rng = np.random.default_rng(np.random.SeedSequence(
                [self.batch.seed, index, self.batch.episode_ids[index], 3]))
            response.append(NOMINAL_RESPONSE_TIME_S * (1. + strength * np.clip(rng.normal(0., .08), -.2, .2)))
            delay.append(int(np.clip(round(strength * rng.normal(.4, .35)), 0, 1)))
        self.controller.configure(ids, delay_steps=delay, response_time_s=self.home.new_tensor(response)[:, None],
                                  action_noise_std=.015 * strength)

    def _reset(self, ids):
        self.batch.reset(ids)
        self.episode_length_buf[ids] = 0
        self.rules.reset(ids)
        self._configure_controller(ids.tolist())
        self.controller.reset(ids)
        if self.cfg.rover:
            self.head_targets[ids] = 0.
        if self.grasp_motion is not None:
            self.grasp_motion.reset(ids)
        if self.capture and not self.cfg.rover:
            self.batch.sim.render()

    def step(self, actions):
        head = None
        if self.cfg.rover:
            arms, gaze = actions.split((ACTION_DIM, len(self.head_limits)), dim=-1)
            gaze = gaze.to(self.head_targets)
            # A zero policy command centers both eyes in the harvest pose despite asymmetric pitch limits.
            head = gaze.clamp(-1., 1.) * torch.where(gaze >= 0., self.head_limits[:, 1], -self.head_limits[:, 0])
            head = torch.where(torch.isfinite(gaze), head, torch.nan)
            actions = arms
        if self.grasp_motion is None:
            command = self.controller.step(actions)
            scored_actions, invalid = command.actions, command.invalid
        else:
            desired, scored_actions, invalid = self.grasp_motion.target(actions, self.phase == LIFT)
            delta = desired - self.home
            scale = torch.where(delta >= 0., self.high - self.home, self.home - self.low).clamp_min(1.e-6)
            command = self.controller.step(torch.where(invalid[:, None], torch.nan, delta / scale))
        return self._advance(command, scored_actions, invalid, head)

    def step_joint_targets(self, angles_rad):
        """Advance the student's applied radians through the same physics and task criteria."""
        head = None
        if self.cfg.rover:
            angles_rad, head = angles_rad.split((ACTION_DIM, len(self.head_limits)), dim=-1)
            head = head.to(self.head_targets)
        command = self.controller.apply_joint_targets(angles_rad)
        return self._advance(command, command.actions, command.invalid, head)

    def _advance(self, command, scored_actions, invalid, head):
        applied = command.joint_command
        if self.cfg.rover:
            invalid = invalid | ~torch.isfinite(head).all(-1)
            head = torch.where(invalid[:, None], self.head_targets, head).clamp(*self.head_limits.unbind(-1))
            self.head_targets.copy_(head)
            applied = torch.cat((applied, head), dim=-1)
        self.batch.step(command.joint_command, head_targets=head, render=self.render)
        self.episode_length_buf += 1
        measurements = self._measure()
        angular_state = self.batch.angular_state()
        rewards, success, fail, dropped, impeded, lift = self.rules.score(
            measurements, angular_state, scored_actions, invalid, command.movement)
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~(success | fail)
        done = success | fail | timeout
        log = {"Metrics/fruit_lift_cm": 100. * torch.nan_to_num(lift).mean(),
               "Metrics/joint_tracking_error_rad": (angular_state[:, :ACTION_DIM] - command.joint_command).abs().mean(),
               "Metrics/forbidden_contact_fraction": impeded.float().mean(),
               "Metrics/drop_fraction": dropped.float().mean()}
        gaze_alignment = None
        if self.cfg.rover:
            gaze_alignment = self.batch.gaze_alignment(measurements[8])
            # Encourage at least one optical axis to follow the fruit. Looking alone never completes a task.
            alignment = gaze_alignment.amax(dim=-1).clamp(-1., 1.)
            rewards -= self.dt * (1. - alignment)
            log["Gaze/best_eye_target_angle_deg"] = alignment.acos().rad2deg().mean()
        center, opposed, normal_force, slip_speed = measurements[8:12]
        near = (measurements[4] - center[:, None, None]).norm(dim=-1).amin(-1) < .07
        allowed = torch.stack((~self.handover | (self.phase >= TRANSFER), torch.ones_like(self.handover)), dim=-1)
        grasp_states = grasp_progress_masks(near, measurements[5], opposed, measurements[6], eligible_hands=allowed)
        if done.any():
            log["Metrics/success_rate"] = success[done].float()
        for group in self.batch.groups:
            crop = group.crop
            selected = slice(group.start, group.stop)
            log[f"Progress/{crop}_phase"] = self.phase[selected].float().mean()
            log[f"Grasp/{crop}/lift_cm"] = 100. * torch.nan_to_num(lift[selected]).amax()
            log[f"Grasp/{crop}/peak_force_n"] = normal_force[selected].amax()
            valid_slip = torch.isfinite(slip_speed[selected])
            log[f"Grasp/{crop}/slip_m_s"] = torch.where(valid_slip, slip_speed[selected], 0.).sum() / valid_slip.sum().clamp_min(1)
            for name, mask in grasp_states.items():
                log[f"Grasp/{crop}/{name}"] = mask[selected].float().mean()
            if done[selected].any():
                log[f"Success/{crop}"] = success[selected][done[selected]].float()
        extras = {"time_outs": timeout, "log": log, "success": success, "dropped": dropped,
                  "invalid_pose": invalid, "obstacle_contact": impeded, "joint_command": applied,
                  "target_pose": measurements[0], "fingertip_contacts": measurements[5],
                  "phase_at_step": self.phase.clone(), "fruit_clearance_m": lift, "held": measurements[6]}
        if self.on_rollout_step is not None:
            self.on_rollout_step()
        ids = done.nonzero(as_tuple=True)[0]
        if ids.numel():
            self._reset(ids)
            self._update_observations()
        else:
            self._update_observations(measurements, angular_state, gaze_alignment)
        if self.on_step is not None:
            self.on_step()
        return self.observation, rewards, done.long(), extras
