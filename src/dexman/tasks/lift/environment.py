"""Direct RL orchestration; robot control, rewards, and resets live in separate modules."""
import json
import torch
from isaaclab.envs import DirectRLEnv

from dexman.paths import JOINT_MANIFEST
from dexman.robot.control import action_limits, decode, encode, joint_limit_violation
from dexman.robot.joints import FINGERTIPS, JOINT_NAMES, PALM
from dexman.sensors.grasp import measured_grasp, forbidden_contact
from .resets import reset_state
from .curriculum import LiftCurriculum
from .rewards import FAILURE_PENALTY, grasp_metrics, lift_reward, lift_potential, progress_reward, object_clearance


class LiftEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        if cfg.reward_version not in (1, 4):
            raise ValueError("Unsupported lift reward/success version; corrected training requires version 4")
        super().__init__(cfg, render_mode, **kwargs)
        self.robot, self.object = self.scene["robot"], self.scene["object"]
        ids, names = self.robot.find_joints(list(JOINT_NAMES), preserve_order=True)
        if tuple(names) != JOINT_NAMES or len(ids) != 30:
            raise RuntimeError(f"Unexpected joint map: {names}")
        self.joint_ids = torch.tensor(ids, device=self.device)
        palm, _ = self.robot.find_bodies(PALM)
        tips, _ = self.robot.find_bodies(list(FINGERTIPS), preserve_order=True)
        self.palm_id, self.tip_ids = palm[0], torch.tensor(tips, device=self.device)
        manifest = json.loads(JOINT_MANIFEST.read_text())
        self.limits = torch.tensor(action_limits(manifest, cfg.arm_half_range, cfg.arm_center),
                                   device=self.device)
        self.physical_limits = torch.tensor(action_limits(manifest, float("inf")), device=self.device)
        self.action_center = None if cfg.action_center is None else torch.tensor(cfg.action_center, device=self.device)
        self.actions = torch.zeros((self.num_envs, 26), device=self.device)
        self.previous_actions = self.actions.clone()
        self.targets = self.robot.data.default_joint_pos.torch[:, self.joint_ids].clone()
        self.home_action = encode(self.targets[0], self.limits, self.action_center)
        self.goal = torch.zeros((self.num_envs, 3), device=self.device)
        self.hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.curriculum = LiftCurriculum(tuple(getattr(cfg, "curriculum_heights", (.15,))),
            getattr(cfg, "curriculum_window", 64), getattr(cfg, "curriculum_success_threshold", .6),
            getattr(cfg, "curriculum_initial_level", 0))
        self.previous_potential = torch.zeros(self.num_envs, device=self.device)
        self.potential_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_failure = self.last_success.clone()
        self.invalid_action = self.last_success.clone()
        self.invalid_pose = self.last_success.clone()
        self.total_episodes = torch.zeros((), device=self.device)
        self.total_successes = torch.zeros((), device=self.device)
        self.episode_max_lift = torch.zeros(self.num_envs, device=self.device)
        self.maximum_lift = torch.zeros((), device=self.device)
        self.maximum_clearance = torch.zeros((), device=self.device)

    @property
    def curriculum_level(self):
        return self.curriculum.level

    def learning_state(self):
        return {"curriculum": self.curriculum.state_dict(), "episodes": int(self.total_episodes.item()),
                "successes": int(self.total_successes.item()), "maximum_lift_m": self.maximum_lift.item(),
                "maximum_clearance_m": self.maximum_clearance.item()}

    def restore_learning_state(self, state):
        self.curriculum.load_state_dict(state["curriculum"])
        if not 0 <= state["successes"] <= state["episodes"]:
            raise ValueError("Invalid episode counters")
        self.total_episodes.fill_(state["episodes"])
        self.total_successes.fill_(state["successes"])
        self.maximum_lift.fill_(state.get("maximum_lift_m", 0.))
        self.maximum_clearance.fill_(state.get("maximum_clearance_m", 0.))

    def _pre_physics_step(self, actions):
        self._accept_action(actions, -1, 1)
        self._update_targets(decode(self.actions, self.limits, self.action_center))

    def _accept_action(self, actions, lower, upper):
        if actions.shape != self.actions.shape:
            raise ValueError(f"Expected actions with shape {tuple(self.actions.shape)}")
        self.invalid_action.copy_(joint_limit_violation(actions, lower, upper, tolerance=1e-5))
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(torch.where(self.invalid_action[:, None], self.actions, actions.clamp(lower, upper)))

    def _update_targets(self, desired):
        # Hold the last applied target for rejected commands.
        desired = torch.where(self.invalid_action[:, None], self.targets, desired)
        self.targets.lerp_(desired, self.cfg.action_smoothing)

    def _apply_action(self):
        self.robot.actuators.target_command.set_position_index(value=self.targets, joint_ids=self.joint_ids)
        if self.cfg.gravity_compensation:
            self.robot.actuators.target_command.set_effort_index(
                value=self.robot.data.gravity_compensation_forces.torch)

    def _geometry(self):
        pos = self.object.data.root_pos_w.torch
        palm = self.robot.data.body_pos_w.torch[:, self.palm_id]
        tips = self.robot.data.body_pos_w.torch[:, self.tip_ids]
        return pos, palm, tips

    def _get_observations(self):
        return self.teacher_observations()

    def teacher_observations(self):
        """Privileged teacher input also available for labelling student-visited states."""
        q = self.robot.data.joint_pos.torch[:, self.joint_ids]
        qd = self.robot.data.joint_vel.torch[:, self.joint_ids]
        low, high = self.limits[:, 0], self.limits[:, 1]
        pos, palm, _ = self._geometry()
        # Expose the controller's filtered position target, not the previous noisy
        # policy sample. This retains the low-pass state and avoids raw-action feedback.
        applied_targets = encode(self.targets, self.limits, self.action_center)
        obs = torch.cat((2 * (q - low) / (high - low) - 1, qd * 0.1,
                         pos - palm, pos - self.scene.env_origins, self.goal - pos, applied_targets), dim=-1)
        return {"policy": obs}

    def _get_rewards(self):
        actions, previous_actions = self._reward_actions()
        pos, palm, tips = self._geometry()
        pd, fd, grasp = grasp_metrics(pos, palm, tips)
        height = pos[:, 2] - self.scene.env_origins[:, 2]
        distance = (self.goal - pos).norm(dim=-1)
        if self.cfg.reward_version == 4:
            xy_error = (pos[:, :2] - self.goal[:, :2]).norm(dim=-1)
            target_lift = self.goal[:, 2] - self.scene.env_origins[:, 2] - self.cfg.object_initial_height
            tip_distances = (tips - pos[:, None]).norm(dim=-1)
            clearance = object_clearance(height, self.object.data.root_quat_w.torch, self.cfg.table_height, self.cfg.asset_shape, self.cfg.asset_dimensions)
            potential, grip, lifted = lift_potential(pd, tip_distances, height,
                xy_error, target_lift, self.cfg.object_initial_height, clearance)
            contact_grasp, contacts, _ = measured_grasp(self)
            potential = potential - 10 * lifted * (1 - contact_grasp.float()) * grip
            grip = grip * contact_grasp.float()
            previous = torch.where(self.potential_valid, self.previous_potential, potential)
            reward = progress_reward(potential, previous, self.last_success, self.last_failure,
                actions, previous_actions, xy_error, grip, lifted, dt=self.step_dt)
            self.previous_potential.copy_(torch.where(self.invalid_pose, 0., potential))
            self.potential_valid.fill_(True)
            self.extras["log"].update({"Metrics/contact_grasp_fraction": contact_grasp.float().mean(),
                "Metrics/fingertip_contacts": contacts.float().sum(-1).mean(),
                "Metrics/soft_grip": torch.where(self.invalid_pose, 0., grip).mean(),
                "Metrics/lift_cm": (torch.where(self.invalid_pose, self.cfg.object_initial_height, height)
                                    - self.cfg.object_initial_height).mean() * 100})
            # Compare commands actually submitted after smoothing with physical state.
            # Implicit-actuator efforts are estimates, not contact-force measurements.
            q = self.robot.data.joint_pos.torch[:, self.joint_ids]
            error = torch.nan_to_num((self.targets - q).abs())
            effort = self.robot.actuators.applied_effort.torch[:, self.joint_ids]
            self.maximum_lift.copy_(torch.maximum(self.maximum_lift,
                torch.where(self.invalid_pose, 0., height - self.cfg.object_initial_height).max()))
            self.maximum_clearance.copy_(torch.maximum(self.maximum_clearance,
                torch.where(self.invalid_pose, 0., clearance).max()))
            self.extras["log"].update({
                "Metrics/best_lift_cm": self.maximum_lift.clone() * 100,
                "Metrics/best_clearance_cm": self.maximum_clearance.clone() * 100,
                "Metrics/wrj0_target_rad": self.targets[:, 7].mean(),
                "Metrics/wrj0_measured_rad": torch.nan_to_num(q[:, 7]).mean(),
                "Metrics/wrj0_tracking_error_rad": error[:, 7].mean(),
                "Metrics/arm_tracking_error_rad": error[:, :6].mean(),
                "Metrics/hand_tracking_error_rad": error[:, 6:].mean(),
                "Metrics/wrj0_effort_estimate_nm": torch.nan_to_num(effort[:, 7]).mean(),
                "Metrics/thumb_distance_cm": torch.nan_to_num(tip_distances[:, -1]).mean() * 100,
                "Metrics/palm_distance_cm": torch.nan_to_num(pd).mean() * 100,
                "Metrics/grasp_fraction": contact_grasp.float().mean(),
                "Metrics/proximity_grasp_fraction": grasp.float().mean(),
                "Metrics/clearance_above_5mm_fraction": (clearance > .005).float().mean()})
        else:
            reward = lift_reward(pd, fd, grasp, height, distance, actions, previous_actions,
                                 self.cfg.object_initial_height, self.cfg.lift_height)
            reward -= FAILURE_PENALTY * self.last_failure.float()
        return torch.where(self.invalid_action | self.invalid_pose, -FAILURE_PENALTY, reward)

    def _reward_actions(self):
        return self.actions, self.previous_actions

    def _check_pose(self, position, rotation, velocity, palm, tips):
        qpos = self.robot.data.joint_pos.torch[:, self.joint_ids]
        qvel = self.robot.data.joint_vel.torch[:, self.joint_ids]
        # PhysX's iterative solver can overshoot hard limits slightly.
        invalid_joints = joint_limit_violation(qpos, self.physical_limits[:, 0], self.physical_limits[:, 1], .05)
        state = torch.cat((qvel, position, rotation, velocity, palm, tips.flatten(1)), dim=-1)
        self.invalid_pose.copy_(invalid_joints | (~torch.isfinite(state)).any(-1)
                                | ((rotation.norm(dim=-1) - 1).abs() > .01))

    def _get_dones(self):
        pos, palm, tips = self._geometry()
        if self.cfg.reward_version == 4:
            grasp, _, _ = measured_grasp(self)
        else:
            _, _, grasp = grasp_metrics(pos, palm, tips)
        rotation = self.object.data.root_quat_w.torch
        velocity = self.object.data.root_lin_vel_w.torch
        self._check_pose(pos, rotation, velocity, palm, tips)
        stable = velocity.norm(dim=-1) < 0.15
        local = pos - self.scene.env_origins
        if self.cfg.reward_version == 4:
            stage_height = self.goal[:, 2] - self.scene.env_origins[:, 2] - self.cfg.object_initial_height
            required_height = self.cfg.object_initial_height + stage_height
            # Compare absolute local heights to avoid subtraction roundoff at the upper boundary.
            # The lower bound is the requested measured lift, with no tolerance below it.
            at_goal = (local[:, 2] >= required_height) & (
                local[:, 2] <= required_height + self.cfg.success_lift_overshoot_m) & (
                (self.goal[:, :2] - pos[:, :2]).norm(dim=-1) < .06)
        else:
            at_goal = (self.goal - pos).norm(dim=-1) < .05
        dropped = local[:, 2] < self.cfg.table_height - 0.15
        escaped = (local[:, :2].abs() > 1.5).any(-1)
        collision = forbidden_contact(self) if self.cfg.reward_version == 4 else torch.zeros_like(dropped)
        failure = dropped | escaped | self.invalid_action | self.invalid_pose | collision
        holding = grasp & stable & at_goal & ~failure
        if self.cfg.reward_version == 4:
            local_height = pos[:, 2] - self.scene.env_origins[:, 2]
            holding &= object_clearance(local_height, rotation, self.cfg.table_height, self.cfg.asset_shape, self.cfg.asset_dimensions) > .005
        self.hold_steps = torch.where(holding, self.hold_steps + 1, 0)
        success = self.hold_steps >= self.cfg.success_hold_steps
        timeout = (self.episode_length_buf >= self.max_episode_length - 1) & ~(success | failure)
        lift = torch.where(self.invalid_pose, 0., (local[:, 2] - self.cfg.object_initial_height).clamp_min(0))
        self.episode_max_lift.copy_(torch.maximum(self.episode_max_lift, lift))
        reason = torch.where(success, 1, torch.where(dropped, 2, torch.where(escaped, 3, torch.where(timeout, 4, 0))))
        reason = torch.where(self.invalid_pose, 6, torch.where(self.invalid_action, 5, reason))
        reason = torch.where(collision, 7, reason)
        self.extras["terminal_reason"] = reason.clone()
        self.extras["episode_max_lift_m"] = self.episode_max_lift.clone()
        self.last_success.copy_(success)
        self.last_failure.copy_(failure)
        self.extras["success"] = success.clone()
        self.extras["log"] = {"Metrics/success": success.float().mean(),
                              "Metrics/forbidden_contact": collision.float().mean()}
        if self.cfg.reward_version == 4:
            self.total_episodes.add_((success | failure | timeout).sum())
            self.total_successes.add_(success.sum())
            self.extras["log"].update({
                "Metrics/episode_success_rate": (self.total_successes / self.total_episodes.clamp(min=1)).clone(),
                "Metrics/curriculum_level": float(self.curriculum_level),
                "Metrics/target_lift_cm": 100 * self.cfg.curriculum_heights[self.curriculum_level]})
        return success | failure, timeout

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if self.cfg.reward_version == 4 and self.cfg.curriculum_enabled:
            finished = self.episode_length_buf[env_ids] > 0
            self.curriculum.observe(int(finished.sum().item()),
                int((self.last_success[env_ids] & finished).sum().item()))
        super()._reset_idx(env_ids)
        reset_state(self, env_ids)
