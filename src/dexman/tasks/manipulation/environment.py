"""Isaac Sim adapter for task resets, observations and scoring."""
import torch
from dexman.tasks.lift.visual_policy_environment import JointAngleLiftEnv
from dexman.tasks.lift.rewards import FAILURE_PENALTY, cube_clearance
from .catalog import TASKS, resolve_instruction
from .rewards import task_metrics, task_reward


class MemoryTaskEnv(JointAngleLiftEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.task_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phrase_ids = self.task_ids.clone()
        self.task_offsets = torch.tensor([task.offset for task in TASKS], device=self.device)
        self.push_violated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.task_potential = torch.zeros(self.num_envs, device=self.device)
        self.task_trials = torch.zeros(len(TASKS), device=self.device)
        self.task_successes = self.task_trials.clone()
        self.fixed_task = None if cfg.instruction_override is None else resolve_instruction(cfg.instruction_override)

    @property
    def instructions(self):
        if self.cfg.instruction_override is not None:
            return [self.cfg.instruction_override] * self.num_envs
        return [TASKS[t].instructions[p] for t, p in zip(self.task_ids.tolist(), self.phrase_ids.tolist())]

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        n = len(env_ids)
        self.task_ids[env_ids] = (torch.randint(len(TASKS), (n,), device=self.device)
                                 if self.fixed_task is None else self.fixed_task)
        phrase_counts = torch.tensor([len(task.instructions) for task in TASKS], device=self.device)
        self.phrase_ids[env_ids] = (torch.rand(n, device=self.device) * phrase_counts[self.task_ids[env_ids]]).long()
        # reset_state already supplies the randomized spawn XY and the lift goal Z.
        self.goal[env_ids, 2] -= self.cfg.curriculum_heights[self.curriculum_level]
        self.goal[env_ids] += self.task_offsets[self.task_ids[env_ids]]
        self.push_violated[env_ids] = False

    def _get_observations(self):
        q = self.robot.data.joint_pos.torch[:, self.joint_ids]
        qd = self.robot.data.joint_vel.torch[:, self.joint_ids]
        pos, palm, _ = self._geometry()
        local = pos - self.scene.env_origins
        task = torch.nn.functional.one_hot(self.task_ids, len(TASKS)).float()
        remaining = (1 - self.episode_length_buf.float() / self.max_episode_length).unsqueeze(-1)
        # 30 + 30 + 3 + 3 + 3 + 3 + 30 + 3 + 1 + 1 = 107.
        critic = torch.cat((q, qd, local, pos - palm, self.goal - pos,
                            self.object.data.root_lin_vel_w.torch, self.actions, task,
                            self.push_violated.float().unsqueeze(-1), remaining), -1)
        return {"policy": q.clone(), "critic": critic}

    def _get_dones(self):
        pos, palm, tips = self._geometry()
        origins = self.scene.env_origins
        local = pos - origins
        rotation = self.object.data.root_quat_w.torch
        velocity = self.object.data.root_lin_vel_w.torch
        self._check_pose(pos, rotation, velocity, palm, tips)
        clearance = cube_clearance(local[:, 2], rotation, self.cfg.table_height)
        self.task_potential, at_goal, self.push_violated = task_metrics(
            self.task_ids, local, self.goal - origins, palm - origins, tips - origins[:, None],
            velocity.norm(dim=-1), clearance, self.push_violated)
        failure = ((local[:, 2] < self.cfg.table_height - .15) | (local[:, :2].abs() > 1.5).any(-1)
                   | self.push_violated | self.invalid_pose | self.invalid_action)
        self.hold_steps = torch.where(at_goal & ~failure, self.hold_steps + 1, 0)
        success = self.hold_steps >= self.cfg.success_hold_steps
        timeout = (self.episode_length_buf >= self.max_episode_length - 1) & ~(success | failure)
        done = success | failure | timeout
        self.last_success.copy_(success)
        self.last_failure.copy_(failure | (timeout & ~success))
        self.task_trials.scatter_add_(0, self.task_ids, done.float())
        self.task_successes.scatter_add_(0, self.task_ids, success.float())
        self.extras["success"] = success.clone()
        # Copy before autoreset replaces instructions/task_ids.
        self.extras["task_ids"] = self.task_ids.clone()
        self.extras["log"] = {
            f"Tasks/{task.name}_success_rate": (self.task_successes[i] / self.task_trials[i].clamp_min(1)).clone()
            for i, task in enumerate(TASKS)}
        return success | failure, timeout

    def _get_rewards(self):
        previous = torch.where(self.potential_valid, self.previous_potential, self.task_potential)
        actions, old_actions = self._reward_actions()
        result = task_reward(self.task_potential, previous, self.last_success | self.last_failure,
                             self.last_success, self.last_failure, actions, old_actions)
        self.previous_potential.copy_(torch.where(self.invalid_pose, 0., self.task_potential))
        self.potential_valid.fill_(True)
        return torch.where(self.invalid_pose | self.invalid_action, -FAILURE_PENALTY, result)
