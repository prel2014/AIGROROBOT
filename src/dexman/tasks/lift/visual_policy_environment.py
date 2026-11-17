"""VLA control path: 30 absolute joint targets, with no teacher synergy decoder."""
import torch

from .environment import LiftEnv


class JointAngleLiftEnv(LiftEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = self.targets.clone()
        self.previous_actions = self.actions.clone()
        self.home_action = self.targets[0].clone()

    def _pre_physics_step(self, actions):
        self._accept_action(actions, self.limits[:, 0], self.limits[:, 1])
        # The prediction is a desired position BEFORE filtering. Apply the filter once.
        self._update_targets(self.actions)

    def _get_observations(self):
        # Object pose and goal are used for scoring only, never as policy inputs.
        qpos = self.robot.data.joint_pos.torch[:, self.joint_ids]
        qvel = self.robot.data.joint_vel.torch[:, self.joint_ids]
        return {"policy": torch.cat((qpos, qvel), dim=-1)}

    def _reward_actions(self):
        low, high = self.limits[:, 0], self.limits[:, 1]
        return (2 * (self.actions - low) / (high - low) - 1,
                2 * (self.previous_actions - low) / (high - low) - 1)
