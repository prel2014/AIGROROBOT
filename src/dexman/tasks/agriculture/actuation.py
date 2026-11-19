"""Batched joint-speed limits, response lag and command disturbances for the teacher."""
import math
from typing import NamedTuple

import torch

from .interface import ACTION_DIM


# Equivalent to applying 35% of the position error at a 30 Hz control step.
NOMINAL_RESPONSE_TIME_S = -(1.0 / 30.0) / math.log1p(-0.35)


class JointCommand(NamedTuple):
    actions: torch.Tensor
    invalid: torch.Tensor
    joint_command: torch.Tensor
    movement: torch.Tensor


class JointCommandController:
    """Execute normalized absolute-position requests while preserving PPO's samples."""

    def __init__(self, home, low, high, velocity_limits, *, dt, response_time_s=NOMINAL_RESPONSE_TIME_S,
                 max_delay_steps=0, seed=0):
        if home.ndim != 2 or home.shape[-1] != ACTION_DIM or not home.is_floating_point():
            raise ValueError(f"Expected floating-point home with shape (replicas, {ACTION_DIM})")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("The control time step must be positive and finite")
        if type(max_delay_steps) is not int or max_delay_steps < 0:
            raise ValueError("Maximum delay must be a nonnegative integer")
        if any(tensor.shape != home.shape for tensor in (low, high, velocity_limits)):
            raise ValueError("Joint bounds and velocity limits must match home")
        self.home = home.detach()
        self._low, self._high = low.detach().to(home), high.detach().to(home)
        velocity = velocity_limits.detach().to(home)
        if (not torch.isfinite(self.home).all() or not torch.isfinite(self._low).all()
                or not torch.isfinite(self._high).all() or not torch.isfinite(velocity).all()
                or (velocity < 0).any() or (self.home < self._low).any() or (self.home > self._high).any()):
            raise ValueError("Home must lie inside finite joint bounds and velocities must be nonnegative")
        self.dt = dt
        self.max_delay_steps = max_delay_steps
        self.targets = self.home.clone()
        self._step_limit = velocity * dt
        self._env_ids = torch.arange(len(home), device=home.device)
        self.delay_steps = torch.zeros_like(self._env_ids)
        self.response_time_s = torch.empty_like(home)
        self._response = torch.empty_like(home)
        self.action_noise_std = torch.zeros_like(home)
        self._noise_enabled = False
        self._generator = torch.Generator(device=home.device).manual_seed(seed)
        self._history = self.home[:, None, :].expand(-1, max_delay_steps + 1, -1).clone()
        self._next_slot = 0
        self.configure(response_time_s=response_time_s)

    def _indices(self, ids):
        if ids is None:
            return self._env_ids
        return torch.as_tensor(ids, device=self.home.device, dtype=torch.long).reshape(-1)

    @torch.no_grad()
    def configure(self, ids=None, *, delay_steps=None, response_time_s=None, action_noise_std=None):
        """Set per-replica parameters before a reset.

        Delay is in whole control steps, with one integer per selected replica.
        Response time and action-noise standard deviation may be scalars, per-joint vectors,
        or tensors of shape (selected replicas, 1 or ACTION_DIM).
        """
        indices = self._indices(ids)
        if delay_steps is not None:
            delay = torch.as_tensor(delay_steps, device=self.home.device)
            if (delay.is_floating_point() or delay.is_complex() or delay.dtype == torch.bool
                    or (delay < 0).any() or (delay > self.max_delay_steps).any()):
                raise ValueError(f"Delay must be an integer between 0 and {self.max_delay_steps}")
            self.delay_steps[indices] = delay.to(torch.long)
        if response_time_s is not None:
            response_time = torch.as_tensor(response_time_s, device=self.home.device, dtype=self.home.dtype)
            if not torch.isfinite(response_time).all() or (response_time <= 0).any():
                raise ValueError("Response time must be positive and finite")
            self.response_time_s[indices] = response_time
            self._response[indices] = -torch.expm1(-self.dt / response_time)
        if action_noise_std is not None:
            noise = torch.as_tensor(action_noise_std, device=self.home.device, dtype=self.home.dtype)
            if not torch.isfinite(noise).all() or (noise < 0).any():
                raise ValueError("Action noise must be nonnegative and finite")
            self.action_noise_std[indices] = noise
            self._noise_enabled = bool(self.action_noise_std.any())

    @torch.no_grad()
    def reset(self, ids=None):
        """Clear targets and command history for the selected replicas."""
        indices = self._indices(ids)
        self.targets[indices] = self.home[indices]
        self._history[indices] = self.home[indices, None, :]

    @property
    def pending_targets(self):
        """Queued position targets, newest first, padded with home after each replica's delay."""
        age = torch.arange(self.max_delay_steps, device=self.home.device)
        slots = (self._next_slot - 1 - age).remainder(self._history.shape[1])
        return torch.where(age[None, :, None] < self.delay_steps[:, None, None],
                           self._history[:, slots], self.home[:, None, :])

    @torch.no_grad()
    def apply_joint_targets(self, targets_rad):
        """Replay post-filter position targets at the same actuator boundary as the demonstrations."""
        if targets_rad.shape != self.targets.shape:
            raise ValueError(f"Expected physical joint targets {tuple(self.targets.shape)}")
        requested = targets_rad.to(self.targets)
        invalid = ~torch.isfinite(requested).all(-1)
        desired = torch.where(invalid[:, None], self.targets, requested).clamp(self._low, self._high)
        movement = (desired - self.targets).clamp(-self._step_limit, self._step_limit)
        command = self.targets + movement
        self.targets.copy_(command)
        delta = command - self.home
        scale = torch.where(delta >= 0, self._high - self.home, self.home - self._low)
        return JointCommand(delta / scale.clamp_min(1.e-6), invalid, command, movement)

    @torch.no_grad()
    def step(self, sampled_actions):
        if sampled_actions.shape != self.targets.shape:
            raise ValueError(f"Expected bimanual actions {tuple(self.targets.shape)}")
        sampled_actions = sampled_actions.to(self.targets)
        invalid = ~torch.isfinite(sampled_actions).all(-1)
        actions = torch.where(invalid[:, None], 0.0, sampled_actions).clamp(-1.0, 1.0)
        noisy_actions = actions
        if self._noise_enabled:
            noise = torch.randn(actions.shape, generator=self._generator, device=actions.device, dtype=actions.dtype)
            noisy_actions = torch.where(invalid[:, None], 0.0,
                                        (actions + noise * self.action_noise_std).clamp(-1.0, 1.0))
        bound = torch.where(noisy_actions >= 0.0, self._high - self.home, self.home - self._low)
        desired = self.home + bound * noisy_actions
        if self.max_delay_steps:
            self._history[:, self._next_slot] = torch.where(invalid[:, None], self.targets, desired)
            due = (self._next_slot - self.delay_steps).remainder(self._history.shape[1])
            desired = self._history[self._env_ids, due]
            self._next_slot = (self._next_slot + 1) % self._history.shape[1]

        movement = (self._response * (desired - self.targets)).clamp(-self._step_limit, self._step_limit)
        movement = torch.where(invalid[:, None], 0.0, movement)
        command = (self.targets + movement).clamp(self._low, self._high)
        self.targets.copy_(command)
        return JointCommand(actions, invalid, command, movement)
