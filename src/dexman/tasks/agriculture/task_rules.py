"""Simulator-independent rewards and phase transitions for agricultural manipulation."""
import torch

from .contact_quality import (MAX_HOLD_LINEAR_SPEED_M_S, MIN_STABLE_HOLD_S,
                              opposing_tip_proximity, steady_contacts)
from .interface import ACTION_DIM
from .objectives import FORBIDDEN_IMPACT_N, LIFT, PLACE, REACH, SETTLE, TRANSFER, TRAY_CENTER
from .randomization import TABLE_Z


class AgriculturalTaskRules:
    """Evaluate measured contacts and motion and retain per-episode task state."""

    def __init__(self, lift_height, deposit, handover, precision, starting_distractors, joint_limits, dt, *, reward_lift):
        self.lift_height = lift_height
        self.deposit = deposit
        self.handover = handover
        self.precision = precision
        self.starting_distractors = starting_distractors
        self.joint_limits = joint_limits
        self.dt = dt
        self.reward_lift = reward_lift
        self.tray = lift_height.new_tensor(TRAY_CENTER)
        self.phase = torch.full_like(deposit, REACH, dtype=torch.long)
        self.hold = torch.zeros_like(self.phase)
        self.overlap_steps = torch.zeros_like(self.phase)
        self.handover_seen = torch.zeros_like(deposit)
        self.previous_potential = torch.zeros_like(lift_height)
        self.potential_valid = torch.zeros_like(deposit)

    def reset(self, ids):
        self.phase[ids] = REACH
        self.hold[ids] = 0
        self.overlap_steps[ids] = 0
        self.handover_seen[ids] = False
        self.potential_valid[ids] = False

    def score(self, measurements, angular_state, actions, invalid, movement):
        pose, velocity, others, palm, tips, contacts, held, collision, center, opposed, _, slip, lift = measurements
        pos = pose[:, :3]
        left, right = held.unbind(-1)
        grasp = torch.where(self.handover, right, held.any(-1))
        stable = held & steady_contacts(slip, contacts)
        stable_grasp = torch.where(self.handover, stable[:, 1], stable.any(-1))
        displacement = (others - self.starting_distractors).norm(dim=-1).amax(-1)
        delta = (pos - self.tray).abs()
        in_tray = (delta[:, 0] < .095) & (delta[:, 1] < .076) & (delta[:, 2] < .018)
        quiet = velocity[:, :3].norm(dim=-1) < MAX_HOLD_LINEAR_SPEED_M_S
        released = ~contacts.any((1, 2))
        impeded = collision > FORBIDDEN_IMPACT_N
        outside = (pos[:, 0] < -.26) | (pos[:, 0] > .94) | (pos[:, 1].abs() > .72)
        dropped = (pos[:, 2] < TABLE_Z - .065) | (outside & (pos[:, 2] < TABLE_Z + .012))
        displaced_off_table = ((others[..., 0] < -.26) | (others[..., 0] > .94)
                               | (others[..., 1].abs() > .72)) & (others[..., 2] < TABLE_Z + .012)
        distractor_dropped = (others[..., 2] < TABLE_Z - .065) | displaced_off_table
        dropped |= (distractor_dropped & (self.starting_distractors[..., 2] > 0.)).any(-1)
        q, dq = angular_state.split(ACTION_DIM, dim=-1)
        low, high = self.joint_limits.unbind(-1)
        impossible = ((q < low - .05) | (q > high + .05) | ~torch.isfinite(q) | ~torch.isfinite(dq)).any(-1)
        invalid |= impossible | ~torch.isfinite(pose).all(-1) | ~torch.isfinite(velocity).all(-1)
        fail = dropped | invalid | (self.precision & (displacement > .018))

        # Dense approach/transport shaping is bounded. Milestones require real contact.
        palm_distance = (palm - center[:, None]).norm(dim=-1)
        approach = (.55 * (1. - torch.tanh(palm_distance / .25))
                    + .45 * opposing_tip_proximity(tips, center, self.precision)
                    + .10 * contacts.float().mean(-1) + .20 * opposed)
        potential = torch.where(self.handover, approach[:, 1], approach.amax(-1))
        lift_progress = (lift / self.lift_height).clamp(0., 1.) * stable_grasp
        potential = torch.where(self.phase == LIFT, .6 * grasp + .4 * lift_progress, potential)
        potential = torch.where(self.phase == TRANSFER, approach[:, 0] + .35 * left, potential)
        potential = torch.where(self.phase >= PLACE, 1. - torch.tanh(delta.norm(dim=-1) / .23), potential)
        progress = torch.where(self.potential_valid, potential - self.previous_potential, 0.)

        overlapped = (self.phase == TRANSFER) & left & right & (lift > .04)
        self.overlap_steps.copy_(torch.where(overlapped & ~impeded & ~fail, self.overlap_steps + 1, 0))
        self.handover_seen |= self.overlap_steps >= 2
        transfer_done = (self.phase == TRANSFER) & self.handover_seen & left & ~contacts[:, 1].any(-1) & (lift > .04)
        criteria = ((self.phase == REACH) & grasp
                    | (self.phase == LIFT) & stable_grasp & quiet & (lift >= self.lift_height)
                    | transfer_done
                    | (self.phase == PLACE) & in_tray & quiet & released
                    | (self.phase == SETTLE) & in_tray & quiet & released)
        self.hold.copy_(torch.where(criteria & ~impeded & ~fail, self.hold + 1, 0))
        direct_success = (self.phase == LIFT) & ~self.deposit
        cutoff = torch.where(direct_success, round(MIN_STABLE_HOLD_S / self.dt),
                             torch.where(self.phase == SETTLE, 9, 3))
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
        if self.reward_lift:
            reward += .12 * lift_progress
        return torch.nan_to_num(reward, nan=-8.), success, fail, dropped, impeded, lift
