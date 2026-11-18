"""Target contact measurements shared by scripted checks and PPO diagnostics."""
from dataclasses import dataclass

import torch

from dexman.sensors.grasp import opposed_contact_grasp


MAX_HOLD_SLIP_M_S = .08
MAX_HOLD_LINEAR_SPEED_M_S = .065
MIN_STABLE_HOLD_S = .5


@dataclass(frozen=True)
class ContactQuality:
    """Per-hand results with fingertip channels in canonical FINGERTIPS order."""

    held: torch.Tensor
    touching: torch.Tensor
    opposed: torch.Tensor
    normal_force_n: torch.Tensor


def measure_contact_quality(forces_w, points_w, object_pos_w, *, precision=False):
    """Score filtered normal forces and contact points in world axes.

    Inputs have shapes [N,5,3], [N,5,3], [N,3]. The thumb is the last
    fingertip. A precision grasp uses the index and thumb, and the other grasps
    require the thumb and two fingers, including at least one opposed pair.
    """
    minimum_force = .003 if precision else .04
    separation = .003 if precision else .015
    fingers = (0, -1) if precision else slice(None)
    opposed, active, measured_force = opposed_contact_grasp(
        forces_w[:, fingers], points_w[:, fingers], object_pos_w,
        minimum_force, minimum_separation=separation, required_fingers=1,
    )
    if precision:
        measured_force = forces_w.norm(dim=-1)
        touching = torch.isfinite(measured_force) & (measured_force >= minimum_force)
        held = opposed
    else:
        touching = active
        held = opposed & (active[:, :-1].sum(-1) >= 2)

    normal = torch.where(torch.isfinite(measured_force), measured_force, 0.)
    return ContactQuality(held, touching, opposed, normal)


def steady_contacts(slip_speed, touching):
    """Require finite, low tangential speed for every loaded fingertip."""
    return (~touching | (slip_speed < MAX_HOLD_SLIP_M_S)).all(-1)


def opposing_tip_proximity(tips_w, center_w, precision):
    """Reward a nearby thumb and eligible finger on opposing sides of the fruit."""
    offsets = tips_w - center_w[:, None, None]
    distance = offsets.norm(dim=-1)
    side = offsets / distance.clamp_min(1.e-6).unsqueeze(-1)
    opposite = (-(side[:, :, :-1] * side[:, :, -1:]).sum(-1)).clamp(0., 1.)
    approach = 1. - torch.tanh(torch.maximum(distance[:, :, :-1], distance[:, :, -1:]) / .09)
    pairs = approach * (.5 + .5 * opposite)
    return torch.where(precision[:, None], pairs[:, :, 0], pairs.amax(-1))


def contact_slip_speed(points_w, forces_w, tip_com_pos_w, tip_com_vel_w, object_com_pos_w,
                       object_com_vel_w, *, minimum_force=.003):
    """Measure relative tangential velocity at the reported contact point (m/s).

    Tip data have shapes [N,F,3] and [N,F,6]. Object data are [N,3] and
    [N,6]. Positions and linear velocities refer to the matching centers of
    mass, with linear velocity followed by angular velocity, all in world axes.
    Missing or invalid contacts produce NaN, so averaging can exclude them.
    """
    normal_force = forces_w.norm(dim=-1)
    normal_hat = forces_w / normal_force.clamp_min(1e-8).unsqueeze(-1)
    tip_linear, tip_angular = tip_com_vel_w.split(3, dim=-1)
    object_linear, object_angular = object_com_vel_w[:, None].split(3, dim=-1)
    tip_velocity = tip_linear + torch.cross(tip_angular, points_w - tip_com_pos_w, dim=-1)
    object_velocity = object_linear + torch.cross(
        object_angular.expand_as(points_w), points_w - object_com_pos_w[:, None], dim=-1,
    )
    relative = tip_velocity - object_velocity
    tangent = relative - (relative * normal_hat).sum(-1, keepdim=True) * normal_hat
    speed = tangent.norm(dim=-1)
    valid = (normal_force >= minimum_force) & torch.isfinite(normal_force) & torch.isfinite(speed)
    return torch.where(valid, speed, torch.nan)


def grasp_progress_masks(near, touching, opposed, held, *, eligible_hands=None):
    """Classify each replica by its deepest achieved instantaneous grasp state.

    near, opposed and held are [N,A], touching is [N,A,F]. Supply a per-hand
    eligibility mask for one-hand phases such as the right-hand banana pickup.
    NEAR threshold selection stays in the caller so it can depend on fruit size.
    """
    allowed = torch.ones_like(near) if eligible_hands is None else eligible_hands
    hold = (held & allowed).any(-1)
    pair = (opposed & allowed).any(-1)
    contact = (touching.any(-1) & allowed).any(-1)
    within_reach = (near & allowed).any(-1)
    return {
        "unreached": ~within_reach & ~contact,
        "near_without_contact": within_reach & ~contact,
        "contact_without_opposition": contact & ~pair,
        "opposition_without_grasp": pair & ~hold,
        "held": hold,
    }
