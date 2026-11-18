"""Measured, object-filtered contact checks; no distance-only grasp acceptance."""
import torch

from dexman.robot.joints import FINGERTIPS


def opposed_contact_grasp(forces, points, object_pos, minimum_force=.05, *,
                          minimum_separation=.015, required_fingers=2):
    """Require a thumb and the specified number of fingers with an opposing pair.

    Forces act on the fingers; filter index zero must be the target object.
    Opposing contacts are a necessary check, not a proof of full force closure.
    """
    magnitude = forces.norm(dim=-1)
    finite = torch.isfinite(forces).all(-1) & torch.isfinite(points).all(-1)
    active = finite & (magnitude >= minimum_force)
    normals = forces / magnitude.clamp_min(1e-8).unsqueeze(-1)
    radial = points - object_pos[:, None]
    radial = radial / radial.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    normal_opposed = (normals[:, :-1] * normals[:, -1:]).sum(-1) < -.5
    position_opposed = (radial[:, :-1] * radial[:, -1:]).sum(-1) < -.25
    separated = (points[:, :-1] - points[:, -1:]).norm(dim=-1) >= minimum_separation
    pair = active[:, :-1] & normal_opposed & position_opposed & separated
    valid = active[:, -1] & (active[:, :-1].sum(-1) >= required_fingers) & pair.any(-1)
    return valid, active, magnitude


def forbidden_contact(env):
    """Measured finger/target contact with barriers or non-target objects.

    This is not a full-arm collision monitor. Other robot links still collide
    through PhysX, but are not included in this measured task signal.
    """
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in FINGERTIPS:
        matrix = env.scene[name].data.normal_force_matrix_w
        if matrix is not None and matrix.torch.shape[2] > 1:
            result |= (matrix.torch[:, 0, 1:].norm(dim=-1) >= env.cfg.minimum_contact_force_n).any(-1)
    if "target_obstacle_contact" in env.scene.sensors:
        matrix = env.scene["target_obstacle_contact"].data.normal_force_matrix_w
        if matrix is not None:
            result |= (matrix.torch[:, 0].norm(dim=-1) >= env.cfg.minimum_contact_force_n).any(-1)
    return result


def measured_grasp(env):
    forces, points = [], []
    for name in FINGERTIPS:
        sensor = env.scene[name].data
        if sensor.normal_force_matrix_w is None or sensor.contact_pos_w is None:
            raise RuntimeError("Grasp scoring requires object-filtered forces and tracked contact points")
        forces.append(sensor.normal_force_matrix_w.torch[:, 0, 0])
        points.append(sensor.contact_pos_w.torch[:, 0, 0])
    forces, points = torch.stack(forces, 1), torch.stack(points, 1)
    valid, active, magnitude = opposed_contact_grasp(
        forces, points, env.object.data.root_pos_w.torch, env.cfg.minimum_contact_force_n)
    return valid, active, magnitude
