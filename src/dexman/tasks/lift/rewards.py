"""Dense approach/grasp/lift shaping, with sustained success checked separately."""
import torch


FAILURE_PENALTY = 10.0


def grasp_metrics(object_pos, palm_pos, fingertip_pos):
    palm_distance = (object_pos - palm_pos).norm(dim=-1)
    distances = (fingertip_pos - object_pos[:, None, :]).norm(dim=-1)
    # Thumb + at least two other fingertips must enclose the lifted cube.
    grasp = (distances[:, -1] < 0.09) & ((distances[:, :-1] < 0.09).sum(-1) >= 2)
    return palm_distance, distances.mean(-1), grasp


def lift_reward(palm_distance, finger_distance, grasp, height, goal_distance, actions, previous,
                initial_height=0.625, target_lift=0.15):
    approach = 1.0 - torch.tanh(4.0 * palm_distance)
    fingers = 1.0 - torch.tanh(8.0 * finger_distance)
    lifted = ((height - initial_height) / target_lift).clamp(0, 1) * grasp.float()
    goal = (1.0 - torch.tanh(8.0 * goal_distance)) * grasp.float() * lifted
    smoothness = (actions - previous).square().mean(-1)
    return 1.5 * approach + fingers + 5.0 * lifted + 4.0 * goal - 0.05 * smoothness


def lift_potential(palm_distance, tip_distances, height, xy_error, target_lift,
                   initial_height=0.625, clearance=None):
    """Continuous progress signal; fingertip proximity is a proxy, not a contact sensor."""
    fingers = 1 - torch.tanh(8 * tip_distances.mean(-1))
    thumb = 1 - torch.tanh(12 * tip_distances[..., -1])
    others = 1 - torch.tanh(12 * tip_distances[..., :-1].topk(2, largest=False).values.mean(-1))
    soft_grip = thumb * others
    lifted = ((height - initial_height) / target_lift).clamp(0, 1)
    if clearance is not None:
        lifted = lifted * (clearance / .01).clamp(0, 1)
    potential = 3 * (1 - torch.tanh(4 * palm_distance)) + 2 * fingers + 10 * lifted * soft_grip
    potential -= 4 * xy_error.clamp(max=1.0)
    return potential, soft_grip, lifted


def progress_reward(potential, previous_potential, success, failure, actions, previous_actions,
                    xy_error, soft_grip, lifted, dt=1 / 30, gamma=0.99):
    """Potential shaping with zero terminal potential, time cost and a terminal success bonus.

    Being stationary near the cube earns no positive approach reward. Timeout is
    a truncation and must not be passed as failure: PPO bootstraps its final value.
    """
    terminal = success | failure
    next_potential = torch.where(terminal, torch.zeros_like(potential), potential)
    shaping = gamma * next_potential - previous_potential
    smoothness = (actions - previous_actions).square().mean(-1)
    return (shaping - 0.01 - 0.05 * smoothness - dt * xy_error.clamp(max=1.0)
            + dt * 4 * soft_grip * lifted + 100 * success.float() - FAILURE_PENALTY * failure.float())


def cube_clearance(center_height, quaternion_xyzw, table_height=.6, half_size=.025):
    """Lowest cube vertex above the table, accounting for orientation (no tipping successes)."""
    x, y, z, w = quaternion_xyzw.unbind(-1)
    row_z = torch.stack((2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2*(x*x + y*y)), -1)
    return center_height - half_size * row_z.abs().sum(-1) - table_height
