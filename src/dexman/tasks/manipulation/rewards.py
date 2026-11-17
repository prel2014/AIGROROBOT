"""Pure tensor task metrics and rewards; no simulator dependencies."""
import torch
from .catalog import TASKS


def task_metrics(task_ids, positions, goals, palms, tips, speeds, clearance, violated, initial_height=.625, contact_grasp=None):
    """Return potential, instantaneous success condition and persistent push violation.

    Measured opposed contact is required for lift/transport acceptance.
    Push tasks must remain on the tabletop throughout the episode.
    """
    lifting = torch.tensor([task.kind != "push" for task in TASKS], device=task_ids.device)[task_ids]
    distance = (positions - goals).norm(dim=-1)
    tip_distance = (tips - positions[:, None]).norm(dim=-1)
    if contact_grasp is None:
        raise ValueError("Measured opposed contact is required for task scoring")
    grasp = contact_grasp
    violated = violated | (~lifting & ((clearance > .015) | (positions[:, 2] < initial_height - .03)))
    stable = speeds < .10
    supported = (clearance.abs() < .01) & ((positions[:, 2] - initial_height).abs() < .015)
    horizontal = (positions[:, :2] - goals[:, :2]).norm(dim=-1) < .025
    height_ok = (positions[:, 2] >= goals[:, 2]) & (positions[:, 2] <= goals[:, 2] + .025)
    success = stable & torch.where(lifting, horizontal & height_ok & grasp & (clearance > .005), (distance < .025) & supported)
    success &= ~violated
    approach = 1 - torch.tanh(4 * (positions - palms).norm(dim=-1))
    fingers = 1 - torch.tanh(8 * tip_distance.mean(-1))
    potential = 3 * approach + lifting.float() * 2 * fingers + 10 * (1 - torch.tanh(8 * distance))
    return potential, success, violated


def task_reward(potential, previous, terminal, success, failure, actions, previous_actions, gamma=.99):
    # Deadline is a finite task boundary: no value bootstrap after it.
    next_potential = torch.where(terminal, torch.zeros_like(potential), potential)
    return (gamma * next_potential - previous - .01
            - .05 * (actions - previous_actions).square().mean(-1)
            + 100 * success.float() - 10 * failure.float())
