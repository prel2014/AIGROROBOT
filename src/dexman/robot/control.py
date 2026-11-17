"""26 normalized absolute commands -> 30 bounded joint-position targets."""
import torch

from .joints import ARM, HAND_DIRECT, JOINT_NAMES


def joint_limit_violation(angles, lower, upper, tolerance=0.0):
    """Flag each robot whose state or command contains a nonfinite or out-of-range value."""
    return ((~torch.isfinite(angles)) | (angles < lower - tolerance) | (angles > upper + tolerance)).any(-1)


def action_limits(manifest, arm_half_range=1.2, arm_center=None):
    """Physical limits in the canonical 30-joint order, narrowed around task home for the arm."""
    from .joints import HOME
    arm_center = HOME if arm_center is None else dict(zip(ARM, arm_center))
    values = {joint["name"]: joint for joint in manifest}
    result = []
    for name in JOINT_NAMES:
        low, high = values[name]["lower_rad"], values[name]["upper_rad"]
        if name in ARM:
            low, high = max(low, arm_center[name] - arm_half_range), min(high, arm_center[name] + arm_half_range)
        result.append((low, high))
    return result


def decode(actions, limits, center=None):
    """Coupled distal/middle joints share equal fractions of their physical ranges.

    This is a position-synergy approximation, not the Shadow hardware tendon dynamics.
    The last 4 commands represent total middle+distal flexion (0 .. approximately pi).
    """
    if actions.shape[-1] != 26 or limits.shape[-2:] != (30, 2):
        raise ValueError("Expected actions[...,26] and limits[...,30,2]")
    if center is not None:
        a = actions.clamp(-1, 1)
        a = torch.cat((a[..., :22], a[..., 22:].repeat_interleave(2, dim=-1)), -1)
        return center + a * torch.where(a >= 0, limits[:, 1] - center, center - limits[:, 0])
    direct_count = len(ARM) + len(HAND_DIRECT)
    fractions = (actions.clamp(-1, 1) + 1) * 0.5
    coupled = fractions[..., direct_count:].repeat_interleave(2, dim=-1)
    fractions = torch.cat((fractions[..., :direct_count], coupled), dim=-1)
    return limits[..., 0] + fractions * (limits[..., 1] - limits[..., 0])


def encode(positions, limits, center=None):
    """Inverse normalized position map for resets and scripted validation."""
    if center is None:
        full = 2 * (positions - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1
    else:
        delta = positions - center
        span = torch.where(delta >= 0, limits[:, 1] - center, center - limits[:, 0]).clamp(min=1e-8)
        full = delta / span
    return torch.cat((full[..., :22], full[..., 22::2]), -1).clamp(-1, 1)
