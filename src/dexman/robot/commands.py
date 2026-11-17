"""Map independent joint commands to bounded absolute angles; no learning backend."""
import torch


class JointCommands:
    """30 independent centered actions -> absolute radians before the environment filter."""
    def __init__(self, limits, center, device="cpu"):
        self.limits = torch.as_tensor(limits, dtype=torch.float32, device=device)
        self.center = torch.as_tensor(center, dtype=torch.float32, device=device)
        if (self.limits.shape != (30, 2) or self.center.shape != (30,)
                or not torch.isfinite(self.limits).all() or not torch.isfinite(self.center).all()
                or not (self.limits[:, 1] > self.limits[:, 0]).all()
                or not ((self.center >= self.limits[:, 0]) & (self.center <= self.limits[:, 1])).all()):
            raise ValueError("Expected finite limits [30,2] and center [30] within limits")

    def normalize(self, angles):
        return 2 * (angles - self.limits[:, 0]) / (self.limits[:, 1] - self.limits[:, 0]) - 1

    def angles(self, actions):
        if actions.ndim != 2 or actions.shape[-1] != 30 or not torch.isfinite(actions).all():
            raise ValueError("Expected finite actions [N,30]")
        actions = actions.clamp(-1, 1)
        span = torch.where(actions >= 0, self.limits[:, 1] - self.center, self.center - self.limits[:, 0])
        return self.center + actions * span
