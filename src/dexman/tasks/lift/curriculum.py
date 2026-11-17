"""Serializable lift progression, independent of the physics runtime."""
from dataclasses import dataclass


@dataclass
class LiftCurriculum:
    heights: tuple
    window: int
    threshold: float
    level: int = 0
    trials: int = 0
    successes: int = 0

    def __post_init__(self):
        if (not self.heights or any(h <= 0 for h in self.heights)
                or any(a >= b for a, b in zip(self.heights, self.heights[1:]))
                or self.window < 1 or not 0 < self.threshold <= 1
                or not 0 <= self.level < len(self.heights)
                or not 0 <= self.successes <= self.trials):
            raise ValueError("Invalid lift curriculum")

    def observe(self, trials, successes):
        if not 0 <= successes <= trials:
            raise ValueError("Success count must be between zero and trial count")
        self.trials += trials
        self.successes += successes
        if self.trials >= self.window:
            if self.successes / self.trials >= self.threshold:
                self.level = min(self.level + 1, len(self.heights) - 1)
            self.trials = self.successes = 0

    def state_dict(self):
        return {"version": 1, "heights": list(self.heights), "window": self.window, "threshold": self.threshold,
                "level": self.level, "trials": self.trials, "successes": self.successes}

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ("version", "heights", "window", "threshold"):
            if state.get(key) != expected[key]:
                raise ValueError(f"Curriculum configuration differs: {key}")
        values = [state[key] for key in ("level", "trials", "successes")]
        if any(type(value) is not int for value in values):
            raise ValueError("Curriculum counters must be integers")
        candidate = LiftCurriculum(self.heights, self.window, self.threshold, *values)
        self.level, self.trials, self.successes = candidate.level, candidate.trials, candidate.successes
