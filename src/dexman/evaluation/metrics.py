"""Episode-level rates, uncertainty and deployment budget measurements."""
import math
from collections import Counter

import numpy as np

REASONS = {1: "success", 2: "dropped", 3: "escaped", 4: "timeout", 5: "invalid_action", 6: "invalid_pose", 7: "forbidden_contact"}


def summarize(records, requested, latencies=(), elapsed_s=None):
    count = len(records)
    successes = sum(bool(row["success"]) for row in records)
    interval = None
    if count:
        # Wilson score interval, 95%; uncertainty is reported separately from acceptance.
        p, z = successes / count, 1.959963984540054
        divisor = 1 + z * z / count
        midpoint = (p + z * z / (2 * count)) / divisor
        radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / divisor
        interval = [max(0., midpoint - radius), min(1., midpoint + radius)]
    return {"completed": count, "requested": requested, "complete": count == requested,
            "successes": successes, "success_rate": successes / count if count else None,
            "success_rate_wilson95": interval, "terminal_reasons": dict(Counter(row["reason"] for row in records)),
            "mean_episode_steps": float(np.mean([row["steps"] for row in records])) if count else None,
            "max_lift_m": max((row.get("max_lift_m", 0.) for row in records), default=0.),
            "inference_ms": {"p50": float(np.percentile(latencies, 50)),
                             "p95": float(np.percentile(latencies, 95))} if len(latencies) else None,
            "elapsed_s": elapsed_s}
