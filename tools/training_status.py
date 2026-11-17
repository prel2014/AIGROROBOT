"""Read TensorBoard scalars from a running PPO job without changing it."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml


def config_integer(path, *keys):
    # Isaac Lab YAML includes Python tuple/object tags. Inspect only scalar nodes;
    # do not construct or execute any Python objects from the configuration.
    node = yaml.compose(path.read_text(encoding="utf-8"))
    for key in keys:
        node = next(value for name, value in node.value if name.value == key)
    return int(node.value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    total = config_integer(run / "params/agent.yaml", "max_iterations")
    rollout_steps = config_integer(run / "params/agent.yaml", "num_steps_per_env")
    robots = config_integer(run / "params/env.yaml", "scene", "num_envs")
    events = EventAccumulator(str(run), size_guidance={"scalars": 0})
    events.Reload()
    metrics = {}
    latest_step = 0
    latest_wall_time = 0
    for tag in events.Tags()["scalars"]:
        if tag.endswith("/time"):
            continue
        values = events.Scalars(tag)
        if not values:
            continue
        last = values[-1]
        latest_step = max(latest_step, last.step)
        latest_wall_time = max(latest_wall_time, last.wall_time)
        metrics[tag] = {"step": last.step, "first": values[0].value,
                        "latest": last.value, "max": max(v.value for v in values),
                        "mean_last_100": sum(v.value for v in values[-100:]) / len(values[-100:])}
    now = datetime.now(timezone.utc)
    status = {"run": str(run), "checked_at_utc": now.isoformat(), "latest_iteration_index": latest_step,
              "completed_iterations": latest_step + 1, "configured_iterations": total,
              "progress_percent": 100 * (latest_step + 1) / total, "robots": robots,
              "approximate_transitions": (latest_step + 1) * robots * rollout_steps,
              "seconds_since_last_event": now.timestamp() - latest_wall_time, "metrics": metrics,
              "note": "Metrics/success is the logged per-step success signal, not an episode success-rate estimate."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
