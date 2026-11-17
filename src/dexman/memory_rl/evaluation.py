"""Evaluate a recurrent actor without constructing or restoring an optimizer."""
import torch
from dexman.agents.recurrent import make_actor
from dexman.tasks.manipulation.catalog import TASKS


def load_actor(saved, contract, device):
    actor = make_actor(contract["feature_dimension"], contract["training_config"]["actor"], device)
    actor.load_state_dict(saved["actor_state_dict"])
    return actor.eval()


@torch.inference_mode()
def evaluate(env, actor, episodes, max_steps):
    if min(episodes, max_steps) < 1:
        raise ValueError("episodes and max_steps must be positive")
    actor.reset()
    observations = env.get_observations()
    records = []
    for _ in range(max_steps):
        observations, _, dones, _ = env.step(actor(observations))
        actor.reset(dones)
        records.extend(env.records)
        if len(records) >= episodes:
            break
    by_task = {}
    for task in TASKS:
        rows = [row for row in records if row["task"] == task.name]
        by_task[task.name] = {"episodes": len(rows), "success_rate":
                             sum(row["success"] for row in rows) / len(rows) if rows else None}
    return {"completed": len(records), "requested": episodes, "by_task": by_task}
