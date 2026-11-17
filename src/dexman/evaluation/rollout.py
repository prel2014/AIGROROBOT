"""Fixed per-replica quotas prevent fast failures from dominating parallel evaluation."""
import time

import numpy as np
import torch

from .metrics import REASONS, summarize


@torch.inference_mode()
def evaluate_rollout(env, policy, episodes, max_steps, seed):
    if min(episodes, max_steps) < 1:
        raise ValueError("Episode and step budgets must be positive")
    raw = env.unwrapped
    quotas = np.full(raw.num_envs, episodes // raw.num_envs)
    quotas[:episodes % raw.num_envs] += 1
    completed = np.zeros(raw.num_envs, dtype=np.int64)
    lengths, returns = np.zeros(raw.num_envs, dtype=np.int64), np.zeros(raw.num_envs)
    obs, _ = env.reset(seed=seed)
    policy.reset(np.ones(raw.num_envs, dtype=bool))
    records, latencies = [], []
    started = time.perf_counter()
    for _ in range(max_steps):
        if torch.cuda.is_available() and str(raw.device).startswith("cuda"):
            torch.cuda.synchronize(raw.device)
        before = time.perf_counter()
        actions = policy.act(raw, obs)
        if torch.cuda.is_available() and str(raw.device).startswith("cuda"):
            torch.cuda.synchronize(raw.device)
        latencies.append(1000 * (time.perf_counter() - before))
        obs, rewards, terminated, truncated, extras = env.step(actions)
        dones = (terminated | truncated).cpu().numpy()
        lengths += 1
        returns += rewards.cpu().numpy()
        for index in np.flatnonzero(dones):
            if completed[index] < quotas[index]:
                success = bool(extras["success"][index])
                reason = int(extras["terminal_reason"][index])
                records.append({"seed": seed, "replica": int(index), "episode": int(completed[index]),
                    "task": "lift", "success": success, "reason": REASONS.get(reason, "unknown"),
                    "steps": int(lengths[index]), "return": float(returns[index]),
                    "max_lift_m": float(extras["episode_max_lift_m"][index])})
                completed[index] += 1
            lengths[index], returns[index] = 0, 0
        policy.reset(dones)
        if np.all(completed >= quotas):
            break
    result = summarize(records, episodes, latencies, time.perf_counter() - started)
    result.update(seed=seed, episodes=records, replica_quotas=quotas.tolist(),
                  inference_scope="observation preprocessing + policy + output conversion; excludes env.step")
    return result
