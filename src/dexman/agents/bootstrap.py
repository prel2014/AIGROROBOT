"""Fresh-teacher imitation from physical calibration, followed by a closed-loop gate.

This module uses the already running environment. Scripted successes remain separate
from learned PPO outcomes, and a resumed teacher never calls this bootstrap.
"""
import copy
import json
import math
from pathlib import Path
from types import MethodType

import torch
from rsl_rl.modules import EmpiricalNormalization
from tensordict import TensorDict

from dexman.agents.models import teacher_architecture
from dexman.experiments import random_state, restore_random_state, save_training_checkpoint, sha256, write_json
from dexman.paths import ROOT
from dexman.robot.control import encode
from dexman.tasks.lift.rewards import cube_clearance


LIFT_REFERENCE_BIAS_M = .012


def _require_start_stage_success(summary, stage_index, height_m, condition):
    """Gate the configured PPO starting stage, even when lower stages succeeded."""
    row = next((item for item in summary["successes_by_height"].values()
                if item["height_index"] == stage_index), None)
    if row is None or row["successes"] < 1:
        raise RuntimeError(f"Bootstrapped actor achieved zero {condition} successes at requested starting stage "
                           f"{stage_index} ({100 * height_m:g} cm); PPO was not started")
    return dict(row)


def _calibrate_exploration(actor, scale):
    """Set and report the exact PPO Gaussian used by every learned calibration."""
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Bootstrap exploration_scale must be finite and in (0,1]")
    distribution = actor.distribution
    before = distribution.log_std_param.detach().exp().cpu().tolist()
    with torch.no_grad():
        distribution.log_std_param.fill_(math.log(.04))
        distribution.log_std_param[:8].fill_(math.log(.02))
        base = distribution.log_std_param.detach().exp().cpu().tolist()
        distribution.log_std_param.add_(math.log(scale)).clamp_(*distribution.log_std_range)
    return {"phase": "after_behavior_cloning_before_closed_loop_calibration",
            "before_std": before, "base_std": base, "scale": scale,
            "after_std": distribution.log_std_param.detach().exp().cpu().tolist(),
            "std_range": list(distribution.std_range)}


class StochasticCalibrationActor(torch.nn.Module):
    """Use the same actor and its PPO Gaussian without changing any parameters."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observations):
        return self.actor(observations, stochastic_output=True).clamp(-1., 1.)


def supervised_step(step):
    """Discard waits and zero-progress boundaries for the state-only actor."""
    return 31 <= step <= 89 or 121 <= step <= 209 or step >= 210


def split_successful_replicas(height_index, successful, seed):
    """Hold out one successful whole replica per height; no frame-level leakage."""
    height_index, successful = height_index.cpu(), successful.cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_ids, validation_ids = [], []
    for level in range(5):
        candidates = ((height_index == level) & successful).nonzero(as_tuple=False).flatten()
        if len(candidates) < 2:
            raise ValueError(f"Height {level} needs two successful expert replicas for separate train/validation; "
                             f"observed {len(candidates)}")
        shuffled = candidates[torch.randperm(len(candidates), generator=generator)]
        validation_ids.append(int(shuffled[0]))
        train_ids.extend(shuffled[1:].tolist())
    return sorted(train_ids), sorted(validation_ids)


def _prepare_calibration(raw, heights):
    raw.episode_length_buf.zero_()
    raw.reset()
    raw.last_success.zero_()
    raw.last_failure.zero_()
    raw.goal[:, 2] = raw.object.data.root_pos_w.torch[:, 2] + heights


def _calibration_rollout(raw, heights, height_index, seed, actor=None):
    """Capture pre-step states/actions; score each replica only until its first exit."""
    _prepare_calibration(raw, heights)
    device = raw.device
    ids = torch.arange(raw.num_envs, device=device)
    alive = torch.ones(raw.num_envs, dtype=torch.bool, device=device)
    successful = torch.zeros_like(alive)
    failed = torch.zeros_like(alive)
    max_lift = torch.zeros(raw.num_envs, device=device)
    max_clearance = torch.full_like(max_lift, -float("inf"))
    first_exit = torch.full_like(ids, -1)
    states, commands, replicas, usable = [], [], [], []
    start = raw.targets.clone()
    closed = raw.action_center[None].expand(raw.num_envs, -1).clone()
    arm_delta = torch.tensor(seed["lift_arm_q"], device=device) - torch.tensor(seed["arm_q"], device=device)
    # Allow 6 mm of calibrated settling plus 6 mm of positive height margin.
    # The environment still requires the actual stage lift and sustained hold.
    lift_arm = closed[:, :6] + arm_delta[None] * ((heights + LIFT_REFERENCE_BIAS_M) / .15)[:, None]
    if actor is not None:
        actor.eval()
    with torch.no_grad():
        for step in range(300):
            observation = raw.teacher_observations()["policy"].detach().clone()
            if actor is None:
                desired = start.lerp(closed, min(max((step - 30) / 60, 0.), 1.))
                if step >= 120:
                    desired[:, :6] = closed[:, :6].lerp(lift_arm, min((step - 120) / 90, 1.))
                action = encode(desired, raw.limits, raw.action_center)
                if supervised_step(step):
                    states.append(observation)
                    commands.append(action.detach().clone())
                    replicas.append(ids.clone())
                    usable.append(alive.clone())
            else:
                action = actor(TensorDict({"policy": observation}, batch_size=[raw.num_envs])).clamp(-1., 1.)
            if not torch.isfinite(observation).all() or not torch.isfinite(action).all():
                raise RuntimeError("Non-finite calibration observation/action")
            _, _, terminated, timeout, extras = raw.step(action)
            position = raw.object.data.root_pos_w.torch
            local_height = position[:, 2] - raw.scene.env_origins[:, 2]
            clearance = cube_clearance(local_height, raw.object.data.root_quat_w.torch, raw.cfg.table_height)
            if not torch.isfinite(position).all() or not torch.isfinite(clearance).all():
                raise RuntimeError("Non-finite physical calibration state")
            max_lift = torch.maximum(max_lift, torch.where(alive, extras["episode_max_lift_m"], max_lift))
            max_clearance = torch.maximum(max_clearance, torch.where(alive, clearance, max_clearance))
            success_now = extras["success"].bool() & alive
            failure_now = raw.last_failure.bool() & alive
            exit_now = alive & (terminated.bool() | timeout.bool())
            successful |= success_now
            failed |= failure_now
            first_exit[exit_now] = step
            alive &= ~exit_now
    summary = {"source": "scripted_calibration" if actor is None else "learned_policy_calibration",
               "steps": 300, "successes_by_height": {}, "success_by_replica": successful.cpu().tolist(),
               "failure_by_replica": failed.cpu().tolist(), "first_exit_step_by_replica": first_exit.cpu().tolist(),
               "max_lift_m_by_replica": max_lift.cpu().tolist(),
               "max_clearance_m_by_replica": max_clearance.cpu().tolist()}
    for level in range(5):
        mask = height_index == level
        if not mask.any():
            continue
        summary["successes_by_height"][str(float(heights[mask][0]))] = {
            "height_index": level, "replicas": int(mask.sum()), "successes": int(successful[mask].sum()),
            "failures": int(failed[mask].sum()), "max_lift_m": float(max_lift[mask].max()),
            "max_clearance_m": float(max_clearance[mask].max())}
    if actor is not None:
        return summary, None
    replica = torch.cat(replicas)
    keep = torch.cat(usable) & successful[replica]
    data = {"observations": torch.cat(states)[keep].clone(), "actions": torch.cat(commands)[keep].clone(),
            "replica": replica[keep].clone(), "successful": successful.clone()}
    return summary, data


def _calibration_sweep(raw, grasp_seed, actor=None, source=None, action_mode=None):
    """Cover at least two episodes per height using the existing replica count."""
    batches = math.ceil(10 / raw.num_envs)
    total = {"source": source or ("scripted_calibration" if actor is None else "learned_policy_calibration"),
             "action_mode": action_mode or ("scripted" if actor is None else "deterministic"),
             "object_xy_range": float(raw.cfg.object_xy_range),
             "steps": 300, "batches": batches, "physical_replicas": raw.num_envs,
             "episodes": batches * raw.num_envs, "successes_by_height": {}}
    sequence_keys = ("success_by_replica", "failure_by_replica", "first_exit_step_by_replica",
                     "max_lift_m_by_replica", "max_clearance_m_by_replica")
    for key in sequence_keys:
        total[key] = []
    parts, height_parts = [], []
    stage_heights = torch.tensor(raw.cfg.curriculum_heights, device=raw.device)
    for batch in range(batches):
        height_index = (torch.arange(raw.num_envs, device=raw.device) + batch * raw.num_envs) % 5
        summary, data = _calibration_rollout(raw, stage_heights[height_index], height_index, grasp_seed, actor)
        height_parts.append(height_index)
        for key in sequence_keys:
            total[key].extend(summary[key])
        for key, row in summary["successes_by_height"].items():
            if key not in total["successes_by_height"]:
                total["successes_by_height"][key] = dict(row)
            else:
                combined = total["successes_by_height"][key]
                for field in ("replicas", "successes", "failures"):
                    combined[field] += row[field]
                for field in ("max_lift_m", "max_clearance_m"):
                    combined[field] = max(combined[field], row[field])
        if data is not None:
            data["replica"] += batch * raw.num_envs
            parts.append(data)
    total["height_index_by_episode"] = torch.cat(height_parts).cpu().tolist()
    total["replica_id_note"] = "Global episode ID = batch * physical_replicas + local replica; split keeps whole episodes."
    data = ({key: torch.cat([part[key] for part in parts]) for key in parts[0]} if parts else None)
    return total, data, torch.cat(height_parts)


def _weighted_loss(actor, observations, actions, weights):
    prediction = actor(TensorDict({"policy": observations}, batch_size=[len(observations)]))
    return ((prediction - actions).square() * weights).mean()


def _validation_loss(actor, observations, actions, weights, batch_size):
    actor.eval()
    total = 0.
    with torch.no_grad():
        for offset in range(0, len(observations), batch_size):
            x, y = observations[offset:offset + batch_size], actions[offset:offset + batch_size]
            total += float(_weighted_loss(actor, x, y, weights)) * len(x)
    return total / len(observations)


def bootstrap_teacher(runner, output, updates=600, batch_size=512, learning_rate=5e-4, seed=42,
                      exploration_scale=1.):
    """Fit only the fresh actor; restore the production environment before PPO starts."""
    if updates < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Bootstrap updates/batch size/learning rate must be positive")
    if not math.isfinite(exploration_scale) or not 0 < exploration_scale <= 1:
        raise ValueError("Bootstrap exploration_scale must be finite and in (0,1]")
    if runner.alg.optimizer.state:
        raise ValueError("Bootstrap is only valid before the first PPO optimizer update")
    raw, actor, critic = runner.raw, runner.alg.actor, runner.alg.critic
    if raw.num_envs < 1 or len(raw.cfg.curriculum_heights) != 5:
        raise ValueError("Five-height bootstrap requires a positive replica count and five heights")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "bootstrap.json"
    seed_path = ROOT / "configs/grasp_seed.json"
    grasp_seed = json.loads(seed_path.read_text())
    learning_before = copy.deepcopy(raw.learning_state())
    start_stage = learning_before["curriculum"]["level"]
    if type(start_stage) is not int or not 0 <= start_stage < len(raw.cfg.curriculum_heights):
        raise ValueError("Invalid requested bootstrap starting curriculum stage")
    start_height = float(raw.cfg.curriculum_heights[start_stage])
    report = {"version": 1, "source": "scripted_calibration", "status": "collecting_expert",
              "seed": seed, "seed_file_sha256": sha256(seed_path), "updates": updates,
              "batch_size": batch_size, "learning_rate": learning_rate, "optimizer": "AdamW",
              "weight_decay": 1e-4, "action_loss_weights": [4.] * 6 + [2.] * 2 + [1.] * 18,
              "fixed_object_pose": grasp_seed["cube_local"], "object_xy_range": 0.,
              "heights_m": list(raw.cfg.curriculum_heights),
              "lift_reference_bias_m": LIFT_REFERENCE_BIAS_M,
              "expert_reference_lifts_m": [height + LIFT_REFERENCE_BIAS_M for height in raw.cfg.curriculum_heights],
              "requested_start_stage": {"height_index": start_stage, "height_m": start_height,
                                        "minimum_successes_per_gate": 1},
              "start_stage_gates": {},
              "schedule": "open30/close60/settle30/lift90/hold90; retain31..89,121..209,210..first-exit",
              "normalization_source": "training replicas only",
              "scope": "calibration on a fixed cube pose; not evidence of generalization or PPO successes",
              "loss_history": []}
    write_json(report_path, report)
    config_before = {key: getattr(raw.cfg, key) for key in ("curriculum_enabled", "object_xy_range", "episode_length_s")}
    rng_before = random_state()
    counter_before = raw.common_step_counter
    reset_override_before = raw.__dict__.get("_reset_envs_from_buffer")
    had_reset_override = "_reset_envs_from_buffer" in raw.__dict__
    actor_mode, critic_mode = actor.training, critic.training
    distribution_before = copy.deepcopy(actor.distribution.state_dict())

    def no_automatic_reset(self):
        return torch.empty(0, device=self.device, dtype=torch.int32)

    raw.cfg.curriculum_enabled = False
    raw.cfg.object_xy_range = 0.
    raw.cfg.episode_length_s = 30.
    raw._reset_envs_from_buffer = MethodType(no_automatic_reset, raw)
    try:
        expert, data, height_index = _calibration_sweep(raw, grasp_seed)
        report["expert"] = expert
        if any(row["successes"] < 1 for row in expert["successes_by_height"].values()):
            raise RuntimeError("Expert calibration did not succeed at all five heights; no behavior-cloning updates performed")
        train_ids, validation_ids = split_successful_replicas(height_index, data["successful"], seed)
        train_mask = torch.isin(data["replica"], torch.tensor(train_ids, device=raw.device))
        validation_mask = torch.isin(data["replica"], torch.tensor(validation_ids, device=raw.device))
        train_x, train_y = data["observations"][train_mask], data["actions"][train_mask]
        validation_x, validation_y = data["observations"][validation_mask], data["actions"][validation_mask]
        if min(len(train_x), len(validation_x)) == 0:
            raise RuntimeError("No valid calibration samples in the training or held-out replica partition")
        report.update(status="fitting_actor", samples=len(data["observations"]), train_samples=len(train_x),
                      validation_samples=len(validation_x), train_replica_ids=train_ids,
                      validation_replica_ids=validation_ids)
        for model in (actor, critic):
            model.obs_normalizer = EmpiricalNormalization(model.obs_dim).to(raw.device)
            model.obs_normalizer.train()
            with torch.no_grad():
                model.obs_normalizer.update(train_x)
        parameters = [parameter for name, parameter in actor.named_parameters()
                      if parameter.requires_grad and not name.startswith("distribution.")]
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
        generator = torch.Generator(device=raw.device).manual_seed(seed)
        torch.manual_seed(seed)
        weights = train_y.new_tensor(report["action_loss_weights"])
        best_validation = _validation_loss(actor, validation_x, validation_y, weights, batch_size)
        initial_validation = best_validation
        best_state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
        best_update = 0
        for update in range(1, updates + 1):
            actor.train()
            indices = torch.randint(len(train_x), (batch_size,), generator=generator, device=raw.device)
            optimizer.zero_grad(set_to_none=True)
            loss = _weighted_loss(actor, train_x[indices], train_y[indices], weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite behavior-cloning loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            if update % 100 == 0 or update == updates:
                validation = _validation_loss(actor, validation_x, validation_y, weights, batch_size)
                if not math.isfinite(validation):
                    raise RuntimeError("Non-finite held-out behavior-cloning loss")
                row = {"update": update, "train_minibatch_weighted_mse": float(loss.detach()),
                       "validation_weighted_mse": validation}
                report["loss_history"].append(row)
                if validation < best_validation:
                    best_validation, best_update = validation, update
                    best_state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
                write_json(report_path, report)
                print(f"Teacher bootstrap {update}/{updates}: train={row['train_minibatch_weighted_mse']:.6f}; "
                      f"held-out={validation:.6f}", flush=True)
        actor.load_state_dict(best_state, strict=True)
        actor.zero_grad(set_to_none=True)
        if any(not torch.equal(value, distribution_before[name])
               for name, value in actor.distribution.state_dict().items()):
            raise RuntimeError("Behavior cloning changed the exploration distribution")
        if runner.alg.optimizer.state:
            raise RuntimeError("Behavior cloning populated PPO optimizer state")
        # BC leaves exploration untouched. Calibrate the production Gaussian before
        # exporting the actor and running either deterministic or stochastic gates.
        report["exploration_calibration"] = _calibrate_exploration(actor, exploration_scale)
        # The exported best actor must include the calibrated Gaussian, not the pre-calibration BC snapshot.
        best_state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
        report.update(status="checking_learned_policy", best_update=best_update,
                      initial_validation_weighted_mse=initial_validation,
                      best_validation_weighted_mse=best_validation)
        save_training_checkpoint(output / "bootstrapped_actor.pt", {
            "format": "dexman_bootstrapped_actor_v1", "source": "scripted_calibration",
            "architecture": teacher_architecture(), "actor_state_dict": best_state,
            "best_update": best_update, "validation_weighted_mse": best_validation,
            "note": "Actor calibration snapshot; not a PPO resume checkpoint."})
        write_json(report_path, report)
        raw.cfg.object_xy_range = 0.
        learned, _, _ = _calibration_sweep(raw, grasp_seed, actor=actor,
                                          source="learned_policy_calibration", action_mode="deterministic")
        report["learned_policy_calibration"] = learned
        report["start_stage_gates"]["deterministic"] = _require_start_stage_success(
            learned, start_stage, start_height, "learned deterministic closed-loop")
        report["status"] = "checking_stochastic_policy"
        write_json(report_path, report)
        stochastic_actor = StochasticCalibrationActor(actor).eval()
        calibration_rng = random_state()
        stochastic, _, _ = _calibration_sweep(raw, grasp_seed, actor=stochastic_actor,
            source="learned_stochastic_calibration", action_mode="stochastic")
        report["learned_stochastic_calibration"] = stochastic
        report["status"] = "checking_production_policy"
        write_json(report_path, report)
        raw.cfg.object_xy_range = config_before["object_xy_range"]
        # Match the random stream: only the reset's XY range differs between these two stochastic checks.
        restore_random_state(calibration_rng)
        production, _, _ = _calibration_sweep(raw, grasp_seed, actor=stochastic_actor,
            source="learned_production_calibration", action_mode="stochastic")
        report["learned_production_calibration"] = production
        report["stochastic_comparison_rng"] = "common initial RNG state for fixed-XY and production-XY sweeps"
        report["start_stage_gates"]["production_stochastic"] = _require_start_stage_success(
            production, start_stage, start_height,
            f"stochastic production-condition (XY range {raw.cfg.object_xy_range:g} m)")
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        write_json(report_path, report)
        raise
    finally:
        for key, value in config_before.items():
            setattr(raw.cfg, key, value)
        if had_reset_override:
            raw._reset_envs_from_buffer = reset_override_before
        else:
            del raw._reset_envs_from_buffer
        raw.restore_learning_state(learning_before)
        raw.episode_length_buf.zero_()
        raw.last_success.zero_()
        raw.last_failure.zero_()
        restore_random_state(rng_before)
        runner.env.reset()
        raw.restore_learning_state(learning_before)
        raw.common_step_counter = counter_before
        for key in ("episode_length_buf", "last_success", "last_failure", "episode_max_lift",
                    "hold_steps", "reset_terminated", "reset_time_outs", "reset_buf", "reward_buf"):
            getattr(raw, key).zero_()
        raw.extras.clear()
        actor.train(actor_mode)
        critic.train(critic_mode)
        actor.reset()
        critic.reset()
    report["ppo_initial_learning_state"] = raw.learning_state()
    report["ppo_optimizer_states"] = len(runner.alg.optimizer.state)
    report["ppo_episode_steps_max"] = int(raw.episode_length_buf.max())
    report["ppo_episode_max_lift_m"] = float(raw.episode_max_lift.max())
    write_json(report_path, report)
    runner.bootstrap_report = report
    return report
