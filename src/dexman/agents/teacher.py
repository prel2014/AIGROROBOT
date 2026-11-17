"""Transformer teacher inference and complete, architecture-checked PPO resume."""
import copy
import math

import torch
from tensordict import TensorDict

from dexman.agents.checkpoint import CHECKPOINT_VERSION, load_teacher_checkpoint
from dexman.agents.models import build_teacher_model, teacher_architecture
from dexman.experiments import random_state, restore_random_state, save_training_checkpoint, sha256
from dexman.paths import JOINT_MANIFEST, ROBOT_USD
from dexman.robot.joints import JOINT_NAMES


def _sync_teacher_learning_rates(runner):
    schedule = runner.training_schedule
    actor_lr = 0. if schedule["remaining"] else schedule["actor_lr"]
    rates = {"actor": actor_lr, "critic": schedule["critic_lr"]}
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = rates[group["name"]]
    runner.alg.learning_rate = actor_lr


def _policy_probe(runner):
    storage = getattr(runner.alg, "storage", None)
    observations = getattr(storage, "observations", None)
    source = observations["policy"] if observations is not None else runner.env.get_observations()["policy"]
    source = source.reshape(-1, 95)
    if not len(source):
        raise ValueError("Cannot guard an actor update without rollout observations")
    indices = torch.linspace(0, len(source) - 1, min(64, len(source)), device=source.device).long()
    return TensorDict({"policy": source[indices].detach().clone()}, batch_size=[len(indices)])


def _policy_mean_std(actor, observations):
    mean = actor(observations)
    distribution = actor.distribution
    std = distribution.log_std_param.clamp(*distribution.log_std_range).exp()
    return mean.detach().clone(), std.detach().clone()


def _install_teacher_update_guard(runner):
    """Wrap the installed RSL update; retain its PPO loop, storage and critic updates."""
    original_update = runner.alg.update

    def guarded_update():
        alg, schedule = runner.alg, runner.training_schedule
        actor = alg.get_policy()
        parameters = list(actor.parameters())
        warmup = schedule["remaining"] > 0
        _sync_teacher_learning_rates(runner)
        actor_lr_used = alg.learning_rate
        actor_state = actor_optimizer_state = observations = old_mean = old_std = None
        if not warmup:
            observations = _policy_probe(runner)
            with torch.no_grad():
                old_mean, old_std = _policy_mean_std(actor, observations)
            actor_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
            actor_optimizer_state = {parameter: copy.deepcopy(alg.optimizer.state.get(parameter))
                                     for parameter in parameters}
        previous_requires_grad = [parameter.requires_grad for parameter in parameters]
        try:
            if warmup:
                for parameter in parameters:
                    parameter.requires_grad_(False)
            losses = original_update()
        finally:
            for parameter, required in zip(parameters, previous_requires_grad):
                parameter.requires_grad_(required)
        proposed_kl, rejected, nonfinite = 0., False, False
        if warmup:
            schedule["remaining"] -= 1
        else:
            with torch.no_grad():
                new_mean, new_std = _policy_mean_std(actor, observations)
                kl = (torch.log(new_std / old_std)
                      + (old_std.square() + (old_mean - new_mean).square()) / (2 * new_std.square()) - .5)
                proposed_kl = float(kl.sum(dim=-1).mean())
            nonfinite = not math.isfinite(proposed_kl)
            rejected = nonfinite or proposed_kl > schedule["kl_limit"]
            if rejected:
                actor.load_state_dict(actor_state, strict=True)
                for parameter, saved_state in actor_optimizer_state.items():
                    if saved_state is None:
                        alg.optimizer.state.pop(parameter, None)
                    else:
                        alg.optimizer.state[parameter] = saved_state
                schedule["actor_lr"] = max(schedule["min_actor_lr"], schedule["actor_lr"] / 2)
                schedule["rejected"] += 1
            else:
                schedule["accepted"] += 1
            # PPO's last mini-batch distribution may describe the rejected parameters. Refresh without sampling.
            with torch.no_grad():
                actor.distribution.update(actor(observations))
        _sync_teacher_learning_rates(runner)
        losses.update(actor_lr=float(alg.learning_rate), actor_lr_used=float(actor_lr_used),
                      critic_lr=float(schedule["critic_lr"]), warmup_remaining=float(schedule["remaining"]),
                      actor_warmup_active=float(warmup), actor_update_rejected=float(rejected),
                      policy_kl_proposed=proposed_kl if not nonfinite else 1e30,
                      policy_kl_nonfinite=float(nonfinite))
        return losses

    runner.alg.update = guarded_update


def teacher_contract(cfg):
    return {"version": 1, "joint_names": list(JOINT_NAMES), "actions": 26,
            "encoding": "normalized_absolute_position_synergy_centered_v1",
            "arm_half_range": cfg.arm_half_range, "arm_center": list(cfg.arm_center),
            "action_center": list(cfg.action_center), "smoothing": cfg.action_smoothing,
            "physics_dt": cfg.sim.dt, "decimation": cfg.decimation, "reward_version": cfg.reward_version,
            "episode_length_s": cfg.episode_length_s, "success_hold_steps": cfg.success_hold_steps,
            "success_criterion": cfg.success_criterion,
            "success_lift_overshoot_m": cfg.success_lift_overshoot_m,
            "object_xy_range": cfg.object_xy_range, "gravity_compensation": cfg.gravity_compensation,
            "robot_sha256": sha256(ROBOT_USD), "joints_sha256": sha256(JOINT_MANIFEST)}


def load_teacher_actor(checkpoint, device="cpu"):
    saved = load_teacher_checkpoint(checkpoint)
    actor = build_teacher_model("actor")
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    return actor.to(device).eval()


def make_teacher_runner(env, config, log_dir, device):
    # Import training only when constructing a trainer; evaluation loads just the actor.
    from rsl_rl.runners import OnPolicyRunner

    class TeacherRunner(OnPolicyRunner):
        def __init__(self):
            self.raw = env.unwrapped
            self.contract = teacher_contract(self.raw.cfg)
            self.architecture = teacher_architecture()
            self.bootstrap_report = None
            for role in ("actor", "critic"):
                if config[role] != self.architecture[role]:
                    raise ValueError(f"Teacher {role} configuration differs from its architecture contract")
            super().__init__(env, copy.deepcopy(config), log_dir=log_dir, device=device)
            self.training_schedule = {"version": 1, "remaining": config["critic_warmup_updates"],
                "total": config["critic_warmup_updates"], "actor_lr": config["actor_learning_rate"],
                "critic_lr": config["critic_learning_rate"], "kl_limit": config["policy_kl_limit"],
                "min_actor_lr": config["minimum_actor_learning_rate"], "accepted": 0, "rejected": 0}
            original_optimizer = self.alg.optimizer
            if original_optimizer.state:
                raise ValueError("Teacher optimizer groups must be configured before any PPO updates")
            self.alg.optimizer = type(original_optimizer)([
                {"name": "actor", "params": list(self.alg.actor.parameters()), "lr": 0.},
                {"name": "critic", "params": list(self.alg.critic.parameters()), "lr": config["critic_learning_rate"]},
            ], **copy.deepcopy(original_optimizer.defaults))
            _sync_teacher_learning_rates(self)
            _install_teacher_update_guard(self)

        def save(self, path, infos=None):
            saved = self.alg.save()
            saved.update(iter=self.current_learning_iteration, infos={"dexman_teacher": {
                "checkpoint_version": CHECKPOINT_VERSION, "architecture": self.architecture,
                "contract": self.contract, "learning_state": self.raw.learning_state(),
                "training_config": copy.deepcopy(self.cfg),
                "bootstrap_report": copy.deepcopy(self.bootstrap_report),
                "training_schedule": copy.deepcopy(self.training_schedule),
                **random_state()}})
            save_training_checkpoint(path, saved)
            self.logger.save_model(path, self.current_learning_iteration)

        def resume(self, path):
            saved = load_teacher_checkpoint(path)
            info = saved["infos"]["dexman_teacher"]
            self.bootstrap_report = copy.deepcopy(info.get("bootstrap_report"))
            if info["contract"] != self.contract:
                raise ValueError("Teacher checkpoint control/physics/reward contract differs from this run")
            self.alg.load(saved, load_cfg=None, strict=True)
            self.training_schedule = copy.deepcopy(info["training_schedule"])
            _sync_teacher_learning_rates(self)
            self.raw.restore_learning_state(info["learning_state"])
            self.current_learning_iteration = saved["iter"] + 1
            # A new simulator starts new episodes at the restored curriculum stage.
            env.reset()
            restore_random_state(info)
            self.alg.actor.reset()
            self.alg.critic.reset()
            print(f"Resumed teacher from {path}; next iteration {self.current_learning_iteration}; "
                  f"curriculum level {self.raw.curriculum_level}; learning rate {self.alg.learning_rate}", flush=True)

    return TeacherRunner()
