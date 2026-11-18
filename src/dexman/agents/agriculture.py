"""Bimanual state Transformer and versioned PPO checkpoints for agriculture."""
from dataclasses import asdict

import torch
from torch import nn
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.runners import OnPolicyRunner

from dexman.experiments import random_state, restore_random_state, save_training_checkpoint
from dexman.tasks.agriculture.interface import ACTION_DIM, ARM_KEYS, JOINT_ORDER, JOINTS_PER_ARM
from dexman.tasks.agriculture.objectives import (ARM_FEATURES, EPISODE_SECONDS, OBJECTIVES, OBS_DIM,
                                                 OBSTACLE_FEATURES, TASK_FEATURES)
from dexman.tasks.agriculture.randomization import OBSTACLE_RADIUS
from .models import RSL_RL_VERSION, StatelessTeacherModel


MODEL_CLASS = "dexman.agents.agriculture:BimanualTeacherModel"
GROUPS = {"actor": ["policy"], "critic": ["policy"]}


class BimanualTeacherModel(StatelessTeacherModel):
    """Attention over both arms, object geometry, task phase and obstacle tokens."""

    def __init__(self, obs, obs_groups, obs_set, output_dim):
        super().__init__()
        self.is_actor = obs_set == "actor"
        if (obs_groups.get(obs_set) != ["policy"] or obs["policy"].shape[-1] != OBS_DIM
                or output_dim != (ACTION_DIM if self.is_actor else 1)):
            raise ValueError("Incompatible agricultural PPO observation/action contract")
        self.obs_groups, self.obs_dim = ["policy"], OBS_DIM
        self.obs_normalization = self.update_observation_normalization = True
        self.obs_normalizer = EmpiricalNormalization(OBS_DIM)
        self.distribution = (GaussianDistribution(ACTION_DIM, init_std=.13, std_type="log",
                                                    std_range=[.025, .35]) if self.is_actor else None)
        width = 128
        self.joint_projection = nn.Linear(3, width)
        self.arm_projection = nn.Linear(ARM_FEATURES, width)
        self.task_projection = nn.Linear(TASK_FEATURES, width)
        self.obstacle_projection = nn.Linear(OBSTACLE_FEATURES, width)
        self.readout_count = len(ARM_KEYS) if self.is_actor else 1
        count = self.readout_count + ACTION_DIM + len(ARM_KEYS) + 1 + len(OBSTACLE_RADIUS)
        self.token_identity = nn.Parameter(torch.empty(count, width))
        nn.init.normal_(self.token_identity, std=.02)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(
            width, 4, 512, dropout=0., activation="gelu", batch_first=True, norm_first=True) for _ in range(3))
        self.output_norm = nn.LayerNorm(width)
        self.output_head = nn.Linear(width, JOINTS_PER_ARM if self.is_actor else 1)
        nn.init.orthogonal_(self.output_head.weight, gain=.01 if self.is_actor else 1.)
        nn.init.zeros_(self.output_head.bias)

    def get_latent(self, obs):
        state = self.obs_normalizer(obs["policy"]).clamp(-10., 10.)
        joint, arms, task, obstacles = state.split(
            (ACTION_DIM * 3, len(ARM_KEYS) * ARM_FEATURES, TASK_FEATURES, len(OBSTACLE_RADIUS) * OBSTACLE_FEATURES),
            dim=-1)
        tokens = torch.cat((
            self.token_identity.new_zeros((len(state), self.readout_count, self.token_identity.shape[-1])),
            self.joint_projection(joint.reshape(-1, ACTION_DIM, 3)),
            self.arm_projection(arms.reshape(-1, len(ARM_KEYS), ARM_FEATURES)),
            self.task_projection(task).unsqueeze(1),
            self.obstacle_projection(obstacles.reshape(-1, len(OBSTACLE_RADIUS), OBSTACLE_FEATURES)),
        ), dim=1) + self.token_identity
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens[:, :self.readout_count])


def contract():
    return {"schema": "dexman.agriculture.ppo.v1", "rsl_rl": RSL_RL_VERSION, "joint_order": list(JOINT_ORDER),
            "observations": OBS_DIM, "actions": ACTION_DIM, "model": MODEL_CLASS,
            "episode_seconds": EPISODE_SECONDS, "control_hz": 30,
            "task_definitions": {name: asdict(spec) for name, spec in OBJECTIVES.items()},
            "obstacle_order": list(OBSTACLE_RADIUS)}


def ppo_config(save_interval):
    return {
        "num_steps_per_env": 32, "save_interval": save_interval, "logger": "tensorboard",
        "obs_groups": GROUPS, "actor": {"class_name": MODEL_CLASS}, "critic": {"class_name": MODEL_CLASS},
        "algorithm": {"class_name": "PPO", "clip_param": .2, "gamma": .99, "lam": .95,
                      "num_learning_epochs": 4, "num_mini_batches": 8, "learning_rate": 1e-4,
                      "schedule": "adaptive", "desired_kl": .015, "entropy_coef": .003, "max_grad_norm": 1.},
    }


def read_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    info = saved.get("infos", {}) if isinstance(saved, dict) else {}
    if (not isinstance(info, dict) or info.get("agriculture") != contract()
            or not isinstance(saved.get("actor_state_dict"), dict) or type(saved.get("iter")) is not int):
        raise ValueError("Se requiere un checkpoint PPO agrícola bimanual compatible")
    return saved


def actor_from_checkpoint(saved, device):
    sample = {"policy": torch.zeros((1, OBS_DIM))}
    actor = BimanualTeacherModel(sample, GROUPS, "actor", ACTION_DIM)
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    return actor.to(device).eval()


class AgriculturalRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        saved = self.alg.save()
        saved.update(iter=self.current_learning_iteration,
                     infos={"agriculture": contract(), "rng": random_state()})
        save_training_checkpoint(path, saved)
        self.logger.save_model(path, self.current_learning_iteration)

    def resume(self, saved):
        self.alg.load(saved, load_cfg=None, strict=True)
        self.current_learning_iteration = saved["iter"] + 1
        restore_random_state(saved["infos"]["rng"])
