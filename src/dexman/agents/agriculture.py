"""Bimanual state Transformer and versioned PPO checkpoints for agriculture."""
from dataclasses import asdict
from time import perf_counter

import torch
from torch import nn
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.runners import OnPolicyRunner

from dexman.experiments import random_state, restore_random_state, save_training_checkpoint
from dexman.tasks.agriculture.grasp_curriculum import GRASP_ACTION_DIM, GRASP_EPISODE_SECONDS
from dexman.tasks.agriculture.interface import ACTION_DIM, ARM_KEYS, JOINT_ORDER
from dexman.tasks.agriculture.objectives import (ARM_FEATURES, EPISODE_SECONDS, JOINT_FEATURES, OBJECTIVES, OBS_DIM,
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
        allowed = (ACTION_DIM, GRASP_ACTION_DIM) if self.is_actor else (1,)
        if (obs_groups.get(obs_set) != ["policy"] or obs["policy"].shape[-1] != OBS_DIM
                or output_dim not in allowed):
            raise ValueError("Incompatible agricultural PPO observation/action contract")
        self.obs_groups, self.obs_dim = ["policy"], OBS_DIM
        self.obs_normalization = self.update_observation_normalization = True
        self.obs_normalizer = EmpiricalNormalization(OBS_DIM)
        self.distribution = (GaussianDistribution(output_dim, init_std=.32 if output_dim == GRASP_ACTION_DIM else .13,
                                                    std_type="log", std_range=[.025, .35]) if self.is_actor else None)
        width = 128
        self.joint_projection = nn.Linear(JOINT_FEATURES, width)
        self.arm_projection = nn.Linear(ARM_FEATURES, width)
        self.task_projection = nn.Linear(TASK_FEATURES, width)
        self.obstacle_projection = nn.Linear(OBSTACLE_FEATURES, width)
        self.readout_count = len(ARM_KEYS) if output_dim == ACTION_DIM else 1
        count = self.readout_count + ACTION_DIM + len(ARM_KEYS) + 1 + len(OBSTACLE_RADIUS)
        self.token_identity = nn.Parameter(torch.empty(count, width))
        nn.init.normal_(self.token_identity, std=.02)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(
            width, 4, 512, dropout=0., activation="gelu", batch_first=True, norm_first=True) for _ in range(3))
        self.output_norm = nn.LayerNorm(width)
        self.output_head = nn.Linear(width, output_dim // self.readout_count)
        nn.init.orthogonal_(self.output_head.weight, gain=.01 if self.is_actor else 1.)
        nn.init.zeros_(self.output_head.bias)

    def get_latent(self, obs):
        state = self.obs_normalizer(obs["policy"]).clamp(-10., 10.)
        joint, arms, task, obstacles = state.split(
            (ACTION_DIM * JOINT_FEATURES, len(ARM_KEYS) * ARM_FEATURES, TASK_FEATURES,
             len(OBSTACLE_RADIUS) * OBSTACLE_FEATURES),
            dim=-1)
        tokens = torch.cat((
            self.token_identity.new_zeros((len(state), self.readout_count, self.token_identity.shape[-1])),
            self.joint_projection(joint.reshape(-1, ACTION_DIM, JOINT_FEATURES)),
            self.arm_projection(arms.reshape(-1, len(ARM_KEYS), ARM_FEATURES)),
            self.task_projection(task).unsqueeze(1),
            self.obstacle_projection(obstacles.reshape(-1, len(OBSTACLE_RADIUS), OBSTACLE_FEATURES)),
        ), dim=1) + self.token_identity
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens[:, :self.readout_count])


def contract(grasp=False):
    return {"schema": "dexman.agriculture.grasp.v1" if grasp else "dexman.agriculture.ppo.v2",
            "rsl_rl": RSL_RL_VERSION, "joint_order": list(JOINT_ORDER),
            "observations": OBS_DIM, "actions": GRASP_ACTION_DIM if grasp else ACTION_DIM, "model": MODEL_CLASS,
            "episode_seconds": GRASP_EPISODE_SECONDS if grasp else EPISODE_SECONDS, "control_hz": 30,
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
    if (not isinstance(info, dict) or info.get("agriculture") not in (contract(), contract(grasp=True))
            or not isinstance(saved.get("actor_state_dict"), dict) or type(saved.get("iter")) is not int):
        raise ValueError("Se requiere un checkpoint PPO agrícola bimanual compatible")
    profile = info.get("simulation_profile")
    if (not isinstance(profile, dict) or type(profile.get("nominal_scene")) is not bool
            or type(profile.get("physics_variation")) not in (int, float)
            or not 0. <= profile["physics_variation"] <= 1.):
        raise ValueError("El checkpoint agrícola tiene un perfil físico inválido")
    return saved


def actor_from_checkpoint(saved, device):
    sample = {"policy": torch.zeros((1, OBS_DIM))}
    actor = BimanualTeacherModel(sample, GROUPS, "actor", saved["infos"]["agriculture"]["actions"])
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    return actor.to(device).eval()


class AgriculturalRunner(OnPolicyRunner):
    def record_epochs(self, recorder):
        """Flush the rollout's frames at the same iteration as the RSL-RL metrics."""
        self.env.on_rollout_step = recorder.capture_step
        log = self.logger.log

        def log_epoch(**metrics):
            start = perf_counter()
            recorder.write_epoch(metrics["it"])
            metrics["learn_time"] += perf_counter() - start
            log(**metrics)

        self.logger.log = log_epoch

    def save(self, path, infos=None):
        saved = self.alg.save()
        saved.update(iter=self.current_learning_iteration,
                     infos={"agriculture": contract(grasp=self.env.cfg.grasp is not None), "rng": random_state(),
                            "simulation_profile": self.simulation_profile()})
        save_training_checkpoint(path, saved)
        self.logger.save_model(path, self.current_learning_iteration)

    def resume(self, saved):
        profile = self.simulation_profile()
        previous = saved["infos"]["simulation_profile"]
        if saved["infos"]["agriculture"] != contract(grasp=self.env.cfg.grasp is not None):
            raise ValueError("La acción del checkpoint no coincide con el modo de entrenamiento")
        if profile.get("grasp") is not None:
            # The reset posture may progress from closed to open with the same action map.
            prior = previous.get("grasp", {})
            geometry = ("crop", "side", "pregrasp", "closed", "lift", "lift_waypoints")
            if any(profile["grasp"][name] != prior.get(name) for name in geometry):
                raise ValueError("Para reanudar el currículo se requiere la misma trayectoria calibrada")
        self.alg.load(saved, load_cfg=None, strict=True)
        self.current_learning_iteration = saved["iter"] + 1
        restore_random_state(saved["infos"]["rng"])

    def simulation_profile(self):
        profile = {"nominal_scene": self.env.cfg.nominal, "physics_variation": self.env.cfg.physics_variation}
        if self.env.cfg.grasp is not None:
            profile["grasp"] = asdict(self.env.cfg.grasp)
        return profile
