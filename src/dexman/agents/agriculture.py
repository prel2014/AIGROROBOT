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
from dexman.tasks.agriculture.interface import ACTION_DIM, ARM_KEYS, EYE_KEYS, JOINT_ORDER, JOINTS_PER_ARM, STATE_DIM
from dexman.tasks.agriculture.mobile_state import HEAD_JOINTS, HEAD_LIMITS_DEG, STATE_ORDER
from dexman.tasks.agriculture.objectives import (ARM_FEATURES, EPISODE_SECONDS, JOINT_FEATURES, OBJECTIVES, OBS_DIM,
                                                 OBSTACLE_FEATURES, ROVER_ACTION_DIM, ROVER_FEATURES, ROVER_OBS_DIM,
                                                 TASK_FEATURES)
from dexman.tasks.agriculture.randomization import OBSTACLE_RADIUS
from .models import RSL_RL_VERSION, StatelessTeacherModel


MODEL_CLASS = "dexman.agents.agriculture:BimanualTeacherModel"
GROUPS = {"actor": ["policy"], "critic": ["policy"]}


class RoverActionHead(nn.Module):
    """Two arms share a joint readout. The third query controls both eyes."""

    def __init__(self, width):
        super().__init__()
        self.arm = nn.Linear(width, JOINTS_PER_ARM)
        self.gaze = nn.Linear(width, len(HEAD_JOINTS))
        for layer in (self.arm, self.gaze):
            nn.init.orthogonal_(layer.weight, gain=.01)
            nn.init.zeros_(layer.bias)

    def forward(self, queries):
        return torch.cat((self.arm(queries[:, :len(ARM_KEYS)]).flatten(1), self.gaze(queries[:, -1])), dim=1)


class BimanualTeacherModel(StatelessTeacherModel):
    """Attention over both arms, object geometry, task phase and obstacle tokens."""

    def __init__(self, obs, obs_groups, obs_set, output_dim):
        super().__init__()
        self.is_actor = obs_set == "actor"
        self.obs_dim = obs["policy"].shape[-1]
        self.rover = self.obs_dim == ROVER_OBS_DIM
        actions = (ROVER_ACTION_DIM,) if self.rover else (ACTION_DIM, GRASP_ACTION_DIM)
        allowed = actions if self.is_actor else (1,)
        if (obs_groups.get(obs_set) != ["policy"] or self.obs_dim not in (OBS_DIM, ROVER_OBS_DIM)
                or output_dim not in allowed):
            raise ValueError("Incompatible agricultural PPO observation/action contract")
        self.obs_groups = ["policy"]
        self.obs_normalization = self.update_observation_normalization = True
        self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        self.distribution = (GaussianDistribution(output_dim, init_std=.32 if output_dim == GRASP_ACTION_DIM else .13,
                                                    std_type="log", std_range=[.025, .35]) if self.is_actor else None)
        width = 128
        self.joint_projection = nn.Linear(JOINT_FEATURES, width)
        self.arm_projection = nn.Linear(ARM_FEATURES, width)
        self.task_projection = nn.Linear(TASK_FEATURES, width)
        self.obstacle_projection = nn.Linear(OBSTACLE_FEATURES, width)
        self.rover_projection = nn.Linear(ROVER_FEATURES, width) if self.rover else None
        self.readout_count = len(ARM_KEYS) if output_dim == ACTION_DIM else 1
        if output_dim == ROVER_ACTION_DIM:
            self.readout_count = len(ARM_KEYS) + 1
        count = self.readout_count + ACTION_DIM + len(ARM_KEYS) + 1 + len(OBSTACLE_RADIUS) + int(self.rover)
        self.token_identity = nn.Parameter(torch.empty(count, width))
        nn.init.normal_(self.token_identity, std=.02)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(
            width, 4, 512, dropout=0., activation="gelu", batch_first=True, norm_first=True) for _ in range(3))
        self.output_norm = nn.LayerNorm(width)
        if output_dim == ROVER_ACTION_DIM:
            self.output_head = RoverActionHead(width)
        else:
            self.output_head = nn.Linear(width, output_dim // self.readout_count)
            nn.init.orthogonal_(self.output_head.weight, gain=.01 if self.is_actor else 1.)
            nn.init.zeros_(self.output_head.bias)

    def get_latent(self, obs):
        state = self.obs_normalizer(obs["policy"]).clamp(-10., 10.)
        joint, arms, task, obstacles = state[:, :OBS_DIM].split(
            (ACTION_DIM * JOINT_FEATURES, len(ARM_KEYS) * ARM_FEATURES, TASK_FEATURES,
             len(OBSTACLE_RADIUS) * OBSTACLE_FEATURES),
            dim=-1)
        tokens = torch.cat((
            self.token_identity.new_zeros((len(state), self.readout_count, self.token_identity.shape[-1])),
            self.joint_projection(joint.reshape(-1, ACTION_DIM, JOINT_FEATURES)),
            self.arm_projection(arms.reshape(-1, len(ARM_KEYS), ARM_FEATURES)),
            self.task_projection(task).unsqueeze(1),
            self.obstacle_projection(obstacles.reshape(-1, len(OBSTACLE_RADIUS), OBSTACLE_FEATURES)),
        ), dim=1)
        if self.rover_projection is not None:
            tokens = torch.cat((tokens, self.rover_projection(state[:, OBS_DIM:]).unsqueeze(1)), dim=1)
        tokens = tokens + self.token_identity
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens[:, :self.readout_count])


def contract(grasp=False, rover=False):
    if rover and grasp:
        raise ValueError("La calibración de agarre del torso fijo es incompatible con el rover")
    result = {"schema": "dexman.agriculture.grasp.v1" if grasp else "dexman.agriculture.ppo.v2",
            "rsl_rl": RSL_RL_VERSION, "joint_order": list(JOINT_ORDER),
            "observations": OBS_DIM, "actions": GRASP_ACTION_DIM if grasp else ACTION_DIM, "model": MODEL_CLASS,
            "episode_seconds": GRASP_EPISODE_SECONDS if grasp else EPISODE_SECONDS, "control_hz": 30,
            "task_definitions": {name: asdict(spec) for name, spec in OBJECTIVES.items()},
            "obstacle_order": list(OBSTACLE_RADIUS)}
    if rover:
        result.update(schema="dexman.agriculture.rover.ppo.v1", observations=ROVER_OBS_DIM,
                      actions=ROVER_ACTION_DIM, joint_order=[*JOINT_ORDER, *HEAD_JOINTS],
                      head_limits_deg=[list(pair) for pair in HEAD_LIMITS_DEG],
                      rover_observation_order=[*STATE_ORDER[STATE_DIM:], *(f"target_rad/{name}" for name in HEAD_JOINTS),
                                              *(f"fruit_alignment_cos/{eye}" for eye in EYE_KEYS)])
    return result


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
    if (not isinstance(info, dict) or info.get("agriculture") not in (contract(), contract(grasp=True), contract(rover=True))
            or not isinstance(saved.get("actor_state_dict"), dict) or type(saved.get("iter")) is not int):
        raise ValueError("Se requiere un checkpoint PPO agrícola bimanual compatible")
    profile = info.get("simulation_profile")
    if (not isinstance(profile, dict) or type(profile.get("nominal_scene")) is not bool
            or type(profile.get("physics_variation")) not in (int, float)
            or not 0. <= profile["physics_variation"] <= 1.):
        raise ValueError("El checkpoint agrícola tiene un perfil físico inválido")
    return saved


def actor_from_checkpoint(saved, device):
    architecture = saved["infos"]["agriculture"]
    sample = {"policy": torch.zeros((1, architecture["observations"]))}
    actor = BimanualTeacherModel(sample, GROUPS, "actor", architecture["actions"])
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
                     infos={"agriculture": contract(grasp=self.env.cfg.grasp is not None, rover=self.env.cfg.rover),
                            "rng": random_state(),
                            "simulation_profile": self.simulation_profile()})
        save_training_checkpoint(path, saved)
        self.logger.save_model(path, self.current_learning_iteration)

    def resume(self, saved):
        profile = self.simulation_profile()
        previous = saved["infos"]["simulation_profile"]
        if saved["infos"]["agriculture"] != contract(grasp=self.env.cfg.grasp is not None, rover=self.env.cfg.rover):
            raise ValueError("La acción del checkpoint no coincide con el modo de entrenamiento")
        if self.env.cfg.rover and previous.get("rover_asset_sha256") != self.env.rover_asset_sha256:
            raise ValueError("El modelo USD del rover es distinto al guardado en el checkpoint")
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
        if self.env.cfg.rover:
            profile["rover_asset_sha256"] = self.env.rover_asset_sha256
        if self.env.cfg.grasp is not None:
            profile["grasp"] = asdict(self.env.cfg.grasp)
        return profile
