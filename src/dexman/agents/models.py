"""Structured state Transformer models using the RSL-RL stochastic-model interface."""
import copy
import math

import torch
from torch import nn

from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import unpad_trajectories

from dexman.robot.joints import ACTION_NAMES, JOINT_NAMES


MODEL_CLASS = "dexman.agents.models:StateTransformerModel"
RSL_RL_VERSION = "5.5.1"
STATE_DIM = 95
ACTION_DIM = 26


def teacher_model_config(stochastic):
    """Serializable constructor arguments shared by training and inference."""
    return {"class_name": MODEL_CLASS, "model_dim": 128, "num_heads": 4,
            "num_layers": 3, "feedforward_dim": 512, "obs_normalization": True,
            "update_observation_normalization": not stochastic,
            "distribution_cfg": ({"class_name": "GaussianDistribution", "init_std": .18,
                                  "std_type": "log", "std_range": [.01, .35]} if stochastic else None)}


def teacher_architecture():
    """Versioned meaning of the observation, token and action axes."""
    return {"version": 3, "type": "structured_state_transformer", "rsl_rl_version": RSL_RL_VERSION,
            "observations": STATE_DIM, "actions": ACTION_DIM,
            "state_layout": [["normalized_joint_position", 30], ["scaled_joint_velocity", 30],
                             ["object_minus_palm", 3], ["object_position_local", 3],
                             ["goal_minus_object", 3], ["filtered_joint_target_encoding", 26]],
            "joint_names": list(JOINT_NAMES), "action_names": list(ACTION_NAMES),
            "tokens": {"joint_position_velocity": 30, "geometry": 3, "filtered_joint_target": 26,
                       "readout": 1, "identity_embeddings": True},
            "attention": "bidirectional_pre_layernorm", "activation": "gelu", "dropout": 0.,
            "normalization": {"algorithm": "rsl_empirical_95_eps_0.01_clamp_10",
                              "actor": "bootstrap_training_observations_fixed",
                              "critic": "empirical_online",
                              "actor_without_bootstrap": "fixed_initial_mean_zero_std_one"},
            "initial_actor_head_gain": .01, "initial_arm_wrist_std": .12,
            "post_bootstrap_exploration_std": {"arm_wrist": .02, "fingers": .04,
                                               "phase": "after_behavior_cloning_before_calibration"},
            "actor": teacher_model_config(True), "critic": teacher_model_config(False)}


class StatelessTeacherModel(nn.Module):
    """RSL-RL interface shared by the privileged Transformer policies."""

    is_recurrent = False

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if hidden_state is not None:
            raise ValueError("State-only teacher does not accept recurrent hidden state")
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        output = self.output_head(self.get_latent(obs)).flatten(1)
        if self.distribution is not None and stochastic_output:
            self.distribution.update(output)
            return self.distribution.sample()
        return output

    def update_normalization(self, obs):
        if self.obs_normalization and self.update_observation_normalization:
            self.obs_normalizer.update(obs["policy"])

    def reset(self, dones=None, hidden_state=None):
        """No temporal state to reset."""

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        """No temporal state to detach."""

    @property
    def output_mean(self):
        return self.distribution.mean

    @property
    def output_std(self):
        return self.distribution.std

    @property
    def output_entropy(self):
        return self.distribution.entropy

    @property
    def output_distribution_params(self):
        return self.distribution.params

    def get_output_log_prob(self, outputs):
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(self, old_params, new_params):
        return self.distribution.kl_divergence(old_params, new_params)


class StateTransformerModel(StatelessTeacherModel):
    """One-step, state-only Transformer actor or critic; no temporal hidden state.

    Thirty tokens pair q/dq by physical joint identity. Three tokens encode geometry,
    twenty-six encode filtered joint targets, and a learned readout token predicts the output.
    RSL-RL owns PPO, the Gaussian distribution and empirical observation statistics.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, model_dim=128, num_heads=4,
                 num_layers=3, feedforward_dim=512, obs_normalization=True, distribution_cfg=None,
                 update_observation_normalization=True):
        super().__init__()
        if obs_set not in ("actor", "critic") or obs_groups.get(obs_set) != ["policy"]:
            raise ValueError("State Transformer requires actor/critic observation group ['policy']")
        if obs["policy"].ndim != 2 or obs["policy"].shape[-1] != STATE_DIM:
            raise ValueError("State Transformer requires policy observations [N,95]")
        if output_dim != (ACTION_DIM if obs_set == "actor" else 1):
            raise ValueError("State Transformer requires 26 actor outputs or one critic output")
        if min(model_dim, num_heads, num_layers, feedforward_dim) < 1 or model_dim % num_heads:
            raise ValueError("Invalid Transformer dimensions or attention head count")
        if (distribution_cfg is not None) != (obs_set == "actor"):
            raise ValueError("Teacher actor requires a Gaussian distribution; critic must be deterministic")
        self.obs_groups, self.obs_dim = ["policy"], STATE_DIM
        self.obs_normalization = obs_normalization
        self.update_observation_normalization = update_observation_normalization
        self.obs_normalizer = EmpiricalNormalization(STATE_DIM) if obs_normalization else nn.Identity()
        self.distribution = None
        if distribution_cfg is not None:
            dist_cfg = copy.deepcopy(distribution_cfg)
            if dist_cfg.pop("class_name") != "GaussianDistribution" or dist_cfg.get("std_type") != "log":
                raise ValueError("Teacher supports the RSL GaussianDistribution with log standard deviation")
            self.distribution = GaussianDistribution(ACTION_DIM, **dist_cfg)
            with torch.no_grad():
                self.distribution.log_std_param[:8].fill_(math.log(.12))
        self.joint_projection = nn.Linear(2, model_dim)
        self.geometry_projection = nn.Linear(3, model_dim)
        self.target_projection = nn.Linear(1, model_dim)
        self.token_identity = nn.Parameter(torch.empty(1, 59, model_dim))
        self.readout_token = nn.Parameter(torch.empty(1, 1, model_dim))
        self.token_types = nn.Parameter(torch.empty(4, model_dim))
        # Construct layers independently rather than cloning identically initialized layers.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(model_dim, num_heads, feedforward_dim, dropout=0.,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_head = nn.Linear(model_dim, output_dim)
        for parameter in (self.token_identity, self.readout_token, self.token_types):
            nn.init.normal_(parameter, std=.02)
        nn.init.orthogonal_(self.output_head.weight, gain=.01 if self.distribution is not None else 1.)
        nn.init.zeros_(self.output_head.bias)

    def get_latent(self, obs, masks=None, hidden_state=None):
        state = self.obs_normalizer(obs["policy"]).clamp(-10., 10.)
        if state.ndim != 2 or state.shape[-1] != STATE_DIM:
            raise ValueError("State Transformer requires policy observations [N,95]")
        joints = torch.stack((state[:, :30], state[:, 30:60]), dim=-1)
        geometry = state[:, 60:69].reshape(-1, 3, 3)
        targets = state[:, 69:].unsqueeze(-1)
        tokens = torch.cat((self.joint_projection(joints) + self.token_types[0],
                            self.geometry_projection(geometry) + self.token_types[1],
                            self.target_projection(targets) + self.token_types[2]), dim=1)
        tokens = tokens + self.token_identity
        readout = self.readout_token.expand(state.shape[0], -1, -1) + self.token_types[3]
        tokens = torch.cat((readout, tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens[:, 0])

def build_teacher_model(obs_set, device="cpu"):
    config = teacher_model_config(obs_set == "actor")
    config.pop("class_name")
    sample = {"policy": torch.zeros(1, STATE_DIM, device=device)}
    return StateTransformerModel(sample, {obs_set: ["policy"]}, obs_set,
                                 ACTION_DIM if obs_set == "actor" else 1, **config).to(device)
