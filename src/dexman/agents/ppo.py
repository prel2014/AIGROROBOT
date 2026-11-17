"""State Transformer PPO teacher; visual language belongs to the student."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from dexman.agents.models import teacher_model_config


@configclass
class StateTransformerModelCfg:
    class_name: str = "dexman.agents.models:StateTransformerModel"
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    feedforward_dim: int = 512
    obs_normalization: bool = True
    update_observation_normalization: bool = True
    distribution_cfg: dict | None = None
    # Isaac Lab's compatibility adapter removes this field for RSL-RL >=5.
    stochastic: bool = False


@configclass
class LiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 64
    max_iterations = 3000
    save_interval = 100
    experiment_name = "ur10e_shadow_lift_transformer"
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor_learning_rate = 1e-6
    critic_learning_rate = 1e-4
    critic_warmup_updates = 32
    policy_kl_limit = .02
    minimum_actor_learning_rate = 1e-7
    actor = StateTransformerModelCfg(**teacher_model_config(True))
    critic = StateTransformerModelCfg(**teacher_model_config(False))
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.003,
        num_learning_epochs=4, num_mini_batches=4, learning_rate=1e-4, schedule="fixed",
        gamma=0.99, lam=0.95, desired_kl=0.02, max_grad_norm=1.0)


@configclass
class CorrectedLiftPPORunnerCfg(LiftPPORunnerCfg):
    experiment_name = "ur10e_shadow_lift_v1_transformer"
