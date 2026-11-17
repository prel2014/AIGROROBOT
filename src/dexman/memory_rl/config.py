"""Recurrent PPO defaults; independent of simulator and training imports."""


def actor_config():
    return {"class_name": "dexman.agents.recurrent:MemoryActor", "hidden_dims": [256, 128],
            "activation": "elu", "obs_normalization": False, "rnn_type": "gru",
            "rnn_hidden_dim": 256, "rnn_num_layers": 1,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": .10,
                                 "std_range": (.02, .30)}}


def training_config():
    return {"num_steps_per_env": 64, "save_interval": 100, "logger": "tensorboard",
            "obs_groups": {"actor": ["policy"], "critic": ["critic"]}, "actor": actor_config(),
            "critic": {"class_name": "MLPModel", "hidden_dims": [256, 128, 64], "activation": "elu",
                       "obs_normalization": True},
            "algorithm": {"class_name": "PPO", "num_learning_epochs": 4, "num_mini_batches": 1,
                          "learning_rate": 3e-4, "schedule": "adaptive", "gamma": .99, "lam": .95,
                          "entropy_coef": .001, "desired_kl": .02, "max_grad_norm": 1.,
                          "value_loss_coef": 1., "use_clipped_value_loss": True, "clip_param": .2}}
