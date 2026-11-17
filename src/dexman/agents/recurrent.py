"""RSL-RL GRU actor and construction adapter."""
import torch
from torch import nn
from rsl_rl.models import RNNModel
from tensordict import TensorDict


class MemoryActor(RNNModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        last = [layer for layer in self.mlp.modules() if isinstance(layer, nn.Linear)][-1]
        nn.init.orthogonal_(last.weight, gain=.01)
        nn.init.zeros_(last.bias)


def make_actor(dimension, config, device):
    config = {key: value for key, value in config.items() if key != "class_name"}
    obs = TensorDict({"policy": torch.zeros(1, dimension, device=device)}, batch_size=[1])
    return MemoryActor(obs, {"actor": ["policy"]}, "actor", 30, **config).to(device)
