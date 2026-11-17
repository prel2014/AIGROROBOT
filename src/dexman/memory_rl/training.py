"""Recurrent PPO training and optimizer resume; no inference entry points."""
import copy
from rsl_rl.runners import OnPolicyRunner
from .checkpoint import load_checkpoint


class MemoryRunner(OnPolicyRunner):
    def __init__(self, env, config, contract, log_dir, device):
        self.contract = copy.deepcopy(contract)
        super().__init__(env, copy.deepcopy(config), log_dir, device)

    def save(self, path, infos=None):
        super().save(path, infos=self.contract)

    def resume(self, path):
        saved, contract = load_checkpoint(path)
        if contract != self.contract:
            raise ValueError("Checkpoint task/control/encoder/training contract differs from this run")
        self.alg.load(saved, load_cfg=None, strict=True)
        self.current_learning_iteration = saved["iter"] + 1
        # Resume learned weights and optimizer; a fresh simulator starts fresh episode memory.
        self.alg.actor.reset()
        self.alg.critic.reset()
