"""Adapter from Isaac observations/actions to the RSL-RL vector environment."""
import json
from pathlib import Path
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from dexman.tasks.manipulation.catalog import TASKS


class VisualVecEnv(VecEnv):
    """Encode observations once per step and retain only embeddings in PPO rollouts."""
    def __init__(self, env, encoder, commands, episodes_path=None):
        self.env, self.raw = env, env.unwrapped
        self.encoder, self.commands = encoder, commands
        self.num_envs, self.device = self.raw.num_envs, self.raw.device
        self.num_actions, self.max_episode_length = 30, self.raw.max_episode_length
        self.cfg = self.raw.cfg
        self.returns = torch.zeros(self.num_envs, device=self.device)
        self.lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.records = []
        self.episodes_path = Path(episodes_path) if episodes_path else None
        obs, _ = env.reset()
        self._obs = self._encode(obs)

    @property
    def episode_length_buf(self):
        return self.raw.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.raw.episode_length_buf = value

    def _encode(self, obs):
        images = self.raw.scene["camera"].data.output["rgb"].torch[..., :3].to(torch.uint8)
        features = self.encoder(images, self.raw.instructions, obs["policy"])
        return TensorDict({"policy": features, "critic": obs["critic"]}, batch_size=[self.num_envs])

    def get_observations(self):
        return self._obs

    def step(self, actions):
        obs, rewards, terminated, truncated, extras = self.env.step(self.commands.angles(actions))
        dones = terminated | truncated
        self.returns += rewards
        self.lengths += 1
        new_records = []
        for index in dones.nonzero().flatten().tolist():
            new_records.append({"replica": index, "task": TASKS[int(extras["task_ids"][index])].name,
                                "success": bool(extras["success"][index]),
                                "return": float(self.returns[index]), "steps": int(self.lengths[index])})
        self.returns[dones], self.lengths[dones] = 0, 0
        # Keep only the latest step in RAM; append all successes AND failures to disk.
        self.records = new_records
        if self.episodes_path and new_records:
            with self.episodes_path.open("a", encoding="utf-8") as stream:
                for record in new_records:
                    stream.write(json.dumps(record) + "\n")
        self._obs = self._encode(obs)
        return self._obs, rewards, dones.long(), extras

    def close(self):
        self.env.close()
