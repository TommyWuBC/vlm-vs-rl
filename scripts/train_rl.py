import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import os
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import RewardWrapper
import json
from datetime import datetime
import math
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class myRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.visited = set()
        self.goal_pos = None
        self.prev_dist = None
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        for i in range(self.env.unwrapped.width):
            for j in range(self.env.unwrapped.height):
                cell = self.env.unwrapped.grid.get(i, j)
                if cell is not None and cell.type == 'goal':
                    self.goal_pos = (i, j)     
        self.visited = set()
        self.prev_dist = None
        return result
    def reward(self, reward):
        dx = self.env.unwrapped.agent_pos[0] - self.goal_pos[0]
        dy = self.env.unwrapped.agent_pos[1] - self.goal_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        if self.env.unwrapped.agent_pos not in self.visited:
            reward = reward + 0.001
            self.visited.add(self.env.unwrapped.agent_pos)
        if self.prev_dist is None or distance < self.prev_dist:
            reward = reward +  0.01
        self.prev_dist = distance
        return reward

# Create the environment
env = gym.make("MiniGrid-FourRooms-v0")
env = myRewardWrapper(env)
env = ImgObsWrapper(env)  # simplifies observation to just the image

# Create a separate environment just for evaluation during training
eval_env = gym.make("MiniGrid-FourRooms-v0")
eval_env = ImgObsWrapper(eval_env)

# This callback evaluates the agent every 10,000 steps and saves the best version
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./results/",
    eval_freq=10000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1
)

# Create the PPO agent
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        # SB3 handles channel ordering automatically, just build the CNN
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    def forward(self, observations):
        return self.linear(self.cnn(observations))

# Create the PPO agent with custom CNN
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=2048,
    device="cuda",
    policy_kwargs=dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
)

training_config = {
    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "total_timesteps": 2_000_000,
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "reward_shaping": "exploration bonus +0.01 per new cell",
    "environment": "MiniGrid-FourRooms-v0",
    "baseline_run": {
        "timesteps": 500_000,
        "final_ep_rew_mean": 0.00811,
        "reward_shaping": "none (sparse)"
    }
}

with open("./results/training_config.json", "w") as f:
    json.dump(training_config, f, indent=2)
print("Config saved to results/training_config.json")
print("Starting training - this will run for ~20 minutes, just let it go...")
model.learn(total_timesteps=2_00_000, callback=eval_callback)
model.save("./models/final_model")

print("Training complete!")
env.close()
eval_env.close()