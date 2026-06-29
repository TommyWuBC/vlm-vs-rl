import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from gymnasium import RewardWrapper
import torch as th
import json
from datetime import datetime
from collections import deque
from minigrid.wrappers import FullyObsWrapper

print(th.cuda.is_available())
print(th.cuda.get_device_name(0))

class MyRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.goal_pos = None
        self.prev_dist = None

    def bfs_distance(self):
        """Distance to goal avoiding lava"""
        print(type(self.env.unwrapped))  # should print MiniGridEnv
        grid = self.env.unwrapped.grid
        width = self.env.unwrapped.width
        height = self.env.unwrapped.height
        start = tuple(self.env.unwrapped.agent_pos)
        
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            (x, y), dist = queue.popleft()
            if (x, y) == self.goal_pos:
                return dist
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = x+dx, y+dy
                if (nx, ny) not in visited and 0 <= nx < width and 0 <= ny < height:
                    cell = grid.get(nx, ny)
                    if cell is None or cell.type == 'goal':
                        visited.add((nx, ny))
                        queue.append(((nx, ny), dist+1))
        return float('inf')  # no path found
    
    def reward(self, reward):
        if reward > 0:
            return 10.0
        if reward < 0:
            return reward *3
        dist = self.bfs_distance()
        if self.prev_dist is not None and dist < self.prev_dist:
            reward += 0.12
        elif self.prev_dist is not None and dist > self.prev_dist:
            reward -= 0.1
        else:
            reward -= 0.04
        reward -= 0.005
        self.prev_dist = dist
        return reward

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.goal_pos = None
        for i in range(self.env.unwrapped.width):
            for j in range(self.env.unwrapped.height):
                cell = self.env.unwrapped.grid.get(i, j)
                if cell is not None and cell.type == 'goal':
                    self.goal_pos = (i, j)
        self.prev_dist = self.bfs_distance()
        return result

def make_env():
    env = gym.make("MiniGrid-LavaCrossingS9N1-v0")
    env = FullyObsWrapper(env)   # full obs first
    env = ImgObsWrapper(env)     # extract image array
    env = MyRewardWrapper(env)   # reward wrapper outermost
    return env

def make_eval_env():
    env = gym.make("MiniGrid-LavaCrossingS9N1-v0")
    env = MyRewardWrapper(env)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)   # extract image array from dict
    return env

# Training env - 8 parallel
env = make_vec_env(make_env, n_envs=8)

# Eval env - single, no reward shaping (measures true performance)
eval_env = make_vec_env(make_eval_env, n_envs=1)

# Save config before training starts
training_config = {
    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "total_timesteps": 30_000_000,
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "n_envs": 8,
    "policy": "MlpPolicy",
    "reward_shaping": "manhattan distance shaping +0.1/-0.05/-0.02 + step penalty -0.001",
    "environment": "MiniGrid-LavaCrossingS9N1-v0",
}
with open("./results/training_config.json", "w") as f:
    json.dump(training_config, f, indent=2)
print("Config saved to results/training_config.json")

# Callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./results/",
    eval_freq=10000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1
)
checkpoint_callback = CheckpointCallback(
    save_freq=1_000_000,
    save_path="./models/checkpoints/",
    name_prefix="ppo_lava"
)

# Model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=2048,
    device="cpu",
)

# Logger
new_logger = configure("./results/logs/", ["stdout", "tensorboard"])
model.set_logger(new_logger)

print("Starting training...")
model.learn(
    total_timesteps=10_000_000,
    callback=[eval_callback, checkpoint_callback]
)
model.save("./models/final_model")
print("Training complete!")

env.close()
eval_env.close()