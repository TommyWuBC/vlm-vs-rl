import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
import time
from minigrid.wrappers import FullyObsWrapper

env = gym.make("MiniGrid-LavaCrossingS9N1-v0", render_mode="human")
env = FullyObsWrapper(env)   # full obs first
env = ImgObsWrapper(env)
model = PPO.load("./models/best_model", env=env)
for episode in range(10):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        env.render()
        time.sleep(0.1)  # slow it down so you can watch
    print(f"Episode {episode+1} reward: {total_reward:.3f}")

env.close()