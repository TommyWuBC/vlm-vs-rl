import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
import json
import numpy as np

model = PPO.load("./models/best_model")

# Create the environment
env = gym.make("MiniGrid-LavaCrossingS9N1-v0")
env = ImgObsWrapper(env)  # simplifies observation to just the image

NUM_EPISODES = 20
MAX_STEPS = 100


results = []
for episode in range(NUM_EPISODES):
    observation, info = env.reset()
    steps = 0
    success = False
    actions_taken = []# start fresh episode, new random goal position
    
    for step in range(MAX_STEPS):
        # Use the model to predict the action
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        actions_taken.append(int(action))
        if terminated:
            success = True
        if terminated or truncated:
            break
    results.append({
        "episode": episode,
        "success": success,
        "steps": steps,
        "actions": actions_taken,
    })

with open("results/rl_results.json", "w") as f:
    json.dump(results, f, indent = 2)
    
print("Done! Results saved.")
env.close()