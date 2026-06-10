import gymnasium as gym
import minigrid

env = gym.make("MiniGrid-FourRooms-v0", render_mode="human")
observation, info = env.reset()

print("Observation shape:", observation['image'].shape)
print("Action space:", env.action_space)
print("Number of possible actions:", env.action_space.n)

# Run a random agent for 200 steps
import time
for step in range(200):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.05)
    if terminated or truncated:
        obs, info = env.reset()

env.close()