import gymnasium as gym
import minigrid

env = gym.make("MiniGrid-LavaCrossingS9N1-v0", render_mode="human")
observation, info = env.reset()

print("Observation shape:", observation['image'].shape)
print("Action space:", env.action_space)
print("Number of possible actions:", env.action_space.n)

# Run a random agent for 200 steps
import time
def reset(self, **kwargs):
    result = self.env.reset(**kwargs)
    for i in range(self.env.unwrapped.width):
        for j in range(self.env.unwrapped.height):
            cell = self.env.unwrapped.grid.get(i, j)
            if cell is not None and cell.type == 'goal':
                self.goal_pos = (i, j)
    print(f"Goal found at: {self.goal_pos}")  # add this temporarily
    self.visited = set()
    self.prev_dist = None
    return result
for step in range(200):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.05)
    if terminated or truncated:
        obs, info = env.reset()

env.close()