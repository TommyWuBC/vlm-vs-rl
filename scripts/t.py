import gymnasium as gym
import minigrid

patterns = set()
env = gym.make("MiniGrid-LavaCrossingS9N1-v0")
for _ in range(1000):
    env.reset()
    # serialize the grid as a simple string
    grid_str = str([(i,j) for i in range(env.unwrapped.width) 
                   for j in range(env.unwrapped.height)
                   if env.unwrapped.grid.get(i,j) is not None 
                   and env.unwrapped.grid.get(i,j).type == 'lava'])
    patterns.add(grid_str)

print(f"Distinct lava patterns: {len(patterns)}")
env.close()