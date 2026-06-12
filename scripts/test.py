import gymnasium as gym
import minigrid

env = gym.make("MiniGrid-FourRooms-v0")
obs, info = env.reset()

raw = env.unwrapped
print("Agent pos:", raw.agent_pos)

for x in range(raw.width):
    for y in range(raw.height):
        cell = raw.grid.get(x, y)
        if cell is not None and cell.type != 'wall':
            print(f"({x},{y}): {cell.type}")

env.close()