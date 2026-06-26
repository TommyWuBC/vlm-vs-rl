import gymnasium as gym
import minigrid
from minigrid.core.constants import OBJECT_TO_IDX

env = gym.make("MiniGrid-LavaCrossingS9N1-v0")
obs, info = env.reset()

print("OBJECT_TO_IDX:", OBJECT_TO_IDX)
print("Goal type integer:", OBJECT_TO_IDX['goal'])

env.close()