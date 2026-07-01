import gymnasium as gym
import minigrid
env = gym.make("MiniGrid-LavaCrossingS9N2-v0")
print(env)
env.close()