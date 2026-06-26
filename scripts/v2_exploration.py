# v2 — Exploration Bonus
# Adds +0.01 for each new cell visited, resets each episode.
# Motivation: encourage the agent to explore rather than sit still.
# Result: agent learned to explore efficiently but stopped prioritizing
# the goal — reward hacking. eval mean_reward dropped to 0.123 despite
# training reward climbing to 0.649. Exploration bonus was too high,
# making cell collection more rewarding than goal completion.

import gymnasium as gym
from gymnasium import RewardWrapper


class ExplorationRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.visited_cells = set()

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.visited_cells = set()
        pos = tuple(self.env.unwrapped.agent_pos)
        self.visited_cells.add(pos)
        return result

    def reward(self, reward):
        if reward != 0:
            return reward

        pos = tuple(self.env.unwrapped.agent_pos)
        if pos not in self.visited_cells:
            reward += 0.01  # bonus for visiting a new cell
            self.visited_cells.add(pos)

        return reward
