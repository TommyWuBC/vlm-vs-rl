# v3 — Manhattan Distance Shaping
# Uses Manhattan distance to goal as the shaping signal.
# Rewards getting closer, penalizes moving away or stagnating.
# Also adds a small step penalty to encourage efficiency.
#
# Problems identified:
#   1. Manhattan distance doesn't respect lava layout — agent is penalized
#      for moving away from goal even when doing so is necessary to navigate
#      around lava barriers.
#   2. Reward signals (~0.1) still weak relative to lava penalty (-1),
#      causing excessive lava aversion and freezing at narrow corridors.
#   3. gen_obs() called every step (redundant — SB3 already generates obs
#      automatically), causing major throughput bottleneck (~200 FPS).
# Result: ~15% success rate. Agent froze at narrow lava corridors.

import math
import gymnasium as gym
from gymnasium import RewardWrapper


class ManhattanDistanceRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.goal_pos = None
        self.prev_dist = None

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.goal_pos = None
        for i in range(self.env.unwrapped.width):
            for j in range(self.env.unwrapped.height):
                cell = self.env.unwrapped.grid.get(i, j)
                if cell is not None and cell.type == 'goal':
                    self.goal_pos = (i, j)
        pos = tuple(self.env.unwrapped.agent_pos)
        self.prev_dist = abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])
        return result

    def reward(self, reward):
        # Pass through terminal rewards unchanged
        if reward != 0:
            return reward

        pos = tuple(self.env.unwrapped.agent_pos)
        dist = abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])

        if dist < self.prev_dist:
            reward += 0.1    # got closer to goal
        elif dist > self.prev_dist:
            reward -= 0.05   # moved away from goal
        else:
            reward -= 0.02   # stagnated

        reward -= 0.001      # step penalty to encourage efficiency
        self.prev_dist = dist
        return reward
