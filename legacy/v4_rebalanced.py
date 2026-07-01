# v4 — Rebalanced Manhattan Distance (Strong Goal Reward)
# Attempt to fix v3's lava-aversion freezing by making goal reward
# much stronger and increasing the step penalty to force commitment.
#
# Changes from v3:
#   - Goal reward overridden to flat +10.0 (removes time decay, massively
#     increases relative value of task completion vs lava penalty)
#   - Step penalty increased from -0.001 to -0.005 (5x) to make
#     hovering near corridors increasingly costly
#   - Movement penalty increased to -0.08 for moving away from goal
#
# Problem introduced: goal reward so strong that agent ignored lava
# entirely, taking straight-line paths through lava to reach goal.
# Manhattan distance still doesn't respect lava layout, so "move toward
# goal" often means "walk through lava."
# Result: agent learned to beeline through lava. ep_len_mean dropped
# to ~8 steps (dying almost immediately every episode).

from gymnasium import RewardWrapper


class RebalancedRewardWrapper(RewardWrapper):
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
        if reward > 0:      # goal reached
            return 10.0     # flat strong reward, ignore time decay
        if reward < 0:      # lava — leave unchanged
            return reward

        pos = tuple(self.env.unwrapped.agent_pos)
        dist = abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])

        if dist < self.prev_dist:
            reward += 0.1
        elif dist > self.prev_dist:
            reward -= 0.08   # stronger penalty for moving away
        else:
            reward -= 0.02

        reward -= 0.005      # increased step penalty vs v3
        self.prev_dist = dist
        return reward
