# v5 — BFS Distance Shaping (Current)
# Uses BFS (breadth-first search) to compute true navigable path distance
# to goal, treating lava as impassable. This replaces Manhattan distance
# which doesn't respect lava layout.
#
# Key insight: Manhattan distance misdirects the agent because the shortest
# straight-line path is often blocked by lava. BFS distance correctly
# captures whether a move genuinely makes progress along a valid path.
# The agent never sees the BFS path directly — it only receives a reward
# signal calibrated against true navigable distance, and must learn
# navigation from its 7x7 partial observation alone.
#
# Changes from v4:
#   - BFS distance replaces Manhattan distance entirely
#   - Movement penalties rebalanced now that distance signal is correct
#   - Step penalty kept at -0.005 to discourage hovering
#   - Goal reward kept at flat +10.0
#
# Result: TBD (currently training at 30M steps)

from collections import deque
from gymnasium import RewardWrapper


class BFSRewardWrapper(RewardWrapper):
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
        self.prev_dist = self.bfs_distance()
        return result

    def bfs_distance(self):
        """Compute shortest navigable path length to goal, treating lava as impassable."""
        grid = self.env.unwrapped.grid
        width = self.env.unwrapped.width
        height = self.env.unwrapped.height
        start = tuple(self.env.unwrapped.agent_pos)

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            (x, y), dist = queue.popleft()
            if (x, y) == self.goal_pos:
                return dist
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and 0 <= nx < width and 0 <= ny < height:
                    cell = grid.get(nx, ny)
                    # Allow traversal of empty cells and the goal; block lava and walls
                    if cell is None or cell.type == 'goal':
                        visited.add((nx, ny))
                        queue.append(((nx, ny), dist + 1))

        return float('inf')  # no valid path found (agent is surrounded)

    def reward(self, reward):
        if reward > 0:      # goal reached
            return 10.0
        if reward < 0:      # lava — leave unchanged
            return reward

        dist = self.bfs_distance()

        if dist < self.prev_dist:
            reward += 0.1    # genuine progress along navigable path
        elif dist > self.prev_dist:
            reward -= 0.05   # moved further from goal along navigable path
        else:
            reward -= 0.02   # stagnated

        reward -= 0.005      # step penalty to discourage hovering
        self.prev_dist = dist
        return reward
