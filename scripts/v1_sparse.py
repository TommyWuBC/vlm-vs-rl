# v1 — Sparse Reward (Baseline)
# No reward shaping. Agent only receives:
#   +1 (time-decayed) for reaching goal
#   -1 for stepping on lava
# Result: agent barely learned anything (~0.00811 ep_rew_mean at 500k steps)
# The sparse signal is too infrequent for PPO to learn meaningful navigation.

# No wrapper needed — use the raw environment directly:
# env = gym.make("MiniGrid-LavaCrossingS9N1-v0")
