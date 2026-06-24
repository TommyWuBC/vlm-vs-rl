# Training Run Notes

## Run 1 — Baseline (sparse reward)
- Timesteps: 500,000
- Reward shaping: none
- Final ep_rew_mean: 0.00811
- Result: agent barely learned anything

## Run 2 — With exploration reward shaping
- Timesteps: 2,000,000
- Reward shaping: +0.01 per new cell visited, resets each episode
- ep_rew_mean at 70k steps: 0.557 (already massively better)
- GPU: GTX 1660 Ti
- Result: TBD
## Run 2 Observation (670k steps)
- eval mean_reward dropped to 0.123 despite training reward climbing to 0.649
- Likely cause: agent overfitting to exploration bonus (0.01 too high)
- Agent learns to collect new cells efficiently but stops prioritizing the goal
- Fix for Run 3: reduce bonus to 0.001
## Run 3 — Distance-based reward shaping
- Timesteps: 2,000,000
- Reward shaping: +0.01 for moving closer to goal, +0.001 for new cells
- Final ep_rew_mean: 0.553
- Best ep_rew_mean seen during training: ~0.6+
- Best model saved: models/best_model.zip
- Result: significantly better than sparse reward baseline

## Run 4 — LavaCrossing, 200k steps (accidental short run)
- Environment: MiniGrid-LavaCrossingS9N1-v0
- ep_rew_mean: 1.23 after just 200k steps
- Agent successfully reaching goal + collecting exploration bonuses
- Note: timesteps typo caused early termination, rerunning for 2M steps

## Run 4 — LavaCrossing, 2M steps (final run)
- Environment: MiniGrid-LavaCrossingS9N1-v0
- Final ep_rew_mean: 1.09 (training)
- Final eval mean_reward: 0.568 ± 0.46
- Agent solves task ~50-60% of episodes
- High variance suggests inconsistent strategy
- ep_len_mean: 141 steps (eval) — sometimes fast, sometimes fails
- Best model saved: models/best_model.zip