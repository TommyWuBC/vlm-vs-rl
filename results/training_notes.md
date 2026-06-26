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

## Run 5 — LavaCrossing, 5M steps, improved reward shaping
- Environment: MiniGrid-LavaCrossingS9N1-v0
- Timesteps: 5,000,000
- Reward shaping:
  - +0.01 for moving closer to goal (distance-based)
  - -0.005 penalty for not moving (wall collision)
  - +0.002 for each step goal is visible in 7x7 observation window
- Motivation: Run 4 (2M steps) produced fixed pattern behavior —
  agent learned "move forward, turn left" loop rather than adaptive
  navigation. Hypothesized causes:
  1. Exploration bonus caused reward hacking (wandering > goal-seeking)
  2. 2M steps insufficient for adaptive policy to emerge
  3. No penalty for wall collisions allowed degenerate looping behavior
- Changes from Run 4:
  - Removed exploration bonus entirely
  - Added wall collision penalty
  - Added goal visibility reward to encourage agent to seek
    and maintain line of sight to goal
  - Increased to 5M steps
- Result: ~15% success rate (see Run 6 for diagnosis)

## Engineering Optimization — Training Throughput
- Problem: training was running at ~200 FPS despite GPU availability
- Root causes identified:
  1. `gen_obs()` called manually inside reward wrapper every step,
     causing observation to be rendered twice per step. This was
     redundant — SB3 already generates observations automatically
     as part of the environment loop.
  2. Single environment instance leaving GPU underutilized between
     environment steps (CPU-bound bottleneck)
  3. Custom CNN feature extractor overkill for 7x7 observation —
     Conv2d operations added overhead without representational benefit
     at this input size
- Fixes applied:
  1. Removed `gen_obs()` call and goal visibility bonus from reward wrapper
  2. Switched to 8 parallel environments via `make_vec_env(make_env, n_envs=8)`
  3. Switched from CnnPolicy + custom MinigridFeaturesExtractor to MlpPolicy,
     which flattens the 7x7x3 observation to a 147-dim vector internally
- Result: 200 FPS → 1600+ FPS (~8x throughput improvement)
- Note: fps naturally drops as training progresses (e.g. to ~177 fps at 11.5M
  steps) because episodes get longer as the agent learns to survive —
  this is expected and indicates learning, not regression

## Run 6 — Reward Wrapper Rewrite + 30M steps
- Environment: MiniGrid-LavaCrossingS9N1-v0
- Timesteps: 30,000,000 (ongoing)
- Policy: MlpPolicy (switched from CnnPolicy)
- Parallel envs: 8
- Reward shaping (full rewrite):
  - Manhattan distance used instead of Euclidean (integer-valued,
    no floating point issues on discrete grid)
  - prev_dist initialized in reset() to avoid NoneType error on
    first step of each episode
  - +0.1 for moving closer to goal
  - -0.05 for moving away from goal
  - -0.02 for stagnating (same distance)
  - -0.001 step penalty to encourage efficiency
  - if reward != 0: return reward — terminal signals (goal/lava)
    no longer diluted by shaping
  - Removed gen_obs() visibility bonus (redundant + major perf bottleneck)
- Motivation: Run 5 achieved only ~15% success rate. Diagnosis:
  1. Reward signals too weak relative to lava penalty (+0.01 shaping
     vs -1 lava death made lava avoidance dominate)
  2. Euclidean distance caused floating point equality issues in
     stagnation detection
  3. prev_dist = None in reset() caused TypeError on first step
  4. Terminal reward dilution — goal reward of +1 was being modified
     by shaping wrapper instead of passed through directly
- Observations at 11.5M steps:
  - ep_rew_mean: 4.88 (strong positive, major improvement over Run 5)
  - ep_len_mean: 324 (agent surviving much longer, learning to avoid lava)
  - eval mean_reward: 0.00 (agent not reaching goal in eval episodes)
  - Likely cause of eval/train discrepancy: agent optimizing for shaped
    rewards without completing task — eval uses sparse rewards only
  - Visualization reveals lava-aversion freezing behavior: agent refuses
    to enter narrow corridors even when goal is directly behind them

## Run 6 Reward Adjustment — Lava Corridor Freezing Fix
- Problem: agent learned excessive lava aversion, freezing at narrow
  corridors rather than crossing to reach goal
- Rejected fixes:
  - Lava proximity survival bonus → agent could farm by wandering near lava
  - Increasing goal reward alone → doesn't resolve freezing without
    also increasing cost of hovering
- Fix applied:
  - Increased step penalty from -0.001 to -0.005 (5x)
  - Override goal reward to flat +10.0 (removes time-decay, massively
    increases relative value of task completion)
  - Lava penalty left unchanged
- Rationale: higher step penalty makes hovering near corridors
  increasingly costly over time, forcing agent to commit to crossing.
  Flat +10 goal reward makes the risk/reward calculation strongly favor
  attempting the corridor. No new reward signals introduced — purely
  a rebalancing of existing ones.
- Result: TBD

## Key Findings So Far

### Reward Shaping Sensitivity
Small changes in reward signal scaling have outsized effects on learned
behavior. The agent is highly sensitive to the relative magnitudes of
shaping rewards vs environment penalties:
- Exploration bonus too high (Run 2) → reward hacking, ignores goal
- Shaping signals too weak vs lava penalty (Run 5) → excessive
  lava aversion, freezing at corridors
- Rebalancing without adding new signals is preferable to adding
  new reward terms which introduce new gaming opportunities

### RL vs VLM Asymmetry
The VLM agent (GPT-4o-mini) requires zero training — inference cost
is purely per-episode API calls. The RL agent by contrast requires
significant engineering investment before it can perform, and that
investment is sensitive to implementation details that compound:
reward misspecification, observation bottlenecks, policy architecture
choices, and parallelization all meaningfully affect outcome. This
asymmetry in setup cost and engineering overhead is itself an
interesting dimension of the comparison, independent of final
performance numbers.

### Reward Misspecification as a Research Theme
The lava-aversion freezing behavior is a concrete instance of reward
misspecification — the agent learned a locally rational policy
(avoid lava) that is globally suboptimal (never reaches goal through
narrow corridors). The VLM agent does not exhibit this failure mode
because it reasons symbolically about paths rather than learning a
fear response through trial and error. This contrast is a meaningful
finding worth discussing in the final report.

## Run 7 — BFS Distance Shaping, Partial Observability, 3x Lava Penalty
- Environment: MiniGrid-LavaCrossingS9N1-v0
- Timesteps: ~7-8M (evaluated mid-training, not final)
- Policy: MlpPolicy
- Parallel envs: 8
- Device: CPU (MlpPolicy too small to benefit from GPU — transfer
  overhead exceeds compute savings)
- Observation: Partial (default 7x7 agent-centric view via ImgObsWrapper)
- Reward shaping (v5_bfs / v6_bfs_partial_obs):
  - BFS distance to goal treating lava as impassable
  - +0.1 for reducing BFS distance
  - -0.05 for increasing BFS distance
  - -0.02 for stagnating
  - -0.005 step penalty
  - Goal reward: flat +10.0 (overrides time-decayed default)
  - Lava penalty: 3x default (-3.0 instead of -1.0)
- Motivation: previous Manhattan distance shaping (v3/v4) misdirected
  agent because straight-line distance doesn't respect lava layout.
  BFS computes true navigable path distance, correctly rewarding moves
  that make genuine progress even when they temporarily increase
  straight-line distance to goal.
- Per-pattern evaluation results (42 distinct patterns):
  - Overall success rate: 39.8%
  - Solved (≥80%): 18/42 patterns
  - Partial (20-80%): 0/42 patterns — stark binary split
  - Failed (<20%): 24/42 patterns

## Run 7 — Key Finding: Partial Observability Generalization Failure
The per-pattern breakdown reveals a systematic, non-random failure mode:
- All 18 solved patterns have lava walls at y=2 or y=4 (horizontal),
  or vertical walls with gaps at y≥2 (middle/bottom of grid)
- All 24 failed patterns share a structural property: lava at y=6
  (bottom rows) or vertical walls where the gap is at y=1 (top row)

This is not random undertraining — it is a systematic generalization
failure caused by partial observability. The agent's 7x7 view combined
with its fixed starting position means it rarely encounters top-row
gaps or bottom lava walls early in episodes. It has learned
position-specific navigation heuristics that don't transfer to
unseen gap locations.

The VLM agent does not exhibit this failure because it receives a full
natural language description of the task and reasons about it globally,
independent of viewport position. This contrast directly illustrates
a fundamental limitation of partial observability in RL navigation tasks.

### Planned Fix: Full Observability (Run 8)
Switch ImgObsWrapper → FullyObsWrapper to give the agent the entire
9x9 grid. If success rate increases significantly, this provides direct
empirical evidence that partial observability is the cause of the
generalization failure — not insufficient training or reward design.
This sets up a clean three-way comparison:
  - VLM agent (GPT-4o-mini): 85% — global symbolic reasoning, zero training
  - RL + full observability: TBD — global visual input, learned policy
  - RL + partial observability: 39.8% — local visual input, learned policy