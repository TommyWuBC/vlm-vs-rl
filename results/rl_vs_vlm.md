## Generalization Test — RL and VLM on Unseen Lava Configurations

### Motivation
Both final agents (RL full obs CNN, VLM zero-shot full obs) achieved high
performance on MiniGrid-LavaCrossingS9N1-v0 (1 lava strip): 95.2% and 100%
respectively. However, the RL agent was trained specifically on this
environment's distribution of 42 layouts, while the VLM received no
environment-specific training at all. This raises the question: does RL's
strong performance reflect genuine navigational understanding, or pattern
memorization specific to the trained distribution?

To test this, both agents were evaluated with zero additional training on:
- MiniGrid-LavaCrossingS9N2-v0 (2 lava strips, structurally novel)
- MiniGrid-LavaCrossingS9N3-v0 (3 lava strips, structurally novel)

### Setup
- **RL agent:** best_model from Run 8 (CnnPolicy, FullyObsWrapper, BFS reward
  shaping, trained on S9N1 only). No retraining. 50 episodes per environment,
  max 500 steps per episode (environment's own step limit ~324).
- **VLM agent:** GPT-4o-mini, zero-shot, full observability (same configuration
  that achieved 100% on S9N1). Prompt updated to mention possible multiple
  lava strips. 20 episodes per environment.

### Results

| Agent | S9N1 (trained) | S9N2 (unseen) | S9N3 (unseen) |
|---|---|---|---|
| RL full obs CNN | 95.2% (40/42 patterns) | 20.0% (10/50) | 10.0% (5/50) |
| VLM zero-shot full obs | 100% (20/20) | 100% (20/20) | 100% (20/20) |

Average steps for successful episodes:
- RL: ~14 steps (S9N2 and S9N3) — comparable to S9N1
- VLM: 9.4 steps (S9N2), 8.9 steps (S9N3) — *faster* than S9N1 (~35 steps)

### Key Finding: RL Collapse vs VLM Robustness

**RL performance collapsed catastrophically under distribution shift** — from
95.2% on the trained distribution to 20% and 10% on structurally novel
environments with zero additional training. Two distinct failure patterns were
observed in the RL episode data:

1. **Step-limit timeout (~50% of failures):** Many episodes hit the environment's
   max step limit (324 steps) without reaching the goal or dying — the same
   freezing/paralysis behavior observed earlier in partial-obs experiments,
   now triggered by genuinely novel lava topology the policy never encountered.

2. **Fast lava death (~50% of failures):** A substantial fraction of episodes
   failed in under 15 steps, suggesting the agent confidently applied a
   memorized single-strip navigation pattern and walked directly into the
   second or third lava strip, which its learned policy had no concept of.

3. **Successes cluster narrowly:** Episodes that succeeded did so in 13–17
   steps — suggesting the agent only succeeded on S9N2/S9N3 layouts that
   happened to closely resemble S9N1 patterns it had memorized, rather than
   demonstrating genuine adaptive navigation.

**VLM performance was unaffected by structural novelty.** Both unseen
environments achieved 100% success, with *faster* average step counts than
on the original S9N1 evaluation. Reasoning traces show the VLM correctly
identifying and sequentially navigating through multiple lava gaps without
any special handling — it reasons about "find the gap, move through it,
repeat" as a general strategy rather than environment-specific memorization.

### Interpretation

This result reframes the entire RL vs VLM comparison. The RL agent's strong
performance on S9N1 (95.2%, beating the VLM's earlier 85% partial-obs result)
reflects **specialized pattern learning**, not general navigational
competence. The CNN policy learned an effective lookup-table-like mapping
from visual layout to action sequence for the 42 trained patterns, but this
mapping carries no transferable concept of "obstacle avoidance" that extends
to structurally different layouts.

The VLM, despite receiving no training at all, demonstrates **genuine
generalization** because its navigation strategy is grounded in symbolic
reasoning about gaps and paths rather than learned visual-to-action mappings.
A third lava strip is not a novel situation requiring new learning — it is
simply one more instance of a general concept ("find the gap") the VLM
already possesses from pretraining.

### Significance

This is the central finding of the project. It provides a clean, concrete
demonstration of a fundamental and actively studied tradeoff in AI: the
distinction between **specialized, sample-efficient policies that excel
within their training distribution but fail to generalize**, versus
**general-purpose reasoning systems that sacrifice peak in-distribution
performance for robustness under distribution shift**. The RL agent required
~10 reward-engineering iterations and millions of environment steps to reach
95.2% on a fixed set of 42 patterns; the VLM required zero training and
achieved perfect generalization to environments it had never encountered,
at the cost of being initially outperformed on the narrow trained
distribution under partial observability (85% vs RL's eventual 95.2%).

### Limitations
- VLM generalization tested with n=20 per environment; RL with n=50.
  Larger sample sizes would strengthen confidence intervals on both estimates.
- Only one VLM configuration (zero-shot, full obs) was tested for
  generalization; CoT full obs was not retested on S9N2/S9N3 despite
  underperforming on S9N1, as the zero-shot configuration was already
  established as the stronger baseline.
- RL generalization failure was not diagnosed at the level of individual
  failed episodes' lava layouts; a pattern-level breakdown analogous to the
  42-pattern evaluation on S9N1 was not performed for S9N2/S9N3 due to time
  constraints.