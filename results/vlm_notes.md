## First run - Zero-Shot VLM Results
- Episodes: 20
- Success rate: 0/20 (0%)
- Primary failure mode: "move forward" loop (stuck hitting walls)
- Cause: no memory between steps, low-res observation, no reasoning

### Key Findings
1. **Move forward loop:** Agent defaulted to "move forward" almost exclusively,
   occasionally turning but never navigating purposefully.

2. **Hallucination detected:** VLM reasoning frequently referenced the green goal
   ("I can see the goal to my right") despite it being outside the 7x7 partial
   observation window. The VLM was confabulating environmental features not
   present in the actual image — a known failure mode of VLMs called hallucination.

3. **No memory between steps:** Each API call is independent. The VLM has no
   knowledge of previous steps, so it cannot recognize it has been hitting the
   same wall repeatedly.

4. **Environment too large:** FourRooms (19x19) is too large for meaningful
   zero-shot navigation with 100 step limit and partial observation.

### Decision
Switched to MiniGrid-LavaCrossing-S9N1-v0 for remaining experiments:
- 9x9 room with lava obstacles
- Smaller, more navigable within step limit
- Obstacles give VLM meaningful visual features to reason about
- Requires RL retraining on new environment

## Prompt Engineering Iterations

### Iteration 1 — Zero-Shot (FourRooms)
- Prompt: simple task description, valid actions, direction info
- Result: 0/20 success rate
- Failure modes:
  - "Move forward" loop — agent defaulted to same action repeatedly
  - Hallucination — VLM referenced goal position outside 7x7 observation window
  - No memory — each step independent, agent unaware of repeating same mistake
- Environment issue: FourRooms 19x19 too large, goal rarely visible in 7x7 window

### Iteration 2 — Chain-of-Thought (FourRooms → LavaCrossing)
- Prompt improvements:
  1. Explicit partial observation notice ("you only see 7x7, not whole map")
  2. Last 5 actions passed in user message as surrogate memory
  3. Explicit lava avoidance instruction
  4. "Done" action instruction when standing on goal
  5. Exploration encouragement when goal not visible
- Environment switched to LavaCrossing (9x9) for smaller, more navigable space
  with meaningful visual obstacles
- Result: TBD

## CoT Evaluation Round 1 (MiniGrid-LavaCrossingS9N1-v0)
- Episodes: 10
- Success rate: 4/10 (40%)
- Model: gpt-4o-mini
- Max steps: 100


### Key Findings
1. **Lucky spawn successes:** All 4 successes were 1-3 steps — goal spawned 
   directly in front of agent. Not genuine navigation ability.

2. **Dead reckoning failure:** VLM ignored image and tried to track orientation 
   mathematically from action history ("after 4 left turns I face right again"). 
   This caused contradictory reasoning and navigation loops.

3. **Parsing inconsistency:** Model output actions with brackets and periods 
   ([move forward].) causing ACTION_MAP mismatches. Fixed with strip('[].')

4. **CoT reasoning quality:** Verbose but unreliable — model reasoned confidently 
   but incorrectly about its orientation and surroundings.

### Changes for Round 2
- Fixed parsing bug: strip('[].')
- Strengthened prompt: explicit instruction to prioritize image over action history
- Explicit warning against dead reckoning
- Cleaner output format specification
- Increased to 20 episodes for statistical validity


## CoT Evaluation Round 2 (MiniGrid-LavaCrossingS9N1-v0)
- Episodes: 20
- Success rate: 17/20 (85%)
- Model: gpt-4o-mini
- Max steps: 100

### Key Improvements Over Round 1
- Prompt change: explicit instruction to prioritize image over action history
- Parsing fix: strip('[].')
- Result: dramatic improvement from 40% to 85% success rate

### Reasoning Quality
- Agent now correctly identifies lava, walls, and open paths from image
- No dead reckoning confusion observed
- Genuine multi-step navigation demonstrated (episodes 2, 19: 31 and 49 steps)
- Short successes (1-9 steps) likely reflect favorable spawn positions

### Remaining Failures
- 3/20 episodes failed at 100 steps
- Failure mode: agent gets lost exploring open areas without systematic strategy
- No memory of visited cells means repeated exploration of same areas

### Prompt Engineering Insight
- Adding "prioritize image over action history" was the critical fix
- Demonstrates that VLM prompt design significantly affects navigation performance
- This is a direct finding relevant to LLM evaluation research

## VLM Experiment 3 — Zero-Shot Full Observability
- Episodes: 20
- Success rate: 20/20 (100%)
- Model: gpt-4o-mini
- Observation: Full 9x9 grid render via env.render() (rgb_array mode)
- Prompt: zero-shot, updated to reflect full map visibility
- Max steps: 100

### Changes from Partial Obs Baseline
- Switched from observation["image"] (7x7 partial) to env.render() (full grid)
- Updated prompt to inform VLM it can see the entire map
- Same zero-shot format as Round 2 baseline (no structured reasoning steps)

### Key Findings
1. **Perfect success rate:** 100% vs 85% with partial obs — full observability
   eliminates the primary VLM failure mode (getting lost in unexplored areas).

2. **Step efficiency is inconsistent:** Episodes ranged from 1 step (episode 8,
   episode 19) to 70 steps (episode 5). Full observability does not guarantee
   efficient navigation — the VLM still reasons one step at a time and
   sometimes wanders despite seeing the full map.

3. **Spatial hallucination persists:** Even with full observability, reasoning
   is sometimes contradictory across steps — e.g. reporting "path is clear below"
   then turning instead of moving forward. The VLM sees the full grid but does
   not maintain a coherent internal spatial model across steps.

4. **Comparison with RL full obs:** RL full obs CNN achieved 95.2%, slightly
   below VLM's 100% on the trained distribution. However RL solved patterns
   in fewer steps and with more consistent behavior, while VLM's step counts
   vary wildly (1 to 70). This suggests VLM navigation is correct but
   not optimal, while RL has learned a more structured policy.

---

## VLM Experiment 4 — Chain-of-Thought Full Observability
- Episodes: 20
- Success rate: 17/20 (85%)
- Model: gpt-4o-mini
- Observation: Full 9x9 grid render (same as Experiment 3)
- Prompt: structured four-step CoT (position → goal → gap → next step)
- Max steps: 100

### Changes from Experiment 3
- Added structured four-step reasoning format:
  1. position: where am I on the map?
  2. goal: where is the green square?
  3. gap: where is the gap in the lava wall?
  4. next_step: what is the single best action?
- max_tokens increased to 200 to accommodate longer responses
- Failed episodes: 3, 5, 14

### Key Finding: CoT Degraded Performance
**Counterintuitively, structured CoT prompting reduced success rate from 100%
to 85% — identical to the partial obs baseline.**

Diagnosis from reasoning logs:
1. **Gap identification is noisy:** The model frequently reported contradictory
   gap locations across consecutive steps on the same map (e.g. "gap: left side"
   then "gap: right side" one step later). The structured format made this
   inconsistency action-consequential on every step.

2. **Re-evaluation introduces instability:** Zero-shot allowed the VLM to
   commit to a direction and move; CoT forced full re-evaluation of the spatial
   layout every step, creating more opportunities for spatial confusion and
   direction reversal.

3. **Step efficiency did not improve:** Average steps per successful episode
   (~42) was higher than zero-shot full obs (~35), contradicting the hypothesis
   that explicit reasoning would produce more efficient paths.

4. **CoT helps reasoning tasks, not spatial navigation:** This finding aligns
   with known limitations of CoT prompting — it improves performance on tasks
   requiring logical or arithmetic chain reasoning, but spatial navigation
   requires consistent commitment to a path, not re-evaluation at each step.

### Implication
The best VLM configuration is zero-shot with full observability. CoT prompting,
despite being a commonly recommended technique for improving LLM performance,
is counterproductive for step-by-step spatial navigation tasks.

---

## VLM Ablation Summary

| Configuration | Obs | Prompt | Success | Notes |
|---|---|---|---|---|
| Iteration 1 (FourRooms) | 7x7 partial | zero-shot | 0% | env too large |
| CoT Round 1 | 7x7 partial | CoT | 40% | lucky spawns only |
| CoT Round 2 (baseline) | 7x7 partial | CoT | 85% | genuine navigation |
| Zero-shot full obs | 9x9 full | zero-shot | 100% | best VLM config |
| CoT full obs | 9x9 full | CoT | 85% | CoT hurt performance |

### Cross-Agent Comparison (Final)

| Agent | Obs | Success | Notes |
|---|---|---|---|
| RL partial obs MLP (Run 7) | 7x7 partial | 47.6% | binary split, 20/42 patterns |
| VLM CoT baseline | 7x7 partial | 85% | genuine navigation with prompt engineering |
| RL full obs CNN (Run 8) | 9x9 full | 95.2% | 40/42 patterns, best RL config |
| VLM zero-shot full obs | 9x9 full | 100% | best VLM config |

### Key Takeaways
1. Observability is the dominant factor for both agents — both improve
   dramatically when given full grid information.
2. VLM benefits more from full observability in relative terms (85% → 100%)
   than RL did (47.6% → 95.2%), because VLM failure under partial obs was
   primarily navigational (getting lost), while RL failure was structural
   (pattern generalization ceiling).
3. Prompt engineering matters as much for VLMs as reward engineering does for
   RL — the VLM's journey from 0% to 100% mirrors the RL agent's journey
   through 9 reward designs.
4. CoT prompting is not universally beneficial — it actively hurts performance
   on spatial navigation where consistency of commitment matters more than
   step-by-step re-evaluation.