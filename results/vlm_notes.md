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