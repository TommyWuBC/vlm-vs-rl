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