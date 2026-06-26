"""
evaluate_patterns.py

Runs the trained PPO agent across all distinct LavaCrossing map patterns
and reports per-pattern success rates. This gives a richer picture than
overall success rate alone — revealing which layout types the agent has
and hasn't generalized to.

Usage:
    python scripts/evaluate_patterns.py

Output:
    - Console: per-pattern results sorted by success rate
    - results/pattern_eval.json: full results for further analysis
    - results/pattern_eval_summary.txt: human-readable summary
"""

import json
import os
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from collections import defaultdict
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "./models/best_model"
ENV_NAME = "MiniGrid-LavaCrossingS9N1-v0"
N_EVAL_EPISODES = 500      # total episodes — spread across all patterns
EPISODES_PER_PATTERN = 20  # minimum episodes per pattern before reporting
MAX_STEPS = 500            # max steps per episode before counting as failure
# ─────────────────────────────────────────────────────────────────────────────


def serialize_grid(env):
    """
    Serialize lava positions into a canonical string key.
    Two episodes with identical lava layouts produce the same key.
    """
    lava_positions = []
    for i in range(env.unwrapped.width):
        for j in range(env.unwrapped.height):
            cell = env.unwrapped.grid.get(i, j)
            if cell is not None and cell.type == 'lava':
                lava_positions.append((i, j))
    return str(sorted(lava_positions))


def evaluate(model_path, env_name, n_episodes, episodes_per_pattern):
    # Load model
    print(f"Loading model from {model_path}...")
    env = gym.make(env_name)
    env = ImgObsWrapper(env)
    model = PPO.load(model_path, env=env)
    print("Model loaded.\n")

    # Track results per pattern
    pattern_results = defaultdict(list)   # pattern_key -> [True/False, ...]
    pattern_examples = {}                 # pattern_key -> lava positions (for display)
    episode = 0

    print(f"Running evaluation ({n_episodes} episodes)...")
    while True:
        obs, _ = env.reset()
        pattern_key = serialize_grid(env)

        # Store a human-readable version of this pattern
        if pattern_key not in pattern_examples:
            lava_positions = []
            for i in range(env.unwrapped.width):
                for j in range(env.unwrapped.height):
                    cell = env.unwrapped.grid.get(i, j)
                    if cell is not None and cell.type == 'lava':
                        lava_positions.append((i, j))
            pattern_examples[pattern_key] = sorted(lava_positions)

        # Run episode
        done = False
        steps = 0
        success = False
        while not done and steps < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if terminated and reward > 0:
                success = True

        pattern_results[pattern_key].append(success)
        episode += 1

        # Stop when every pattern has been seen enough times
        min_episodes = min(len(v) for v in pattern_results.values()) if pattern_results else 0
        all_covered = (
            len(pattern_results) >= 2 and  # found at least some patterns
            min_episodes >= episodes_per_pattern and
            episode >= n_episodes
        )
        if all_covered:
            break

        if episode % 100 == 0:
            print(f"  {episode} episodes, {len(pattern_results)} patterns discovered...")

    env.close()
    return pattern_results, pattern_examples


def report(pattern_results, pattern_examples):
    os.makedirs("./results", exist_ok=True)

    # Compute per-pattern stats
    stats = []
    for key, results in pattern_results.items():
        success_rate = sum(results) / len(results)
        stats.append({
            "pattern_id": len(stats) + 1,
            "lava_positions": pattern_examples[key],
            "n_episodes": len(results),
            "n_success": sum(results),
            "success_rate": round(success_rate, 3),
        })

    # Sort by success rate
    stats.sort(key=lambda x: x["success_rate"], reverse=True)

    # Assign clean pattern IDs after sorting
    for i, s in enumerate(stats):
        s["pattern_id"] = i + 1

    # Overall stats
    all_results = [r for results in pattern_results.values() for r in results]
    overall_success = sum(all_results) / len(all_results)
    n_patterns = len(stats)
    n_solved = sum(1 for s in stats if s["success_rate"] >= 0.8)
    n_partial = sum(1 for s in stats if 0.2 <= s["success_rate"] < 0.8)
    n_failed = sum(1 for s in stats if s["success_rate"] < 0.2)

    # ── Console output ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total episodes:       {len(all_results)}")
    print(f"Distinct patterns:    {n_patterns}")
    print(f"Overall success rate: {overall_success:.1%}")
    print(f"  Solved (≥80%):      {n_solved}/{n_patterns} patterns")
    print(f"  Partial (20-80%):   {n_partial}/{n_patterns} patterns")
    print(f"  Failed (<20%):      {n_failed}/{n_patterns} patterns")
    print("="*60)

    print("\nPer-pattern results (sorted by success rate):")
    print(f"{'ID':>4}  {'Success':>8}  {'Episodes':>9}  Lava positions")
    print("-"*60)
    for s in stats:
        bar = "█" * int(s["success_rate"] * 10) + "░" * (10 - int(s["success_rate"] * 10))
        print(f"  {s['pattern_id']:>2}  {s['success_rate']:>7.1%}  {s['n_episodes']:>9}  {bar}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "overall_success_rate": round(overall_success, 3),
        "total_episodes": len(all_results),
        "n_patterns": n_patterns,
        "n_solved": n_solved,
        "n_partial": n_partial,
        "n_failed": n_failed,
        "per_pattern": stats,
    }
    with open("./results/pattern_eval.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nFull results saved to ./results/pattern_eval.json")

    # ── Save summary txt ──────────────────────────────────────────────────────
    with open("./results/pattern_eval_summary.txt", "w") as f:
        f.write("PATTERN EVALUATION SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Overall success rate: {overall_success:.1%}\n")
        f.write(f"Distinct patterns:    {n_patterns}\n")
        f.write(f"Solved (>=80%):       {n_solved}\n")
        f.write(f"Partial (20-80%):     {n_partial}\n")
        f.write(f"Failed (<20%):        {n_failed}\n\n")
        f.write("Per-pattern breakdown:\n")
        for s in stats:
            f.write(f"  Pattern {s['pattern_id']:>2}: {s['success_rate']:.1%} "
                    f"({s['n_success']}/{s['n_episodes']} episodes)\n")
            f.write(f"    Lava: {s['lava_positions']}\n")
    print("Summary saved to ./results/pattern_eval_summary.txt")

    return overall_success, stats


if __name__ == "__main__":
    pattern_results, pattern_examples = evaluate(
        MODEL_PATH, ENV_NAME, N_EVAL_EPISODES, EPISODES_PER_PATTERN
    )
    overall, stats = report(pattern_results, pattern_examples)
