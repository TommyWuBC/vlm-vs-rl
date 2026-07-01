"""
evaluate_generalization_rl.py

Tests the best RL agent (trained on S9N1) on harder unseen environments:
- MiniGrid-LavaCrossingS9N2-v0 (2 lava strips)
- MiniGrid-LavaCrossingS9N3-v0 (3 lava strips)

The agent receives no additional training — this is a pure generalization test.
Results reveal whether the learned navigation policy transfers to harder layouts.

Usage:
    python scripts/evaluate_generalization_rl.py
"""

import json
import os
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from stable_baselines3 import PPO
import numpy as np

MODEL_PATH = "./models/best_model"
ENVIRONMENTS = [
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
]
N_EPISODES = 50
MAX_STEPS = 500  # longer limit since harder envs may need more steps


def evaluate_env(model, env_name, n_episodes):
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    results = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        success = False

        while not done and steps < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
            if terminated and reward > 0:
                success = True

        results.append({
            "episode": episode,
            "success": success,
            "steps": steps,
        })

        if (episode + 1) % 10 == 0:
            so_far = sum(r["success"] for r in results)
            print(f"  {env_name} — {episode+1}/{n_episodes} episodes, "
                  f"success so far: {so_far}/{episode+1} "
                  f"({100*so_far/(episode+1):.1f}%)")

    env.close()
    return results


def main():
    os.makedirs("./results", exist_ok=True)

    # load model using a temp env to get obs space right
    temp_env = gym.make("MiniGrid-LavaCrossingS9N1-v0")
    temp_env = FullyObsWrapper(temp_env)
    temp_env = ImgObsWrapper(temp_env)
    model = PPO.load(MODEL_PATH, env=temp_env)
    temp_env.close()
    print(f"Model loaded from {MODEL_PATH}\n")

    all_results = {}
    summary = {}

    for env_name in ENVIRONMENTS:
        print(f"Evaluating on {env_name}...")
        results = evaluate_env(model, env_name, N_EPISODES)

        successes = sum(r["success"] for r in results)
        success_rate = successes / N_EPISODES
        avg_steps = sum(r["steps"] for r in results) / N_EPISODES
        avg_steps_success = (
            sum(r["steps"] for r in results if r["success"]) / successes
            if successes > 0 else None
        )

        all_results[env_name] = results
        summary[env_name] = {
            "n_episodes": N_EPISODES,
            "n_success": successes,
            "success_rate": round(success_rate, 3),
            "avg_steps_all": round(avg_steps, 1),
            "avg_steps_success": round(avg_steps_success, 1) if avg_steps_success else None,
        }

        print(f"  Result: {successes}/{N_EPISODES} ({100*success_rate:.1f}%)\n")

    # save full results
    with open("./results/generalization_rl.json", "w") as f:
        json.dump({"summary": summary, "episodes": all_results}, f, indent=2)

    # print summary
    print("=" * 50)
    print("RL GENERALIZATION RESULTS")
    print("=" * 50)
    print(f"{'Environment':<35} {'Success':>8} {'Rate':>8} {'Avg Steps':>10}")
    print("-" * 50)

    # include S9N1 trained performance for reference
    trained_ref = {
        "MiniGrid-LavaCrossingS9N1-v0 (trained)": {
            "success_rate": 0.952,
            "n_success": 40,
            "n_episodes": 42,
        }
    }
    for env_name, stats in trained_ref.items():
        print(f"{env_name:<35} "
              f"{stats['n_success']:>4}/{stats['n_episodes']:<3} "
              f"{100*stats['success_rate']:>7.1f}%  "
              f"{'N/A':>10}")

    for env_name, stats in summary.items():
        short_name = env_name.replace("MiniGrid-", "").replace("-v0", "")
        print(f"{short_name:<35} "
              f"{stats['n_success']:>4}/{stats['n_episodes']:<3} "
              f"{100*stats['success_rate']:>7.1f}%  "
              f"{stats['avg_steps_all']:>10.1f}")

    print("\nFull results saved to ./results/generalization_rl.json")


if __name__ == "__main__":
    main()
