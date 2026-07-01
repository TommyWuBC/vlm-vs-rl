"""
evaluate_generalization_vlm.py

Tests the best VLM agent (zero-shot, full observability) on harder unseen environments:
- MiniGrid-LavaCrossingS9N2-v0 (2 lava strips)
- MiniGrid-LavaCrossingS9N3-v0 (3 lava strips)

Uses the same configuration that achieved 100% on S9N1:
- GPT-4o-mini
- Full observability (env.render())
- Zero-shot prompt
- 20 episodes per environment (cost-conscious)

Usage:
    python scripts/evaluate_generalization_vlm.py
"""

from dotenv import load_dotenv
import gymnasium as gym
import minigrid
import os
from openai import OpenAI
import base64
from PIL import Image
import io
import json
import numpy as np
import time

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

ENVIRONMENTS = [
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
]
N_EPISODES = 20
MAX_STEPS = 100

# same prompt that achieved 100% on S9N1 — zero-shot full obs
# updated to mention multiple lava strips
PROMPT = """You are navigating a grid world. You can see the ENTIRE map.
Your location is the red triangle. Your goal is the green square.
Lava is orange — never move onto it. There may be multiple lava walls
with gaps you need to navigate through in sequence.

Since you can see the full map, plan your route carefully before acting.
Find the gaps in the lava walls and navigate through them to reach the goal.

Rules:
- Plan the optimal path using the full map view
- Never move onto lava
- There may be multiple lava strips — find and navigate through each gap in sequence
- Navigate toward the goal via the safe path through all lava gaps
- Avoid backtracking

Valid actions: turn left, turn right, move forward
Direction reference: 0=right, 1=down, 2=left, 3=up

Respond ONLY in this exact format:
reasoning: [one sentence about the full map layout and your planned path]
action: [turn left / turn right / move forward]"""


def encode_observation(obs):
    image = Image.fromarray(obs.astype(np.uint8))
    image = image.resize((224, 224), Image.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode()
    return image_b64


ACTION_MAP = {
    "turn left": 0,
    "turn right": 1,
    "move forward": 2,
}


def evaluate_env(env_name, n_episodes):
    env = gym.make(env_name, render_mode="rgb_array")
    results = []

    for episode in range(n_episodes):
        observation, info = env.reset()
        steps = 0
        success = False
        actions_taken = []
        reasoning_taken = []

        for step in range(MAX_STEPS):
            full_image = env.render()
            image_b64 = encode_observation(full_image)
            direction = observation["direction"]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": (f"Current direction: {direction}. "
                                         f"Last 5 actions: {actions_taken[-5:]}. "
                                         f"What action do you take?")
                            }
                        ]
                    }
                ],
                max_tokens=120
            )

            action_text = response.choices[0].message.content.strip().strip('[].').lower()
            reasoning = ""
            if "action:" in action_text:
                parts = action_text.split("action:")
                reasoning = parts[0].replace("reasoning:", "").strip()
                action_text = parts[-1].strip()
                reasoning_taken.append(reasoning)

            action_int = ACTION_MAP.get(action_text, 2)
            observation, reward, terminated, truncated, info = env.step(action_int)
            steps += 1
            actions_taken.append(action_text)

            preview = reasoning[:60] if reasoning else action_text
            print(f"  Ep {episode} Step {step}: {preview}... → {action_text}")

            time.sleep(3)

            if terminated:
                success = True
            if terminated or truncated:
                break

        results.append({
            "episode": episode,
            "success": success,
            "steps": steps,
            "actions": actions_taken,
            "reasoning": reasoning_taken,
        })
        print(f"Episode {episode} done — success: {success}, steps: {steps}")

    env.close()
    return results


def main():
    os.makedirs("./results", exist_ok=True)

    all_results = {}
    summary = {}

    for env_name in ENVIRONMENTS:
        print(f"\nEvaluating VLM on {env_name}...")
        results = evaluate_env(env_name, N_EPISODES)

        successes = sum(r["success"] for r in results)
        success_rate = successes / N_EPISODES
        avg_steps = sum(r["steps"] for r in results) / N_EPISODES

        all_results[env_name] = results
        summary[env_name] = {
            "n_episodes": N_EPISODES,
            "n_success": successes,
            "success_rate": round(success_rate, 3),
            "avg_steps": round(avg_steps, 1),
        }

        # save after each env in case of API errors midway
        with open("./results/generalization_vlm.json", "w") as f:
            json.dump({"summary": summary, "episodes": all_results}, f, indent=2)

        print(f"\n{env_name}: {successes}/{N_EPISODES} ({100*success_rate:.1f}%)")

    # final summary
    print("\n" + "=" * 55)
    print("VLM GENERALIZATION RESULTS")
    print("=" * 55)
    print(f"{'Environment':<30} {'Success':>10} {'Rate':>8} {'Avg Steps':>10}")
    print("-" * 55)

    # S9N1 reference
    print(f"{'S9N1 (trained dist.)' :<30} {'20/20':>10} {'100.0%':>8} {'~35':>10}")

    for env_name, stats in summary.items():
        short = env_name.replace("MiniGrid-LavaCrossing", "").replace("-v0", "")
        print(f"{short:<30} "
              f"{stats['n_success']:>4}/{stats['n_episodes']:<5} "
              f"{100*stats['success_rate']:>7.1f}%  "
              f"{stats['avg_steps']:>10.1f}")

    print("\nFull results saved to ./results/generalization_vlm.json")


if __name__ == "__main__":
    main()


