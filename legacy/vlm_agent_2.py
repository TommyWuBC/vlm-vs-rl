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

NUM_EPISODES = 20
MAX_STEPS = 100

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

env = gym.make("MiniGrid-LavaCrossingS9N1-v0", render_mode="rgb_array")

prompt = """You are navigating a grid world. You can see the ENTIRE map.
Your location is the red triangle. Your goal is the green square. 
Lava is orange — never move onto it.

Since you can see the full map, plan your route carefully before acting.
Find the gap in the lava wall and navigate through it to reach the goal.

Rules:
- Plan the optimal path using the full map view
- Never move onto lava
- If you see the goal: navigate toward it via the safe path
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

results = []
ACTION_MAP = {
    "turn left": 0,
    "turn right": 1,
    "move forward": 2,
}

for episode in range(NUM_EPISODES):
    observation, info = env.reset()
    steps = 0
    success = False
    actions_taken = []
    reasoning_taken = []

    for step in range(MAX_STEPS):
        full_image = env.render()          # get full grid render
        image_b64 = encode_observation(full_image)  # encode it
        direction = observation["direction"]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": prompt
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
                            "text": f"Current direction: {direction}. Last 5 actions: {actions_taken[-5:]}. What action do you take?"
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
        print(f"Ep {episode} Step {step}: {reasoning[:50]}... → {action_text}")
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
        "reasoning": reasoning_taken
    })

with open("results/vlm_results_full_obs.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done! Results saved.")
env.close()