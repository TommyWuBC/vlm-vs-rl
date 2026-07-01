from dotenv import load_dotenv
import gymnasium as gym
import minigrid
import os
from openai import OpenAI
import base64 #converts image bytes to text so we can send it in an API call
from PIL import Image 
import io #memory manipulation
import json
import numpy as np
import time

NUM_EPISODES = 20
MAX_STEPS = 100

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

env = gym.make("MiniGrid-LavaCrossingS9N1-v0")
prompt = """You are a robot in a grid world. Your location is the red triangle. Your goal is the green square. Lava is orange — never move onto it.

CRITICAL: Base your decision PRIMARILY on what you SEE in the image. Do not try to mathematically track your orientation from past actions — trust the image instead.

You can only see a 7x7 area around you. The goal may not be visible. If you cannot see the goal, explore systematically. Your last 5 actions are provided to help you avoid backtracking — use them as a hint, not as your primary reasoning source.

Rules:
- If you see the green goal square: navigate toward it
- If you see lava immediately ahead: turn instead of moving forward  
- If you keep repeating the same actions: try something different
- Trust what you see in the image over what you calculate from history

Valid actions: turn left, turn right, move forward
Direction reference: 0=right, 1=down, 2=left, 3=up

Respond ONLY in this exact format:
reasoning: [one sentence about what you see and why]
action: [turn left / turn right / move forward]"""

def encode_observation(obs): 
    image = Image.fromarray(obs.astype(np.uint8))
    image = image.resize((224, 224), Image.NEAREST)
    buffer = io.BytesIO() #memory buffer - so doesnt save on the disk
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
    actions_taken = []# start fresh episode, new random goal position
    reasoning_taken = []
    
    for step in range(MAX_STEPS):
        image_b64 = encode_observation(observation["image"])
        direction = observation["direction"]
        # Dynamic - built each step inside the loop
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": [ {
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
        action_int = ACTION_MAP.get(action_text, 2)  #defaults to move forward if unrecognized
        observation, reward, terminated, truncated, info = env.step(action_int)
        steps += 1
        actions_taken.append(action_text)
        print(f"Ep {episode} Step {step}: {reasoning[:50]}... → {action_text}")
        time.sleep(3) 
        if terminated:
            success = True
        if terminated or truncated:
            break
    # record results for this episode
    results.append({
        "episode": episode,
        "success": success,
        "steps": steps,
        "actions": actions_taken,
        "reasoning": reasoning_taken
    })
    
#save results
with open("results/vlm_results_cot.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done! Results saved.")
env.close()