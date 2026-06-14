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

NUM_EPISODES = 1
MAX_STEPS = 30

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

env = gym.make("MiniGrid-FourRooms-v0")
prompt = """You are a robot in this minigrid world whose location is denoted by the red triangle. Your goal is to 
reach the goal (preferably as quickly as possible), denoted by the green square. Each time, you may move once by stating an action
from: (turn left, turn right, move forward, pick up, done). You will also be given an initial direction from 
0-3, where 0=right, 1=down, 2=left, 3=up. STRICTLY ONLY RESPOND WITH THE ACTION AND NOTHING ELSE"""

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
    "pick up": 3,
    "done": 6
}
for episode in range(NUM_EPISODES):
    observation, info = env.reset()
    steps = 0
    success = False
    actions_taken = []# start fresh episode, new random goal position
    
    for step in range(MAX_STEPS):
        image_b64 = encode_observation(observation["image"])
        direction = observation["direction"]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": prompt  #static instructions
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
                            "text": f"Current direction: {direction}. What action do you take?"
                        }
                    ]
                }
            ],
            max_tokens=20
        )
        action_text = response.choices[0].message.content.strip().lower()
        action_int = ACTION_MAP.get(action_text, 2)  #defaults to move forward if unrecognized
        observation, reward, terminated, truncated, info = env.step(action_int)
        steps += 1
        actions_taken.append(action_text)
        time.sleep(1) 
        if terminated:
            success = True
        if terminated or truncated:
            break
    # record results for this episode
    results.append({
        "episode": episode,
        "success": success,
        "steps": steps,
        "actions": actions_taken
    })
#save results
with open("results/vlm_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done! Results saved.")
env.close()