"""
make_gif.py — render a navigation episode of the trained RL agent as a GIF.

Usage:
    python scripts/make_gif.py

Output:
    results/agent_navigation.gif   — single best episode as a GIF
    results/agent_navigation_fail.gif — one failure episode (if found)

Requirements: pip install imageio[ffmpeg] Pillow
The script uses env.render(mode="rgb_array") so no display window is needed.
"""

import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
import numpy as np
from PIL import Image
import imageio
import os
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ── Reproduce the exact wrapper / architecture used in Run 8 ──────────────────

class MyRewardWrapper(gym.Wrapper):
    """Reward wrapper used during training — kept here so the model loads cleanly.
    During evaluation we ignore the shaped reward; we only need the wrapper to
    match the environment interface the model was trained on."""
    def __init__(self, env):
        super().__init__(env)
        self.goal_pos = None
        self.prev_dist = None

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.goal_pos = None
        for i in range(self.env.unwrapped.width):
            for j in range(self.env.unwrapped.height):
                cell = self.env.unwrapped.grid.get(i, j)
                if cell is not None and cell.type == "goal":
                    self.goal_pos = (i, j)
        if self.goal_pos is not None:
            pos = tuple(self.env.unwrapped.agent_pos)
            self.prev_dist = abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info


def make_eval_env(render_mode=None):
    """Recreate the exact env stack from Run 8 (no reward wrapper needed for eval)."""
    env = gym.make("MiniGrid-LavaCrossingS9N1-v0", render_mode=render_mode)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    return env


# ── Load model ────────────────────────────────────────────────────────────────

print("Loading model…")
model = PPO.load("./models/best_model")
print("Model loaded.")


# ── Run episodes and capture frames ───────────────────────────────────────────

MAX_STEPS = 100
N_SEARCH = 100        # search this many episodes to find a good success + a failure
FRAME_DURATION = 0.12  # seconds per frame in the GIF

success_frames = None
failure_frames = None
success_steps = None

for ep in range(N_SEARCH):
    if success_frames is not None and failure_frames is not None:
        break

    # Use rgb_array render mode so we can capture frames without a window
    env_render = make_eval_env(render_mode="rgb_array")
    obs, _ = env_render.reset()

    frames = []
    success = False
    steps = 0

    for _ in range(MAX_STEPS):
        # Capture the current frame before taking the action
        frame = env_render.render()
        if frame is not None:
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env_render.step(action)
        steps += 1

        if terminated or truncated:
            # Capture final frame
            frame = env_render.render()
            if frame is not None:
                frames.append(frame)
            if terminated:
                success = True
            break

    env_render.close()

    if success and success_frames is None:
        success_frames = frames
        success_steps = steps
        print(f"  Episode {ep+1}: SUCCESS in {steps} steps — captured for GIF")
    elif not success and failure_frames is None:
        failure_frames = frames
        print(f"  Episode {ep+1}: FAILURE after {steps} steps — captured for GIF")


# ── Save GIFs ─────────────────────────────────────────────────────────────────

os.makedirs("results", exist_ok=True)

def save_gif(frames, path, duration=FRAME_DURATION, scale=4):
    """Save a list of numpy RGB frames as a GIF.
    scale: upscale factor so the small MiniGrid grid is visible."""
    if not frames:
        print(f"  No frames to save for {path}")
        return
    pil_frames = []
    for f in frames:
        img = Image.fromarray(f.astype(np.uint8))
        w, h = img.size
        img = img.resize((w * scale, h * scale), Image.NEAREST)
        pil_frames.append(np.array(img))
    imageio.mimsave(path, pil_frames, duration=duration, loop=0)
    print(f"  Saved {path}  ({len(pil_frames)} frames)")


if success_frames:
    save_gif(success_frames, "results/agent_navigation.gif")
    print(f"  Success GIF: {success_steps} steps")
else:
    print("  WARNING: no successful episode found in search window — try increasing N_SEARCH")

if failure_frames:
    save_gif(failure_frames, "results/agent_navigation_fail.gif")
else:
    print("  No failure episode found (agent may be solving everything) — skipping failure GIF")

print("\nDone. Check results/ for GIFs.")