import torch
import stable_baselines3
import gymnasium
import minigrid
import openai
import PIL
import pandas

print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("Stable Baselines3 version:", stable_baselines3.__version__)
print("All imports successful!")