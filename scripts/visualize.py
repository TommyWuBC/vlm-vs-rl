import numpy as np
import matplotlib.pyplot as plt
import json

# Load RL training curve
eval_data = np.load("results/evaluations.npz")
timesteps = eval_data['timesteps']
mean_rewards = eval_data['results'].mean(axis=1)  # average across eval episodes

# Load VLM and RL evaluation results
with open("results/vlm_results_cot.json") as f:
    vlm_data = json.load(f)
with open("results/rl_results.json") as f:
    rl_data = json.load(f)

# Calculate success rates
vlm_success = sum(1 for e in vlm_data if e['success']) / len(vlm_data)
rl_success = sum(1 for e in rl_data if e['success']) / len(rl_data)

print(f"VLM Success Rate: {vlm_success:.1%}")
print(f"RL Success Rate: {rl_success:.1%}")

# Plot 1: Training curve
plt.figure(figsize=(10, 4))
plt.plot(timesteps, mean_rewards)
plt.xlabel("Training Steps")
plt.ylabel("Mean Eval Reward")
plt.title("RL Agent Learning Curve (LavaCrossing)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/learning_curve.png", dpi=150)
plt.close()

# Plot 2: Success rate comparison
plt.figure(figsize=(6, 4))
plt.bar(["RL (PPO)", "VLM (GPT-4o-mini CoT)"],
        [rl_success, vlm_success],
        color=["coral", "steelblue"])
plt.ylabel("Success Rate")
plt.title("RL vs VLM: Navigation Success Rate")
plt.ylim(0, 1)
for i, v in enumerate([rl_success, vlm_success]):
    plt.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("results/success_comparison.png", dpi=150)
plt.close()

print("Charts saved to results/")