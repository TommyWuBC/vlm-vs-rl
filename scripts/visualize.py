import matplotlib.pyplot as plt
import numpy as np

# Generalization data from rl_vs_vlm.md
environments = ["S9N1\n(trained)", "S9N2\n(unseen)", "S9N3\n(unseen)"]
rl_success = [95.2, 20.0, 10.0]
vlm_success = [100.0, 100.0, 100.0]

x = np.arange(len(environments))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, rl_success, width, label="RL (CNN, full obs)", color="coral")
bars2 = ax.bar(x + width/2, vlm_success, width, label="VLM (GPT-4o-mini, zero-shot)", color="steelblue")

ax.set_ylabel("Success Rate (%)")
ax.set_title("Generalization to Unseen Lava Configurations")
ax.set_xticks(x)
ax.set_xticklabels(environments)
ax.legend()
ax.set_ylim(0, 110)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("results/generalization_comparison.png", dpi=150)
plt.close()
print("Saved generalization_comparison.png")