import json
import matplotlib.pyplot as plt
import numpy as np

with open("results/pattern_eval.json") as f:
    data = json.load(f)

patterns = data["per_pattern"]

def find_gap_and_orientation(lava_positions):
    """Each wall is 6 cells along a fixed x or fixed y (with one gap in the 7-cell line).
    Returns (orientation, wall_index, gap_coord)."""
    xs = [p[0] for p in lava_positions]
    ys = [p[1] for p in lava_positions]
    if len(set(xs)) == 1:
        # vertical wall: fixed x, gap is the missing y in range 1-7
        wall_x = xs[0]
        present_ys = set(ys)
        gap_y = [y for y in range(1, 8) if y not in present_ys][0]
        return ("vertical", wall_x, gap_y)
    else:
        # horizontal wall: fixed y, gap is the missing x in range 1-7
        wall_y = ys[0]
        present_xs = set(xs)
        gap_x = [x for x in range(1, 8) if x not in present_xs][0]
        return ("horizontal", wall_y, gap_x)

# Build a grid: rows = wall position (1-7), cols = gap position (1-7), separately for each orientation
rows = []
for p in patterns:
    orientation, wall_pos, gap_pos = find_gap_and_orientation(p["lava_positions"])
    rows.append({
        "pattern_id": p["pattern_id"],
        "orientation": orientation,
        "wall_pos": wall_pos,
        "gap_pos": gap_pos,
        "success_rate": p["success_rate"]
    })

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, orientation, title in zip(
    axes, ["vertical", "horizontal"],
    ["Vertical Walls (gap = y position)", "Horizontal Walls (gap = x position)"]
):
    grid = np.full((7, 7), np.nan)  # wall_pos 1-7 x gap_pos 1-7
    for r in rows:
        if r["orientation"] == orientation:
            grid[r["wall_pos"] - 1, r["gap_pos"] - 1] = r["success_rate"] * 100

    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(7))
    ax.set_xticklabels(range(1, 8))
    ax.set_yticks(range(7))
    ax.set_yticklabels(range(1, 8))
    ax.set_xlabel("Gap position")
    ax.set_ylabel("Wall position")
    ax.set_title(title)

    for i in range(7):
        for j in range(7):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.0f}", ha="center", va="center",
                         color="black", fontsize=9, fontweight="bold")

fig.colorbar(im, ax=axes, label="Success Rate (%)", shrink=0.8)
fig.suptitle("Run 8 Per-Pattern Success Rate (Full Observability CNN, 40/42 Solved)", fontsize=13)
plt.savefig("results/pattern_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pattern_heatmap.png")

# Print the two failures explicitly for the report
print("\nFailures:")
for r in rows:
    if r["success_rate"] < 0.2:
        print(f"  Pattern {r['pattern_id']}: {r['orientation']} wall at {r['wall_pos']}, gap at {r['gap_pos']} -> {r['success_rate']:.1%}")
