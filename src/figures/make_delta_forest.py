import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# palette
BLUE = "#1C7293"
RED  = "#A32D2D"
INK  = "#1A1A1A"
GREY = "#666666"

# canonical numbers (RESULTS_SUMMARY_June6): BERT, Gemini-iso, contamination flag
# film, bert, gemini, contaminated
rows = [
    ("Tears of Steel",      0.356,  0.467, False),
    ("Sintel",              0.214,  0.000, False),
    ("The Secret Number",   0.159,  0.201, False),
    ("Payload",             0.140,  0.085, False),
    ("Chatter",             0.136,  0.514, True),
    ("Between Viewings",    0.114,  0.239, False),
    ("Superhero",           0.025,  0.305, True),
    ("To Claire From Sonny",-0.040,  0.121, True),
    ("Lesson Learned",     -0.085,  0.112, False),
    ("You Again",          -0.123,  0.048, True),
    ("After the Rain",     -0.197, -0.075, False),
    ("Spaceman",           -0.292, -0.096, True),
]

# delta = gemini - bert
data = [(f, g - b, c) for (f, b, g, c) in rows]
# sort by delta descending so the eye reads the shift magnitude
data.sort(key=lambda r: r[1], reverse=True)

labels  = [d[0] for d in data]
deltas  = [d[1] for d in data]
contam  = [d[2] for d in data]

n = len(data)
ypos = list(range(n))[::-1]  # top = first

fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for y, dv, c in zip(ypos, deltas, contam):
    color = RED if c else BLUE
    ax.plot([0, dv], [y, y], color=color, lw=3, solid_capstyle="round", zorder=2)
    ax.scatter([dv], [y], s=80, color=color, zorder=3,
               edgecolor="white", linewidth=0.8)
    # value label
    ha = "left" if dv >= 0 else "right"
    off = 0.012 if dv >= 0 else -0.012
    ax.text(dv + off, y, f"{dv:+.2f}", va="center", ha=ha,
            fontsize=10.5, color=color, fontweight="bold")

ax.axvline(0, color=INK, lw=1.2, zorder=1)

# --- legend (explains the two colors) ---
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color=BLUE, lw=3, marker="o", markersize=8,
           markeredgecolor="white", label="clean transcript"),
    Line2D([0], [0], color=RED, lw=3, marker="o", markersize=8,
           markeredgecolor="white", label="contaminated transcript (*)"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=10.5,
          frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
          handletextpad=0.6, borderpad=0.7)

ax.set_yticks(ypos)
# mark contaminated films with an asterisk in the tick label
ylabels = [f"{lab} *" if c else lab for lab, c in zip(labels, contam)]
ax.set_yticklabels(ylabels, fontsize=11, color=INK)
ax.set_ylim(-0.7, n - 0.3)

ax.set_xlim(-0.30, 0.45)
ax.set_xlabel("Change in agreement with human valence  (Gemini \u2212 BERT, Pearson r)",
              fontsize=12, color=INK)
ax.tick_params(axis="x", labelsize=10.5, colors=INK)

# title
ax.set_title("Context-aware LLM shifts agreement upward on 10 of 12 films",
             fontsize=14, color=INK, fontweight="bold", pad=30, loc="center")
ax.text(0.5, 1.015,
        "Per-film paired difference; each film is its own control. Positive = Gemini agrees with humans more than BERT.",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=10, color=GREY)

# legend-ish note moved to slide caption to avoid collision

for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GREY)

# subtle vertical gridlines
for xg in [-0.2, -0.1, 0.1, 0.2, 0.3, 0.4]:
    ax.axvline(xg, color="#E6E6E6", lw=0.6, zorder=0)

plt.tight_layout()
import os
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/forest_delta.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
print("saved figures/forest_delta.png and .pdf")
# report
pos = sum(1 for d in deltas if d > 0)
print(f"positive deltas: {pos}/{n}")
