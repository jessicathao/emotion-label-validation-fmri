#!/usr/bin/env python3
"""
make_context_progression.py
Slope chart: valence-agreement r across three conditions
(word-level BERT -> isolated LLM -> context LLM) for 4 films.
Shows context rescues dialogue-driven films (upward slopes) but not the
dialogue-light film (AfterTheRain, flat). Human reference lines included.
Outputs PNG + PDF to figures/ (repo-relative).
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

CONDS = ["per-segment\nBERT", "isolated\nLLM", "context\nLLM"]
X = [0, 1, 2]

# film: (values across 3 conditions, color, is_flat)
RISE = "#1C7293"     # teal (deck accent) for dialogue-driven
FLAT = "#C44E52"     # red for the dialogue-light null
DATA = [
    ("Tears of Steel",  [0.356, 0.467, 0.591], RISE, False),
    ("Payload",         [0.140, 0.085, 0.321], RISE, False),
    ("Lesson Learned",  [-0.085, 0.112, 0.225], RISE, False),
    ("After the Rain",  [-0.197, -0.075, -0.080], FLAT, True),
]

fig, ax = plt.subplots(figsize=(8.4, 6.0))

# human reference bands
# Human reference lines, cited from Morgenroth et al. 2025 (not recomputed here:
# the per-rater series are not in the public derivatives). 0.58 = their reported
# PleasantOther inter-rater agreement (highest-agreement item); 0.39 = their mean
# agreement across all items and films (abstract rounds it to 0.38).
ax.axhline(0.58, color="#1B9E77", lw=1.4, ls="--", zorder=1)
ax.axhline(0.39, color="#7A8B2B", lw=1.2, ls=":", zorder=1)
ax.axhline(0.0, color="#999999", lw=1.0, zorder=1)
ax.text(-0.20, 0.585, "PleasantOther inter-rater 0.58 (different denominator)",
        color="#1B9E77", fontsize=8, va="bottom", ha="left")
ax.text(-0.20, 0.395, "dataset mean inter-rater 0.39",
        color="#7A8B2B", fontsize=8, va="bottom", ha="left")

for name, vals, color, is_flat in DATA:
    lw = 2.6 if not is_flat else 2.2
    ls = "-" if not is_flat else (0, (4, 2))
    ax.plot(X, vals, color=color, lw=lw, ls=ls, zorder=3,
            marker="o", ms=8, markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=1.2)
    # label at right end
    ax.text(2.06, vals[2], f"  {name}", color=color, fontsize=10,
            va="center", ha="left", fontweight="bold" if not is_flat else "normal")
    # value at each node (small)
    for xi, v in zip(X, vals):
        ax.text(xi, v + 0.028, f"{v:+.2f}", color=color, fontsize=7.5,
                ha="center", va="bottom")

ax.set_xticks(X)
ax.set_xticklabels(CONDS)
ax.set_xlim(-0.25, 3.45)
ax.set_ylim(-0.32, 0.72)
ax.set_ylabel("Pearson r with human PleasantOther valence")
ax.set_title("More context, better agreement, only where dialogue carries emotion",
             fontsize=12, pad=34)
fig.suptitle("4-film preliminary probe", fontsize=10, color="#777", y=0.965)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#EEEEEE", lw=0.6)
ax.set_axisbelow(True)

legend_handles = [
    Line2D([0],[0], color=RISE, lw=2.6, marker="o", markerfacecolor=RISE,
           markeredgecolor="white", label="dialogue-driven (rescued)"),
    Line2D([0],[0], color=FLAT, lw=2.2, ls=(0,(4,2)), marker="o",
           markerfacecolor=FLAT, markeredgecolor="white",
           label="dialogue-light (no rescue)"),
]
ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
          ncol=2, frameon=False, fontsize=9)

fig.text(0.5, 0.015,
         "Same pipeline; only the model input changes. Descriptive CIs not shown; "
         "significance withheld. All 4 films uncontaminated.",
         ha="center", va="bottom", fontsize=8, color="#666")

plt.subplots_adjust(left=0.11, right=0.80, top=0.82, bottom=0.22)
import os
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/context_progression.{ext}", dpi=200)
print("saved figures/context_progression.png and .pdf")
