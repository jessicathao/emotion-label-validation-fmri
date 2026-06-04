import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

films = [
    ("Tears of Steel",       0.36,  0.11,  0.51, "pos"),
    ("The Secret Number",    0.16,  0.03,  0.31, "pos"),
    ("Sintel",               0.21, -0.37,  0.38, "ns"),
    ("Payload",              0.14, -0.07,  0.32, "ns"),
    ("Chatter",              0.14, -0.23,  0.82, "ns"),
    ("Between Viewings",     0.11, -0.06,  0.30, "ns"),
    ("Superhero",            0.03, -0.19,  0.25, "ns"),
    ("To Claire From Sonny",-0.04, -0.26,  0.31, "ns"),
    ("Lesson Learned",      -0.09, -0.31,  0.07, "ns"),
    ("You Again",           -0.12, -0.26,  0.04, "ns"),
    ("After the Rain",      -0.20, -0.35,  0.03, "ns"),
    ("Spaceman",            -0.29, -0.47, -0.04, "neg"),
]
COL = {"pos": "#185FA5", "neg": "#A32D2D", "ns": "#888780"}
LABEL = {"pos": "significant positive", "neg": "significant negative",
         "ns": "not significant (CI crosses 0)"}

fig, ax = plt.subplots(figsize=(9, 6))
ys = list(range(len(films)))[::-1]
for y, (name, r, lo, hi, sig) in zip(ys, films):
    c = COL[sig]; lw = 2.6 if sig != "ns" else 1.6; ms = 9 if sig != "ns" else 6
    ax.plot([lo, hi], [y, y], color=c, lw=lw, solid_capstyle="round", zorder=2)
    ax.plot([lo, lo], [y-0.12, y+0.12], color=c, lw=lw, zorder=2)
    ax.plot([hi, hi], [y-0.12, y+0.12], color=c, lw=lw, zorder=2)
    ax.plot(r, y, "o", color=c, markersize=ms, zorder=3)
ax.axvline(0, color="#5F5E5A", lw=1.8, zorder=1)
ax.axvline(0.40, color="#1D9E75", lw=1.6, ls="--", zorder=1)
ax.axvline(0.78, color="#534AB7", lw=1.6, ls="--", zorder=1)
ax.text(0.40, len(films)-0.2, "human-human\nceiling (r~0.40)", color="#0F6E56",
        fontsize=9, ha="center", va="bottom")
ax.text(0.78, len(films)-0.2, "human-human\nagreement (r~0.78)", color="#3C3489",
        fontsize=9, ha="center", va="bottom")
ax.set_yticks(ys); ax.set_yticklabels([f[0] for f in films], fontsize=11)
ax.set_xlabel("Correlation with human valence (Pearson r)", fontsize=12)
ax.set_xlim(-0.6, 0.95); ax.set_ylim(-0.8, len(films)+0.8)
ax.set_title("BERT-vs-human valence agreement across 12 films\n(fixed +2s offset, 95% block-bootstrap CI)", fontsize=13)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#888780", alpha=0.12)
legend_items = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=COL["pos"], markersize=9, label=LABEL["pos"]),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=COL["neg"], markersize=9, label=LABEL["neg"]),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=COL["ns"], markersize=7, label=LABEL["ns"]),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=9, frameon=False)
plt.tight_layout()
outdir = os.path.expanduser("~/emofilm/figures")
fig.savefig(os.path.join(outdir,"forest_bert_human.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(outdir,"forest_bert_human.pdf"), bbox_inches="tight")
print("saved to", outdir)
