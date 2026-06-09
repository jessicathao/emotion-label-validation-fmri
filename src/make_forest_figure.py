import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# v3 effect sizes (real-timeline bootstrap, +2s offset). June 5 2026.
# CIs are DESCRIPTIVE only. Significance NOT tested: PleasantOther autocorrelation
# time is ~90-110s, exceeding what these film durations support for valid block
# resampling. Make this point on the methods slide / in speech, NOT in the title.
films = [
    ("Tears of Steel",       0.356,  0.134,  0.535, "hi"),
    ("Sintel",               0.214, -0.251,  0.456, "lo"),
    ("The Secret Number",    0.159, -0.010,  0.319, "lo"),
    ("Payload",              0.140, -0.077,  0.330, "lo"),
    ("Chatter",              0.136, -0.484,  0.839, "lo"),
    ("Between Viewings",     0.114, -0.055,  0.282, "lo"),
    ("Superhero",            0.025, -0.188,  0.241, "lo"),
    ("To Claire From Sonny",-0.040, -0.283,  0.286, "lo"),
    ("Lesson Learned",      -0.085, -0.272,  0.078, "lo"),
    ("You Again",           -0.123, -0.261,  0.015, "lo"),
    ("After the Rain",      -0.197, -0.395,  0.080, "lo"),
    ("Spaceman (all)",      -0.292, -0.455, -0.052, "art"),
    ("Spaceman (lyrics out)",-0.107,-0.308,  0.105, "lo"),
]
COL = {"hi": "#185FA5", "art": "#A32D2D", "lo": "#8A8780"}
LABEL = {"hi": "sizeable effect, replicates across 3 models",
         "art": "Spaceman all-dialogue (lyric artifact; see paired row)",
         "lo": "near-zero / no consistent effect"}

fig, ax = plt.subplots(figsize=(9.6, 6.8))
ys = list(range(len(films)))[::-1]
for y, (name, r, lo, hi, role) in zip(ys, films):
    c = COL[role]; lw = 2.6 if role != "lo" else 1.6; ms = 9 if role != "lo" else 6
    ax.plot([lo, hi], [y, y], color=c, lw=lw, solid_capstyle="round", zorder=2)
    ax.plot([lo, lo], [y-0.12, y+0.12], color=c, lw=lw, zorder=2)
    ax.plot([hi, hi], [y-0.12, y+0.12], color=c, lw=lw, zorder=2)
    ax.plot(r, y, "o", color=c, markersize=ms, zorder=3)
ax.axvline(0, color="#5F5E5A", lw=1.8, zorder=1)
ax.axvline(0.58, color="#1D9E75", lw=1.6, ls="--", zorder=1)
ax.axvline(0.39, color="#7A9E1D", lw=1.4, ls=":", zorder=1)
ax.text(0.36, len(films)-0.15, "dataset mean\ninter-rater (r=0.39)", color="#5A7016",
        fontsize=8.5, ha="right", va="bottom")
ax.text(0.60, len(films)-0.15, "PleasantOther\ninter-rater (r=0.58)", color="#0F6E56",
        fontsize=8.5, ha="left", va="bottom")
ax.set_yticks(ys); ax.set_yticklabels([f[0] for f in films], fontsize=10.5)
ax.set_xlabel("Correlation with human valence (Pearson r)", fontsize=12)
ax.set_xlim(-0.6, 0.95); ax.set_ylim(-0.8, len(films)+1.3)
ax.set_title("BERT-vs-human valence agreement across 12 films\n"
             "effect sizes with descriptive 95% bootstrap CIs (+2s offset)",
             fontsize=12, pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#888780", alpha=0.12)
legend_items = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=COL["hi"], markersize=9, label=LABEL["hi"]),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=COL["art"], markersize=9, label=LABEL["art"]),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=COL["lo"], markersize=7, label=LABEL["lo"]),
]
ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(0.0, 0.95),
          fontsize=8.5, frameon=True, framealpha=0.9, edgecolor="none")
plt.tight_layout()
# write to the repo's figures/ folder (this script is in src/, so go up one level)
outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(outdir, exist_ok=True)
fig.savefig(os.path.join(outdir,"forest_bert_human.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(outdir,"forest_bert_human.pdf"), bbox_inches="tight")
print("saved to", outdir)
