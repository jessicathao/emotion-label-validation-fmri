import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# Corrected v3 results (real-timeline bootstrap, +2s offset). June 5 2026.
# Only Tears of Steel is significant. Secret Number and Spaceman are NOT
# (Spaceman's all-dialogue negative is a lyric artifact; see clean value).
films = [
    ("Tears of Steel",       0.356,  0.134,  0.535, "pos"),
    ("Sintel",               0.214, -0.251,  0.456, "ns"),
    ("The Secret Number",    0.159, -0.010,  0.319, "ns"),
    ("Payload",              0.140, -0.077,  0.330, "ns"),
    ("Chatter",              0.136, -0.484,  0.839, "ns"),
    ("Between Viewings",     0.114, -0.055,  0.282, "ns"),
    ("Superhero",            0.025, -0.188,  0.241, "ns"),
    ("To Claire From Sonny",-0.040, -0.283,  0.286, "ns"),
    ("Lesson Learned",      -0.085, -0.272,  0.078, "ns"),
    ("You Again",           -0.123, -0.261,  0.015, "ns"),
    ("After the Rain",      -0.197, -0.395,  0.080, "ns"),
    ("Spaceman (all)",      -0.292, -0.455, -0.052, "neg"),
    ("Spaceman (lyrics out)",-0.107,-0.308,  0.105, "ns"),
]
COL = {"pos": "#185FA5", "neg": "#A32D2D", "ns": "#888780"}
LABEL = {"pos": "significant positive", "neg": "significant (artifact, see text)",
         "ns": "not significant (CI crosses 0)"}

fig, ax = plt.subplots(figsize=(9, 6.5))
ys = list(range(len(films)))[::-1]
for y, (name, r, lo, hi, sig) in zip(ys, films):
    c = COL[sig]; lw = 2.6 if sig != "ns" else 1.6; ms = 9 if sig != "ns" else 6
    ax.plot([lo, hi], [y, y], color=c, lw=lw, solid_capstyle="round", zorder=2)
    ax.plot([lo, lo], [y-0.12, y+0.12], color=c, lw=lw, zorder=2)
    ax.plot([hi, hi], [y-0.12, y+0.12], color=c, lw=lw, zorder=2)
    ax.plot(r, y, "o", color=c, markersize=ms, zorder=3)
ax.axvline(0, color="#5F5E5A", lw=1.8, zorder=1)
# Human reference: PleasantOther inter-rater agreement r=0.58 (paper, highest item);
# dataset-wide mean inter-rater r=0.39 (paper).
ax.axvline(0.58, color="#1D9E75", lw=1.6, ls="--", zorder=1)
ax.axvline(0.39, color="#7A9E1D", lw=1.4, ls=":", zorder=1)
ax.text(0.58, len(films)-0.2, "PleasantOther\ninter-rater (r=0.58)", color="#0F6E56",
        fontsize=8.5, ha="center", va="bottom")
ax.text(0.39, -1.1, "dataset mean\ninter-rater (r=0.39)", color="#5A7016",
        fontsize=8.5, ha="center", va="top")
ax.set_yticks(ys); ax.set_yticklabels([f[0] for f in films], fontsize=10.5)
ax.set_xlabel("Correlation with human valence (Pearson r)", fontsize=12)
ax.set_xlim(-0.6, 0.95); ax.set_ylim(-1.8, len(films)+0.8)
ax.set_title("BERT-vs-human valence agreement across 12 films\n(+2s offset, real-timeline block-bootstrap 95% CI)", fontsize=13)
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
os.makedirs(outdir, exist_ok=True)
fig.savefig(os.path.join(outdir,"forest_bert_human.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(outdir,"forest_bert_human.pdf"), bbox_inches="tight")
print("saved to", outdir)
