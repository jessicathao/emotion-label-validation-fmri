#!/usr/bin/env python3
"""
make_forest_bert_gemini.py
Forest plot comparing BERT vs Gemini-3.5-Flash agreement (Pearson r) with human
PleasantOther valence across 12 Emo-FilM films. Descriptive 95% CIs. Films
ordered by BERT r (the baseline method) so the ordering encodes no hypothesis.

Outputs both PNG (slides) and PDF (vector, publication) to figures/.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42,  # editable text in PDF
})

BERT_C = "#4C72B0"      # blue
GEM_C  = "#DD8452"      # orange (colorblind-safe pair)
ZERO_C = "#444444"

# film: (bert r, bert lo, bert hi, gem r, gem lo, gem hi, n, blk, contaminated)
DATA = {
 "TearsOfSteel":      (+0.356,+0.134,+0.535, +0.467,+0.199,+0.634, 147,22, False),
 "Sintel":            (+0.214,-0.251,+0.456, +0.000,-0.333,+0.440,  96,14, False),
 "AfterTheRain":      (-0.197,-0.395,+0.080, -0.075,-0.235,+0.237, 347,24, False),
 "BetweenViewings":   (+0.114,-0.055,+0.282, +0.239,+0.071,+0.390, 470,34, False),
 "Chatter":           (+0.136,-0.484,+0.839, +0.514,-0.469,+0.671,  81, 9, True),
 "LessonLearned":     (-0.085,-0.272,+0.078, +0.112,-0.050,+0.278, 375,32, False),
 "Payload":           (+0.140,-0.077,+0.330, +0.085,-0.140,+0.260, 443,44, False),
 "Spaceman":          (-0.292,-0.455,-0.052, -0.096,-0.329,+0.216, 612,39, True),
 "Superhero":         (+0.025,-0.188,+0.241, +0.305,+0.132,+0.472, 651,47, True),
 "TheSecretNumber":   (+0.159,-0.010,+0.319, +0.201,-0.006,+0.381, 411,32, False),
 "ToClaireFromSonny": (-0.040,-0.283,+0.286, +0.121,-0.131,+0.515, 253,15, True),
 "YouAgain":          (-0.123,-0.261,+0.015, +0.048,-0.137,+0.224, 469,36, True),
}

# Order by BERT r ascending so highest BERT sits at TOP (barh plots bottom-up).
order = sorted(DATA.items(), key=lambda kv: kv[1][0])
films = [k for k, _ in order]
n = len(films)
ys = list(range(n))
off = 0.18  # vertical separation of the two models within a film row

fig, ax = plt.subplots(figsize=(9.5, 8.0))

for i, (film, d) in enumerate(order):
    br, blo, bhi, gr, glo, ghi, nn, blk, contam = d
    y = ys[i]
    # BERT (upper of the pair)
    ax.plot([blo, bhi], [y+off, y+off], color=BERT_C, lw=1.6, zorder=2)
    ax.plot(br, y+off, "o", color=BERT_C, ms=7, zorder=3)
    # Gemini (lower of the pair)
    ax.plot([glo, ghi], [y-off, y-off], color=GEM_C, lw=1.6, zorder=2)
    ax.plot(gr, y-off, "s", color=GEM_C, ms=6.5, zorder=3)

ax.axvline(0, color=ZERO_C, lw=1.0, ls="--", zorder=1)

# Human inter-rater reference lines, cited from Morgenroth et al. 2025 (not
# recomputed here: the per-rater series are not in the public derivatives), matching
# the BERT forest figure so the two are directly comparable. 0.58 = their reported
# PleasantOther inter-rater agreement (highest-agreement item); 0.39 = their mean
# agreement across all items and films (abstract rounds it to 0.38). These are a
# reference for how well humans agree with each other, on a different (pairwise)
# denominator than the averaged consensus the models are scored against.
ax.axvline(0.39, color="#7A8B2B", lw=1.4, ls=":", zorder=1)
ax.axvline(0.58, color="#1B9E77", lw=1.6, ls="--", zorder=1)
ymax = n - 0.4
ax.text(0.39, ymax, "dataset mean\ninter-rater r=0.39", color="#7A8B2B",
        fontsize=7.5, ha="center", va="top", linespacing=1.1)
ax.text(0.58, ymax - 1.15, "PleasantOther\ninter-rater r=0.58", color="#1B9E77",
        fontsize=7.5, ha="center", va="top", linespacing=1.1)

# y labels: film name + (n), asterisk for contaminated, dagger for low-block
ylabels = []
for film, d in order:
    contam = d[8]; blk = d[7]
    mark = ""
    if contam: mark += "*"
    if blk < 16: mark += "†"
    ylabels.append(f"{film} (n={d[6]}){(' '+mark) if mark else ''}")
ax.set_yticks(ys)
ax.set_yticklabels(ylabels)
ax.set_ylim(-0.7, n-0.3)

ax.set_xlim(-0.6, 0.9)
ax.set_xlabel("Pearson r with human PleasantOther valence  (+2 s offset)")
ax.set_title("12 naturalistic films, Emo-FilM. Points = effect size; bars = "
             "descriptive 95% CI.\nFilms ordered by BERT r.",
             fontsize=9.5, fontweight="normal", color="#555", pad=10)
fig.suptitle("Agreement with human valence: per-segment BERT vs a stronger model (Gemini)",
             fontsize=14, fontweight="bold", y=0.98)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", color="#DDDDDD", lw=0.6)
ax.set_axisbelow(True)

legend_handles = [
    Line2D([0],[0], color=BERT_C, marker="o", lw=1.6, ms=7,
           label="BERT (per-segment sentiment)"),
    Line2D([0],[0], color=GEM_C, marker="s", lw=1.6, ms=6.5,
           label="Gemini 3.5 Flash (isolated segments)"),
    Line2D([0],[0], color="white", label="*  Whisper contamination (songs/credits)"),
    Line2D([0],[0], color="white", label="†  <16 bootstrap blocks (CI unreliable)"),
]
ax.legend(handles=legend_handles, loc="lower right", frameon=True,
          framealpha=0.95, edgecolor="#CCCCCC")

plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"figures/forest_bert_vs_gemini.{ext}",
                dpi=200, bbox_inches="tight")
print("saved forest_bert_vs_gemini.png and .pdf")
