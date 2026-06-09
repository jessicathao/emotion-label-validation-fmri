#!/usr/bin/env python3
"""
make_spaceman_mechanism.py
Spaceman decontamination: same 94 s of Whisper song-lyrics + credits removed from
all four model signals, run through correlate_v3 (+2s, PleasantOther). Shows the
SAME cut moves every model the same direction but to different destinations:
BERT-family phantom negatives collapse toward 0; the graded LLM reveals a real
positive. Canonical numbers (correlate_v3, n=519 clean / 612 all).
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 11, "ytick.labelsize": 10.5, "xtick.labelsize": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

ALL_C   = "#C44E52"
CLEAN_C = "#55A868"
ZERO_C  = "#444444"

DATA = [
    ("BERT (binary)",            -0.292, -0.107),
    ("SiEBERT (RoBERTa-lg)",     -0.282, -0.113),
    ("DistilBERT (SST-2)",       -0.125, +0.021),
    ("Gemini 3.5 Flash (graded LLM)", -0.096, +0.235),
]

fig, ax = plt.subplots(figsize=(8.5, 5.6))
ys = list(range(len(DATA)))[::-1]

for y, (name, r_all, r_clean) in zip(ys, DATA):
    ax.annotate("", xy=(r_clean, y), xytext=(r_all, y),
                arrowprops=dict(arrowstyle="-|>", color="#aaaaaa", lw=1.8,
                                shrinkA=7, shrinkB=7))
    ax.plot(r_all, y, "o", color=ALL_C, ms=11, zorder=3)
    ax.plot(r_clean, y, "o", color=CLEAN_C, ms=11, zorder=3)
    ax.text(r_all, y+0.22, f"{r_all:+.3f}", ha="center", va="bottom",
            fontsize=9, color=ALL_C, fontweight="bold")
    ax.text(r_clean, y+0.22, f"{r_clean:+.3f}", ha="center", va="bottom",
            fontsize=9, color=CLEAN_C, fontweight="bold")

ax.axvline(0, color=ZERO_C, lw=1.0, ls="--", zorder=1)
ax.set_yticks(ys)
ax.set_yticklabels([d[0] for d in DATA])
ax.set_ylim(-0.5, len(DATA)-0.05)
ax.set_xlim(-0.45, 0.42)
ax.set_xlabel("Pearson r with human PleasantOther valence  (+2 s offset)")

fig.suptitle("Transcription contamination corrupts emotion labels differently by model",
             fontsize=13, fontweight="bold", y=0.985)
ax.set_title("Spaceman: same 94 s of Whisper song + credits removed",
             fontsize=11, color="#444", pad=12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", color="#DDDDDD", lw=0.6)
ax.set_axisbelow(True)

legend_handles = [
    Line2D([0],[0], marker="o", color="white", markerfacecolor=ALL_C, ms=11,
           label="all dialogue (lyrics included, n=612)"),
    Line2D([0],[0], marker="o", color="white", markerfacecolor=CLEAN_C, ms=11,
           label="decontaminated (lyrics+credits removed, n=519)"),
]
ax.legend(handles=legend_handles, loc="upper right", frameon=True,
          framealpha=0.95, edgecolor="#CCCCCC", fontsize=9)

fig.text(0.5, 0.02,
         "Polarized classifiers scored the despairing lyrics as strong negatives -> phantom "
         "negative that collapses when removed.\nThe graded LLM scored them mildly -> a real "
         "positive emerges once lyrics are removed. CIs are descriptive; the contrast is the point.",
         ha="center", va="bottom", fontsize=8.4, color="#666")

plt.subplots_adjust(left=0.27, right=0.97, top=0.88, bottom=0.20)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/spaceman_decontam_mechanism.{ext}", dpi=200)
print("saved")
