#!/usr/bin/env python3
"""make_context_slopegraph.py - isolated -> +context, BERT vs Gemini (slopegraph).

Two panels (BERT per-segment | Gemini), each a slopegraph: every context film is a
line from its ISOLATED r to its +CONTEXT r vs human valence. Upward slope = context
helped. Gemini rises on every dialogue-driven film and is flat on the dialogue-light
one; per-segment BERT scatters. The one BERT rise (Tears of Steel) is a smoothing
artifact (window sweep / sanity_bert_context_tos.py), flagged on the figure.
Numbers are the canonical correlate_v3 (+2s, PleasantOther) values; colors match
forest_bert_vs_gemini.png (BERT blue, Gemini orange).
Writes figures/context_slopegraph_iso_to_ctx.{png,pdf}.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BERT_C = "#4C72B0"; GEM_C = "#DD8452"; INK = "#1A1A1A"; GREY = "#666666"; GOLD = "#C9A227"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "figures")

# film: bert_iso, bert_ctx, gem_iso, gem_ctx, bert_smoothing_flag
data = [
    ("Lesson Learned", -0.085, -0.056, +0.112, +0.225, False),
    ("Payload",        +0.140, -0.118, +0.085, +0.321, False),
    ("Tears of Steel", +0.356, +0.513, +0.467, +0.591, True),
    ("After the Rain", -0.197, -0.217, -0.075, -0.080, False),
]


def panel(ax, color, which, title):
    for film, bi, bc, gi, gc, sm in data:
        v0, v1 = (bi, bc) if which == "bert" else (gi, gc)
        ax.plot([0, 1], [v0, v1], color=color, lw=2.4, zorder=2, alpha=0.92)
        ax.scatter([0], [v0], s=46, facecolor="white", edgecolor=color, linewidth=1.8, zorder=3)
        ax.scatter([1], [v1], s=52, color=color, zorder=3)
        ax.text(-0.04, v0, f"{v0:+.2f}", va="center", ha="right", fontsize=8.5, color=color)
        ax.text(1.04, v1, f"{film}  {v1:+.2f}", va="center", ha="left", fontsize=9.3, color=INK)
        if sm and which == "bert":
            ax.text(1.04, v1 - 0.045, "smoothing artifact", va="center", ha="left",
                    fontsize=7.6, color=GOLD, fontstyle="italic")
    ax.axhline(0, color="#999999", lw=0.8, zorder=0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["isolated", "+context"], fontsize=10.5, color=INK)
    ax.set_xlim(-0.5, 2.0)
    ax.set_title(title, fontsize=11.5, color=color, fontweight="bold")
    for sp in ["top", "right", "bottom"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=0)


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 5.6), sharey=True)
    fig.suptitle("From isolated to +context: Gemini rises across films, per-segment BERT does not",
                 fontsize=13, fontweight="bold", y=0.99, color=INK)
    panel(axL, BERT_C, "bert", "BERT (per-segment sentiment)")
    panel(axR, GEM_C, "gem", "Gemini 3.5 Flash")
    axL.set_ylabel("agreement with human valence  (Pearson r)", fontsize=10.5, color=INK)
    axL.set_ylim(-0.27, 0.64)
    for ax in (axL, axR):
        ax.tick_params(axis="y", labelsize=9.5, colors=INK)
        ax.spines["left"].set_color("#CCCCCC")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUTDIR, exist_ok=True)
    png = os.path.join(OUTDIR, "context_slopegraph_iso_to_ctx.png")
    pdf = os.path.join(OUTDIR, "context_slopegraph_iso_to_ctx.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print("wrote", png); print("wrote", pdf)


if __name__ == "__main__":
    main()
