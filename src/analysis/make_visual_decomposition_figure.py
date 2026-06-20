#!/usr/bin/env python3
"""make_visual_decomposition_figure.py - companion panel to brain_decode_two_film_null.png.

Reads results/visual_decomposition_<film>.json for Payload and TearsOfSteel and
renders, for each target, the leave-one-subject-out decode MINUS its own shift-null
(beats_null), with the shift-null sd as the error cap. A bar that clears its cap
decodes above its own chance level. Visual features clear it; valence does not, on
both films. The visual-set R2 predicting valence is annotated (the mechanism behind
the elevated Tears of Steel null). Effect sizes only; no p-values.

USAGE
  python make_visual_decomposition_figure.py
Writes figures/visual_decomposition_two_film.png and .pdf
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "results")
FIGURES = os.path.join(REPO, "figures")
FILMS = [("Payload", "Payload (dialogue-dense)", "#2E6CB4"),
         ("TearsOfSteel", "Tears of Steel (visually intense)", "#C8902F")]
NAVY, GOLD, MUTE = "#13335B", "#B8902F", "#6B7682"
VISUAL = ["luminance", "rms_contrast", "motion", "edges", "cuts", "saturation"]
VAL = ["valence", "valence_resid_visual"]
DISPLAY = {"luminance": "luminance", "rms_contrast": "contrast", "motion": "motion",
           "edges": "edges", "cuts": "cuts", "saturation": "saturation",
           "valence": "valence", "valence_resid_visual": "valence\n(visual out)"}


def load(film):
    p = os.path.join(RESULTS, f"visual_decomposition_{film}.json")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}; run decode_target_multi.py {film} <subs> first")
    return json.load(open(p))


def main():
    data = {f: load(f) for f, _, _ in FILMS}

    # order visual features by mean beats_null across films (descending), valence block last
    mean_beats = {k: np.mean([data[f]["targets"][k]["beats_null"] for f, _, _ in FILMS])
                  for k in VISUAL}
    vis_order = sorted(VISUAL, key=lambda k: mean_beats[k], reverse=True)
    cats = vis_order + VAL
    x = np.arange(len(cats), dtype=float)
    x[len(vis_order):] += 0.6           # gap before the valence block
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 4.7))
    for j, (film, label, color) in enumerate(FILMS):
        t = data[film]["targets"]
        beats = [t[c]["beats_null"] for c in cats]
        sds = [t[c]["shift_null_sd"] for c in cats]
        off = (j - 0.5) * w
        ax.bar(x + off, beats, w, color=color, edgecolor="white", linewidth=0.6,
               yerr=sds, capsize=3, ecolor=MUTE, error_kw={"elinewidth": 1, "alpha": 0.8},
               label=label, zorder=3)

    ax.axhline(0, color=NAVY, lw=1.2, zorder=2)
    div = (x[len(vis_order) - 1] + x[len(vis_order)]) / 2
    ax.axvline(div, color="#C9D2DE", lw=1, ls="--", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[c] for c in cats], fontsize=10, color=NAVY)
    ax.set_ylabel("decode r  minus its own shift-null", fontsize=11, color=NAVY)
    ax.set_title("Occipital decodes the film's visual dynamics, not its valence",
                 fontsize=15, weight="bold", color=NAVY, pad=26)
    ax.text(0.5, 1.025,
            "Leave-one-subject-out, 5 subjects/film. Bars clearing their error cap decode "
            "above their own chance level; valence does not, on either film.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color=MUTE)

    r2 = {f: data[f]["stimulus_side"]["visual_set_R2_predicting_valence"] for f, _, _ in FILMS}
    ax.text(0.985, 0.96,
            f"visual set $\\rightarrow$ valence  ($R^2$):\n"
            f"  Payload {r2['Payload']:.2f}      Tears of Steel {r2['TearsOfSteel']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=NAVY,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F7F0DA", ec=GOLD, lw=1))

    ax.legend(loc="upper left", frameon=False, fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTE); ax.spines["bottom"].set_color(MUTE)
    ax.tick_params(colors=MUTE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EAEFF5", lw=0.8)
    fig.tight_layout()

    os.makedirs(FIGURES, exist_ok=True)
    png = os.path.join(FIGURES, "visual_decomposition_two_film.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {png.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
