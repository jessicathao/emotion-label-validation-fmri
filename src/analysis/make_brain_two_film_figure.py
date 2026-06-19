#!/usr/bin/env python3
"""make_brain_two_film_figure.py - the two-film controlled-null figure (canonical).

Reads results/brain_decode_<film>.json for two films (default Payload + TearsOfSteel)
and builds a two-panel figure: positive control on both films (left), real-vs-shift-null
on both films (right). The ToS null is elevated because its visually intense occipital
signal correlates with any film-autocorrelated target; a zero-test would false-positive.
Significance withheld; descriptive effect sizes only.
"""
import os, sys, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "results")

FILMS = [("Payload", "dialogue-dense", "#4C72B0"),
         ("TearsOfSteel", "dialogue-light, visual", "#DD8452")]
C_NULL = "#BBBBBB"; C_ZERO = "#444444"; C_VAL = "#C44E52"


def find_json(film, override=None):
    if override and os.path.exists(override):
        return override
    for name in (f"brain_decode_{film}.json", f"brain_decode_{film.lower()}.json"):
        p = os.path.join(RES, name)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(RES, f"brain_decode_{film}*.json"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"no results JSON for {film} in {RES}")


def main(payload_path=None, tos_path=None):
    overrides = {"Payload": payload_path, "TearsOfSteel": tos_path}
    data = {}
    for film, _, _ in FILMS:
        data[film] = json.load(open(find_json(film, overrides.get(film))))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.suptitle("Brain arm: a validated decoder recovers a planted signal but reads "
                 "no real valence, on two films",
                 fontsize=13, fontweight="bold", y=0.99)

    ymin, ymax = -0.10, 0.36

    for film, desc, color in FILMS:
        pc = data[film]["positive_control_pooled"]
        ts = list(pc["timescales_s"]); r1 = list(pc["recovery_r_snr1"])
        order = np.argsort(ts)
        ts = np.array(ts)[order]; r1 = np.array(r1)[order]
        axL.plot(ts, r1, "-o", color=color, lw=2, ms=6, label=f"{film} ({desc})")
        vi = list(ts).index(pc.get("valence_timescale_s", 100))
        axL.scatter([ts[vi]], [r1[vi]], s=140, facecolor="none",
                    edgecolor=C_VAL, linewidth=2, zorder=6)
    axL.axhline(0, color=C_ZERO, lw=0.8)
    axL.axvline(100, color="#cccccc", lw=1, ls=":")
    axL.text(100, ymax - 0.02, "valence\ntimescale", color=C_VAL, fontsize=8.5,
             ha="center", va="top", fontweight="bold")
    axL.set_xlabel("planted signal timescale (s)")
    axL.set_ylabel("recovery r  (leave-one-subject-out)")
    axL.set_title("Positive control PASSES on both films\nestimator recovers a planted signal",
                  fontsize=11)
    axL.set_ylim(ymin, ymax)
    axL.legend(frameon=False, fontsize=9, loc="lower center")
    axL.spines[["top", "right"]].set_visible(False)

    rng = np.random.default_rng(1)
    xpos = {f[0]: i for i, f in enumerate(FILMS)}
    for film, desc, color in FILMS:
        rv = data[film]["real_valence_pooled"]
        x = xpos[film]
        nm, ns = rv["shift_null_mean"], rv["shift_null_sd"]
        real = rv["loso_mean"]; pts = rv["loso_r_per_subject"]
        axR.fill_between([x - 0.32, x + 0.32], nm - 2 * ns, nm + 2 * ns,
                         color=C_NULL, alpha=0.25, zorder=1)
        axR.fill_between([x - 0.32, x + 0.32], nm - ns, nm + ns,
                         color=C_NULL, alpha=0.55, zorder=1)
        axR.plot([x - 0.32, x + 0.32], [nm, nm], color="#777777", lw=1.2, ls="--", zorder=2)
        xs = x + (rng.random(len(pts)) - 0.5) * 0.34
        axR.scatter(xs, pts, s=55, color=color, edgecolor="white", linewidth=0.7, zorder=5)
        axR.plot([x - 0.34, x + 0.34], [real, real], color=C_VAL, lw=3, zorder=6)
        axR.annotate(f"real +{real:.3f}\nnull +{nm:.3f}+/-{ns:.3f}",
                     xy=(x, max(pts) + 0.015), ha="center", va="bottom",
                     fontsize=8.5, color="#333333")
    axR.axhline(0, color=C_ZERO, lw=0.8)
    axR.set_xticks(list(xpos.values()))
    axR.set_xticklabels([f"{f[0]}\n({f[1]})" for f in FILMS], fontsize=9)
    axR.set_xlim(-0.6, len(FILMS) - 0.4)
    axR.set_ylim(ymin, ymax)
    axR.set_title("Real valence at CHANCE on both films\n(ToS chance level is elevated)",
                  fontsize=11)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axR.legend(handles=[
        Line2D([0], [0], color=C_VAL, lw=3, label="real pooled mean"),
        Line2D([0], [0], color="#777777", lw=1.2, ls="--", label="shift-null mean"),
        Patch(facecolor=C_NULL, alpha=0.55, label="null +/-1 sd"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=7, label="per-subject r")],
        frameon=False, fontsize=8.5, loc="upper left")
    axR.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.005,
             "Two-film controlled null: occipital valence decoding is not estimable above "
             "chance on a dialogue-dense (Payload) or a dialogue-light (Tears of Steel) film. "
             "The elevated ToS null shows a zero-test would false-positive. Descriptive; no p-values.",
             ha="center", fontsize=8.5, color="#555555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outdir = os.path.join(REPO, "figures"); os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, "brain_decode_two_film_null.png")
    pdf = os.path.join(outdir, "brain_decode_two_film_null.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("wrote", png)
    print("wrote", pdf)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else None
    b = sys.argv[2] if len(sys.argv) > 2 else None
    main(a, b)
