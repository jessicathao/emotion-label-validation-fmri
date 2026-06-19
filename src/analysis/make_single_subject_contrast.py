#!/usr/bin/env python3
"""make_single_subject_contrast.py - the honest self-correction figure.

Two timeseries panels, same format as the retired PoC plot, but showing the
INSTABILITY rather than one seductive fit:
  LEFT  sub-S07 ToS (highest-motion run): within-fold CV prediction tracks truth, r>0
  RIGHT sub-S31 ToS (lowest-motion run):  same pipeline, prediction fails, r<0
Opposite signs on the SAME film => one subject cannot establish a sign. This is
"the single-subject decode we almost reported," the brain-arm parallel to the
Spaceman "false negative we almost reported."

Reads data/decode_cache/{sub}_TearsOfSteel.npz (needs the tr_idx field).
Writes figures/brain_decode_single_subject_contrast.{png,pdf}.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_brain as D

FILM = "TearsOfSteel"
PAIR = [("sub-S07", "highest-motion run (fd 0.55)", "#C44E52"),
        ("sub-S31", "lowest-motion run (fd 0.09)", "#4C72B0")]
ALPHA = 10000.0

def zscore(a):
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-12)

def panel(ax, sub, label, color):
    d = D.load_cache(sub, FILM)
    X = d["X"].astype(np.float64)
    y = d["y_human"].astype(np.float64)
    TR = float(d["TR"])
    tr_idx = d["tr_idx"] if "tr_idx" in d.files else np.arange(len(y))
    t = tr_idx * TR
    r, per, pred = D.cv_within_fold(X, y, alpha=ALPHA, return_pred=True)
    ax.plot(t, zscore(y), color="#333333", lw=1.8, label="true human valence (z)")
    ax.plot(t, zscore(pred), color=color, lw=1.0, alpha=0.9, label="CV prediction (z)")
    ax.axhline(0, color="#999999", lw=0.6, zorder=0)
    sign = "tracks truth" if r > 0 else "fails (opposite sign)"
    ax.set_title(f"{sub}  -  {label}\nwithin-fold r = {r:+.2f}   {sign}",
                 fontsize=11, color=color, fontweight="bold")
    ax.set_xlabel("scan time (s)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    return r

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
fig.suptitle("The single-subject decode we almost reported: same film, opposite signs",
             fontsize=13, fontweight="bold", y=1.0)
r07 = panel(axL, *PAIR[0])
r31 = panel(axR, *PAIR[1])
axL.set_ylabel("valence (z-scored)")
fig.text(0.5, -0.02,
         "Same mask, same pipeline, same film (Tears of Steel). The +0.21 came from the "
         "highest-motion subject and reverses on the cleanest one: a single subject "
         "cannot establish a sign. Descriptive; no p-values.",
         ha="center", fontsize=8.5, color="#555555")
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
outdir = os.path.join(D.REPO, "figures"); os.makedirs(outdir, exist_ok=True)
png = os.path.join(outdir, "brain_decode_single_subject_contrast.png")
pdf = os.path.join(outdir, "brain_decode_single_subject_contrast.pdf")
fig.savefig(png, dpi=200, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"sub-S07 r = {r07:+.3f}")
print(f"sub-S31 r = {r31:+.3f}")
print("wrote", png)
print("wrote", pdf)
