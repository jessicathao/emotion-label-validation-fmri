#!/usr/bin/env python3
"""'Go through one participant' fMRI sanity check (Joshua's ask).
Loads one preprocessed BOLD file, prints structural summary, plots whole-brain
and posterior-ROI mean BOLD time courses. NOT a decoder — data inspection only.
Run from anywhere; needs the file materialized via datalad get."""
import nibabel as nib, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
f = os.path.expanduser("~/ds004892/derivatives/preprocessing/sub-S07/ses-1/func/"
                       "sub-S07_ses-1_task-TearsOfSteel_space-MNI_desc-ppres_bold.nii.gz")
img = nib.load(f)
data = img.get_fdata()
TR = float(img.header.get_zooms()[3])
X, Y, Z, T = data.shape
FILM_ON, FILM_OFF = 94.19, 94.19 + 588.03      # scan events: onset 94.19s, dur 588.03s
print("=== Participant fMRI: sub-S07, Tears of Steel (MNI, preprocessed) ===")
print(f"shape (X,Y,Z,T) : {data.shape}")
print(f"voxel size (mm) : {tuple(round(z,2) for z in img.header.get_zooms()[:3])}")
print(f"TR              : {TR}s")
print(f"scan duration   : {T} volumes x {TR}s = {T*TR:.0f}s  (~{T*TR/60:.1f} min)")
print(f"film block      : {FILM_ON:.1f}s - {FILM_OFF:.1f}s (from scan events)")
# whole-brain mean over time (exclude zero/background voxels)
brain = data != 0
wb = np.array([data[..., t][brain[..., t]].mean() for t in range(T)])
# posterior ROI box (visual cortex sees the film) — approximate, index-based
y0 = int(Y*0.78); z0, z1 = int(Z*0.35), int(Z*0.65)
roi = data[:, y0:, z0:z1, :]
roi_mask = roi != 0
occ = np.array([roi[..., t][roi_mask[..., t]].mean() for t in range(T)])
t = np.arange(T)*TR
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(t, wb, color="#065A82", lw=1.0)
ax[0].set_ylabel("whole-brain\nmean BOLD")
ax[0].set_title("sub-S07 - Tears of Steel - preprocessed BOLD (MNI)")
ax[0].axvspan(FILM_ON, FILM_OFF, color="#1C7293", alpha=0.08, label="film block")
ax[0].legend(loc="upper right", fontsize=8, frameon=False)
ax[1].plot(t, occ, color="#A32D2D", lw=1.0)
ax[1].set_ylabel("posterior ROI\nmean BOLD")
ax[1].set_xlabel("time (s)")
ax[1].axvspan(FILM_ON, FILM_OFF, color="#1C7293", alpha=0.08)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out = os.path.expanduser("~/emofilm/figures/fmri_sub-S07_ToS.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved figure ->", out)
