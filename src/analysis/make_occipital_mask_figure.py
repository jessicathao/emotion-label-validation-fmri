#!/usr/bin/env python3
"""make_occipital_mask_figure.py - show WHERE the brain-arm decode reads from.

Builds the decoding ROI with the EXACT definition used in decode_brain.py
(posterior MNI gray matter, y < -60 mm) directly in MNI152 2mm space, verifies the
voxel count matches the decode runs (43683), and renders it two ways:

  TOP    glass brain (the recognizable projection view) - answers "which voxels"
  BOTTOM ROI on the MNI anatomical template (ortho slices) - shows it sits at the
         posterior/occipital pole

This is the MASK only, NOT a per-voxel weight map: in a null result the weights are
not stable, so a weight map would imply structure the decode says is not reliably there.
The mask honestly answers "which voxels"; it is a valence-CORRELATED VISUAL ROI, not a
claim of affect-specific representation.

Writes figures/occipital_mask.{png,pdf}.
"""
import os
import numpy as np
import nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn.datasets import load_mni152_gm_mask, load_mni152_template
from nilearn.plotting import plot_glass_brain, plot_anat

OCC_Y_MM = -60          # identical to decode_brain.py
EXPECTED_VOXELS = 43683  # from the decode runs; this script verifies it
DECK_BLUE = "#1C7293"

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "figures")


def build_occipital_mask():
    """Posterior (y < -60 mm) MNI gray matter, in MNI152 2mm space."""
    gm = load_mni152_gm_mask(resolution=2)
    aff = gm.affine
    ii, jj, kk = np.where(gm.get_fdata() > 0)
    xyz = nib.affines.apply_affine(aff, np.c_[ii, jj, kk])
    occ = xyz[:, 1] < OCC_Y_MM
    m = np.zeros(gm.shape, dtype=np.int8)
    m[ii[occ], jj[occ], kk[occ]] = 1
    return nib.Nifti1Image(m, aff), int(occ.sum())


def main():
    mask_img, nvox = build_occipital_mask()
    print(f"occipital mask voxels: {nvox}  (expected {EXPECTED_VOXELS})")
    if nvox != EXPECTED_VOXELS:
        print(f"  WARNING: voxel count differs from the decode runs by {nvox - EXPECTED_VOXELS}; "
              "check the nilearn MNI template resolution.")
    else:
        print("  MATCH: same voxels as the decode runs.")

    os.makedirs(OUTDIR, exist_ok=True)
    template = load_mni152_template(resolution=2)

    fig = plt.figure(figsize=(11, 7.4))
    fig.suptitle("Decoding ROI: posterior occipital gray matter "
                 f"(y < {OCC_Y_MM} mm, {nvox:,} voxels)",
                 fontsize=13, fontweight="bold", y=0.99, color="#1A1A1A")

    ax_top = fig.add_subplot(2, 1, 1)
    g = plot_glass_brain(None, axes=ax_top, display_mode="lyrz",
                         title=None, black_bg=False)
    g.add_contours(mask_img, levels=[0.5], colors=[DECK_BLUE], linewidths=2.0)

    ax_bot = fig.add_subplot(2, 1, 2)
    d = plot_anat(template, axes=ax_bot, display_mode="ortho", colorbar=False,
                  cut_coords=(-12, -85, 6), draw_cross=False, annotate=True, title=None)
    d.add_overlay(mask_img, cmap=ListedColormap([DECK_BLUE]))
    d.add_contours(mask_img, levels=[0.5], colors=[DECK_BLUE], linewidths=1.5)

    fig.text(0.5, 0.02,
             "The mask only (not a per-voxel weight map): a valence-correlated VISUAL ROI, "
             "not an affect-specific region. Same voxels as the two-film decode.",
             ha="center", fontsize=9, color="#666666")

    png = os.path.join(OUTDIR, "occipital_mask.png")
    pdf = os.path.join(OUTDIR, "occipital_mask.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print("wrote", png)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
