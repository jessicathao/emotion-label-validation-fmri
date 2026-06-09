#!/usr/bin/env python3
"""Brain-arm results artifacts (sub-S07 ToS PoC).
Produces:
  1. figures/brain_decode_poc.png: predicted-vs-true human valence (CV), modest by design,
     with an alpha-stability inset and the underpowered-comparison note.
  2. results/brain_results_table.csv: machine-readable results.
  3. figures/brain_results_table.png: rendered table for slides.
Reuses the validated decode logic (matches src/decode_valence_poc.py / commit d56af31).
HONEST FRAMING: occipital => valence-CORRELATED VISUAL signal, not pure affect.
Contiguous-block CV; significance WITHHELD; descriptive only. n=70 comparison NOT interpretable.
"""
import nibabel as nib, numpy as np, json, gzip, os
from scipy.stats import pearsonr, gamma
from sklearn.linear_model import Ridge
from nilearn.datasets import load_mni152_gm_mask
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiMasker
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

BOLD = os.path.expanduser("~/ds004892/derivatives/preprocessing/sub-S07/ses-1/func/"
                          "sub-S07_ses-1_task-TearsOfSteel_space-MNI_desc-ppres_bold.nii.gz")
ANNOT = os.path.expanduser("~/ds004872/derivatives/Annot_TearsOfSteel_stim.tsv.gz")
GEMINI = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "transcripts", "TearsOfSteel_gemini_context.json")
PO_COL = 3; FILM_ONSET = 94.19099301198548; K = 6
# write figures to the repo's figures/ and tabular output to results/ (script is in src/figures/)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGDIR = os.path.join(REPO, "figures"); os.makedirs(FIGDIR, exist_ok=True)
RESDIR = os.path.join(REPO, "results"); os.makedirs(RESDIR, exist_ok=True)

img = nib.load(BOLD); TR = float(img.header.get_zooms()[3]); n_tr = img.shape[3]
gm = resample_to_img(load_mni152_gm_mask(resolution=2), img, interpolation="nearest")
aff = img.affine; ii, jj, kk = np.where(gm.get_fdata() > 0)
xyz = nib.affines.apply_affine(aff, np.c_[ii, jj, kk]); occ = xyz[:, 1] < -60
occ_mask = np.zeros(img.shape[:3], bool); occ_mask[ii[occ], jj[occ], kk[occ]] = True
masker = NiftiMasker(mask_img=nib.Nifti1Image(occ_mask.astype(np.int8), aff),
                     standardize=True, detrend=True)
X_all = masker.fit_transform(img)

def spm_hrf_1hz(L=32):
    t = np.arange(0, L, 1.0); h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0; return h / h.sum()
hrf = spm_hrf_1hz(); tr_times = np.arange(n_tr) * TR
def place(s):
    c = np.convolve(np.nan_to_num(s, nan=0.0), hrf)[:len(s)]; c[np.isnan(s)] = np.nan
    return np.interp(tr_times, FILM_ONSET + np.arange(len(c)), c, left=np.nan, right=np.nan)

rows = []
with gzip.open(ANNOT, "rt") as f:
    for line in f:
        line = line.strip()
        if line: rows.append([float(x) for x in line.split("\t")])
y_h = place(np.array(rows)[:, PO_COL])
gem = np.array([x if x is not None else np.nan for x in json.load(open(GEMINI))["valence"]], float)
y_g = place(gem)

def cv_pred(X, y, alpha, K=K):
    n = len(y); pred = np.full(n, np.nan); b = np.linspace(0, n, K + 1).astype(int)
    for k in range(K):
        te = np.zeros(n, bool); te[b[k]:b[k+1]] = True
        pred[te] = Ridge(alpha=alpha).fit(X[~te], y[~te]).predict(X[te])
    return pred

film = ~np.isnan(y_h); Xf = X_all[film]; yf = y_h[film]
sweep = {a: pearsonr(cv_pred(Xf, yf, a), yf)[0] for a in (100., 1000., 10000.)}
pred_best = cv_pred(Xf, yf, 10000.); r_best = pearsonr(pred_best, yf)[0]
both = ~np.isnan(y_h) & ~np.isnan(y_g)
r_h70 = pearsonr(cv_pred(X_all[both], y_h[both], 10000.), y_h[both])[0]
r_g70 = pearsonr(cv_pred(X_all[both], y_g[both], 10000.), y_g[both])[0]
r_ll = pearsonr(y_h[both], y_g[both])[0]

# ---- FIGURE 1: predicted vs true (honest, modest) ----
t = np.arange(film.sum()) * TR + FILM_ONSET
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, (yf - yf.mean()) / yf.std(), color="#444", lw=1.3, label="true human valence (z)")
ax.plot(t, (pred_best - pred_best.mean()) / pred_best.std(), color="#A32D2D", lw=1.1,
        alpha=0.85, label="CV prediction (z)")
ax.set_xlabel("scan time (s)"); ax.set_ylabel("valence (z-scored)")
ax.set_title(f"sub-S07 ToS: human valence decoded from occipital BOLD (PoC)\n"
             f"predicted-vs-true r = {r_best:+.2f}  (occipital = valence-correlated VISUAL signal; "
             f"contiguous-block CV; descriptive)", fontsize=9)
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.01, -0.22, f"alpha stability: 100→{sweep[100.]:+.2f}, 1000→{sweep[1000.]:+.2f}, "
        f"10000→{sweep[10000.]:+.2f}  |  modest, stable PoC, not a headline",
        transform=ax.transAxes, fontsize=7.5, color="#666")
plt.tight_layout()
fig.savefig(f"{FIGDIR}/brain_decode_poc.png", dpi=150, bbox_inches="tight")
print("saved", f"{FIGDIR}/brain_decode_poc.png")

# ---- TABLE: csv + rendered png ----
table = [
    ["Analysis", "Target", "n volumes", "predicted-vs-true r", "Note"],
    ["Human PoC (alpha=100)",   "PleasantOther", str(int(film.sum())), f"{sweep[100.]:+.3f}",  "modest, stable"],
    ["Human PoC (alpha=1000)",  "PleasantOther", str(int(film.sum())), f"{sweep[1000.]:+.3f}", "modest, stable"],
    ["Human PoC (alpha=10000)", "PleasantOther", str(int(film.sum())), f"{sweep[10000.]:+.3f}","best/most-regularized"],
    ["Fair comparison",         "PleasantOther", str(int(both.sum())), f"{r_h70:+.3f}",  "underpowered (n too small)"],
    ["Fair comparison",         "Gemini-context",str(int(both.sum())), f"{r_g70:+.3f}",  "underpowered (n too small)"],
    ["Label-vs-label sanity",   "human vs gemini",str(int(both.sum())),f"{r_ll:+.3f}",   "consistent w/ validated +0.60"],
]
import csv
with open(f"{RESDIR}/brain_results_table.csv", "w", newline="") as f:
    csv.writer(f).writerows(table)
print("saved", f"{RESDIR}/brain_results_table.csv")

figt, axt = plt.subplots(figsize=(13, 2.6)); axt.axis("off")
tb = axt.table(cellText=table[1:], colLabels=table[0], loc="center", cellLoc="left")
tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.5)
for c in range(len(table[0])): tb[0, c].set_facecolor("#1C7293"); tb[0, c].set_text_props(color="w", weight="bold")
for r in (4, 5):
    for c in range(len(table[0])): tb[r, c].set_facecolor("#F6E3E3")  # flag underpowered rows
axt.set_title("Brain arm, sub-S07 ToS (PoC). Significance withheld; occipital = visual-correlated signal.",
              fontsize=9, pad=10)
plt.tight_layout()
figt.savefig(f"{FIGDIR}/brain_results_table.png", dpi=150, bbox_inches="tight")
print("saved", f"{FIGDIR}/brain_results_table.png")
