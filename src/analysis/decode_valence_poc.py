#!/usr/bin/env python3
"""Valence decoding proof-of-concept: sub-S07, Tears of Steel (single run).

Decodes human-annotated valence (PleasantOther consensus) from occipital BOLD,
and attempts a fair human-vs-automatic (Gemini-context) comparison on the SAME
volumes/mask/CV.

HONEST FRAMING (do not overclaim):
  * RAW-ish single-subject preprocessed BOLD (WM/CSF + motion regressed; no
    smoothing/decode-filtering). PRELIMINARY proof-of-concept only.
  * Occipital mask => this is valence-CORRELATED VISUAL signal, not a pure
    affective representation.
  * Contiguous-block CV (NOT random) because annotation autocorrelation ~80-110s
    would leak across random splits. Significance WITHHELD project-wide; report
    descriptive effect sizes only.
  * Film onset 94.19s, duration 588.03s taken from the scan events file
    (sub-S07_ses-1_task-scan_acq-TearsOfSteel_events.tsv), NOT eyeballed.

RESULTS (this session, alpha=10000):
  * Human PleasantOther, 451 film volumes: predicted-vs-true r ~= +0.21 (stable
    across alpha 100/1000/10000 -> +0.16/+0.18/+0.21).
  * Fair human-vs-Gemini comparison: UNDERPOWERED on ToS, only 70 shared
    volumes (Gemini-context scored on 147/567 s; ToS is dialogue-light), so the
    comparison r's are NOT interpretable. Label-vs-label sanity on shared set
    r(human,gemini)=+0.63 (consistent with validated +0.60). A powered
    comparison needs a dialogue-DENSE film (Payload / LessonLearned).
"""
import nibabel as nib, numpy as np, json, gzip, os
from scipy.stats import pearsonr, gamma
from sklearn.linear_model import Ridge
from nilearn.datasets import load_mni152_gm_mask
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiMasker
import warnings; warnings.filterwarnings("ignore")

# ---- paths ----
BOLD = os.path.expanduser("~/ds004892/derivatives/preprocessing/sub-S07/ses-1/func/"
                          "sub-S07_ses-1_task-TearsOfSteel_space-MNI_desc-ppres_bold.nii.gz")
ANNOT = os.path.expanduser("~/ds004872/derivatives/Annot_TearsOfSteel_stim.tsv.gz")
GEMINI = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "transcripts", "TearsOfSteel_gemini_context.json")
PLEASANTOTHER_COL = 3            # column index in the annotation table
FILM_ONSET = 94.19099301198548  # scan events: film onset (s)
ALPHA = 10000.0
K = 6                            # contiguous CV blocks

# ---- load BOLD, build occipital mask (posterior MNI gray matter, y < -60 mm) ----
img = nib.load(BOLD); TR = float(img.header.get_zooms()[3]); n_tr = img.shape[3]
gm = resample_to_img(load_mni152_gm_mask(resolution=2), img, interpolation="nearest")
aff = img.affine; gm_data = gm.get_fdata() > 0
ii, jj, kk = np.where(gm_data)
xyz = nib.affines.apply_affine(aff, np.c_[ii, jj, kk])
occ = xyz[:, 1] < -60
occ_mask = np.zeros(img.shape[:3], bool); occ_mask[ii[occ], jj[occ], kk[occ]] = True
print(f"occipital voxels: {int(occ_mask.sum())}")
masker = NiftiMasker(mask_img=nib.Nifti1Image(occ_mask.astype(np.int8), aff),
                     standardize=True, detrend=True)
X_all = masker.fit_transform(img)            # (n_tr, n_voxels)

# ---- HRF (SPM double-gamma, 1 Hz) ----
def spm_hrf_1hz(length=32):
    t = np.arange(0, length, 1.0)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return h / h.sum()
hrf = spm_hrf_1hz()
tr_times = np.arange(n_tr) * TR

def place_on_tr(series_1hz):
    """HRF-convolve a 1 Hz label, place at film onset, resample to TR grid (NaN outside)."""
    c = np.convolve(np.nan_to_num(series_1hz, nan=0.0), hrf)[:len(series_1hz)]
    c[np.isnan(series_1hz)] = np.nan
    return np.interp(tr_times, FILM_ONSET + np.arange(len(c)), c, left=np.nan, right=np.nan)

# ---- targets ----
rows = []
with gzip.open(ANNOT, "rt") as f:
    for line in f:
        line = line.strip()
        if line: rows.append([float(x) for x in line.split("\t")])
human = np.array(rows)[:, PLEASANTOTHER_COL]
y_h = place_on_tr(human)

gem_raw = json.load(open(GEMINI))["valence"]
gem = np.array([x if x is not None else np.nan for x in gem_raw], dtype=float)
y_g = place_on_tr(gem)

# ---- contiguous-block CV ridge, predicted-vs-true r ----
def cv_r(X, y, K=K, alpha=ALPHA):
    n = len(y); pred = np.full(n, np.nan); b = np.linspace(0, n, K + 1).astype(int)
    for k in range(K):
        te = np.zeros(n, bool); te[b[k]:b[k+1]] = True
        pred[te] = Ridge(alpha=alpha).fit(X[~te], y[~te]).predict(X[te])
    return pearsonr(pred, y)[0]

# (1) human PoC on all film volumes
film_h = ~np.isnan(y_h)
print(f"\n[1] HUMAN PoC: {int(film_h.sum())} film volumes, alpha sweep:")
for a in (100., 1000., 10000.):
    print(f"    alpha={a:7.0f}  predicted-vs-true r = {cv_r(X_all[film_h], y_h[film_h], alpha=a):+.3f}")

# (2) fair comparison on shared (both-defined) volumes, UNDERPOWERED on ToS
both = ~np.isnan(y_h) & ~np.isnan(y_g)
Xb = X_all[both]; yh = y_h[both]; yg = y_g[both]
print(f"\n[2] FAIR COMPARISON: {int(both.sum())} shared volumes (alpha={ALPHA:.0f})")
print(f"    human  target : r = {cv_r(Xb, yh):+.3f}")
print(f"    gemini target : r = {cv_r(Xb, yg):+.3f}")
print(f"    label-vs-label sanity r(human,gemini) on shared set = {pearsonr(yh, yg)[0]:+.3f}")
if both.sum() < 150:
    print("    !! UNDERPOWERED: too few shared volumes; comparison r's NOT interpretable.")
    print("    !! Needs a dialogue-DENSE film (Payload / LessonLearned) for a powered test.")
