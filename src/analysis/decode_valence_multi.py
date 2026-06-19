#!/usr/bin/env python3
# DEPRECATED: raw pooled predicted-vs-true r is the drift-prone metric. Use decode_brain.py
# (within-fold + leave-one-subject-out, validated by a planted-signal positive control).
"""Powered human-vs-automatic valence decode, Payload, multi-subject.

Extends decode_valence_poc.py to a dialogue-DENSE film across QC-selected
subjects, reporting an effect-size DISTRIBUTION (per-subject r's), not one
number. Significance WITHHELD project-wide; descriptive effect sizes only.

Two arms per subject:
  [1] HUMAN-only decode on all film volumes (PoC-style; alpha sweep).
  [2] FAIR comparison on shared (both-defined) volumes: human vs Gemini-context
      target, same mask/CV. CV blocks are cut on the REAL film timeline then
      subset to dialogue volumes (NOT on the collapsed index) so autocorrelation
      does not leak across folds.

Occipital mask => valence-CORRELATED VISUAL signal, not pure affect. PoC-grade
claim, now powered. Film onset/duration read from each subject's scan events
file, never eyeballed.
"""
import nibabel as nib, numpy as np, json, gzip, os, sys
import pandas as pd
from scipy.stats import pearsonr, gamma
from sklearn.linear_model import Ridge
from nilearn.datasets import load_mni152_gm_mask
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiMasker
import warnings; warnings.filterwarnings("ignore")

FILM = "Payload"
PLEASANTOTHER_COL = 3
ALPHAS = (100., 1000., 10000.)
ALPHA_CMP = 10000.0     # comparison arm uses the PoC's reported alpha
K_HUMAN = 6             # human-only arm, full film volumes (matches PoC)
K_SHARED = 10           # shared arm; blocks cut on real timeline (see below)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREP = os.path.expanduser("~/ds004892/derivatives/preprocessing")
EVENTS = os.path.expanduser("~/ds004892")
ANNOT = os.path.expanduser(f"~/ds004872/derivatives/Annot_{FILM}_stim.tsv.gz")
GEMINI = os.path.join(REPO, "data", "transcripts", f"{FILM}_gemini_context.json")

# subject -> session (from the path listing; QC-selected order, lowest motion first)
SUBJECTS = {"sub-S08": "ses-4", "sub-S05": "ses-4", "sub-S11": "ses-2",
            "sub-S15": "ses-2", "sub-S06": "ses-1"}

def spm_hrf_1hz(length=32):
    t = np.arange(0, length, 1.0)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return h / h.sum()
HRF = spm_hrf_1hz()

def read_film_onset(sub, ses):
    ev = os.path.join(EVENTS, sub, ses, "func",
                      f"{sub}_{ses}_task-scan_acq-{FILM}_events.tsv")
    df = pd.read_csv(ev, sep="\t")
    row = df[df["trial_type"] == "film"].iloc[0]
    return float(row["onset"]), float(row["duration"])

def load_human():
    rows = []
    with gzip.open(ANNOT, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split("\t")])
    return np.array(rows)[:, PLEASANTOTHER_COL]

def load_gemini():
    raw = json.load(open(GEMINI))["valence"]
    return np.array([x if x is not None else np.nan for x in raw], dtype=float)

def place_on_tr(series_1hz, onset, tr_times):
    c = np.convolve(np.nan_to_num(series_1hz, nan=0.0), HRF)[:len(series_1hz)]
    c[np.isnan(series_1hz)] = np.nan
    return np.interp(tr_times, onset + np.arange(len(c)), c, left=np.nan, right=np.nan)

def occ_masker(img):
    gm = resample_to_img(load_mni152_gm_mask(resolution=2), img, interpolation="nearest")
    aff = img.affine; gmd = gm.get_fdata() > 0
    ii, jj, kk = np.where(gmd)
    xyz = nib.affines.apply_affine(aff, np.c_[ii, jj, kk])
    occ = xyz[:, 1] < -60
    m = np.zeros(img.shape[:3], bool); m[ii[occ], jj[occ], kk[occ]] = True
    return NiftiMasker(mask_img=nib.Nifti1Image(m.astype(np.int8), aff),
                       standardize=True, detrend=True), int(m.sum())

def cv_r_contig(X, y, K, alpha):
    """CV on contiguous chunks of the array as given (use for full-film arm)."""
    n = len(y); pred = np.full(n, np.nan); b = np.linspace(0, n, K + 1).astype(int)
    for k in range(K):
        te = np.zeros(n, bool); te[b[k]:b[k+1]] = True
        pred[te] = Ridge(alpha=alpha).fit(X[~te], y[~te]).predict(X[te])
    return pearsonr(pred, y)[0]

def cv_r_realtime(X, y, tr_idx, K, alpha, TR):
    """CV with folds cut on the REAL TR timeline, then applied to retained vols.
    tr_idx = real volume index of each retained sample. Blocks are contiguous
    spans of real time; a retained sample is tested in the block its real index
    falls into. Prevents autocorrelation leak across folds."""
    lo, hi = tr_idx.min(), tr_idx.max() + 1
    edges = np.linspace(lo, hi, K + 1).astype(int)
    pred = np.full(len(y), np.nan)
    sizes = []
    for k in range(K):
        te = (tr_idx >= edges[k]) & (tr_idx < edges[k+1])
        sizes.append((int(te.sum()), round((edges[k+1]-edges[k]) * TR)))
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        pred[te] = Ridge(alpha=alpha).fit(X[~te], y[~te]).predict(X[te])
    ok = ~np.isnan(pred)
    return pearsonr(pred[ok], y[ok])[0], sizes

def run_subject(sub, ses, human_1hz, gem_1hz, verbose=True):
    bold = os.path.join(PREP, sub, ses, "func",
                        f"{sub}_{ses}_task-{FILM}_space-MNI_desc-ppres_bold.nii.gz")
    img = nib.load(bold); TR = float(img.header.get_zooms()[3]); n_tr = img.shape[3]
    onset, dur = read_film_onset(sub, ses)
    tr_times = np.arange(n_tr) * TR
    masker, nvox = occ_masker(img)
    X_all = masker.fit_transform(img)
    y_h = place_on_tr(human_1hz, onset, tr_times)
    y_g = place_on_tr(gem_1hz, onset, tr_times)

    film_h = ~np.isnan(y_h)
    human_sweep = {a: cv_r_contig(X_all[film_h], y_h[film_h], K_HUMAN, a) for a in ALPHAS}

    both = (~np.isnan(y_h)) & (~np.isnan(y_g))
    tr_idx = np.where(both)[0]
    Xb, yh, yg = X_all[both], y_h[both], y_g[both]
    r_h, sizes = cv_r_realtime(Xb, yh, tr_idx, K_SHARED, ALPHA_CMP, TR)
    r_g, _ = cv_r_realtime(Xb, yg, tr_idx, K_SHARED, ALPHA_CMP, TR)
    r_ll = pearsonr(yh, yg)[0]

    if verbose:
        print(f"\n=== {sub} ({ses}) ===")
        print(f"  occipital voxels: {nvox} | film onset {onset:.1f}s dur {dur:.1f}s | {n_tr} TR")
        print(f"  [1] HUMAN-only, {int(film_h.sum())} film volumes:")
        for a in ALPHAS:
            print(f"        alpha={a:7.0f}  r = {human_sweep[a]:+.3f}")
        print(f"  [2] SHARED set: {int(both.sum())} volumes (alpha={ALPHA_CMP:.0f}, K={K_SHARED})")
        print(f"        per-fold (n_vol, span_s): {sizes}")
        print(f"        human  target r = {r_h:+.3f}")
        print(f"        gemini target r = {r_g:+.3f}")
        print(f"        label-vs-label sanity r(human,gemini) = {r_ll:+.3f}")
    return {"subject": sub, "n_film": int(film_h.sum()), "n_shared": int(both.sum()),
            **{f"human_a{int(a)}": human_sweep[a] for a in ALPHAS},
            "shared_human": r_h, "shared_gemini": r_g, "label_vs_label": r_ll}

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    human_1hz, gem_1hz = load_human(), load_gemini()
    print(f"FILM={FILM} | human defined {np.isfinite(human_1hz).sum()}/{len(human_1hz)}"
          f" | gemini defined {np.isfinite(gem_1hz).sum()}/{len(gem_1hz)}")
    subs = {only: SUBJECTS[only]} if only else SUBJECTS
    res = [run_subject(s, ses, human_1hz, gem_1hz) for s, ses in subs.items()]
    df = pd.DataFrame(res)
    if len(df) > 1:
        print("\n==== DISTRIBUTION ACROSS SUBJECTS ====")
        print(df.to_string(index=False))
        for col, lab in [("human_a10000", "human-only (full film)"),
                         ("shared_human", "shared: human"),
                         ("shared_gemini", "shared: gemini")]:
            v = df[col].values
            print(f"  {lab:28s} median {np.median(v):+.3f}  range [{v.min():+.3f}, {v.max():+.3f}]")
        out = os.path.join(REPO, "results", f"decode_{FILM}_multi.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False); print(f"\nsaved {out}")

if __name__ == "__main__":
    main()
