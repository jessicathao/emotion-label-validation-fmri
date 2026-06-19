#!/usr/bin/env python3
"""decode_brain.py - canonical occipital valence-decode pipeline for Emo-FilM.

Within-fold Pearson (single subject) + leave-one-subject-out Ridge (pooled), each
gated by a planted-signal positive control. See RESULTS_brain_arm_June19.md.
Occipital mask => valence-correlated VISUAL signal, not pure affect. One ROI, one
target. Significance withheld project-wide; descriptive effect sizes only.

USAGE
  python decode_brain.py extract <film> <sub> [sub ...]   # cache occipital matrices
  python decode_brain.py single  <film> <sub>             # within-fold r + controls
  python decode_brain.py pooled  <film> <sub> <sub> ...   # leave-one-subject-out + gates
"""
import os, sys, glob, gzip, json, argparse
import numpy as np, pandas as pd, nibabel as nib
from scipy.stats import pearsonr, gamma
from sklearn.linear_model import Ridge
from nilearn.datasets import load_mni152_gm_mask
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiMasker
import warnings; warnings.filterwarnings("ignore")

TR_DEFAULT = 1.3
PLEASANTOTHER_COL = 3
ALPHA = 10000.0
OCC_Y_MM = -60
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DS_FMRI = os.path.expanduser("~/ds004892")
DS_ANNOT = os.path.expanduser("~/ds004872")
CACHE = os.path.join(REPO, "data", "decode_cache")
RESULTS = os.path.join(REPO, "results")
GEMINI_DIR = os.path.join(REPO, "data", "transcripts")

def spm_hrf_1hz(length=32):
    t = np.arange(0, length, 1.0)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return h / h.sum()
HRF = spm_hrf_1hz()

def place_on_tr(series_1hz, onset, tr_times):
    c = np.convolve(np.nan_to_num(series_1hz, nan=0.0), HRF)[:len(series_1hz)]
    c[np.isnan(series_1hz)] = np.nan
    return np.interp(tr_times, onset + np.arange(len(c)), c, left=np.nan, right=np.nan)

def load_human(film):
    path = os.path.join(DS_ANNOT, "derivatives", f"Annot_{film}_stim.tsv.gz")
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split("\t")])
    return np.array(rows)[:, PLEASANTOTHER_COL]

def load_gemini_context(film):
    path = os.path.join(GEMINI_DIR, f"{film}_gemini_context.json")
    raw = json.load(open(path))["valence"]
    return np.array([x if x is not None else np.nan for x in raw], dtype=float)

def occ_masker(img):
    gm = resample_to_img(load_mni152_gm_mask(resolution=2), img, interpolation="nearest")
    aff = img.affine
    ii, jj, kk = np.where(gm.get_fdata() > 0)
    xyz = nib.affines.apply_affine(aff, np.c_[ii, jj, kk])
    occ = xyz[:, 1] < OCC_Y_MM
    m = np.zeros(img.shape[:3], bool)
    m[ii[occ], jj[occ], kk[occ]] = True
    masker = NiftiMasker(mask_img=nib.Nifti1Image(m.astype(np.int8), aff),
                         standardize=True, detrend=True)
    return masker, int(m.sum())

def find_run(sub, film):
    bold = glob.glob(os.path.join(DS_FMRI, "derivatives", "preprocessing", sub, "ses-*",
                     "func", f"{sub}_ses-*_task-{film}_space-MNI_desc-ppres_bold.nii.gz"))
    ev = glob.glob(os.path.join(DS_FMRI, sub, "ses-*", "func",
                   f"{sub}_ses-*_task-scan_acq-{film}_events.tsv"))
    if not bold or not ev:
        raise FileNotFoundError(f"run not found for {sub} {film} (datalad get it first)")
    return bold[0], ev[0]

def film_onset(events_path):
    df = pd.read_csv(events_path, sep="\t")
    row = df[df["trial_type"] == "film"].iloc[0]
    return float(row["onset"]), float(row["duration"])

def extract_and_cache(sub, film, overwrite=False):
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, f"{sub}_{film}.npz")
    if os.path.exists(out) and not overwrite:
        print(f"[{sub}] cache exists, skipping"); return out
    bold, ev = find_run(sub, film)
    onset, dur = film_onset(ev)
    img = nib.load(bold)
    TR = float(img.header.get_zooms()[3]) or TR_DEFAULT
    masker, nvox = occ_masker(img)
    X = masker.fit_transform(img)
    tr_times = np.arange(img.shape[3]) * TR
    y_h = place_on_tr(load_human(film), onset, tr_times)
    try:
        y_g = place_on_tr(load_gemini_context(film), onset, tr_times)
    except FileNotFoundError:
        y_g = np.full_like(y_h, np.nan)
    fm = ~np.isnan(y_h)
    np.savez_compressed(out, X=X[fm].astype(np.float32), y_human=y_h[fm].astype(np.float32),
                        y_gemini=y_g[fm].astype(np.float32), nvox=nvox,
                        onset=onset, dur=dur, TR=TR, tr_idx=np.where(fm)[0].astype(np.int32))
    print(f"[{sub}] cached {int(fm.sum())} vols x {nvox} vox -> {out}")
    return out

def load_cache(sub, film):
    return np.load(os.path.join(CACHE, f"{sub}_{film}.npz"))

def cv_within_fold(X, y, K=6, alpha=ALPHA, return_pred=False):
    n = len(y); b = np.linspace(0, n, K + 1).astype(int)
    rs = []; pred = np.full(n, np.nan)
    for k in range(K):
        te = np.zeros(n, bool); te[b[k]:b[k + 1]] = True
        if (~te).sum() < 5 or te.sum() < 5:
            rs.append(np.nan); continue
        p = Ridge(alpha=alpha).fit(X[~te], y[~te]).predict(X[te])
        pred[te] = p
        rs.append(np.nan if np.std(p) < 1e-9 or np.std(y[te]) < 1e-9 else pearsonr(p, y[te])[0])
    rs = np.array(rs)
    return (float(np.nanmean(rs)), rs, pred) if return_pred else (float(np.nanmean(rs)), rs)

def loso(Xs, targets, alpha=ALPHA):
    rs = []
    for i in range(len(Xs)):
        Xtr = np.vstack([Xs[j] for j in range(len(Xs)) if j != i])
        ytr = np.concatenate([targets[j] for j in range(len(Xs)) if j != i])
        p = Ridge(alpha=alpha).fit(Xtr, ytr).predict(Xs[i])
        rs.append(np.nan if np.std(p) < 1e-9 else pearsonr(p, targets[i])[0])
    return np.array(rs)

def _planted(n, TR, period):
    t = np.arange(n) * TR
    s = np.sin(2*np.pi*t/period) + 0.5*np.sin(2*np.pi*t/(period*0.6)+1.0)
    return (s - s.mean()) / s.std()

def positive_control_pooled(Xs, TR, timescales=(250,150,100,60,30), snrs=(1.0,0.3), alpha=ALPHA, seed=0):
    rng = np.random.default_rng(seed); n_min = min(x.shape[0] for x in Xs)
    v = rng.standard_normal(Xs[0].shape[1]); v /= np.linalg.norm(v)
    Xz = [(x - x.mean(0))/(x.std(0)+1e-8) for x in Xs]
    out = {}
    for p in timescales:
        tc = _planted(n_min, TR, p); row = {}
        for snr in snrs:
            Xi = [Xz[i][:n_min] + snr*np.outer(tc, v) for i in range(len(Xz))]
            row[f"snr{snr}"] = float(np.nanmean(loso(Xi, [tc.copy() for _ in Xz], alpha)))
        out[p] = row
    return out

def positive_control_single(X, TR, timescales=(250,150,100,60,30), snrs=(2.0,1.0,0.3), alpha=ALPHA, seed=0):
    rng = np.random.default_rng(seed); Xz = (X - X.mean(0))/(X.std(0)+1e-8)
    v = rng.standard_normal(X.shape[1]); v /= np.linalg.norm(v); out = {}
    for p in timescales:
        tc = _planted(len(X), TR, p)
        out[p] = {f"snr{s}": cv_within_fold(Xz + s*np.outer(tc, v), tc, alpha=alpha)[0] for s in snrs}
    return out

def shift_null_pooled(Xs, ys, fracs=(0.2,0.35,0.5,0.65,0.8), alpha=ALPHA):
    return np.array([float(np.nanmean(loso(Xs, [np.roll(y, int(len(y)*f)) for y in ys], alpha))) for f in fracs])

def cmd_extract(a):
    for s in a.subjects: extract_and_cache(s, a.film, overwrite=a.overwrite)

def cmd_single(a):
    d = load_cache(a.subjects[0], a.film); X, y, TR = d["X"], d["y_human"], float(d["TR"])
    print(f"{a.subjects[0]} {a.film}: {len(y)} vols, {int(d['nvox'])} vox, TR {TR}")
    for al in (100.,1000.,10000.):
        m, per = cv_within_fold(X, y, alpha=al)
        print(f"  within-fold r (alpha={al:7.0f}) = {m:+.3f}  folds={[round(x,2) for x in per]}")
    print("  positive control (single-subject):")
    for p, row in positive_control_single(X, TR).items():
        print(f"    {p:>4}s  " + "  ".join(f"{k}={v:+.3f}" for k, v in row.items()))

def cmd_pooled(a):
    Xs, ys = [], []; TR = TR_DEFAULT
    for s in a.subjects:
        d = load_cache(s, a.film); Xs.append(d["X"].astype(np.float64))
        ys.append((d["y_human"] - d["y_human"].mean()).astype(np.float64)); TR = float(d["TR"])
        print(f"  {s}: {Xs[-1].shape[0]} vols x {Xs[-1].shape[1]} vox")
    print(f"pooled: {len(a.subjects)} subjects, {sum(x.shape[0] for x in Xs)} total volumes")
    print("\n[GATE 1] positive control (planted, leave-one-subject-out):")
    pc = positive_control_pooled(Xs, TR)
    for p, row in pc.items():
        print(f"  {p:>4}s  " + "  ".join(f"{k}={v:+.3f}" for k, v in row.items()))
    print("\n[GATE 2] real valence (leave-one-subject-out + shift null):")
    real = loso(Xs, ys); nulls = shift_null_pooled(Xs, ys)
    print(f"  per-subject r = {[round(x,3) for x in real]}")
    print(f"  real mean = {np.nanmean(real):+.3f}")
    print(f"  shift-null mean {nulls.mean():+.3f}  sd {nulls.std():.3f}")
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"brain_decode_{a.film}.json")
    json.dump({"film": a.film, "subjects": list(a.subjects),
               "n_volumes_total": int(sum(x.shape[0] for x in Xs)),
               "positive_control_pooled": {"timescales_s": list(pc.keys()),
                    "recovery_r_snr1": [pc[p].get("snr1.0") for p in pc],
                    "recovery_r_snr0_3": [pc[p].get("snr0.3") for p in pc],
                    "valence_timescale_s": 100},
               "real_valence_pooled": {"loso_r_per_subject": [round(float(x),3) for x in real],
                    "loso_mean": round(float(np.nanmean(real)),3),
                    "shift_null_mean": round(float(nulls.mean()),3),
                    "shift_null_sd": round(float(nulls.std()),3)}},
              open(out, "w"), indent=2)
    print(f"\nsaved {out}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("extract", cmd_extract), ("single", cmd_single), ("pooled", cmd_pooled)):
        p = sub.add_parser(name); p.add_argument("film"); p.add_argument("subjects", nargs="+")
        if name == "extract": p.add_argument("--overwrite", action="store_true")
        p.set_defaults(func=fn)
    a = ap.parse_args(); a.func(a)

if __name__ == "__main__":
    main()
