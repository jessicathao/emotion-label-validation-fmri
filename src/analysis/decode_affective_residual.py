#!/usr/bin/env python
"""decode_affective_residual.py - visual-residual check on the affective ROIs.
Framing-critical: does removing the low-level visual confound from the TARGET
(valence_resid_visual) expose valence in the cortical ROIs on Tears of Steel (=> 'less
sensitive test'), or does it stay null (=> 'true absence')? Payload is the CONTROL: the
affective valence decode should SURVIVE residualization if it is affect, not a visual confound.

Reuse discipline: residualize_1hz and load_visual are IMPORTED from decode_target_multi, so
the residual is the SAME construction used in the June-20 occipital decomposition; validated
by reproducing the cached R2 in results/visual_decomposition_<film>.json (stops on mismatch).
Decode reuses decode_brain.loso / positive_control_pooled / shift_null_pooled / ALPHA
unchanged. Strict-vmPFC via the same column subset as decode_vmpfc_strict. No BOLD, no OpenCV.
Valence is recomputed here on the SAME (residual-finite) volumes as the residual, so the two
are a matched paired contrast; with 0 volumes dropped these reproduce the canonical valence."""
import os, sys, json, numpy as np, nibabel as nib
sys.path.insert(0, "src/analysis")
import decode_brain as D
from decode_target_multi import residualize_1hz, load_visual, load_human
try:
    from decode_target_multi import FEATURES
except Exception:
    from build_visual_regressors import FEATURES
from decode_vmpfc_strict import vmpfc_masks, find_bold

ALPHA = D.ALPHA
FILMS = {"Payload":      ["sub-S08","sub-S05","sub-S11","sub-S15","sub-S06"],
         "TearsOfSteel": ["sub-S31","sub-S08","sub-S11","sub-S25","sub-S04"]}
ROIS  = ["network","insula","vmpfc","vmpfc_strict","amygdala"]

def resid_1hz(film):
    y1 = np.asarray(load_human(film), float)
    feat = load_visual(film)
    if isinstance(feat, dict) or hasattr(feat, "keys"):
        V1 = np.column_stack([np.asarray(feat[f], float) for f in FEATURES])
    else:
        V1 = np.asarray(feat, float)
    n = min(len(y1), V1.shape[0]); y1, V1 = y1[:n], V1[:n]
    yr, R2 = residualize_1hz(y1, V1)
    R2c = json.load(open(f"results/visual_decomposition_{film}.json"))\
              ["stimulus_side"]["visual_set_R2_predicting_valence"]
    ok = abs(float(R2) - float(R2c)) < 0.005
    print(f"  {film}: reproduced R2={float(R2):+.3f}  cached={float(R2c):+.3f}  {'OK' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit("Residual differs from June-20. Paste: sed -n '50,105p' "
                 "src/analysis/decode_target_multi.py  and I will match V1 exactly.")
    return np.asarray(yr, float)

def strict_member():
    full, strict = vmpfc_masks(nib.load(find_bold("sub-S08","Payload")))
    return strict.ravel()[np.flatnonzero(full.ravel())]

def load_roi(roi, sub, film, member):
    src = "vmpfc" if roi == "vmpfc_strict" else roi
    z = np.load(f"data/decode_cache_{src}/{sub}_{film}.npz", allow_pickle=True)
    X = z["X"].astype(float)
    if roi == "vmpfc_strict": X = X[:, member]
    return X, z["y_human"].astype(float), float(z["TR"]), float(z["onset"]), z["tr_idx"].astype(int)

def clean(Xs):
    g = np.ones(Xs[0].shape[1], bool)
    for X in Xs: g &= np.isfinite(X).all(0) & (np.nanstd(X,0) > 1e-6)
    return [X[:, g] for X in Xs]

def decode(Xs, ys):
    real = D.loso(Xs, ys, alpha=ALPHA); nul = D.shift_null_pooled(Xs, ys, alpha=ALPHA)
    rm, nm, ns = float(np.nanmean(real)), float(nul.mean()), float(nul.std())
    return rm, nm, ns, ((rm-nm)/ns if ns > 0 else float("nan"))

def main():
    member = strict_member()
    resid = {f: resid_1hz(f) for f in FILMS}
    rows = []
    for film, subs in FILMS.items():
        yr1 = resid[film]
        for roi in ROIS:
            loaded = [load_roi(roi, s, film, member) for s in subs]
            Xs, yv, yd, drop = [], [], [], 0
            for X, yh, TR, onset, tr_idx in loaded:
                rt = D.place_on_tr(yr1, onset, tr_idx*TR)
                yhr = D.place_on_tr(load_human(film), onset, tr_idx*TR)
                assert np.corrcoef(yhr, yh)[0,1] > 0.999, f"{roi} {film}: y grid mismatch"
                m = np.isfinite(rt); drop += int((~m).sum())
                Xs.append(X[m]); yv.append(yh[m]); yd.append(rt[m])
            Xs = clean(Xs); TR = loaded[0][2]
            r100 = float(D.positive_control_pooled(Xs, TR, alpha=ALPHA)[100]["snr1.0"])
            v = decode(Xs, yv); r = decode(Xs, yd)
            print(f"\n=== {roi.upper():12s} | {film} | {Xs[0].shape[1]:5d} vox | "
                  f"pc100={r100:+.3f} | dropped {drop} vol ===")
            print(f"  valence        real={v[0]:+.3f}  null={v[1]:+.3f}+/-{v[2]:.3f}  clears={v[3]:+.1f}sd")
            print(f"  valence_resid  real={r[0]:+.3f}  null={r[1]:+.3f}+/-{r[2]:.3f}  clears={r[3]:+.1f}sd")
            rows.append(dict(roi=roi, film=film, nvox=int(Xs[0].shape[1]), pc100=round(r100,3),
                dropped=drop,
                valence=dict(real=round(v[0],3), null=round(v[1],3), sd=round(v[2],3), clears=round(v[3],2)),
                resid=dict(real=round(r[0],3), null=round(r[1],3), sd=round(r[2],3), clears=round(r[3],2))))
    json.dump(rows, open("results/brain_decode_affective_residual.json","w"), indent=2)
    print("\n===== SUMMARY: valence vs valence_resid_visual =====")
    print(f"{'roi':14s}{'film':14s}{'valence real/clr':>20s}{'resid real/clr':>20s}")
    for o in rows:
        print(f"{o['roi']:14s}{o['film']:14s}"
              f"{o['valence']['real']:+8.3f}/{o['valence']['clears']:+4.1f}sd "
              f"{o['resid']['real']:+8.3f}/{o['resid']['clears']:+4.1f}sd")
    print("\nsaved results/brain_decode_affective_residual.json")

if __name__ == "__main__":
    main()
