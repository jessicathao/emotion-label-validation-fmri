#!/usr/bin/env python3
"""decode_target_multi.py - occipital visual-vs-valence decomposition for Emo-FilM.

Reuses the CANONICAL estimator from decode_brain.py (within-subject standardized
occipital matrices, leave-one-subject-out Ridge, planted-signal positive control,
circular-shift null) and runs it on several 1 Hz targets so they are strictly
comparable:

  valence                 the human PleasantOther consensus (the published null)
  <each visual feature>   luminance, rms_contrast, motion, edges, cuts, saturation
                          = the REAL-FEATURE positive control: does occipital carry
                            decodable low-level visual info under the same pipeline?
  valence_resid_visual    valence with the visual-feature set regressed out at 1 Hz
                          (leak-free: residualization touches no BOLD)

Plus a stimulus-side check: how much of the valence timecourse the visual set
predicts (this is what makes the Tears of Steel chance level elevated).

Significance withheld project-wide; descriptive effect sizes only. A target
"decodes" when its real LOSO mean clears its OWN shift-null by more than 1 null sd.
Every target is judged against its own elevated null, never against zero.

USAGE
  python build_visual_regressors.py <film>                    # make features first
  python decode_target_multi.py <film> <sub> <sub> ...        # >=2 subjects
Output: results/visual_decomposition_<film>.json  (mirrors brain_decode_<film>.json)
"""
import os, sys, json
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_brain import (  # canonical pipeline, imported verbatim
    REPO, RESULTS, TR_DEFAULT, ALPHA,
    place_on_tr, load_human, load_cache,
    loso, shift_null_pooled, positive_control_pooled,
)

FEATURES = ["luminance", "rms_contrast", "motion", "edges", "cuts", "saturation"]
VISUAL_DIR = os.path.join(REPO, "data", "visual")


def load_visual(film):
    p = os.path.join(VISUAL_DIR, f"{film}_features_1hz.npz")
    if not os.path.exists(p):
        sys.exit(f"missing {p}\n  run: python build_visual_regressors.py {film}")
    d = np.load(p)
    return {f: d[f].astype(np.float64) for f in FEATURES}


def zscore(a):
    s = np.nanstd(a)
    return (a - np.nanmean(a)) / (s + 1e-12)


def residualize_1hz(y1, V1):
    """Regress the visual set out of valence at 1 Hz. No BOLD involved => leak-free
    w.r.t. the brain decode. Returns (y_resid, R2 of visual set predicting valence)."""
    Vz = (V1 - V1.mean(0)) / (V1.std(0) + 1e-12)
    A = np.column_stack([np.ones(len(y1)), Vz])
    beta, *_ = np.linalg.lstsq(A, y1, rcond=None)
    yhat = A @ beta
    R2 = 1.0 - np.var(y1 - yhat) / (np.var(y1) + 1e-12)
    return y1 - yhat, float(R2)


def recover_tr_times(y_cached, film, onset, TR, kmax=80):
    """Old-format caches store X + the aligned human target y, but not tr_idx/TR.
    The retained volumes are a contiguous block starting at some TR index k0
    (head/tail dropped where place_on_tr returns NaN). Recover k0 by matching the
    cached y against load_human re-aligned on a candidate grid; the right grid
    reproduces y almost exactly. Returns (tr_times, match_r). Self-validating: if
    match_r is not ~1.0 the contiguous assumption is wrong and you must re-extract."""
    human = load_human(film)
    n = len(y_cached); best = (None, -2.0, None)
    for k0 in range(kmax):
        tt = (k0 + np.arange(n)) * TR
        yh = place_on_tr(human, onset, tt)
        m = np.isfinite(yh) & np.isfinite(y_cached)
        if m.sum() < max(10, n // 2) or np.std(yh[m]) < 1e-9 or np.std(y_cached[m]) < 1e-9:
            continue
        r = float(np.corrcoef(yh[m], y_cached[m])[0, 1])
        if r > best[1]:
            best = (k0, r, tt)
    if best[0] is None:
        raise RuntimeError("could not recover TR grid; re-extract with decode_brain.py")
    return best[2], best[1]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    film, subjects = sys.argv[1], sys.argv[2:]
    feat = load_visual(film)

    # --- stimulus-side: residualize valence on the visual set at 1 Hz ---
    y1 = load_human(film).astype(np.float64)
    V1 = np.column_stack([feat[f] for f in FEATURES])
    L = min(len(y1), len(V1))
    y1, V1 = y1[:L], V1[:L]
    y_resid1, R2_visual = residualize_1hz(y1, V1)
    feat_1hz_full = {f: feat[f] for f in FEATURES}  # full length for per-subject alignment
    stim_corr = {f: float(pearsonr(V1[:, k], y1)[0]) for k, f in enumerate(FEATURES)}

    target_names = ["valence"] + FEATURES + ["valence_resid_visual"]

    # --- align every target to each subject's retained volumes (identical to cache) ---
    Xs, T = [], {n: [] for n in target_names}
    TR = TR_DEFAULT
    for s in subjects:
        d = load_cache(s, film); files = set(d.files)
        X = d["X"].astype(np.float64)
        onset = float(d["onset"])
        TR = float(d["TR"]) if "TR" in files else TR_DEFAULT
        ycache = (d["y_human"] if "y_human" in files else d["y"]).astype(np.float64)
        if "tr_idx" in files:
            tr_times = d["tr_idx"].astype(float) * TR
        else:
            tr_times, mr = recover_tr_times(ycache, film, onset, TR)
            tag = "OK" if mr > 0.99 else "CHECK (re-extract if low)"
            print(f"    [{s}] old-format cache: recovered TR grid, match r={mr:.4f}  [{tag}]")
        raw = {"valence": ycache}                                  # cached human target
        for f in FEATURES:
            raw[f] = place_on_tr(feat_1hz_full[f], onset, tr_times)
        raw["valence_resid_visual"] = place_on_tr(y_resid1, onset, tr_times)

        keep = np.ones(len(X), bool)
        for v in raw.values():
            keep &= np.isfinite(v)
        Xs.append(X[keep])
        for n in target_names:
            T[n].append(zscore(raw[n][keep]))
        print(f"  {s}: {int(keep.sum())} vols x {X.shape[1]} vox")

    print(f"pooled: {len(subjects)} subjects, {sum(x.shape[0] for x in Xs)} volumes, TR {TR}")

    # --- positive control once (planted signal, target-agnostic) ---
    print("\n[GATE 1] planted-signal positive control (leave-one-subject-out):")
    pc = positive_control_pooled(Xs, TR)
    for p, row in pc.items():
        print(f"  {p:>4}s  " + "  ".join(f"{k}={v:+.3f}" for k, v in row.items()))

    # --- each target through the identical LOSO + shift-null ---
    print("\n[GATE 2] real decode vs each target's OWN shift-null:")
    print(f"  {'target':22s} {'real':>7} {'null':>7} {'nullsd':>7} {'beats':>7}  verdict")
    res = {}
    for n in target_names:
        real = loso(Xs, T[n], ALPHA)
        nulls = shift_null_pooled(Xs, T[n], alpha=ALPHA)
        rm, nm, nsd = float(np.nanmean(real)), float(nulls.mean()), float(nulls.std())
        beats = rm - nm
        verdict = "decodes" if beats > nsd else "null"
        res[n] = {"loso_r_per_subject": [round(float(x), 3) for x in real],
                  "loso_mean": round(rm, 3), "shift_null_mean": round(nm, 3),
                  "shift_null_sd": round(nsd, 3), "beats_null": round(beats, 3),
                  "verdict": verdict}
        print(f"  {n:22s} {rm:+7.3f} {nm:+7.3f} {nsd:7.3f} {beats:+7.3f}  {verdict}")

    print(f"\nstimulus-side: visual set predicts valence  R2 = {R2_visual:+.3f}")
    print("  per-feature corr with valence (1 Hz): " +
          "  ".join(f"{f}={stim_corr[f]:+.2f}" for f in FEATURES))

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"visual_decomposition_{film}.json")
    json.dump({
        "film": film, "subjects": list(subjects),
        "n_volumes_total": int(sum(x.shape[0] for x in Xs)),
        "alpha": ALPHA, "roi": "posterior occipital (y < -60 mm)",
        "positive_control_pooled": {
            "timescales_s": list(pc.keys()),
            "recovery_r_snr1": [pc[p].get("snr1.0") for p in pc],
            "recovery_r_snr0_3": [pc[p].get("snr0.3") for p in pc],
            "valence_timescale_s": 100},
        "targets": res,
        "stimulus_side": {"visual_set_R2_predicting_valence": round(R2_visual, 3),
                          "feature_corr_with_valence_1hz": {f: round(stim_corr[f], 3) for f in FEATURES}},
        "note": ("Real-feature positive control: occipital should decode visual features "
                 "(beats their own elevated nulls) while valence does not. valence_resid_visual "
                 "tests whether any occipital valence survives removing the visual confound. "
                 "Effect sizes only; each target judged against its own shift-null, never zero."),
    }, open(out, "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
