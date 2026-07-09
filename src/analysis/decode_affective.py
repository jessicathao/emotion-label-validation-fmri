#!/usr/bin/env python
"""decode_affective.py - affective-ROI decode for Emo-FilM.
Mirrors decode_brain.py EXACTLY: imports its loso, positive_control_pooled,
shift_null_pooled, ALPHA. Only the cache directory (the mask) changes. Same target
(human PleasantOther), same 5 lowest-motion subjects per film, same two films.
Primary test = the a priori affective NETWORK (insula u vmPFC u amygdala); insula,
vmPFC, amygdala are reported separately as secondary localization.

Pre-registered before any real decode:
  * voxel-drop: drop any voxel non-finite OR dead (std<1e-6) in ANY pooled subject,
    identically across subjects so the LOSO feature columns stay in correspondence.
    (Extraction showed all voxels finite, so this is a no-op safeguard here.)
  * verdict (descriptive, no p-values): a target DECODES iff its real LOSO mean clears
    its OWN circular-shift null by more than 1 null sd AND the planted-signal positive
    control recovers at the ~100s valence timescale. A ROI whose positive control does
    not recover is 'uninterpretable', not 'null'. Each target judged vs its OWN null.
The pos-control PASS flag (>=0.10 at ~100s, SNR1.0) is a convenience marker; the real
read is the full timescale grid, judged qualitatively as occipital was (strong+uniform).
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_brain as D

ALPHA = D.ALPHA
ROIS  = ["network", "insula", "vmpfc", "amygdala"]        # network first = primary
RUNS  = {"Payload":      ["sub-S08","sub-S05","sub-S11","sub-S15","sub-S06"],
         "TearsOfSteel": ["sub-S31","sub-S08","sub-S11","sub-S25","sub-S04"]}
ALPHA_GRID = [1000.0, 10000.0, 100000.0]                  # stability check for small ROIs
POS_TS = 100                                              # valence timescale (headline)

def load_roi(roi, sub, film):
    z = np.load(f"data/decode_cache_{roi}/{sub}_{film}.npz", allow_pickle=True)
    return z["X"].astype(float), z["y_human"].astype(float), float(z["TR"])

def clean_columns(Xs):
    good = np.ones(Xs[0].shape[1], bool)
    for X in Xs:
        good &= np.isfinite(X).all(0) & (np.nanstd(X, 0) > 1e-6)
    return [X[:, good] for X in Xs], int((~good).sum()), int(good.sum())

def run(roi, film):
    subs = RUNS[film]
    loaded = [load_roi(roi, s, film) for s in subs]
    Xs = [x for x, _, _ in loaded]; ys = [y for _, y, _ in loaded]; TR = loaded[0][2]
    Xs, dropped, kept = clean_columns(Xs)
    n_tot = sum(x.shape[0] for x in Xs)

    pc    = D.positive_control_pooled(Xs, TR, alpha=ALPHA)          # GATE 1
    r100  = float(pc[POS_TS]["snr1.0"])
    pos_ok = r100 >= 0.10

    real  = D.loso(Xs, ys, alpha=ALPHA)                            # GATE 2
    nulls = D.shift_null_pooled(Xs, ys, alpha=ALPHA)
    real_m, null_m, null_sd = float(np.nanmean(real)), float(nulls.mean()), float(nulls.std())
    clears = (real_m - null_m) > null_sd
    verdict = "DECODES" if (pos_ok and clears) else ("null" if pos_ok else
              "uninterpretable (positive control weak)")

    alpha_stab = {f"alpha_{int(a)}":
                  (round(real_m, 3) if a == ALPHA
                   else round(float(np.nanmean(D.loso(Xs, ys, alpha=a))), 3))
                  for a in ALPHA_GRID}

    print(f"\n=== {roi.upper():9s} | {film} | {len(subs)} subj | "
          f"{kept} vox (dropped {dropped}) | {n_tot} vols ===")
    print("  [GATE 1] positive control (planted, LOSO):")
    for p, row in pc.items():
        print(f"    {p:>4}s  " + "  ".join(f"{k}={v:+.3f}" for k, v in row.items()))
    print(f"  headline ~{POS_TS}s SNR1.0 = {r100:+.3f}  ({'PASS' if pos_ok else 'WEAK'})")
    print("  [GATE 2] real valence (LOSO + shift null):")
    print(f"    per-subject r = {[round(x,3) for x in real]}")
    print(f"    real mean   = {real_m:+.3f}")
    print(f"    shift-null  = {null_m:+.3f}  sd {null_sd:.3f}")
    print(f"    alpha sweep = {alpha_stab}")
    print(f"  VERDICT: {verdict}   (real {real_m:+.3f} vs null {null_m:+.3f} +/- {null_sd:.3f})")

    out = {"roi": roi, "film": film, "subjects": subs, "alpha": ALPHA,
           "nvox_kept": kept, "nvox_dropped": dropped, "n_volumes_total": n_tot,
           "positive_control": {str(p): row for p, row in pc.items()},
           "positive_control_valence_timescale_snr1": round(r100, 3),
           "positive_control_pass": bool(pos_ok),
           "real_valence_pooled": {
               "loso_r_per_subject": [round(float(x), 3) for x in real],
               "loso_mean": round(real_m, 3), "shift_null_mean": round(null_m, 3),
               "shift_null_sd": round(null_sd, 3), "clears_null_by_1sd": bool(clears)},
           "alpha_stability_real_mean": alpha_stab, "verdict": verdict}
    os.makedirs("results", exist_ok=True)
    path = f"results/brain_decode_affective_{roi}_{film}.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"  saved {path}")
    return out

def main():
    which = sys.argv[1:] or ROIS
    rows = [run(roi, film) for roi in which for film in RUNS]
    print("\n================ SUMMARY (real vs its OWN shift-null) ================")
    print(f"{'ROI':10s} {'film':13s} {'pc~100':>7s} {'real':>7s} {'null':>7s} {'sd':>6s}  verdict")
    for o in rows:
        rv = o["real_valence_pooled"]
        print(f"{o['roi']:10s} {o['film']:13s} "
              f"{o['positive_control_valence_timescale_snr1']:+7.3f} "
              f"{rv['loso_mean']:+7.3f} {rv['shift_null_mean']:+7.3f} "
              f"{rv['shift_null_sd']:6.3f}  {o['verdict']}")

if __name__ == "__main__":
    main()
