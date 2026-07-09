#!/usr/bin/env python
"""decode_affective_scaling.py - subject-scaling decode for the affective-ROI arm (Payload).
Reruns the IDENTICAL estimator (decode_brain.loso / positive_control_pooled / shift_null_pooled,
ALPHA) at n = 5,10,15,20,24 in strict fd_mean-ascending order (nested), for the network (primary)
and insula / vmPFC(strict) / amygdala. Pre-registered before this run:
  * voxel-drop: at each n, drop any voxel non-finite OR dead (std<1e-6) in ANY pooled subject,
    uniformly across subjects so LOSO columns stay in correspondence (handles S20's 16 dead vox
    at n>=12; a no-op at n=5,10 so the n=5 row reproduces the canonical decode_affective numbers).
  * read rule: HOLDS if at n=24 the network clears its own shift-null by >1 sd with a passing
    positive control AND the null has not drifted materially above the n=5 null; STRENGTHENS if
    clearance rises with n; FAILS if the real trends to the null or the null rises to meet it.
Also traces the null center and per-subject positive fraction as n grows, and runs the
subsample-to-five fluke test: from the full 24, draw random 5-subject sets, place the original
five's real mean in that distribution (was n=5 a lucky draw?).
No BOLD, no datalad; reads the cached matrices. Long run (~45-60 min, network pos-control at
large n dominates); prints progress. Env: FLUKE_K (default 200), FLUKE=0 to skip.
"""
import os, sys, json, numpy as np, nibabel as nib, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_brain as D
from decode_vmpfc_strict import vmpfc_masks, find_bold

ALPHA = D.ALPHA
FILM  = "Payload"
NS    = [5, 10, 15, 20, 24]
ROIS  = ["network", "insula", "vmpfc", "amygdala"]      # vmpfc = strict (2102)
FLUKE_ROIS = ["network", "insula", "amygdala"]
FLUKE_K = int(os.environ.get("FLUKE_K", "200"))
DO_FLUKE = os.environ.get("FLUKE", "1") == "1"
DS = os.path.expanduser("~/ds004892")
GROUP = os.path.join(DS, "derivatives", "mriqc", "group_bold.tsv")
CAP = 0.25

def order_subjects():
    df = pd.read_csv(GROUP, sep="\t")
    idc = next((c for c in df.columns if c.lower() in ("bids_name","name","run_id")), df.columns[0])
    fdc = next((c for c in df.columns if c.lower()=="fd_mean"),
               next((c for c in df.columns if "fd_mean" in c.lower()), None))
    pay = df[df[idc].astype(str).str.contains(f"task-{FILM}", case=False, na=False)].copy().sort_values(fdc)
    pay = pay[pay[fdc] < CAP]
    subs, fds = [], []
    for _, r in pay.iterrows():
        s = str(r[idc]).split("_")[0]
        if all(os.path.exists(f"data/decode_cache_{k}/{s}_{FILM}.npz") for k in ROIS):
            subs.append(s); fds.append(float(r[fdc]))
    return subs, fds

# strict-vmPFC column membership over the full (5222) vmPFC columns
_full, _strict = vmpfc_masks(nib.load(find_bold("sub-S08", FILM)))
MEMBER = _strict.ravel()[np.flatnonzero(_full.ravel())]

def load_roi(roi, sub):
    src = "vmpfc" if roi == "vmpfc" else roi
    z = np.load(f"data/decode_cache_{src}/{sub}_{FILM}.npz", allow_pickle=True)
    X = z["X"].astype(float)
    if roi == "vmpfc": X = X[:, MEMBER]
    return X, z["y_human"].astype(float), float(z["TR"])

def clean_pool(Xs):
    good = np.ones(Xs[0].shape[1], bool)
    for X in Xs:
        good &= np.isfinite(X).all(0) & (np.nanstd(X, 0) > 1e-6)
    return [X[:, good] for X in Xs], int(good.sum()), int((~good).sum())

def cell(roi, subs):
    loaded = [load_roi(roi, s) for s in subs]
    Xs = [x for x, _, _ in loaded]; ys = [y for _, y, _ in loaded]; TR = loaded[0][2]
    Xs, kept, dropped = clean_pool(Xs)
    pc = D.positive_control_pooled(Xs, TR, alpha=ALPHA)
    real = D.loso(Xs, ys, alpha=ALPHA)
    nulls = D.shift_null_pooled(Xs, ys, alpha=ALPHA)
    rm, nm, sd = float(np.nanmean(real)), float(nulls.mean()), float(nulls.std())
    return dict(n=len(subs), nvox=kept, dropped=dropped,
                pc100=round(float(pc[100]["snr1.0"]), 3),
                real=round(rm, 3), null=round(nm, 3), null_sd=round(sd, 3),
                clears=round((rm - nm) / sd, 2) if sd > 0 else None,
                pos_frac=round(float(np.mean(np.asarray(real) > 0)), 2),
                per_subject=[round(float(x), 3) for x in real])

def fluke(roi, subs_full, orig_real):
    arrs = [load_roi(roi, s) for s in subs_full]
    Xall = [a[0] for a in arrs]; yall = [a[1] for a in arrs]; n = len(subs_full)
    rng = np.random.default_rng(0); means = []
    for k in range(FLUKE_K):
        idx = rng.choice(n, 5, replace=False)
        Xs, _, _ = clean_pool([Xall[i] for i in idx])
        means.append(float(np.nanmean(D.loso(Xs, [yall[i] for i in idx], alpha=ALPHA))))
        if (k + 1) % 50 == 0: print(f"      fluke {roi} {k+1}/{FLUKE_K}")
    m = np.array(means)
    return dict(k=FLUKE_K, orig_five_real=round(orig_real, 3),
                median=round(float(np.median(m)), 3), mean=round(float(m.mean()), 3),
                p10=round(float(np.percentile(m, 10)), 3), p90=round(float(np.percentile(m, 90)), 3),
                orig_percentile=round(float(100 * np.mean(m <= orig_real)), 1),
                frac_subsets_positive=round(float(np.mean(m > 0)), 2))

def main():
    subs, fds = order_subjects()
    print(f"{len(subs)} nested subjects (fd_mean {fds[0]:.3f} to {fds[-1]:.3f}); scaling at n={NS}\n")
    res = {r: {} for r in ROIS}
    for roi in ROIS:
        for N in NS:
            print(f"  [{roi} n={N}] ...", flush=True)
            res[roi][N] = cell(roi, subs[:N])
            c = res[roi][N]
            print(f"    nvox={c['nvox']} (drop {c['dropped']})  pc100={c['pc100']:+.3f}  "
                  f"real={c['real']:+.3f}  null={c['null']:+.3f}+/-{c['null_sd']:.3f}  "
                  f"clears={c['clears']:+.1f}sd  pos={c['pos_frac']:.2f}")
    flk = {}
    if DO_FLUKE:
        print("\n  subsample-to-five fluke test:")
        for roi in FLUKE_ROIS:
            flk[roi] = fluke(roi, subs, res[roi][5]["real"])
            f = flk[roi]
            print(f"    {roi:9s} orig5={f['orig_five_real']:+.3f}  median5={f['median']:+.3f}  "
                  f"[p10 {f['p10']:+.3f}, p90 {f['p90']:+.3f}]  orig at {f['orig_percentile']:.0f}th pct  "
                  f"frac>0={f['frac_subsets_positive']:.2f}")

    json.dump({"subjects": subs, "fd_mean": fds, "ns": NS, "scaling": res, "fluke": flk},
              open("results/brain_decode_affective_scaling.json", "w"), indent=2)

    print("\n===== SCALING CURVE (network = primary) =====")
    print(f"{'roi':9s}" + "".join(f"{'n='+str(N):>14s}" for N in NS))
    for roi in ROIS:
        row = "".join(f"{res[roi][N]['real']:+.3f}/{res[roi][N]['clears']:+.1f}sd".rjust(14) for N in NS)
        print(f"{roi:9s}{row}")
    print("(cell = real / clearance-in-sd)")

    net5, net24 = res["network"][5], res["network"][24]
    null_drift = net24["null"] - net5["null"]
    holds = (net24["clears"] is not None and net24["clears"] > 1.0
             and net24["pc100"] >= 0.10 and null_drift <= net5["null_sd"])
    print(f"\nREAD (network, pre-registered rule): n5 clears {net5['clears']:+.1f}sd -> "
          f"n24 clears {net24['clears']:+.1f}sd; null drift {null_drift:+.3f} "
          f"(vs n5 sd {net5['null_sd']:.3f}); pc100 n24={net24['pc100']:+.3f}")
    print("  tentative: " + ("HOLDS" if holds else "does NOT meet HOLDS as literally stated; read the curve")
          + f"; clearance trend {net5['clears']:+.1f} -> {net24['clears']:+.1f}sd")
    print("\nsaved results/brain_decode_affective_scaling.json")

if __name__ == "__main__":
    main()
