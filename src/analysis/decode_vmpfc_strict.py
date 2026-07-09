#!/usr/bin/env python
"""decode_vmpfc_strict.py - strict-vmPFC (drop Frontal Orbital) rerun, BOLD-free.
Subsets the existing full-vmPFC caches by column. standardize+detrend are per-voxel, so
column-subsetting is EXACTLY equivalent to re-masking with the tighter mask. Reruns the
identical loso / positive_control_pooled / shift_null_pooled from decode_brain.py on both
the full (FM+SC+FO, 5222) and strict (FM+SC, 2102) masks, same code path, same seed.
Answers: is the Payload vmPFC decode carried by strict ventromedial cortex, or by orbital
voxels (then it is an OFC result)? Re-measures the soft 2.2 sd clearance on the tight mask."""
import os, sys, glob, json, numpy as np, nibabel as nib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_brain as D
from nilearn.datasets import load_mni152_gm_mask, fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img

DS = os.path.expanduser("~/ds004892"); ALPHA = D.ALPHA
RUNS = {"Payload":      ["sub-S08","sub-S05","sub-S11","sub-S15","sub-S06"],
        "TearsOfSteel": ["sub-S31","sub-S08","sub-S11","sub-S25","sub-S04"]}

def find_bold(sub, film):
    h = glob.glob(os.path.join(DS,"derivatives","preprocessing",sub,"ses-*","func",
                  f"{sub}_ses-*_task-{film}_space-MNI_desc-ppres_bold.nii.gz"))
    return h[0] if h else None

def vmpfc_masks(ref):
    gm = np.asarray(resample_to_img(load_mni152_gm_mask(resolution=2), ref,
                    interpolation="nearest").get_fdata()).round().astype(bool)
    cort = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    data = np.asarray(resample_to_img(cort.maps, ref, interpolation="nearest").get_fdata()).round().astype(int)
    names = list(cort.labels); bg = names and names[0].strip().lower() in ("background","")
    def lab(name):
        idx = [(i if bg else i+1) for i,n in enumerate(names) if name.lower() in n.lower()]
        return np.isin(data, idx) & gm
    fm, sc, fo = lab("Frontal Medial Cortex"), lab("Subcallosal Cortex"), lab("Frontal Orbital Cortex")
    return (fm|sc|fo), (fm|sc)

def summarize(tag, Xs, ys, TR):
    pc = D.positive_control_pooled(Xs, TR, alpha=ALPHA); r100 = float(pc[100]["snr1.0"])
    real = D.loso(Xs, ys, alpha=ALPHA); nulls = D.shift_null_pooled(Xs, ys, alpha=ALPHA)
    rm, nm, ns = float(np.nanmean(real)), float(nulls.mean()), float(nulls.std())
    clr = (rm-nm)/ns if ns > 0 else float("nan")
    print(f"  {tag:16s} nvox={Xs[0].shape[1]:5d}  pc100={r100:+.3f}  "
          f"real={rm:+.3f}  null={nm:+.3f}+/-{ns:.3f}  clears={clr:+.1f}sd")
    return dict(nvox=Xs[0].shape[1], pc100=round(r100,3),
                real_per_subject=[round(float(x),3) for x in real],
                real_mean=round(rm,3), null_mean=round(nm,3), null_sd=round(ns,3),
                clears_sd=round(clr,2))

def main():
    ref = nib.load(find_bold("sub-S08","Payload"))
    full_m, strict_m = vmpfc_masks(ref)
    member = strict_m.ravel()[np.flatnonzero(full_m.ravel())]        # bool over full columns
    print(f"full vmPFC/OFC {int(full_m.sum())}  strict vmPFC {int(strict_m.sum())}  "
          f"(strict = {int(member.sum())} of {member.size} full columns)")
    assert int(strict_m.sum()) == int(member.sum()), "membership mismatch"
    for film, subs in RUNS.items():
        zs = [np.load(f"data/decode_cache_vmpfc/{s}_{film}.npz", allow_pickle=True) for s in subs]
        for z in zs:
            assert int(z["nvox"]) == int(full_m.sum()) == zs[0]["X"].shape[1], "cache/mask drift"
        ys = [z["y_human"].astype(float) for z in zs]; TR = float(zs[0]["TR"])
        Xf = [z["X"].astype(float) for z in zs]; Xs = [X[:, member] for X in Xf]
        print(f"\n=== {film} ===")
        out = {"full": summarize("full vmPFC/OFC", Xf, ys, TR),
               "strict": summarize("strict vmPFC", Xs, ys, TR)}
        json.dump(out, open(f"results/brain_decode_affective_vmpfc_strict_{film}.json","w"), indent=2)
    print("\nSaved results/brain_decode_affective_vmpfc_strict_{Payload,TearsOfSteel}.json")

if __name__ == "__main__":
    main()
