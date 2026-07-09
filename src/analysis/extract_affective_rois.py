#!/usr/bin/env python
"""extract_affective_rois.py - affective-ROI arm for Emo-FilM.
Mirrors decode_brain.occ_masker (same grid, same NiftiMasker kwargs); only the MASK changes.
FORMAT-INDEPENDENT: tr_idx and the human target are RECOMPUTED via decode_brain.place_on_tr
+ load_human from the onset stored in every cache, never read from a stored tr_idx. Each run
is validated by reproducing the cached occipital X and target to r=1.0. Per run: fetch the
BOLD one at a time, apply insula / vmPFC / amygdala / their union (the a priori NETWORK,
primary test), cache one matrix per ROI, drop only BOLDs we fetched. Standardize+detrend over
the full run THEN select the film volumes, exactly as occipital did. No decode here."""
import os, sys, glob, subprocess, numpy as np, nibabel as nib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_brain as D
from nilearn.datasets import load_mni152_gm_mask, fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiMasker

DS  = os.path.expanduser("~/ds004892")
OCC = "data/decode_cache"
TR_DEFAULT = getattr(D, "TR_DEFAULT", 1.3)
ROIS = ["insula","vmpfc","amygdala","network"]
RUNS = {"Payload":      ["sub-S08","sub-S05","sub-S11","sub-S15","sub-S06"],
        "TearsOfSteel": ["sub-S31","sub-S08","sub-S11","sub-S25","sub-S04"]}

def find_bold(sub, film):
    h = glob.glob(os.path.join(DS,"derivatives","preprocessing",sub,"ses-*","func",
                  f"{sub}_ses-*_task-{film}_space-MNI_desc-ppres_bold.nii.gz"))
    return h[0] if h else None

def present(p):
    try: return os.path.exists(p) and os.path.getsize(p) > 1_000_000
    except OSError: return False

def corr(a, b):
    return float(np.corrcoef(np.asarray(a,float).ravel(), np.asarray(b,float).ravel())[0,1])

def build_masks(ref):
    gm = np.asarray(resample_to_img(load_mni152_gm_mask(resolution=2), ref,
                    interpolation="nearest").get_fdata()).round().astype(bool)
    cort = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    sub  = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    def sel(atlas, wanted):
        data = np.asarray(resample_to_img(atlas.maps, ref,
               interpolation="nearest").get_fdata()).round().astype(int)
        names = list(atlas.labels); bg = names and names[0].strip().lower() in ("background","")
        m = np.zeros(ref.shape[:3], bool)
        for i,n in enumerate(names):
            if any(w.lower() in n.lower() for w in wanted):
                m |= (data == (i if bg else i+1))
        return m & gm
    ins = sel(cort, ["Insular Cortex"])
    vmp = sel(cort, ["Frontal Medial Cortex","Subcallosal Cortex","Frontal Orbital Cortex"])
    amy = sel(sub,  ["Amygdala"])
    return {"insula":ins, "vmpfc":vmp, "amygdala":amy, "network":(ins|vmp|amy)}, ref.affine

def main():
    ref_path = find_bold("sub-S08","Payload")
    if not (ref_path and present(ref_path)):
        sys.exit("reference BOLD sub-S08 Payload not resident; cannot define grid.")
    ref = nib.load(ref_path)
    masks, aff = build_masks(ref)
    for k in ROIS:
        os.makedirs(f"data/decode_cache_{k}", exist_ok=True)
        print(f"mask {k:10s} {int(masks[k].sum()):6d} voxels")
    print()

    for film, subs in RUNS.items():
        for sub in subs:
            outs = {k: f"data/decode_cache_{k}/{sub}_{film}.npz" for k in ROIS}
            if all(os.path.exists(p) for p in outs.values()):
                print(f"skip  {sub} {film} (cached)"); continue
            occ = f"{OCC}/{sub}_{film}.npz"
            if not os.path.exists(occ):
                print(f"  !! missing occipital cache {occ}; skip"); continue
            z = np.load(occ, allow_pickle=True)
            onset = float(z["onset"]); y_cache = z["y_human"] if "y_human" in z.files else z["y"]

            path = find_bold(sub, film)
            if path is None: print(f"  !! no BOLD path for {sub} {film}; skip"); continue
            fetched = False
            if not present(path):
                print(f"  datalad get {os.path.relpath(path, DS)} ...")
                subprocess.run(["datalad","get",path], cwd=DS, check=True); fetched = True

            img = nib.load(path)
            TR = float(img.header.get_zooms()[3]) or TR_DEFAULT
            tr_times = np.arange(img.shape[3]) * TR
            y_h = D.place_on_tr(D.load_human(film), onset, tr_times)
            fm = ~np.isnan(y_h); tr_idx = np.where(fm)[0].astype(np.int32)

            if int(fm.sum()) != len(y_cache):
                print(f"  !! {sub} {film} vol mismatch fm={int(fm.sum())} cache={len(y_cache)}; skip")
                if fetched: subprocess.run(["datalad","drop",path], cwd=DS, check=False)
                continue
            r_y = corr(y_h[fm], y_cache)
            r_x = corr(D.occ_masker(img)[0].fit_transform(img)[fm], z["X"])
            ok = (r_y > 0.999 and r_x > 0.999)
            print(f"  {sub} {film}: vols={int(fm.sum())} r_y={r_y:.4f} r_x={r_x:.4f} "
                  f"{'OK' if ok else 'MISMATCH -> skip'}")
            if not ok:
                if fetched: subprocess.run(["datalad","drop",path], cwd=DS, check=False)
                continue

            try:
                y_g = D.place_on_tr(D.load_gemini_context(film), onset, tr_times)[fm].astype(np.float32)
            except Exception:
                y_g = np.full(int(fm.sum()), np.nan, np.float32)

            diag = []
            for k in ROIS:
                msk = NiftiMasker(mask_img=nib.Nifti1Image(masks[k].astype(np.int8), aff),
                                  standardize=True, detrend=True)
                X = msk.fit_transform(img)[fm].astype(np.float32)
                nan = int(np.isnan(X).sum()); dead = int((np.nanstd(X,0) < 1e-6).sum())
                if nan or dead: diag.append(f"{k}[nan={nan},dead={dead}/{X.shape[1]}]")
                np.savez_compressed(outs[k], X=X, y_human=y_h[fm].astype(np.float32), y_gemini=y_g,
                    nvox=int(masks[k].sum()), onset=onset,
                    dur=float(z["dur"]) if "dur" in z.files else np.nan, TR=TR, tr_idx=tr_idx)
            print("    cached all ROIs" + ("  WARN " + " ".join(diag) if diag else "  (all voxels finite)"))
            if fetched: subprocess.run(["datalad","drop",path], cwd=DS, check=False)

    print("\nDone. Matrices in data/decode_cache_{insula,vmpfc,amygdala,network}/")

if __name__ == "__main__":
    main()
