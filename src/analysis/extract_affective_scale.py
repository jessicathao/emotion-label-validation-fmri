#!/usr/bin/env python
"""extract_affective_scale.py - subject-scaling extraction for the affective-ROI decode (Payload).
Selects Payload runs from mriqc group_bold.tsv by fd_mean (ascending, NESTED with the current
five) up to MOTION_CAP. Same four HO masks, same NiftiMasker kwargs, same target. Onset is read
per subject from the SCAN events file find_run uses: {sub}_{ses}_task-scan_acq-{film}_events.tsv
in the raw func/ dir (tiny; datalad-get it if dropped). For a subject without an occipital cache
it BOOTSTRAPS one from the occipital X it computes for validation (free occipital-at-n control);
existing runs are skipped. One BOLD fetch per new run, all masks applied, drop.
Run DRYRUN=1 first to resolve events + onset for every selected run WITHOUT fetching any BOLD.
MOTION_CAP is the pre-registered fd_mean ceiling, set BEFORE any decode."""
import os, sys, glob, subprocess, numpy as np, nibabel as nib, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_brain as D
from nilearn.datasets import load_mni152_gm_mask, fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiMasker

FILM = "Payload"
MOTION_CAP = float(os.environ.get("MOTION_CAP", "0.25"))   # <-- pre-registered ceiling
DRYRUN = os.environ.get("DRYRUN", "0") == "1"
DS  = os.path.expanduser("~/ds004892")
OCC = "data/decode_cache"
GROUP = os.path.join(DS, "derivatives", "mriqc", "group_bold.tsv")
ROIS = ["insula", "vmpfc", "amygdala", "network"]
TR_DEFAULT = getattr(D, "TR_DEFAULT", 1.3)

def bold_path(sub, ses):
    return os.path.join(DS, "derivatives", "preprocessing", sub, ses, "func",
                        f"{sub}_{ses}_task-{FILM}_space-MNI_desc-ppres_bold.nii.gz")
def events_path(sub, ses):
    return os.path.join(DS, sub, ses, "func", f"{sub}_{ses}_task-scan_acq-{FILM}_events.tsv")
def present(p, mn=1_000_000):
    try: return os.path.exists(p) and os.path.getsize(p) > mn
    except OSError: return False
def corr(a, b): return float(np.corrcoef(np.asarray(a,float).ravel(), np.asarray(b,float).ravel())[0,1])

def ensure_events(sub, ses):
    ep = events_path(sub, ses)
    if not os.path.exists(ep):
        subprocess.run(["datalad", "get", ep], cwd=DS, check=False)
    return ep if os.path.exists(ep) else None

def select():
    df = pd.read_csv(GROUP, sep="\t")
    idc = next((c for c in df.columns if c.lower() in ("bids_name","name","run_id")), df.columns[0])
    fdc = next((c for c in df.columns if c.lower()=="fd_mean"),
               next((c for c in df.columns if "fd_mean" in c.lower()), None))
    pay = df[df[idc].astype(str).str.contains(f"task-{FILM}", case=False, na=False)].copy().sort_values(fdc)
    pay = pay[pay[fdc] < MOTION_CAP]
    out = []
    for _, r in pay.iterrows():
        parts = str(r[idc]).split("_"); out.append((parts[0], parts[1], float(r[fdc])))
    return out

def build_masks(ref):
    gm = np.asarray(resample_to_img(load_mni152_gm_mask(resolution=2), ref,
                    interpolation="nearest").get_fdata()).round().astype(bool)
    cort = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    sub  = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    def sel(atlas, wanted):
        data = np.asarray(resample_to_img(atlas.maps, ref, interpolation="nearest").get_fdata()).round().astype(int)
        names = list(atlas.labels); bg = names and names[0].strip().lower() in ("background","")
        m = np.zeros(ref.shape[:3], bool)
        for i,n in enumerate(names):
            if any(w.lower() in n.lower() for w in wanted): m |= (data == (i if bg else i+1))
        return m & gm
    ins = sel(cort, ["Insular Cortex"])
    vmp = sel(cort, ["Frontal Medial Cortex","Subcallosal Cortex","Frontal Orbital Cortex"])
    amy = sel(sub,  ["Amygdala"])
    return {"insula":ins, "vmpfc":vmp, "amygdala":amy, "network":(ins|vmp|amy)}

def main():
    subs = select()
    print(f"MOTION_CAP={MOTION_CAP}  ->  {len(subs)} Payload runs (nested, fd_mean ascending)"
          + ("   [DRYRUN: events + onset only]" if DRYRUN else ""))
    if DRYRUN:
        ok = 0
        for sub, ses, fd in subs:
            cached = all(os.path.exists(f"data/decode_cache_{k}/{sub}_{FILM}.npz") for k in ROIS)
            ep = ensure_events(sub, ses)
            if ep is None:
                print(f"  {sub} {ses}  fd={fd:.3f}  !! NO scan events at {os.path.relpath(ep or events_path(sub,ses), DS)}")
                continue
            onset, dur = D.film_onset(ep); ok += 1
            print(f"  {sub} {ses}  fd={fd:.3f}  onset={onset:.3f} dur={dur:.3f}"
                  + ("   (affective cached)" if cached else "   NEW"))
        print(f"\n{ok}/{len(subs)} runs resolve an onset. If all resolve, rerun WITHOUT DRYRUN to extract.")
        return

    ref = nib.load(bold_path("sub-S08", "ses-4"))
    if not present(bold_path("sub-S08","ses-4")): sys.exit("reference BOLD sub-S08 ses-4 not resident.")
    masks = build_masks(ref); aff = ref.affine
    for k in ROIS: os.makedirs(f"data/decode_cache_{k}", exist_ok=True)
    os.makedirs(OCC, exist_ok=True)
    n_new = 0
    for sub, ses, fd in subs:
        if all(os.path.exists(f"data/decode_cache_{k}/{sub}_{FILM}.npz") for k in ROIS):
            print(f"skip  {sub} (affective cached)"); continue
        ep = ensure_events(sub, ses)
        if ep is None: print(f"  !! no scan events {sub} {ses}; skip"); continue
        onset, dur = D.film_onset(ep)
        bp = bold_path(sub, ses); fetched = False
        if not present(bp):
            print(f"  datalad get {os.path.relpath(bp, DS)} ...")
            subprocess.run(["datalad","get",bp], cwd=DS, check=True); fetched = True
        img = nib.load(bp)
        TR = float(img.header.get_zooms()[3]) or TR_DEFAULT
        tr_times = np.arange(img.shape[3]) * TR
        y_h = D.place_on_tr(D.load_human(FILM), onset, tr_times)
        fm = ~np.isnan(y_h); tr_idx = np.where(fm)[0].astype(np.int32)
        try: y_g = D.place_on_tr(D.load_gemini_context(FILM), onset, tr_times)[fm].astype(np.float32)
        except Exception: y_g = np.full(int(fm.sum()), np.nan, np.float32)
        occX = D.occ_masker(img)[0].fit_transform(img)[fm].astype(np.float32)
        occ_path = f"{OCC}/{sub}_{FILM}.npz"
        if os.path.exists(occ_path):
            rx = corr(occX, np.load(occ_path, allow_pickle=True)["X"])
            print(f"  {sub}: vols={int(fm.sum())} fd={fd:.3f} r_x={rx:.4f} {'OK' if rx>0.999 else 'MISMATCH -> skip'}")
            if rx <= 0.999:
                if fetched: subprocess.run(["datalad","drop",bp], cwd=DS, check=False)
                continue
        else:
            np.savez_compressed(occ_path, X=occX, y_human=y_h[fm].astype(np.float32), y_gemini=y_g,
                nvox=occX.shape[1], onset=onset, dur=dur, TR=TR, tr_idx=tr_idx)
            print(f"  {sub}: vols={int(fm.sum())} fd={fd:.3f} occ bootstrapped ({occX.shape[1]} vox)")
        diag = []
        for k in ROIS:
            msk = NiftiMasker(mask_img=nib.Nifti1Image(masks[k].astype(np.int8), aff),
                              standardize=True, detrend=True)
            X = msk.fit_transform(img)[fm].astype(np.float32)
            nan = int(np.isnan(X).sum()); dead = int((np.nanstd(X,0) < 1e-6).sum())
            if nan or dead: diag.append(f"{k}[nan={nan},dead={dead}/{X.shape[1]}]")
            np.savez_compressed(f"data/decode_cache_{k}/{sub}_{FILM}.npz",
                X=X, y_human=y_h[fm].astype(np.float32), y_gemini=y_g, nvox=int(masks[k].sum()),
                onset=onset, dur=dur, TR=TR, tr_idx=tr_idx)
        print("    cached all ROIs" + ("  WARN " + " ".join(diag) if diag else "  (all voxels finite)"))
        n_new += 1
        if fetched: subprocess.run(["datalad","drop",bp], cwd=DS, check=False)
    print(f"\nDone. {n_new} new runs extracted; affective caches now cover fd_mean < {MOTION_CAP}.")

if __name__ == "__main__":
    main()
