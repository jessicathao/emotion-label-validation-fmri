#!/usr/bin/env python3
"""fetch one fMRI run at a time, cache its occipital matrix, drop the bold.
USAGE: python fetch_extract_cache.py <film> <sub> [sub ...]"""
import os, sys, glob, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_brain as D

def sh(cmd):
    print("  $", cmd)
    return subprocess.run(cmd, shell=True, cwd=D.DS_FMRI)

def free_space():
    return subprocess.run("df -h ~ | awk 'NR==2{print $4}'", shell=True,
                          capture_output=True, text=True).stdout.strip()

def process(film, subjects, drop=True):
    os.makedirs(D.CACHE, exist_ok=True)
    for sub in subjects:
        if os.path.exists(os.path.join(D.CACHE, f"{sub}_{film}.npz")):
            print(f"[{sub}] cache exists, skipping"); continue
        bold = glob.glob(os.path.join(D.DS_FMRI, "derivatives", "preprocessing", sub,
               "ses-*", "func", f"{sub}_ses-*_task-{film}_space-MNI_desc-ppres_bold.nii.gz"))
        ev = glob.glob(os.path.join(D.DS_FMRI, sub, "ses-*", "func",
             f"{sub}_ses-*_task-scan_acq-{film}_events.tsv"))
        if not bold or not ev:
            print(f"[{sub}] run not listed; skipping"); continue
        print(f"\n[{sub}] fetch")
        sh(f"datalad get {os.path.relpath(ev[0], D.DS_FMRI)}")
        sh(f"datalad get {os.path.relpath(bold[0], D.DS_FMRI)}")
        D.extract_and_cache(sub, film)
        if drop:
            print(f"[{sub}] drop bold")
            sh(f"datalad drop {os.path.relpath(bold[0], D.DS_FMRI)}")
        print(f"[{sub}] done. free space: {free_space()}")
    print(f"\nAll requested subjects cached in {D.CACHE}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    process(sys.argv[1], sys.argv[2:])
