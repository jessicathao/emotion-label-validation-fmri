#!/usr/bin/env python3
"""spaceman_decontam_all.py

Reproduce spaceman_lyric_check.py's EXACT contamination cut, but apply it to
every model signal (BERT, Gemini, DistilBERT, SiEBERT) so we can see whether
Gemini's apparent Spaceman result survives lyric/credit removal the same way
BERT's did (BERT: -0.292 -> -0.107).

Method matches spaceman_lyric_check.py precisely:
  - same PleasantOther column (idx 3), same +2s offset
  - same CONTAM second-ranges masked out of the dialogue mask
  - Pearson on finite, in-mask, offset-aligned pairs
No new API calls: reuses the existing *_gemini.json / *_bert.json signals.
"""
import json, gzip, os
import numpy as np

ANNOT = os.path.expanduser("~/ds004872/derivatives/Annot_Spaceman_stim.tsv.gz")
PO_IDX, OFFSET = 3, 2

# IDENTICAL to spaceman_lyric_check.py
CONTAM = [(526, 596), (703, 709), (787, 805)]

REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "transcripts")
MEDIA = os.path.expanduser("~/emofilm/media")

# model -> candidate signal paths (repo first, then media)
SIGNALS = {
    "BERT":       ["Spaceman_bert.json"],
    "Gemini-3.5": ["Spaceman_gemini.json"],
    "DistilBERT": ["Spaceman_distilbert.json"],
    "SiEBERT":    ["Spaceman_siebert.json"],
}

def find(fname):
    for base in (REPO, MEDIA):
        p = os.path.join(base, fname)
        if os.path.exists(p):
            return p
    return None

def load_po():
    rows = []
    with gzip.open(ANNOT, "rt") as f:
        for line in f:
            rows.append(float(line.rstrip("\n").split("\t")[PO_IDX]))
    return np.array(rows)

def shift(x, k):
    out = np.full_like(x, np.nan, dtype=float)
    if k > 0: out[k:] = x[:-k]
    elif k < 0: out[:k] = x[-k:]
    else: out[:] = x
    return out

def corr(a, b, mask):
    sel = mask & np.isfinite(a) & np.isfinite(b)
    if sel.sum() < 3:
        return float("nan"), int(sel.sum())
    return float(np.corrcoef(a[sel], b[sel])[0, 1]), int(sel.sum())

po = load_po()

print(f"{'Model':12s} {'all r':>8s} {'n':>5s} | {'clean r':>8s} {'n':>5s} | {'delta':>7s}")
print("-" * 56)
for model, names in SIGNALS.items():
    path = None
    for nm in names:
        path = find(nm)
        if path: break
    if not path:
        print(f"{model:12s}  (signal not found)")
        continue
    v = np.array(json.load(open(path))["valence"], dtype=float)
    n = min(len(po), len(v))
    pov, vv = po[:n], v[:n]
    dialogue = ~np.isnan(vv)
    r_all, n_all = corr(shift(vv, OFFSET), pov, dialogue)
    clean = dialogue.copy()
    for a, b in CONTAM:
        clean[a:b] = False
    r_clean, n_clean = corr(shift(vv, OFFSET), pov, clean)
    d = r_clean - r_all
    print(f"{model:12s} {r_all:+8.3f} {n_all:>5d} | {r_clean:+8.3f} {n_clean:>5d} | {d:+7.3f}")

print("-" * 56)
print(f"Removed seconds (same as BERT lyric check): "
      f"{sum(b-a for a,b in CONTAM)}s across {len(CONTAM)} ranges")
print("Reading: if Gemini's 'all r' collapses toward 0 under 'clean r' like BERT's,")
print("the Spaceman signal was shared transcription contamination in BOTH models.")
