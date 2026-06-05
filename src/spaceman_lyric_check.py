#!/usr/bin/env python3
"""Does Spaceman's negative survive removing diegetic song lyrics + credits?"""
import json, gzip, os
import numpy as np

ANNOT = os.path.expanduser("~/ds004872/derivatives/Annot_Spaceman_stim.tsv.gz")
BERT  = os.path.expanduser("~/lang_brain_project/data/transcripts/Spaceman_bert.json")
PO_IDX, OFFSET = 3, 2

# Lyric / credits second-ranges from the Whisper transcript (inclusive starts, exclusive ends)
CONTAM = [(526, 596),   # seg 88-91 song: "you're nothing / die in Hollywood / candy dilated"
          (703, 709),   # seg 115 song: "walking out of my dreams"
          (787, 805)]   # seg 122 credits hallucination: "Thank you for watching"

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
    return np.corrcoef(a[sel], b[sel])[0,1], int(sel.sum())

po = load_po()
v = np.array(json.load(open(BERT))["valence"], dtype=float)
n = min(len(po), len(v))
po, v = po[:n], v[:n]
dialogue = ~np.isnan(v)

# baseline: BERT valence (offset +2s) vs PleasantOther, on all dialogue seconds
r_all, n_all = corr(shift(v, OFFSET), po, dialogue)

# contaminated seconds removed
clean = dialogue.copy()
for a, b in CONTAM:
    clean[a:b] = False
r_clean, n_clean = corr(shift(v, OFFSET), po, clean)

print(f"Spaceman BERT vs PleasantOther (+2s offset):")
print(f"  all dialogue seconds : r = {r_all:+.3f}  (n = {n_all})")
print(f"  lyrics+credits removed: r = {r_clean:+.3f}  (n = {n_clean})")
print(f"  seconds removed       : {n_all - n_clean}")
