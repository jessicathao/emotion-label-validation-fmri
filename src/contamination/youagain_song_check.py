#!/usr/bin/env python3
"""You Again: does the (already near-zero) correlation change when the
sung Pushkin passage (~582-688s) is removed? Parallels spaceman_lyric_check."""
import json, gzip, csv, math, os
import numpy as np

ANNOT = os.path.expanduser("~/ds004872/derivatives/Annot_YouAgain_stim.tsv.gz")
JSON  = os.path.expanduser("~/ds004872/derivatives/Annot_YouAgain_stim.json")
BERT  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "transcripts", "YouAgain_bert.json")
OFFSET = 2
SONG = [(582, 689)]   # sung Pushkin passage, 09:42-11:28

def load_po():
    idx = json.load(open(JSON))["Columns"].index("PleasantOther")
    rows=[]
    with gzip.open(ANNOT,"rt") as f:
        for r in csv.reader(f,delimiter="\t"):
            rows.append(float(r[idx]))
    return np.array(rows)

def shift(x,k):
    out=np.full_like(x,np.nan,dtype=float)
    if k>0: out[k:]=x[:-k]
    elif k<0: out[:k]=x[-k:]
    else: out[:]=x
    return out

def corr(a,b,mask):
    sel=mask & np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[sel],b[sel])[0,1], int(sel.sum())

po=load_po()
v=np.array(json.load(open(BERT))["valence"],dtype=float)
n=min(len(po),len(v)); po,v=po[:n],v[:n]
dialogue=~np.isnan(v)

r_all,n_all=corr(shift(v,OFFSET),po,dialogue)
clean=dialogue.copy()
for a,b in SONG: clean[a:b]=False
r_clean,n_clean=corr(shift(v,OFFSET),po,clean)

print("You Again BERT vs PleasantOther (+2s):")
print(f"  all dialogue   : r={r_all:+.3f}  (n={n_all})")
print(f"  song removed   : r={r_clean:+.3f}  (n={n_clean})")
print(f"  seconds removed: {n_all-n_clean}")
