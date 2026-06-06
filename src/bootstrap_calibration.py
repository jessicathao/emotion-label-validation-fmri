#!/usr/bin/env python3
"""Sanity-check correlate_v3's block bootstrap:
  (A) NULL calibration: on uncorrelated-but-autocorrelated data, the 95% CI
      should exclude zero ~5% of the time. More = still overstating significance.
  (B) SIGNAL recovery: a known strong correlation should be bracketed and
      flagged significant.
Tested on a high-coverage film (Spaceman) and a low-coverage one (Sintel)."""
import json, gzip, csv, math, random, os
import numpy as np

ANNOT = os.path.expanduser("~/ds004872/derivatives")
TRANS = "data/transcripts"
OFFSET = 2

def pearson(xs, ys):
    n=len(xs)
    if n<3: return None
    mx=sum(xs)/n; my=sum(ys)/n
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx=math.sqrt(sum((x-mx)**2 for x in xs)); sy=math.sqrt(sum((y-my)**2 for y in ys))
    if sx==0 or sy==0: return None
    return cov/(sx*sy)

def block_bootstrap_ci(x_full, y_full, n_boot=2000, block=20, seed=0):
    random.seed(seed); T=len(x_full)
    if T<block*2: block=max(2,T//4)
    n_blocks=math.ceil(T/block); starts_max=T-block; rs=[]
    for _ in range(n_boot):
        bx=[]; by=[]
        for _b in range(n_blocks):
            s=random.randint(0,max(0,starts_max))
            bx.extend(x_full[s:s+block]); by.extend(y_full[s:s+block])
        px=[]; py=[]
        for a,b in zip(bx,by):
            if a is None or b is None: continue
            px.append(a); py.append(b)
        r=pearson(px,py)
        if r is not None: rs.append(r)
    rs.sort()
    if len(rs)<20: return None,None
    return rs[int(0.025*len(rs))], rs[int(0.975*len(rs))]

def load_po(film):
    j=json.load(open(f"{ANNOT}/Annot_{film}_stim.json"))
    idx=j["Columns"].index("PleasantOther")
    out=[]
    with gzip.open(f"{ANNOT}/Annot_{film}_stim.tsv.gz","rt") as f:
        for row in csv.reader(f,delimiter="\t"):
            out.append(float(row[idx]))
    return np.array(out)

def dialogue_mask(film):
    v=np.array(json.load(open(f"{TRANS}/{film}_bert.json"))["valence"],dtype=float)
    return ~np.isnan(v), len(v)

def circular_shift(x, k):
    return np.roll(x, k)

def to_full(po, mask, target):
    x=[]; y=[]
    for t in range(len(mask)):
        if mask[t] and np.isfinite(po[t]) and np.isfinite(target[t]):
            x.append(float(po[t])); y.append(float(target[t]))
        else:
            x.append(None); y.append(None)
    return x, y

for film in ["Spaceman", "Sintel"]:
    po = load_po(film)
    mask, n = dialogue_mask(film)
    m = min(len(po), n); po=po[:m]; mask=mask[:m]

    shifted = np.full_like(po, np.nan); shifted[OFFSET:] = po[:-OFFSET]
    xs, ys = to_full(po, mask, shifted)
    lo, hi = block_bootstrap_ci(xs, ys)
    pt = pearson([a for a,b in zip(xs,ys) if a is not None and b is not None],
                 [b for a,b in zip(xs,ys) if a is not None and b is not None])
    print(f"\n=== {film} ===")
    print(f"(B) signal recovery: r={pt:+.3f} CI[{lo:+.3f},{hi:+.3f}] "
          f"-> {'SIG (correct)' if (lo>0 or hi<0) else 'FAILED to detect known signal'}")

    n_trials=40; excl=0
    rng=np.random.default_rng(0)
    span=len(po)
    for i in range(n_trials):
        k = int(rng.integers(span//5, span - span//5))
        null_target = circular_shift(po, k)
        xs, ys = to_full(po, mask, null_target)
        lo, hi = block_bootstrap_ci(xs, ys, seed=i)
        if lo is None: continue
        if lo>0 or hi<0: excl+=1
    rate = excl/n_trials
    flag = "OK (~5% expected)" if rate<=0.15 else "TOO HIGH - still overstating"
    print(f"(A) null calibration: CI excluded 0 in {excl}/{n_trials} = {rate:.0%}  [{flag}]")
