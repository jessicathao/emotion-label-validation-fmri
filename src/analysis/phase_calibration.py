#!/usr/bin/env python3
"""Phase-randomized calibration of correlate_v3's block bootstrap.
Phase randomization preserves the power spectrum (autocorrelation) while
destroying cross-alignment -> the correct null for a smooth signal.
A calibrated 95% CI should exclude zero ~5% of the time."""
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

def phase_randomize(x, rng):
    n=len(x); X=np.fft.rfft(x-x.mean()); mag=np.abs(X)
    ph=rng.uniform(0,2*np.pi,size=len(X)); ph[0]=0.0
    if n%2==0: ph[-1]=0.0
    return np.fft.irfft(mag*np.exp(1j*ph), n=n) + x.mean()

def to_full(po, mask, target):
    x=[]; y=[]
    for t in range(len(mask)):
        if mask[t] and np.isfinite(po[t]) and np.isfinite(target[t]):
            x.append(float(po[t])); y.append(float(target[t]))
        else:
            x.append(None); y.append(None)
    return x, y

for film in ["Spaceman", "Sintel"]:
    po=load_po(film); mask,n=dialogue_mask(film)
    m=min(len(po),n); po=po[:m]; mask=mask[:m]
    print(f"\n=== {film} ===")
    shifted=np.full_like(po,np.nan); shifted[OFFSET:]=po[:-OFFSET]
    xs,ys=to_full(po,mask,shifted); lo,hi=block_bootstrap_ci(xs,ys)
    pt=pearson([a for a,b in zip(xs,ys) if a is not None and b is not None],
               [b for a,b in zip(xs,ys) if a is not None and b is not None])
    print(f"(B) signal recovery: r={pt:+.3f} CI[{lo:+.3f},{hi:+.3f}] "
          f"-> {'SIG (correct)' if (lo>0 or hi<0) else 'FAILED'}")
    n_trials=100; excl=0; done=0; rng=np.random.default_rng(0)
    for i in range(n_trials):
        surr=phase_randomize(po,rng); xs,ys=to_full(po,mask,surr)
        lo,hi=block_bootstrap_ci(xs,ys,seed=i)
        if lo is None: continue
        done+=1
        if lo>0 or hi<0: excl+=1
    rate=excl/done if done else float('nan')
    if rate<=0.15: flag="OK - well calibrated (~5% expected)"
    elif rate<=0.25: flag="marginal - slightly liberal"
    else: flag="TOO HIGH - v3 overcovers; widen CIs / soften 'significant'"
    print(f"(A) phase-randomized null: CI excluded 0 in {excl}/{done} = {rate:.0%}  [{flag}]")
