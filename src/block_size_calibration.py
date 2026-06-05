#!/usr/bin/env python3
"""Find a block size that calibrates the v3 bootstrap.
Estimates integrated autocorrelation time (IAT) of PleasantOther per film,
then sweeps block sizes and reports the phase-randomized false-positive rate
(target ~5-15%). Pick the smallest block whose rate is in range."""
import json, gzip, csv, math, random, os
import numpy as np

ANNOT = os.path.expanduser("~/ds004872/derivatives")
TRANS = "data/transcripts"

def pearson(xs, ys):
    n=len(xs)
    if n<3: return None
    mx=sum(xs)/n; my=sum(ys)/n
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx=math.sqrt(sum((x-mx)**2 for x in xs)); sy=math.sqrt(sum((y-my)**2 for y in ys))
    if sx==0 or sy==0: return None
    return cov/(sx*sy)

def block_bootstrap_ci(x_full, y_full, n_boot=1500, block=20, seed=0):
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

def iat(x):
    """Integrated autocorrelation time (sum of positive-lag autocorrelations
    until first crossing of zero). A rough measure of the signal's 'memory'."""
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    if var == 0: return 1.0
    tau = 1.0
    for lag in range(1, n//2):
        c = np.dot(x[:-lag], x[lag:]) / (n * var)
        if c <= 0: break
        tau += 2*c
    return tau

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

def fp_rate(po, mask, block, n_trials=80):
    rng=np.random.default_rng(0); excl=0; done=0
    for i in range(n_trials):
        surr=phase_randomize(po,rng); xs,ys=to_full(po,mask,surr)
        lo,hi=block_bootstrap_ci(xs,ys,block=block,seed=i)
        if lo is None: continue
        done+=1
        if lo>0 or hi<0: excl+=1
    return (excl/done if done else float('nan')), done

BLOCKS = [20, 40, 60, 90, 120]
for film in ["Spaceman", "Sintel", "TearsOfSteel"]:
    po=load_po(film); mask,n=dialogue_mask(film)
    m=min(len(po),n); po=po[:m]; mask=mask[:m]
    t = iat(po)
    print(f"\n=== {film} ===  IAT(PleasantOther) ~ {t:.1f}s  (suggests block ~ {int(2*t)}-{int(3*t)}s)")
    for b in BLOCKS:
        rate, done = fp_rate(po, mask, b)
        tag = "OK" if 0.0 < rate <= 0.15 else ("marginal" if rate <= 0.25 else "too high")
        print(f"   block={b:>3}s  false-positive={rate:.0%}  ({done} trials)  [{tag}]")
