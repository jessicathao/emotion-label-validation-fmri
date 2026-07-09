#!/usr/bin/env python
"""sanity_chatter_context.py - is Chatter's BERT-context gain (+0.136 -> +0.651) real or artifact?
Chatter is the one film of 11 where dumb smoothing does NOT reproduce the context r. Three checks,
reusing the ToS diagnostic:
  1. WINDOW SWEEP + block-bootstrap CI (rebuild BERT-context at W=0,1,2,3,5). Smoothing/instability
     signature: r climbs WHILE the CI lower bound falls through zero (losing effective samples).
     A real gain tightens the CI as it rises.
  2. LEAVE-MOST-CHANGED-OUT on the +/-3 context: drop the seconds context changed most; collapse
     => outlier-driven, not broad.
  3. CONTAMINATION: print the dialogue covering the most-changed seconds; Chatter is transcription-
     contaminated, so check whether the gain rides on credits/sound-event hallucinations.
Local BERT inference for the sweep (Chatter is short); no API."""
import json, math, gzip, csv, os, sys, subprocess, random

OFF = 2
HOME = os.path.expanduser("~")
FILM = "Chatter"
FILT = f"{HOME}/emofilm/media/{FILM}_filt3.json"
ISO  = f"data/transcripts/{FILM}_bert.json"
CTX  = f"{HOME}/emofilm/media/{FILM}_bert_context.json"
ANN  = f"{HOME}/ds004872/derivatives/Annot_{FILM}_stim"
PY = sys.executable

def pearson(xs,ys):
    n=len(xs)
    if n<3: return None
    mx=sum(xs)/n; my=sum(ys)/n
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx=math.sqrt(sum((x-mx)**2 for x in xs)); sy=math.sqrt(sum((y-my)**2 for y in ys))
    return cov/(sx*sy) if sx and sy else None
def is_nan(v): return v is None or (isinstance(v,float) and math.isnan(v))
def load_human():
    cols=json.load(open(ANN+".json"))["Columns"]; po=cols.index("PleasantOther"); h=[]
    with gzip.open(ANN+".tsv.gz","rt") as f:
        for row in csv.reader(f,delimiter="\t"): h.append(float(row[po]))
    return h
def aligned(sig,human):
    H=len(human); x=[]; y=[]
    for t in range(len(sig)):
        ht=t-OFF
        if is_nan(sig[t]) or not (0<=ht<H): x.append(None); y.append(None)
        else: x.append(sig[t]); y.append(human[ht])
    return x,y
def r_finite(xf,yf):
    xs=[a for a,b in zip(xf,yf) if a is not None and b is not None]
    ys=[b for a,b in zip(xf,yf) if a is not None and b is not None]
    return pearson(xs,ys), len(xs)
def block_ci(xf,yf,n_boot=2000,block=20,seed=0):
    random.seed(seed); T=len(xf)
    if T<block*2: block=max(2,T//4)
    n_blocks=math.ceil(T/block); smax=T-block; rs=[]
    for _ in range(n_boot):
        bx=[]; by=[]
        for _b in range(n_blocks):
            s=random.randint(0,max(0,smax)); bx.extend(xf[s:s+block]); by.extend(yf[s:s+block])
        px=[a for a,b in zip(bx,by) if a is not None and b is not None]
        py=[b for a,b in zip(bx,by) if a is not None and b is not None]
        r=pearson(px,py)
        if r is not None: rs.append(r)
    rs.sort()
    if len(rs)<20: return None,None
    return rs[int(0.025*len(rs))], rs[int(0.975*len(rs))]
def lag1(sig):
    a=[];b=[]
    for t in range(len(sig)-1):
        if is_nan(sig[t]) or is_nan(sig[t+1]): continue
        a.append(sig[t]); b.append(sig[t+1])
    return pearson(a,b)

human=load_human()
iso=json.load(open(ISO))["valence"]

print(f"=== {FILM}: WINDOW SWEEP + CI (real gain tightens the CI; smoothing widens it) ===")
print(f"  {'window':>7} {'r':>8} {'95% CI':>18} {'lag1':>7}")
for W in [0,1,2,3,5]:
    if W==0:
        sig=iso
    else:
        tmp=f"/tmp/{FILM}_ctx_w{W}.json"
        subprocess.run([PY,"src/signals/make_bert_signal.py",FILT,tmp,"--window",str(W)],
                       capture_output=True,text=True)
        sig=json.load(open(tmp))["valence"]
    xf,yf=aligned(sig,human); r,n=r_finite(xf,yf); lo,hi=block_ci(xf,yf)
    ci=f"[{lo:+.2f},{hi:+.2f}]" if lo is not None else "[   n/a   ]"
    print(f"  {('iso' if W==0 else '+/-'+str(W)):>7} {r:+.3f} {ci:>18} {lag1(sig):+.2f}")

ctx=json.load(open(CTX))["valence"]
print(f"\n=== LEAVE-MOST-CHANGED-OUT (+/-3 context; collapse => outlier-driven) ===")
def chg(t):
    return abs(ctx[t]-iso[t]) if (t<len(iso) and not is_nan(ctx[t]) and not is_nan(iso[t])) else -1
changed=sorted(range(len(ctx)), key=lambda t: -chg(t))
for k in [0,5,10,20]:
    drop=set(changed[:k])
    sig=[float("nan") if t in drop else ctx[t] for t in range(len(ctx))]
    xf,yf=aligned(sig,human); r,n=r_finite(xf,yf)
    print(f"  drop top {k:2d} changed seconds: r={r:+.3f} (n={n})")

print(f"\n=== CONTAMINATION: dialogue covering the most-changed seconds ===")
segs=json.load(open(FILT))["segments"]
def seg_at(t):
    for s in segs:
        if s["start"]<=t<s["end"]: return s["text"]
    return "(no covering segment)"
seen=set()
for t in changed[:20]:
    if chg(t)<0: continue
    txt=seg_at(t)
    if txt in seen: continue
    seen.add(txt)
    print(f"  [{t:3d}s] iso={iso[t]:+.2f} ctx={ctx[t]:+.2f}  \"{txt[:72]}\"")
    if len(seen)>=8: break

print("\nREAD: REAL -> r rises while the CI tightens, survives leave-most-changed-out, and the")
print("driving lines are genuine dialogue. ARTIFACT -> r rises while the CI lower bound falls")
print("through zero, collapses when a few seconds drop, or rides on credits/sound-event lines.")
