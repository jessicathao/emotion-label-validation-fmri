#!/usr/bin/env python
"""sanity_bert_context_seven.py - generalize the ToS smoothing control to the 7 new films.
For each film: score isolated and +/-3 context BERT vs consensus (+2s), then sweep a dumb
centered moving average over the ISOLATED signal and find the width whose r best matches the
context r. If plain smoothing reaches the context r (no neighbouring text, no model call), the
context 'gain' is smoothing, not comprehension. Same smooth/r_vs_human/lag1 as sanity_..._tos.py."""
import json, math, gzip, csv, os

OFF = 2
HOME = os.path.expanduser("~")
FILMS = ["BetweenViewings","Chatter","Spaceman","Superhero","TheSecretNumber","ToClaireFromSonny","YouAgain"]
WSWEEP = [0,3,5,7,9,11,15,21,31,45,61]

def pearson(xs, ys):
    n=len(xs)
    if n<3: return None
    mx=sum(xs)/n; my=sum(ys)/n
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx=math.sqrt(sum((x-mx)**2 for x in xs)); sy=math.sqrt(sum((y-my)**2 for y in ys))
    return cov/(sx*sy) if sx and sy else None
def is_nan(v): return v is None or (isinstance(v,float) and math.isnan(v))
def load_human(film):
    base=f"{HOME}/ds004872/derivatives/Annot_{film}_stim"
    cols=json.load(open(base+".json"))["Columns"]; po=cols.index("PleasantOther"); h=[]
    with gzip.open(base+".tsv.gz","rt") as f:
        for row in csv.reader(f,delimiter="\t"): h.append(float(row[po]))
    return h
def r_vs_human(sig,human):
    H=len(human); xs=[]; ys=[]
    for t in range(len(sig)):
        ht=t-OFF
        if is_nan(sig[t]) or not (0<=ht<H): continue
        xs.append(sig[t]); ys.append(human[ht])
    return pearson(xs,ys)
def smooth(sig,W):
    T=len(sig); half=W//2; out=[None]*T
    for t in range(T):
        if is_nan(sig[t]): continue
        vals=[sig[u] for u in range(max(0,t-half),min(T,t+half+1)) if not is_nan(sig[u])]
        out[t]=sum(vals)/len(vals)
    return out
def lag1(sig):
    a=[];b=[]
    for t in range(len(sig)-1):
        if is_nan(sig[t]) or is_nan(sig[t+1]): continue
        a.append(sig[t]); b.append(sig[t+1])
    return pearson(a,b)

print(f"{'film':17s} {'iso':>7s} {'ctx':>7s} {'ctxL1':>6s} | {'bestW':>5s} {'smooth@W':>8s} {'match?':>6s}")
print("-"*70)
verdicts=[]
for film in FILMS:
    human=load_human(film)
    iso=json.load(open(f"data/transcripts/{film}_bert.json"))["valence"]
    ctx=json.load(open(f"{HOME}/emofilm/media/{film}_bert_context.json"))["valence"]
    r_iso=r_vs_human(iso,human); r_ctx=r_vs_human(ctx,human); l_ctx=lag1(ctx)
    # find the smoothing width whose r is closest to the context r
    best=None
    for W in WSWEEP:
        r=r_vs_human(smooth(iso,W),human)
        if r is None: continue
        if best is None or abs(r-r_ctx)<abs(best[1]-r_ctx): best=(W,r)
    bw,br=best
    # "match" = plain smoothing reaches the context r within 0.03 (reproduces the gain)
    reproduced = abs(br-r_ctx)<=0.03 and bw>0
    verdicts.append((film,r_iso,r_ctx,bw,br,reproduced))
    print(f"{film:17s} {r_iso:+.3f} {r_ctx:+.3f} {l_ctx:+.2f} | {bw:>5} {br:+.3f}  {'SMOOTH' if reproduced else 'check'}")

n_sm=sum(1 for v in verdicts if v[5])
print(f"\n{n_sm}/{len(verdicts)} films: the context r is reproduced by dumb smoothing of the isolated")
print("signal (no context) => that 'gain' is smoothing, not comprehension. Films marked 'check'")
print("are where smoothing does NOT reach the context r; those are the only candidates for a real")
print("effect and get the per-film sweep next.")
