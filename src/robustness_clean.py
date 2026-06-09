#!/usr/bin/env python3
"""Re-run 3 sentiment models through corrected (real-timeline) bootstrap,
with diegetic-lyric/credits seconds excluded. Spaceman + Tears of Steel."""
import json, math, gzip, csv, random, os

ANNOT = os.path.expanduser("~/ds004872/derivatives")
TRANS = "data/transcripts"
OFFSET = 2

# contaminated second-ranges per film (inclusive start, exclusive end)
CONTAM = {
    "Spaceman": [(526, 596), (703, 709), (787, 805)],
    "TearsOfSteel": [],   # <-- fill after reading its transcript
}
MODELS = {"BERT": "_bert", "DistilBERT": "_distilbert", "SiEBERT": "_siebert"}

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

def load_human(film):
    j=json.load(open(f"{ANNOT}/Annot_{film}_stim.json"))
    idx=j["Columns"].index("PleasantOther")
    out=[]
    with gzip.open(f"{ANNOT}/Annot_{film}_stim.tsv.gz","rt") as f:
        for row in csv.reader(f,delimiter="\t"):
            out.append(float(row[idx]))
    return out

def run(film, model_suffix, exclude):
    human=load_human(film); H=len(human)
    v=json.load(open(f"{TRANS}/{film}{model_suffix}.json"))["valence"]
    bad=set()
    if exclude:
        for a,b in CONTAM.get(film,[]):
            bad.update(range(a,b))
    x_full=[]; y_full=[]
    for t in range(len(v)):
        val=v[t]
        nan=(val is None) or (isinstance(val,float) and math.isnan(val))
        ht=t-OFFSET
        if nan or not(0<=ht<H) or (t in bad):
            x_full.append(None); y_full.append(None)
        else:
            x_full.append(val); y_full.append(human[ht])
    xs=[a for a,b in zip(x_full,y_full) if a is not None and b is not None]
    ys=[b for a,b in zip(x_full,y_full) if a is not None and b is not None]
    r=pearson(xs,ys); lo,hi=block_bootstrap_ci(x_full,y_full)
    sig = "CI excludes 0" if (lo is not None and (lo>0 or hi<0)) else "CI spans 0  "
    ci = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None else "[n/a]"
    return f"r={r:+.3f} {ci} n={len(xs)} {sig}"

for film in ["Spaceman", "TearsOfSteel"]:
    print(f"\n=== {film} ===")
    for mname, suf in MODELS.items():
        if not os.path.exists(f"{TRANS}/{film}{suf}.json"):
            print(f"  {mname:11s} (no file)"); continue
        full  = run(film, suf, exclude=False)
        clean = run(film, suf, exclude=True)
        print(f"  {mname:11s} all:   {full}")
        print(f"  {' ':11s} clean: {clean}")
