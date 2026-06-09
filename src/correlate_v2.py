# DEPRECATED (June 5 2026): broken block bootstrap, resampled blocks on the
# dialogue-only collapsed signal, breaking real-time autocorrelation and
# overstating significance. Superseded by correlate_v3.py. Kept ONLY to
# reproduce the old archived numbers. Do not use for new analysis.
import json, sys, math, gzip, csv, random

def pearson(xs, ys):
    n=len(xs)
    if n<3: return None
    mx=sum(xs)/n; my=sum(ys)/n
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx=math.sqrt(sum((x-mx)**2 for x in xs)); sy=math.sqrt(sum((y-my)**2 for y in ys))
    if sx==0 or sy==0: return None
    return cov/(sx*sy)

def block_bootstrap_ci(xs, ys, n_boot=2000, block=20, seed=0):
    random.seed(seed)
    n=len(xs)
    if n < block*2: block=max(2,n//4)
    n_blocks=math.ceil(n/block)
    rs=[]; starts_max=n-block
    for _ in range(n_boot):
        bx=[]; by=[]
        for _b in range(n_blocks):
            s=random.randint(0, max(0,starts_max))
            bx.extend(xs[s:s+block]); by.extend(ys[s:s+block])
        bx=bx[:n]; by=by[:n]
        r=pearson(bx,by)
        if r is not None: rs.append(r)
    rs.sort()
    return rs[int(0.025*len(rs))], rs[int(0.975*len(rs))]

print("[DEPRECATED] correlate_v2.py reproduces pre-June-5 archived numbers only.")
print("[DEPRECATED] Its block bootstrap is anticonservative; significance is")
print("[DEPRECATED] withdrawn project-wide (see correlate_v3.py). CI is descriptive.")
bert_path, tsv_path, json_path = sys.argv[1], sys.argv[2], sys.argv[3]
fixed_off = int(sys.argv[4]) if len(sys.argv)>4 else 2
bert_val = json.load(open(bert_path))["valence"]
cols = json.load(open(json_path))["Columns"]
po_idx = cols.index("PleasantOther")
human=[]
with gzip.open(tsv_path,"rt") as f:
    for row in csv.reader(f, delimiter="\t"):
        human.append(float(row[po_idx]))
H=len(human)
xs=[]; ys=[]
for t,v in enumerate(bert_val):
    if v is None or (isinstance(v,float) and math.isnan(v)): continue
    ht=t-fixed_off
    if 0<=ht<H:
        xs.append(v); ys.append(human[ht])
r=pearson(xs,ys)
lo,hi=block_bootstrap_ci(xs,ys)
sig = "[archived: CI excludes 0]" if (lo>0 or hi<0) else "[archived: CI spans 0]"
name=bert_path.split("/")[-1].replace("_bert.json","")
print(f"{name:18s} r={r:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  n={len(xs)}  offset={fixed_off}s  {sig}")
