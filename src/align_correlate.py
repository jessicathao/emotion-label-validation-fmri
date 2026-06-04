import json, sys, math, gzip, csv

def pearson(pairs):
    n=len(pairs)
    if n<3: return None
    mx=sum(p[0] for p in pairs)/n; my=sum(p[1] for p in pairs)/n
    cov=sum((p[0]-mx)*(p[1]-my) for p in pairs)
    sx=math.sqrt(sum((p[0]-mx)**2 for p in pairs))
    sy=math.sqrt(sum((p[1]-my)**2 for p in pairs))
    if sx==0 or sy==0: return None
    return cov/(sx*sy)

bert_path, tsv_path, json_path = sys.argv[1], sys.argv[2], sys.argv[3]
max_off = int(sys.argv[4]) if len(sys.argv)>4 else 30
min_off = -10

bert = json.load(open(bert_path))
bert_val = bert["valence"]

cols = json.load(open(json_path))["Columns"]
po_idx = cols.index("PleasantOther")
human=[]
with gzip.open(tsv_path, "rt") as f:
    for row in csv.reader(f, delimiter="\t"):
        human.append(float(row[po_idx]))
H=len(human)
print(f"human PleasantOther: {H}s   bert grid: {len(bert_val)}s  (dialogue secs: {sum(1 for v in bert_val if not math.isnan(v))})")

results=[]
for off in range(min_off, max_off+1):
    pairs=[]
    for t,v in enumerate(bert_val):
        if math.isnan(v): continue
        ht=t-off
        if 0<=ht<H:
            pairs.append((v, human[ht]))
    r=pearson(pairs)
    if r is not None:
        results.append((off, r, len(pairs)))

results_sorted=sorted(results, key=lambda x: x[1], reverse=True)
best=results_sorted[0]
print(f"\nBEST offset = {best[0]}s   Pearson r = {best[1]:+.3f}   (n={best[2]} dialogue secs)")
print("human-human ceiling for reference ~0.40\n")
print("top 5 offsets by r:")
for off,r,n in results_sorted[:5]:
    print(f"  offset {off:3d}s -> r={r:+.3f}  (n={n})")
print("\ncurve (every 10s):")
for off,r,n in results:
    if off % 10 == 0:
        bar = "#"*int(abs(r)*40)
        print(f"  {off:3d}s  r={r:+.3f} {bar}")
