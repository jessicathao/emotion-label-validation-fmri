import sys, re, glob, os

def parse_srt(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    lines=[]
    for block in txt.strip().split("\n\n"):
        rows=block.strip().split("\n")
        if len(rows)>=3:
            lines.append(" ".join(rows[2:]).strip())
    return [l for l in lines if l]

def norm(t): return re.sub(r"[^a-z0-9 ]","",t.lower()).strip()

folder = sys.argv[1] if len(sys.argv)>1 else "."
print(f"{'film':28s} {'segs':>5s} {'uniq':>5s} {'rep%':>5s} {'words':>6s}  flag")
print("-"*70)
for srt in sorted(glob.glob(os.path.join(folder,"*.srt"))):
    segs = parse_srt(srt)
    if not segs:
        print(f"{os.path.basename(srt)[:28]:28s} {'EMPTY':>5s}"); continue
    normed=[norm(s) for s in segs]
    uniq=len(set(normed))
    rep = 100*(1 - uniq/len(segs))
    words=sum(len(s.split()) for s in segs)
    flag = "  <-- CHECK (repetitive)" if rep>30 else ("  <-- low words" if words<80 else "")
    name=os.path.basename(srt).replace("_exp.srt","")
    print(f"{name[:28]:28s} {len(segs):5d} {uniq:5d} {rep:5.0f} {words:6d}{flag}")
