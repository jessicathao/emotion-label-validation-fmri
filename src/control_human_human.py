# POSITIVE CONTROL: human-vs-human correlations, to confirm the pipeline math.
# The CI flag below is DESCRIPTIVE (does the bootstrap CI exclude 0); significance
# is withheld project-wide. This checks the method, it is not a reported result.
import json, sys, math, gzip, csv, random

def pearson(xs, ys):
    n=len(xs)
    if n<3: return None
    mx=sum(xs)/n; my=sum(ys)/n
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx=math.sqrt(sum((x-mx)**2 for x in xs)); sy=math.sqrt(sum((y-my)**2 for y in ys))
    if sx==0 or sy==0: return None
    return cov/(sx*sy)

def boot_ci(xs, ys, n_boot=2000, block=20, seed=0):
    random.seed(seed); n=len(xs)
    if n < block*2: block=max(2,n//4)
    nb=math.ceil(n/block); rs=[]; smax=n-block
    for _ in range(n_boot):
        bx=[]; by=[]
        for _b in range(nb):
            s=random.randint(0,max(0,smax)); bx+=xs[s:s+block]; by+=ys[s:s+block]
        r=pearson(bx[:n],by[:n])
        if r is not None: rs.append(r)
    rs.sort(); return rs[int(.025*len(rs))], rs[int(.975*len(rs))]

tsv, jsn = sys.argv[1], sys.argv[2]
cols = json.load(open(jsn))["Columns"]
data={c:[] for c in cols}
with gzip.open(tsv,"rt") as f:
    for row in csv.reader(f, delimiter="\t"):
        for i,c in enumerate(cols):
            data[c].append(float(row[i]))

def test(a,b,expect):
    xs,ys=data[a],data[b]
    r=pearson(xs,ys); lo,hi=boot_ci(xs,ys)
    sig = "CI excludes 0" if (lo>0 or hi<0) else "CI spans 0  "
    print(f"  {a:16s} vs {b:16s}  r={r:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  {sig}   (expect {expect})")

print("POSITIVE CONTROL - human vs human, same film, same pipeline math\n")
print("Should be STRONGLY POSITIVE:")
test("PleasantOther","Good","++")
test("Happiness","Satisfaction","++")
print("\nShould be STRONGLY NEGATIVE:")
test("Happiness","Sad","--")
test("PleasantOther","Bad","--")
print("\nShould be NEAR ZERO (unrelated):")
test("PleasantOther","Suddenly","~0")
test("Heartrate","SocialNorms","~0")
