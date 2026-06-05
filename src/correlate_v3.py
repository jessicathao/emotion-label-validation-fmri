import json, sys, math, gzip, csv, random

def pearson(xs, ys):
    n = len(xs)
    if n < 3: return None
    mx = sum(xs)/n; my = sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x-mx)**2 for x in xs))
    sy = math.sqrt(sum((y-my)**2 for y in ys))
    if sx == 0 or sy == 0: return None
    return cov/(sx*sy)

def block_bootstrap_ci(x_full, y_full, n_boot=2000, block=20, seed=0):
    """Block on the FULL timeline so blocks are real-time-contiguous.
    x_full / y_full are aligned, same length, may contain None (NaN).
    Each resample drops NaN pairs only when computing Pearson."""
    random.seed(seed)
    T = len(x_full)
    if T < block*2: block = max(2, T//4)
    n_blocks = math.ceil(T/block)
    starts_max = T - block
    rs = []
    for _ in range(n_boot):
        bx = []; by = []
        for _b in range(n_blocks):
            s = random.randint(0, max(0, starts_max))
            bx.extend(x_full[s:s+block])
            by.extend(y_full[s:s+block])
        # drop NaN pairs within the resample
        px = []; py = []
        for a, b in zip(bx, by):
            if a is None or b is None: continue
            px.append(a); py.append(b)
        r = pearson(px, py)
        if r is not None: rs.append(r)
    rs.sort()
    if len(rs) < 20: return None, None
    return rs[int(0.025*len(rs))], rs[int(0.975*len(rs))]

bert_path, tsv_path, json_path = sys.argv[1], sys.argv[2], sys.argv[3]
fixed_off = int(sys.argv[4]) if len(sys.argv) > 4 else 2

bert_val = json.load(open(bert_path))["valence"]
cols = json.load(open(json_path))["Columns"]
po_idx = cols.index("PleasantOther")
human = []
with gzip.open(tsv_path, "rt") as f:
    for row in csv.reader(f, delimiter="\t"):
        human.append(float(row[po_idx]))
H = len(human)
T = len(bert_val)

# Build aligned full-timeline arrays (None where missing), offset applied to BERT.
# Pairing rule identical to v2: BERT second t pairs with human second t-off.
x_full = []; y_full = []
for t in range(T):
    v = bert_val[t]
    is_nan = (v is None) or (isinstance(v, float) and math.isnan(v))
    ht = t - fixed_off
    if is_nan or not (0 <= ht < H):
        x_full.append(None); y_full.append(None)
    else:
        x_full.append(v); y_full.append(human[ht])

# point estimate on finite pairs
xs = [a for a, b in zip(x_full, y_full) if a is not None and b is not None]
ys = [b for a, b in zip(x_full, y_full) if a is not None and b is not None]
r = pearson(xs, ys)

# real-time block count: contiguous timeline blocks containing >=1 finite pair
finite = [1 if (a is not None and b is not None) else 0 for a, b in zip(x_full, y_full)]
blk = 20
real_blocks = sum(1 for i in range(0, T, blk) if any(finite[i:i+blk]))

lo, hi = block_bootstrap_ci(x_full, y_full)
if lo is None:
    sig = "CI unavailable"
    ci_str = "[   n/a   ]"
else:
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not sig (CI spans 0)"
    ci_str = f"[{lo:+.3f},{hi:+.3f}]"

name = bert_path.split("/")[-1].replace("_bert.json", "")
print(f"{name:18s} r={r:+.3f}  95%CI{ci_str}  n={len(xs)}  realblk={real_blocks}  off={fixed_off}s  {sig}")
