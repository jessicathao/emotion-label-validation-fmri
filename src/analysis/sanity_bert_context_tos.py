#!/usr/bin/env python3
"""sanity_bert_context_tos.py - is the ToS BERT-context rise context, or smoothing?

Take the ISOLATED BERT signal and smooth it with a dumb centered moving average
(NO context, no neighbouring text, just blurring). If plain smoothing of the
isolated signal reaches the same r-vs-human as the +/-3 context condition, the
"context gain" carries no context information: it is entirely smoothing.
"""
import json, sys, math, gzip, csv, os

OFF = 2
FILM = "TearsOfSteel"
ISO_PATH = sys.argv[1] if len(sys.argv) > 1 else f"data/transcripts/{FILM}_bert.json"
CTX_PATH = sys.argv[2] if len(sys.argv) > 2 else f"data/transcripts/{FILM}_bert_context.json"
TSV = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser(f"~/ds004872/derivatives/Annot_{FILM}_stim.tsv.gz")
JSN = sys.argv[4] if len(sys.argv) > 4 else os.path.expanduser(f"~/ds004872/derivatives/Annot_{FILM}_stim.json")
CONTEXT_SWEEP = {"iso": 0.356, "w1": 0.483, "w2": 0.508, "w3": 0.513, "w5": 0.566}

def pearson(xs, ys):
    n = len(xs)
    if n < 3: return None
    mx = sum(xs)/n; my = sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x-mx)**2 for x in xs)); sy = math.sqrt(sum((y-my)**2 for y in ys))
    if sx == 0 or sy == 0: return None
    return cov/(sx*sy)

def is_nan(v):
    return (v is None) or (isinstance(v, float) and math.isnan(v))

def load_human():
    cols = json.load(open(JSN))["Columns"]; po = cols.index("PleasantOther"); h = []
    with gzip.open(TSV, "rt") as f:
        for row in csv.reader(f, delimiter="\t"):
            h.append(float(row[po]))
    return h

def r_vs_human(sig, human):
    H = len(human); xs = []; ys = []
    for t in range(len(sig)):
        ht = t - OFF
        if is_nan(sig[t]) or not (0 <= ht < H): continue
        xs.append(sig[t]); ys.append(human[ht])
    return pearson(xs, ys), len(xs)

def smooth(sig, W):
    T = len(sig); half = W // 2; out = [None]*T
    for t in range(T):
        if is_nan(sig[t]): continue
        vals = [sig[u] for u in range(max(0, t-half), min(T, t+half+1)) if not is_nan(sig[u])]
        out[t] = sum(vals)/len(vals)
    return out

def lag1_autocorr(sig):
    a = []; b = []
    for t in range(len(sig)-1):
        if is_nan(sig[t]) or is_nan(sig[t+1]): continue
        a.append(sig[t]); b.append(sig[t+1])
    return pearson(a, b)

def main():
    human = load_human()
    iso = json.load(open(ISO_PATH))["valence"]
    ctx = json.load(open(CTX_PATH))["valence"]
    r_iso, n = r_vs_human(iso, human); r_ctx, _ = r_vs_human(ctx, human)
    print(f"{FILM}: n={n} paired seconds, off={OFF}s")
    print(f"  isolated BERT     r={r_iso:+.3f}   lag1 autocorr={lag1_autocorr(iso):+.3f}")
    print(f"  +/-3 context BERT r={r_ctx:+.3f}   lag1 autocorr={lag1_autocorr(ctx):+.3f}  (smoother)")
    print("\nDUMB-SMOOTHING CONTROL: moving-average the ISOLATED signal (no context at all)")
    print(f"  {'smooth W (s)':>12}   {'r vs human':>10}   {'lag1':>7}")
    for W in (0, 5, 11, 21, 31, 45, 61):
        sm = smooth(iso, W); r, _ = r_vs_human(sm, human)
        print(f"  {W:>12}   {r:+.3f}      {lag1_autocorr(sm):+.3f}")
    print("\nCONTEXT-WINDOW SWEEP (for comparison; validated run):")
    print("  " + "  ".join(f"{k}={v:+.3f}" for k, v in CONTEXT_SWEEP.items()))
    print("\nVERDICT: if dumb smoothing of the isolated signal reaches the context r")
    print("(~+0.51 to +0.57) WITHOUT any neighbouring text, the context 'gain' is")
    print("smoothing, not context comprehension.")

if __name__ == "__main__":
    main()
