#!/usr/bin/env python3
"""
sanity_context_tos.py
Sanity-check why Tears of Steel context r (+0.591) > isolation r (+0.467).
No API calls. Reads the two existing signal files + the human annotation, and
reports the diagnostics that distinguish a REAL context effect from an artifact:

  1. signal spread (sd) and lag-1 autocorrelation: did context just SMOOTH the
     signal? (smoother signal can correlate higher with smooth consensus for
     reasons unrelated to reading valence better)
  2. how many segments changed, and by how much
  3. leave-most-changed-out: drop the k segments whose score changed most
     between isolation and context, recompute r. If r survives, it's robust;
     if it collapses, a few points carry it.
  4. paired view: r(isolation), r(context), and r of each vs the other.

Usage:
  python sanity_context_tos.py \
      data/transcripts/TearsOfSteel_gemini.json \
      data/transcripts/TearsOfSteel_gemini_context.json \
      ~/ds004872/derivatives/Annot_TearsOfSteel_stim.tsv.gz \
      ~/ds004872/derivatives/Annot_TearsOfSteel_stim.json \
      [offset=2]
"""
import sys, os, json, gzip, csv, math

OFFSET_DEFAULT = 2
PLEASANTOTHER_FALLBACK_IDX = 3  # headerless; col index 3 per project notes


def load_signal(path):
    with open(path) as f:
        d = json.load(f)
    return d["valence"], d


def load_annotation(tsv_gz, json_sidecar):
    # find PleasantOther column index from sidecar "Columns"
    idx = PLEASANTOTHER_FALLBACK_IDX
    try:
        with open(json_sidecar) as f:
            meta = json.load(f)
        cols = meta.get("Columns") or meta.get("columns")
        if cols and "PleasantOther" in cols:
            idx = cols.index("PleasantOther")
    except Exception:
        pass
    vals = []
    op = gzip.open if tsv_gz.endswith(".gz") else open
    with op(tsv_gz, "rt") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            if not row:
                continue
            try:
                vals.append(float(row[idx]))
            except (ValueError, IndexError):
                vals.append(float("nan"))
    return vals, idx


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None
             and not (isinstance(x, float) and math.isnan(x))
             and not (isinstance(y, float) and math.isnan(y))]
    n = len(pairs)
    if n < 3:
        return None, n
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    cov = sum((p[0]-mx)*(p[1]-my) for p in pairs)
    sx = math.sqrt(sum((p[0]-mx)**2 for p in pairs))
    sy = math.sqrt(sum((p[1]-my)**2 for p in pairs))
    if sx == 0 or sy == 0:
        return None, n
    return cov/(sx*sy), n


def real_vals(sig):
    return [v for v in sig if v is not None and not (isinstance(v, float) and math.isnan(v))]


def sd(vs):
    n = len(vs)
    if n < 2:
        return 0.0
    m = sum(vs)/n
    return math.sqrt(sum((v-m)**2 for v in vs)/(n-1))


def lag1_autocorr(sig):
    # on the real-valued run, consecutive non-NaN seconds
    seq = []
    for v in sig:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            if seq and seq[-1] is not None:
                seq.append(None)
        else:
            seq.append(v)
    xs, ys = [], []
    for a, b in zip(seq, seq[1:]):
        if a is not None and b is not None:
            xs.append(a); ys.append(b)
    r, _ = pearson(xs, ys)
    return r


def shift(sig, off):
    if off == 0:
        return sig[:]
    if off > 0:
        return [float("nan")]*off + sig[:-off]
    return sig[-off:] + [float("nan")]*(-off)


def main():
    iso_p, ctx_p, tsv, js = sys.argv[1:5]
    off = int(sys.argv[5]) if len(sys.argv) > 5 else OFFSET_DEFAULT

    iso, _ = load_signal(iso_p)
    ctx, _ = load_signal(ctx_p)
    ann, idx = load_annotation(os.path.expanduser(tsv), os.path.expanduser(js))
    print(f"PleasantOther column idx = {idx}")

    L = min(len(iso), len(ctx), len(ann))
    iso, ctx, ann = iso[:L], ctx[:L], ann[:L]

    iso_s = shift(iso, off)
    ctx_s = shift(ctx, off)

    r_iso, n_iso = pearson(iso_s, ann)
    r_ctx, n_ctx = pearson(ctx_s, ann)
    print(f"\nr(isolation vs human) = {r_iso:+.3f}  (n={n_iso})")
    print(f"r(context   vs human) = {r_ctx:+.3f}  (n={n_ctx})")

    # 1. spread + autocorrelation
    iso_r = real_vals(iso); ctx_r = real_vals(ctx)
    print(f"\n--- spread / smoothness ---")
    print(f"isolation: sd={sd(iso_r):.3f}  lag1-autocorr={lag1_autocorr(iso):+.3f}")
    print(f"context:   sd={sd(ctx_r):.3f}  lag1-autocorr={lag1_autocorr(ctx):+.3f}")
    print("(if context sd is much lower AND autocorr much higher, context mainly SMOOTHED)")

    # 2. how many segments changed (compare second-by-second real values)
    diffs = []
    for a, b in zip(iso, ctx):
        if (a is not None and not (isinstance(a, float) and math.isnan(a))
                and b is not None and not (isinstance(b, float) and math.isnan(b))):
            diffs.append(abs(a-b))
    if diffs:
        changed = sum(1 for d in diffs if d > 0.01)
        print(f"\n--- change isolation->context (per real second) ---")
        print(f"real seconds compared: {len(diffs)}")
        print(f"seconds that changed (>0.01): {changed} ({100*changed/len(diffs):.0f}%)")
        print(f"mean |change|: {sum(diffs)/len(diffs):.3f}  max |change|: {max(diffs):.3f}")

    # 3. leave-most-changed-out: drop top-k most-changed SECONDS, recompute r(context)
    print(f"\n--- robustness: drop most-changed seconds, recompute r(context vs human) ---")
    # build list of (idx, change) on aligned real seconds vs annotation
    triples = []
    for i in range(L):
        a, b, h = iso[i], ctx_s[i], ann[i]
        if (b is not None and not (isinstance(b, float) and math.isnan(b))
                and h is not None and not (isinstance(h, float) and math.isnan(h))):
            ch = abs((a if a is not None and not (isinstance(a,float) and math.isnan(a)) else b) - b)
            triples.append((i, ch))
    triples.sort(key=lambda t: -t[1])
    for k in (0, 5, 10, 20):
        drop = set(t[0] for t in triples[:k])
        xs = [ctx_s[i] for i in range(L) if i not in drop]
        ys = [ann[i] for i in range(L) if i not in drop]
        r, n = pearson(xs, ys)
        print(f"  drop top-{k:2d} changed: r={r:+.3f}  (n={n})")
    print("(if r holds as k grows, the effect is broad; if it collapses, a few points carry it)")


if __name__ == "__main__":
    main()
