#!/usr/bin/env python3
"""
make_gemini_context_signal.py
CONTEXT-CONDITION probe (companion to make_gemini_signal.py).

Difference from the isolation probe: when scoring each dialogue segment, the model
is shown a window of +/-N surrounding segments as CONTEXT, but is asked to score
ONLY the target segment's valence. Output is one score per segment on the same
1 Hz grid as the isolation run, so correlate_v3 consumes it unchanged. The ONLY
variable that differs from the isolation Gemini result is the presence of context.

Usage:
  python make_gemini_context_signal.py TRANSCRIPT OUT [grid_len] \
      [--window 3] [--model gemini-3.5-flash] [--cache CACHE] [--sleep 0.0]

TRANSCRIPT : the SAME _filt3.json used by BERT/isolation Gemini
             format: {film, source?, segments:[{start,end,text}, ...]}
OUT        : output signal json (same schema as make_bert_signal output)
grid_len   : optional int; total seconds in the 1 Hz grid (default: ceil(max end))

Env: GEMINI_API_KEY must be set (same as your isolation probe).
"""
import sys, os, json, math, time, argparse, urllib.request, urllib.error

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT_TEMPLATE = """You are rating the emotional VALENCE of one line of film dialogue.
Valence = how pleasant (positive) vs unpleasant (negative) the moment feels.

You are given the surrounding dialogue as CONTEXT, then the TARGET line.
Use the context only to understand the target. Score ONLY the target line.

Return a single number from 0.0 to 1.0:
  0.0 = very unpleasant/negative, 0.5 = neutral, 1.0 = very pleasant/positive.
Return ONLY the number, no words.

CONTEXT (surrounding lines, for understanding only):
{context}

TARGET line to score:
{target}

Valence (0.0-1.0):"""


def load_transcript(path):
    with open(path) as f:
        d = json.load(f)
    segs = d["segments"]
    # normalize
    out = []
    for s in segs:
        out.append({"start": float(s["start"]), "end": float(s["end"]),
                    "text": (s.get("text") or "").strip()})
    return d.get("film"), d.get("source"), out


def build_context(segs, i, window):
    lo = max(0, i - window)
    hi = min(len(segs), i + window + 1)
    lines = []
    for j in range(lo, hi):
        tag = ">>> " if j == i else "    "
        lines.append(f"{tag}{segs[j]['text']}")
    return "\n".join(lines)


def call_gemini(model, prompt, retries=5):
    url = ENDPOINT.format(model=model) + f"?key={API_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 16,
                             "thinkingConfig": {"thinkingBudget": 0}},
    }
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read().decode())
            cand = resp.get("candidates", [{}])[0]
            parts = cand.get("content", {}).get("parts", [{}])
            txt = (parts[0].get("text") or "").strip() if parts else ""
            return parse_score(txt)
        except urllib.error.HTTPError as e:
            code = e.code
            if code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            msg = e.read().decode()[:200]
            print(f"  HTTP {code}: {msg}", file=sys.stderr)
            if code in (400, 404):
                raise SystemExit(f"Fatal API error {code}; check model id / key.")
            return None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"  network error: {e}", file=sys.stderr)
            return None
    return None


def parse_score(txt):
    if not txt:
        return None
    # grab first float-looking token
    import re
    m = re.search(r"[01](?:\.\d+)?|\.\d+", txt)
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    return max(0.0, min(1.0, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("out")
    ap.add_argument("grid_len", nargs="?", type=int, default=None)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    if not API_KEY:
        raise SystemExit("Set GEMINI_API_KEY in your environment first.")

    film, source, segs = load_transcript(args.transcript)
    if not segs:
        raise SystemExit("No segments in transcript.")

    grid_len = args.grid_len or int(math.ceil(max(s["end"] for s in segs)))

    # cache keyed by (model, window) so context runs never mix with isolation
    cache_path = args.cache or (os.path.splitext(args.out)[0] + "_cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("model") == args.model and rec.get("window") == args.window:
                        cache[rec["i"]] = rec["score"]
                except Exception:
                    pass

    scores = [None] * len(segs)
    cf = open(cache_path, "a")
    n_called = 0
    for i, seg in enumerate(segs):
        if i in cache and cache[i] is not None:
            scores[i] = cache[i]
            continue
        ctx = build_context(segs, i, args.window)
        prompt = PROMPT_TEMPLATE.format(context=ctx, target=seg["text"])
        sc = call_gemini(args.model, prompt)
        scores[i] = sc
        cf.write(json.dumps({"model": args.model, "window": args.window,
                             "i": i, "score": sc}) + "\n")
        cf.flush()
        n_called += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(segs)}] last={sc}")
        if args.sleep:
            time.sleep(args.sleep)
    cf.close()

    # build 1 Hz valence grid, last-write-wins on overlap, map 0..1 -> [-1,1]
    valence = [float("nan")] * grid_len
    for seg, sc in zip(segs, scores):
        if sc is None:
            continue
        v = 2.0 * sc - 1.0
        a = int(math.floor(seg["start"]))
        b = int(math.ceil(seg["end"]))
        for t in range(max(0, a), min(grid_len, b)):
            valence[t] = v

    out = {
        "film": film,
        "source": source,
        "fps": 1,
        "duration_s": grid_len,
        "valence": valence,
        "n_segments": len(segs),
        "n_scored": sum(1 for s in scores if s is not None),
        "model": args.model,
        "condition": f"context_w{args.window}",
        "window": args.window,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}: {out['n_scored']}/{len(segs)} segments scored, "
          f"grid {grid_len}s, {n_called} new API calls")


if __name__ == "__main__":
    main()
