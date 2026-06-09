#!/usr/bin/env python3
"""
make_gemini_signal.py  --  BOUNDED EXPLORATORY LLM PROBE (Tears of Steel only)

Drop-in sibling of make_bert_signal.py. Scores ToS dialogue segments with
Gemini 2.5 Flash, ONE segment in isolation at a time (Option 1), to match
phanerozoic/BERT-Sentiment-Classifier's per-segment design EXACTLY, and emits a
signal in the IDENTICAL format make_bert_signal.py produces, so correlate_v3.py
consumes it unchanged and the r is directly comparable to BERT (+0.356 on ToS).

Scope (deliberate):
  - ONE film (ToS): the clean, subtitle-sourced, contamination-free positive.
  - DIALOGUE-ONLY input, ONE segment per call, NO context, NO neighbouring lines.
    (dialogue+context is a SEPARATE later experiment, not this.)
  - Clearly EXPLORATORY. Asks one question on the single film where word-level
    sentiment SUCCEEDED: does a context-aware model do comparably/better/worse
    under identical conditions?

Interface parity with make_bert_signal.py (verified against repo):
  * Input transcript: {"film":..., "segments":[{"start","end","text"}, ...]}.
  * BERT maps P(pos) in [0,1] -> valence in [-1,1] via  v = 2*p - 1.
    We map Gemini's 0..1 reply the SAME way: v = 2*g - 1.  (Pearson is
    scale-invariant, so this changes nothing statistically, but it keeps the
    two signals on a LITERALLY identical axis -- no surprises downstream.)
  * Grid: per-second list length N = ceil(max end) (or grid_len arg), NaN where
    no dialogue. Overlap handling is LAST-WRITE-WINS (valence[t] = v), exactly as
    make_bert_signal does -- NOT averaging.
  * Output dict keys identical to BERT's, plus a "source":"gemini-2.5-flash" tag.

Usage
-----
  export GEMINI_API_KEY=...                       # set in the environment
  conda activate lang_brain_project
  python make_gemini_signal.py \
      data/transcripts/TearsOfSteel.json \
      data/transcripts/TearsOfSteel_gemini.json \
      567 \
      --cache data/transcripts/TearsOfSteel_gemini_cache.jsonl \
      --sleep 4

  # then IDENTICAL to BERT (positional: signal, annotation tsv.gz, json sidecar, offset):
  python src/correlate_v3.py \
      data/transcripts/TearsOfSteel_gemini.json \
      ~/ds004872/derivatives/Annot_TearsOfSteel_stim.tsv.gz \
      <pleasantother_sidecar.json> 2
  # compare the printed r to BERT's +0.356.

  Pass the SAME grid_length_s used for TearsOfSteel_bert.json (its
  "duration_s" = 567) so the two grids line up second-for-second.
"""

import argparse
import json
import math
import os
import socket
import sys
import time
import urllib.request
import urllib.error

DEFAULT_MODEL = "gemini-2.5-flash"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def gemini_url(model):
    return f"{GEMINI_BASE}/{model}:generateContent"

# Single segment, no context, scalar valence. We ask on 0..1 (intuitive for the
# model) then map to BERT's [-1,1] axis in code, mirroring 2*P(pos)-1.
SYSTEM_INSTRUCTION = (
    "You are a sentiment valence rater. You will be given a SINGLE short snippet of "
    "film dialogue, with no surrounding context. Rate ONLY the emotional valence of "
    "that snippet's wording, as a single number between 0.00 and 1.00, where 0.00 = "
    "maximally negative/unpleasant, 0.50 = neutral, and 1.00 = maximally positive/"
    "pleasant. Judge the words themselves, not any imagined scene. Respond with the "
    "number ONLY, nothing else."
)


def call_gemini(text, api_key, model, thinking=False, max_out=None,
                max_retries=5, timeout=60):
    """One isolated segment -> raw 0..1 score (NaN on failure). Retries 429/5xx.

    Thinking behaviour:
      - thinking=False (default): send thinkingConfig.thinkingBudget=0 to suppress
        reasoning. Works for Flash-tier models, which otherwise spend a tiny output
        budget on hidden reasoning and return a candidate with NO 'parts'.
      - thinking=True: OMIT thinkingConfig entirely. Required for models that only
        run in thinking mode (e.g. gemini-3.1-pro-preview, which 400s on budget 0).
        We then need a larger maxOutputTokens so the visible number survives after
        the reasoning tokens are spent.
    Text is extracted defensively; finishReason is surfaced when nothing usable
    comes back, so failures are diagnosable instead of opaque."""
    gen = {
        "temperature": 0.0,
        "maxOutputTokens": max_out if max_out else (2048 if thinking else 32),
        "responseMimeType": "text/plain",
    }
    if not thinking:
        gen["thinkingConfig"] = {"thinkingBudget": 0}
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": gen,
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"{gemini_url(model)}?key={api_key}"
    backoff = 2.0
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            raw = extract_text(body)
            if raw is None:
                fr = (body.get("candidates", [{}])[0].get("finishReason", "?")
                      if body.get("candidates") else "no-candidates")
                return float("nan"), f"<no-text finishReason={fr}>"
            return parse_unit(raw), raw
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            if e.code in (400, 404):
                try:
                    msg = json.loads(e.read().decode()).get("error", {}).get("message", "")
                except Exception:
                    msg = ""
                hint = ""
                if "thinking" in msg.lower() or "budget" in msg.lower():
                    hint = "  -> this model requires thinking mode; rerun with --thinking"
                sys.exit(f"\nHTTP {e.code} from model '{model}': {msg}{hint}\n"
                         f"(check the model id with the /models curl, or add --thinking)")
            raise
        except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
            # transient network failure (read timeout, dropped connection): retry
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            return float("nan"), f"<network: {type(e).__name__}>"
    return float("nan"), "<exhausted-retries>"


def extract_text(body):
    """Pull the model's visible text out of a response, tolerating shape
    variation. Returns the concatenated text, or None if there is none."""
    cands = body.get("candidates")
    if not cands:
        return None
    parts = cands[0].get("content", {}).get("parts")
    if not parts:
        return None
    chunks = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
    joined = "".join(chunks).strip()
    return joined if joined else None


def parse_unit(raw):
    """Extract a float in [0,1] from the model reply (tolerates a 0-100 slip)."""
    s = raw.strip().split()[0] if raw.strip() else ""
    s = s.rstrip(".,")
    try:
        v = float(s)
    except ValueError:
        return float("nan")
    if 1.0 < v <= 100.0:
        v = v / 100.0
    return min(1.0, max(0.0, v))


def load_cache(path, model):
    """Only reuse scores produced by THE SAME model, so switching models can
    never silently mix two label sources into one signal. Legacy records with
    no 'model' field are treated as DEFAULT_MODEL (the original 2.5-flash run)."""
    cache = {}
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("model", DEFAULT_MODEL) != model:
                        continue
                    u = rec["unit"]
                    # Do NOT trust failed (NaN) entries: re-query them next run.
                    if isinstance(u, float) and math.isnan(u):
                        cache.pop(rec["text"], None)
                        continue
                    cache[rec["text"]] = u
    return cache


def append_cache(path, model, text, unit, raw):
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"model": model, "text": text,
                                "unit": unit, "raw": raw}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", help="transcript json: {'segments':[{start,end,text}]}")
    ap.add_argument("out", help="output signal json (BERT-identical format)")
    ap.add_argument("grid_len", nargs="?", type=int, default=None,
                    help="grid length in s; pass TearsOfSteel_bert.json's duration_s (567)")
    ap.add_argument("--cache", default=None, help="jsonl cache (resume-safe, avoids re-billing)")
    ap.add_argument("--sleep", type=float, default=0.0, help="s between calls (RPM throttle)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Gemini model id (default {DEFAULT_MODEL}). Copy the EXACT id "
                         "from AI Studio, e.g. gemini-3.5-flash. Cache is keyed by model, "
                         "so switching models will NOT mix label sources in one signal.")
    ap.add_argument("--thinking", action="store_true",
                    help="OMIT thinkingBudget=0 (let the model reason). REQUIRED for "
                         "thinking-only models like gemini-3.1-pro-preview, which reject "
                         "budget 0. Uses a larger output budget so the number survives.")
    ap.add_argument("--max-out", type=int, default=None, dest="max_out",
                    help="override maxOutputTokens (default 32 normal / 2048 thinking)")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set in environment.")

    data = json.load(open(args.transcript))
    assert "segments" in data, "transcript must have a top-level 'segments' list"
    segs = data["segments"]
    assert segs and all(k in segs[0] for k in ("start", "end", "text")), \
        "each segment needs start/end/text"
    print(f"{data.get('film','?')}: {len(segs)} segments, "
          f"span {segs[0]['start']:.1f}-{segs[-1]['end']:.1f}s "
          f"| model={args.model}, isolation=per-segment")

    cache = load_cache(args.cache, args.model)
    units = []
    for i, s in enumerate(segs):
        text = s["text"].strip()
        if not text:
            units.append(float("nan"))
            continue
        if text in cache:
            units.append(cache[text])
            continue
        u, raw = call_gemini(text, api_key, args.model,
                             thinking=args.thinking, max_out=args.max_out)
        units.append(u)
        append_cache(args.cache, args.model, text, u, raw)
        if (i + 1) % 25 == 0 or i == len(segs) - 1:
            print(f"  scored {i+1}/{len(segs)}")
        if args.sleep:
            time.sleep(args.sleep)

    # Map 0..1 -> [-1,1] exactly like BERT's 2*P(pos)-1.
    seg_val = [(2.0 * u - 1.0) if not (isinstance(u, float) and math.isnan(u))
               else float("nan") for u in units]

    # Grid build: last-write-wins, NaN gaps -- IDENTICAL to make_bert_signal.
    end_time = max(s["end"] for s in segs)
    N = args.grid_len if args.grid_len else int(math.ceil(end_time))
    valence = [float("nan")] * N
    for s, v in zip(segs, seg_val):
        if isinstance(v, float) and math.isnan(v):
            continue
        a = int(math.floor(s["start"])); b = int(math.ceil(s["end"]))
        for t in range(max(0, a), min(N, b)):
            valence[t] = v
    coverage = sum(1 for v in valence if not math.isnan(v))

    out = {"film": data.get("film", "?"), "fps": 1, "duration_s": N,
           "valence": valence, "n_segments": len(segs), "coverage_s": coverage,
           "subtitle_offset_s": 0, "source": args.model}
    json.dump(out, open(args.out, "w"))

    print(f"grid length: {N}s, dialogue coverage: {coverage}s ({100*coverage/N:.1f}% of grid)")
    valid = [u for u in units if not (isinstance(u, float) and math.isnan(u))]
    if valid:
        print(f"per-segment unit score: mean={sum(valid)/len(valid):.2f}, "
              f"min={min(valid):.2f}, max={max(valid):.2f}, "
              f"n_valid={len(valid)}/{len(segs)}")
    print("saved:", args.out)
    print("NEXT: run correlate_v3 on this file exactly as for *_bert.json "
          "(+2s offset, PleasantOther). Compare r to BERT +0.356.")


if __name__ == "__main__":
    main()
