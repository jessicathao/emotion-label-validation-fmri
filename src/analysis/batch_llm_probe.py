#!/usr/bin/env python3
"""
batch_llm_probe.py  --  run the Gemini 3.5 Flash probe + correlate_v3 across all
dialogue films, and print one BERT-vs-Gemini comparison table.

What it does, per film:
  1. Score that film's _filt3.json transcript with Gemini 3.5 Flash (the EXACT
     transcript BERT used; confirmed n_segments-matched). Per-segment isolation,
     same as the ToS/Sintel runs. Cached + resume-safe.
  2. Run correlate_v3 on BOTH the BERT signal and the new Gemini signal, against
     the same PleasantOther annotation, +2s offset.
  3. Parse the r and CI from each and collect into a table.

Design decisions baked in (from the project's own methodology):
  - Model = gemini-3.5-flash, NO thinking. (Pro was checked on Sintel only and
    confirmed the null is model-independent; Pro adds nothing to the batch.)
  - Effect sizes only. The table reports r and a DESCRIPTIVE CI. No significance
    verdict (June-5 calibration: the bootstrap is too liberal here to test sig).
  - Contaminated films (Whisper songs/verse scored as dialogue) are TAGGED, not
    dropped. BERT read the same contaminated _filt3 transcript, so the model-vs-
    model comparison is still fair; the tag just flags "don't over-read absolute r."
  - Non-verbal films (FirstBite, BigBuckBunny) are OUT OF SCOPE for an isolation
    probe (no dialogue to read) and are not run here. They need the separate
    context experiment.

This wrapper SHELLS OUT to the existing scripts rather than importing them, so it
uses the identical, already-validated code paths (no reimplementation risk).

Usage
-----
  conda activate lang_brain_project
  cd ~/emotion-label-validation-fmri
  python batch_llm_probe.py
  # add --dry-run to print the plan without calling the API
  # add --skip-probe to ONLY re-run correlations on existing _gemini.json files
"""

import argparse
import json
import os
import re
import subprocess
import sys

HOME = os.path.expanduser("~")
MEDIA = os.path.join(HOME, "emofilm", "media")            # *_filt3.json, *_bert.json live here
ANNOT = os.path.join(HOME, "ds004872", "derivatives")     # Annot_<Film>_stim.{tsv.gz,json}
OUTDIR = "data/transcripts"                                # where _gemini.json signals go
PROBE = "src/signals/make_gemini_signal.py"
CORRELATE = "src/analysis/correlate_v3.py"
MODEL = "gemini-3.5-flash"
OFFSET = "2"

# 10 dialogue films (all confirmed _filt3 n_seg == bert n_seg).
FILMS = ["AfterTheRain", "BetweenViewings", "Chatter", "LessonLearned", "Payload",
         "Spaceman", "Superhero", "TheSecretNumber", "ToClaireFromSonny", "YouAgain"]

# Per project notes (June 5 transcript skim): Whisper transcribed songs/verse/
# sound-effects as dialogue in these. Both BERT and Gemini read the same _filt3,
# so model-vs-model is fair; tag so absolute r is not over-read.
CONTAMINATED = {"Spaceman", "YouAgain", "Chatter", "ToClaireFromSonny", "Superhero"}

# Already done earlier in the session (clean, subtitle-sourced); included in the
# final table for completeness. Their _gemini.json already exist in OUTDIR.
PRESCORED = ["TearsOfSteel", "Sintel"]

R_RE = re.compile(r"r=([+-]?\d+\.\d+)\s+95%CI(\[[^\]]+\])\s+n=(\d+)\s+realblk=(\d+)")


def run_probe(film, dry):
    transcript = os.path.join(MEDIA, f"{film}_filt3.json")
    out = os.path.join(OUTDIR, f"{film}_gemini.json")
    cache = os.path.join(OUTDIR, f"{film}_gemini_cache.jsonl")
    cmd = [sys.executable, PROBE, transcript, out, bert_grid_len(film),
           "--cache", cache, "--model", MODEL]
    if dry:
        print("  PROBE:", " ".join(cmd))
        return out
    print(f"  scoring {film} with {MODEL} ...")
    subprocess.run(cmd, check=True)
    return out


def run_correlate(signal_path, film):
    tsv = os.path.join(ANNOT, f"Annot_{film}_stim.tsv.gz")
    sidecar = os.path.join(ANNOT, f"Annot_{film}_stim.json")
    res = subprocess.run([sys.executable, CORRELATE, signal_path, tsv, sidecar, OFFSET],
                         capture_output=True, text=True, check=True)
    line = res.stdout.strip()
    m = R_RE.search(line)
    if not m:
        return None
    return {"r": float(m.group(1)), "ci": m.group(2),
            "n": int(m.group(3)), "blk": int(m.group(4))}


def find_bert_signal(film):
    """BERT signals are split: ToS/Sintel in data/transcripts/, the rest in
    ~/emofilm/media/. Search both, prefer the repo copy. Returns path or None."""
    for cand in (os.path.join(OUTDIR, f"{film}_bert.json"),
                 os.path.join(MEDIA, f"{film}_bert.json")):
        if os.path.exists(cand):
            return cand
    return None


def signal_path_for(film):
    """BERT signal: wherever it actually is. Gemini signal: always OUTDIR."""
    return {
        "bert": find_bert_signal(film),
        "gemini": os.path.join(OUTDIR, f"{film}_gemini.json"),
    }


def bert_grid_len(film):
    """Read the film's BERT signal duration_s so the Gemini grid matches it."""
    bp = find_bert_signal(film)
    if bp is None:
        sys.exit(f"[fatal] no BERT signal found for {film} in {OUTDIR} or {MEDIA}")
    with open(bp) as f:
        return str(json.load(f)["duration_s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the commands; no API calls")
    ap.add_argument("--skip-probe", action="store_true",
                    help="don't call Gemini; only correlate existing _gemini.json")
    args = ap.parse_args()

    rows = []

    if args.dry_run:
        print("[dry-run] plan (no API calls, no correlations):\n")
        print(f"  prescored films (already have _gemini.json): {', '.join(PRESCORED)}")
        for film in FILMS:
            run_probe(film, dry=True)
        print(f"\n  contaminated (tagged in table): {', '.join(sorted(CONTAMINATED))}")
        print("  then correlate BERT + Gemini for all "
              f"{len(PRESCORED) + len(FILMS)} films -> comparison table.")
        return

    # 1) the two clean films already scored this session
    for film in PRESCORED:
        gem = os.path.join(OUTDIR, f"{film}_gemini.json")
        if not os.path.exists(gem):
            print(f"[warn] {film}_gemini.json not found, skipping in table")
            continue
        rows.append((film, run_correlate(signal_path_for(film)["bert"], film),
                     run_correlate(gem, film)))

    # 2) the 10 dialogue films
    failed = []
    for film in FILMS:
        try:
            if not args.skip_probe:
                run_probe(film, dry=False)
            gem = os.path.join(OUTDIR, f"{film}_gemini.json")
            rows.append((film, run_correlate(signal_path_for(film)["bert"], film),
                         run_correlate(gem, film)))
        except subprocess.CalledProcessError:
            print(f"  [skip] {film}: probe/correlate failed (likely transient). "
                  f"Cache kept; re-run later to resume.")
            failed.append(film)
            continue

    # 3) the table
    print("\n" + "=" * 92)
    print("BERT vs Gemini-3.5-Flash  --  PleasantOther, +2s offset, EFFECT SIZES "
          "(descriptive CI; sig withdrawn)")
    print("=" * 92)
    hdr = f"{'Film':18s} {'BERT r':>8s} {'BERT CI':>18s} | {'Gemini r':>9s} {'Gemini CI':>18s}  {'n':>4s} {'blk':>3s}  flag"
    print(hdr)
    print("-" * len(hdr))
    for film, b, g in rows:
        if b is None or g is None:
            print(f"{film:18s}  (correlate parse failed)")
            continue
        flag = "CONTAM" if film in CONTAMINATED else ("clean" if film in PRESCORED else "")
        delta = g["r"] - b["r"]
        print(f"{film:18s} {b['r']:+8.3f} {b['ci']:>18s} | {g['r']:+9.3f} {g['ci']:>18s}  "
              f"{g['n']:>4d} {g['blk']:>3d}  {flag}  (d={delta:+.3f})")
    print("-" * len(hdr))
    print("d = Gemini r - BERT r. CONTAM = Whisper sang/verse-as-dialogue (per June-5 "
          "skim);\nboth models read the same _filt3 transcript, so d is still a clean "
          "model effect,\nbut absolute r on those rows is not a pure dialogue signal.")
    print("Non-verbal films (FirstBite, BigBuckBunny) omitted: no dialogue for an "
          "isolation probe.")
    if failed:
        print(f"\n[!] SKIPPED (re-run to finish): {', '.join(failed)}  "
              f"-- cache preserved, resumes where it stopped.")


if __name__ == "__main__":
    main()
