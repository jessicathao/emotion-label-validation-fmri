#!/usr/bin/env python
"""extend_bert_context.py - extend BERT-context past the original 4 films to test whether the
dialogue-density ceiling holds. The 7 candidate films already have a filtered transcript
(<Film>_filt3.json) and an isolation signal on disk, so this builds ONLY the new window-3 context
signal from the SAME filt3, then scores existing-isolation (anchor) and new-context against the
consensus PleasantOther (+2s, correlate_v3). If a film's filt3 is missing it is rebuilt from the
SRT (srt_to_json -> filter@3). Local BERT inference for context only; no API."""
import subprocess, sys, os, re

HOME = os.path.expanduser("~")
SRT_DIR = f"{HOME}/emofilm/media/whisper_out"
WORK = f"{HOME}/emofilm/media"
SIG  = "data/transcripts"
ANN  = f"{HOME}/ds004872/derivatives/Annot_{{film}}_stim"
PY = sys.executable

FILMS = {
 "BetweenViewings":  "Between_Viewings_exp",
 "Chatter":          "Chatter_exp",
 "Spaceman":         "Spaceman_exp",
 "Superhero":        "Superhero_exp",
 "TheSecretNumber":  "The_secret_number_exp",
 "ToClaireFromSonny":"To_Claire_From_Sonny_exp",
 "YouAgain":         "You_Again_exp",
}

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True); return r.returncode, r.stdout, r.stderr

def score(sig_path, film):
    if not os.path.exists(sig_path): return None
    ann = ANN.format(film=film)
    rc, out, err = run([PY, "src/analysis/correlate_v3.py", sig_path, ann+".tsv.gz", ann+".json"])
    if rc != 0: return None
    m = re.search(r"r=([+-]?\d+\.\d+)", out); return float(m.group(1)) if m else None

def ensure_filt(film, base):
    filt = f"{WORK}/{film}_filt3.json"
    if os.path.exists(filt): return filt
    srt = f"{SRT_DIR}/{base}.srt"; raw = f"{WORK}/{film}_whisper.json"
    if not os.path.exists(srt): return None
    run([PY, "src/signals/srt_to_json.py", srt, film, raw])
    run([PY, "src/signals/filter_transcript.py", raw, filt, "3"])
    return filt if os.path.exists(filt) else None

print(f"{'film':18s} {'iso (anchor)':>13s} {'ctx(+/-3)':>10s} {'delta':>8s}  note")
print("-"*62)
rows = []
for film, base in FILMS.items():
    filt = ensure_filt(film, base)
    if not filt:
        print(f"{film:18s}  no filt3 and no SRT - skip"); continue
    iso_disk = f"{SIG}/{film}_bert.json"
    ctx = f"{WORK}/{film}_bert_context.json"
    if not os.path.exists(ctx):
        rc, out, err = run([PY, "src/signals/make_bert_signal.py", filt, ctx, "--window", "3"])
        if not os.path.exists(ctx):
            print(f"{film:18s}  context build FAILED: {(err.strip().splitlines() or [''])[-1]}"); continue
    r_iso = score(iso_disk, film); r_ctx = score(ctx, film)
    d = (r_ctx - r_iso) if (r_iso is not None and r_ctx is not None) else None
    def f(x): return f"{x:+.3f}" if x is not None else "  n/a"
    note = "" if d is None else ("up" if d > 0.03 else ("down" if d < -0.03 else "flat"))
    rows.append((film, r_iso, r_ctx, d))
    print(f"{film:18s} {f(r_iso):>13s} {f(r_ctx):>10s} {f(d):>8s}  {note}")

ups = [r for r in rows if r[3] is not None and r[3] > 0.03]
downs = [r for r in rows if r[3] is not None and r[3] < -0.03]
flats = [r for r in rows if r[3] is not None and abs(r[3]) <= 0.03]
print(f"\nsummary of {len(rows)} new films: {len(ups)} up, {len(downs)} down, {len(flats)} flat")
print("Directionless (mix of up/down/flat, no systematic gain) => dialogue-density ceiling holds")
print("for word-level BERT beyond the original four films. A systematic up would be the surprise.")
