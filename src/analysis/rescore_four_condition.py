#!/usr/bin/env python
"""rescore_four_condition.py - verify-first re-score of the text arm from existing signals.
Calls correlate_v3.py unchanged (consensus PleasantOther, +2s) on the four context films across
BERT-iso / BERT-ctx / Gemini-iso / Gemini-ctx, and on the three isolation-calibration films.
Prints each r beside its canonical value with OK / DRIFT. No API, no transcripts."""
import subprocess, re, sys, os

ANN = os.path.expanduser("~/ds004872/derivatives/Annot_{film}_stim")
SIG = "data/transcripts/{film}_{cond}.json"
COND = {"bert":"bert", "bertctx":"bert_context", "gem":"gemini", "gemctx":"gemini_context"}

CTX_CANON = {  # film: (bert-iso, bert-ctx, gem-iso, gem-ctx)
 "LessonLearned": (-0.085, -0.056, +0.112, +0.225),
 "Payload":       (+0.140, -0.118, +0.085, +0.321),
 "TearsOfSteel":  (+0.356, +0.513, +0.467, +0.591),
 "AfterTheRain":  (-0.197, -0.217, -0.075, -0.080),
}
ISO_CANON = {"AfterTheRain": -0.197, "LessonLearned": -0.085, "Payload": +0.140}

def score(film, cond):
    sig = SIG.format(film=film, cond=COND[cond])
    ann = ANN.format(film=film)
    if not os.path.exists(sig): return None, f"missing {sig}"
    out = subprocess.run([sys.executable, "src/analysis/correlate_v3.py",
                          sig, ann+".tsv.gz", ann+".json"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return None, (out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "error")
    m = re.search(r"r=([+-]?\d+\.\d+)", out.stdout)
    return (float(m.group(1)), out.stdout.strip()) if m else (None, out.stdout.strip())

def flag(got, canon, tol=0.005):
    return "OK" if (got is not None and abs(got-canon) <= tol) else "DRIFT"

print("=== ISOLATION CALIBRATION (BERT-iso vs canonical) ===")
for film, canon in ISO_CANON.items():
    got, _ = score(film, "bert")
    g = f"{got:+.3f}" if got is not None else "  n/a"
    print(f"  {film:14s} BERT-iso  got {g}  canon {canon:+.3f}   {flag(got,canon)}")

print("\n=== FOUR-CONDITION TABLE (got / canon) ===")
hdr = f"{'film':14s} {'BERT-iso':>20s} {'BERT-ctx':>20s} {'Gem-iso':>20s} {'Gem-ctx':>20s}"
print(hdr); print("-"*len(hdr))
order = ["bert","bertctx","gem","gemctx"]
drift = 0
for film,(cb,cbx,cg,cgx) in CTX_CANON.items():
    canon = {"bert":cb,"bertctx":cbx,"gem":cg,"gemctx":cgx}
    cells = []
    for cond in order:
        got,_ = score(film, cond)
        f = flag(got, canon[cond]); drift += (f=="DRIFT")
        g = f"{got:+.3f}" if got is not None else " n/a "
        cells.append(f"{g}/{canon[cond]:+.3f} {f}")
    print(f"{film:14s} " + " ".join(f"{c:>20s}" for c in cells))
print(f"\n{'ALL OK - signals reproduce canonical' if drift==0 else str(drift)+' cell(s) DRIFT - investigate before extending'}")
