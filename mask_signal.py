#!/usr/bin/env python3
"""mask_signal.py -- null out contaminated second-ranges from a signal file,
writing a *_clean.json that runs through correlate_v3 IDENTICALLY to any other
signal (so the resulting r/CI match the rest of the table's methodology).

The masked seconds become NaN, exactly how a no-dialogue second is represented,
so correlate_v3 drops them from the Pearson pairs just like real gaps.

Usage:
  python mask_signal.py IN.json OUT.json 526:596 703:709 787:805
  # ranges are START:END  (inclusive start, exclusive end -- same convention as
  #   spaceman_lyric_check.py CONTAM)

Then correlate exactly as for any signal:
  python src/correlate_v3.py OUT.json \\
      ~/ds004872/derivatives/Annot_<Film>_stim.tsv.gz \\
      ~/ds004872/derivatives/Annot_<Film>_stim.json 2
"""
import json, sys, math

if len(sys.argv) < 4:
    sys.exit("Usage: python mask_signal.py IN.json OUT.json START:END [START:END ...]")

in_path, out_path = sys.argv[1], sys.argv[2]
ranges = []
for r in sys.argv[3:]:
    a, b = r.split(":")
    ranges.append((int(a), int(b)))

d = json.load(open(in_path))
val = d["valence"]
N = len(val)

removed = 0
for a, b in ranges:
    for t in range(max(0, a), min(N, b)):
        if not (val[t] is None or (isinstance(val[t], float) and math.isnan(val[t]))):
            removed += 1
        val[t] = float("nan")

d["valence"] = val
d["masked_ranges"] = ranges
d["masked_from"] = in_path
json.dump(d, open(out_path, "w"))

cov = sum(1 for v in val if not (v is None or (isinstance(v, float) and math.isnan(v))))
print(f"masked {removed} dialogue-seconds across {len(ranges)} ranges {ranges}")
print(f"wrote {out_path}: coverage now {cov}s (was {cov + removed}s)")
print("NEXT: run correlate_v3 on the OUT file exactly as for any signal.")
