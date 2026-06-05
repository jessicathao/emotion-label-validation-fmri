#!/usr/bin/env python3
"""
Round-trip pipeline validation control.
Tests whether the expansion + offset + bootstrap machinery preserves a REAL
signal, by pushing the human PleasantOther annotation through the same steps
the BERT signal went through, restricted to the same dialogue-second mask.

Outputs per film:
  - roundtrip r        : PleasantOther(+2s) vs PleasantOther(0s) on dialogue mask
  - sweep peak offset  : offset that maximises human self-correlation (should be 0)
  - dialogue coverage  : fraction of film-seconds with dialogue
  - n blocks (20s)     : independent blocks available -> CI reliability
  - PleasantOther idx  : confirms column 3 read
"""
import json, gzip, glob, os
import numpy as np

ANNOT_DIR = os.path.expanduser("~/ds004872/derivatives")
BERT_DIR  = os.path.expanduser("~/lang_brain_project/data/transcripts")
PLEASANT_OTHER_IDX = 3          # zero-indexed column from the JSON sidecar
OFFSET = 2                       # the fixed +2s used in the main analysis
SWEEP = range(-10, 31)           # match align_correlate.py sweep
BLOCK = 20

def load_annotation(film):
    """Read headerless TSV, return PleasantOther column as float array."""
    path = os.path.join(ANNOT_DIR, f"Annot_{film}_stim.tsv.gz")
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append(float(parts[PLEASANT_OTHER_IDX]))
    return np.array(rows)

def load_dialogue_mask(film):
    """Dialogue seconds = where BERT valence is not NaN."""
    path = os.path.join(BERT_DIR, f"{film}_bert.json")
    d = json.load(open(path))
    v = np.array(d["valence"], dtype=float)   # NaN preserved
    return ~np.isnan(v), len(v)

def shift(x, k):
    """Shift by k seconds; pad with NaN so length is preserved."""
    out = np.full_like(x, np.nan, dtype=float)
    if k == 0:
        out[:] = x
    elif k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out

def corr_on_mask(a, b, mask):
    """Pearson r on positions where mask True and both finite."""
    sel = mask & np.isfinite(a) & np.isfinite(b)
    if sel.sum() < 3:
        return np.nan, int(sel.sum())
    return np.corrcoef(a[sel], b[sel])[0, 1], int(sel.sum())

def main():
    films = sorted(
        os.path.basename(p).replace("_bert.json", "")
        for p in glob.glob(os.path.join(BERT_DIR, "*_bert.json"))
    )
    print(f"{'Film':<18}{'rt_r(+2s)':>10}{'peak_off':>9}{'cover':>7}{'n_blk':>7}{'n_sec':>7}")
    print("-" * 58)
    for film in films:
        annot_path = os.path.join(ANNOT_DIR, f"Annot_{film}_stim.tsv.gz")
        if not os.path.exists(annot_path):
            print(f"{film:<18}  (no annotation file)")
            continue
        po = load_annotation(film)
        mask, n_bert = load_dialogue_mask(film)
        n = min(len(po), n_bert)
        po, mask = po[:n], mask[:n]

        # round-trip at +2s
        rt_r, n_sec = corr_on_mask(shift(po, OFFSET), po, mask)

        # sweep: where does human self-correlation peak?
        best_r, best_off = -2, None
        for k in SWEEP:
            r, _ = corr_on_mask(shift(po, k), po, mask)
            if np.isfinite(r) and r > best_r:
                best_r, best_off = r, k

        coverage = mask.sum() / n
        n_blocks = int(np.floor(mask.sum() / BLOCK))

        print(f"{film:<18}{rt_r:>10.3f}{best_off:>9d}{coverage:>7.2f}{n_blocks:>7d}{n_sec:>7d}")

if __name__ == "__main__":
    main()
