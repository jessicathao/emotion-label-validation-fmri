# Brain arm: occipital valence decoding (controlled null, two films)

Occipital decoding of human PleasantOther valence is **not estimable above chance** on
Emo-FilM, single-subject or pooled across five subjects, and the null **replicates on two
films**. We prove it with a planted-signal positive control that the estimator passes
while the real decode does not.

## Result (two-film controlled null)

| film | character | positive control (~100s) | real mean | shift-null | verdict |
|---|---|---|---|---|---|
| Payload | dialogue-dense | +0.276 (pass) | +0.033 | +0.044 +/- 0.013 | null |
| Tears of Steel | dialogue-light, visual | +0.249 (pass) | +0.089 | +0.127 +/- 0.042 | null |

On both films the pooled leave-one-subject-out estimator recovers a planted signal at the
valence timescale (positive control passes) while the real human-valence decode sits inside
its own circular-shift null. Five lowest-motion subjects per film, pre-registered by motion;
sub-S07 (the retracted PoC subject) is excluded from both.

**The elevated Tears of Steel null is itself a finding.** Its shift-null sits at +0.127 (vs
Payload's +0.044) and is wider, because ToS is visually intense and dialogue-light: occipital
BOLD carries strong stimulus-locked visual structure that any film-autocorrelated target, even
a brain-decoupled shifted one, partly correlates with. A naive test against zero would have
called the real +0.089 a positive hit. The circular-shift null catches it. This is the
effect-sizes-not-p-values thesis reaching the brain arm from the neural side.

Single-subject estimates are unstable and non-replicating (within-fold Pearson): S07 +0.21
(highest motion), S31 -0.16 (cleanest), opposite signs on the SAME film; a planted ~100s
signal fails to recover at the single-subject level (+0.15 at SNR 2.0). Pooling is what
validated the estimator.

Figures: `figures/brain_decode_two_film_null.png` (canonical two-film result),
`figures/brain_decode_single_subject_contrast.png` (the retracted PoC shown for contrast,
self-correction only). Numbers: `results/brain_decode_Payload.json`,
`results/brain_decode_TearsOfSteel.json`.

## Why this estimator (metric history)

The cross-validated point estimate is itself destabilised by the ~80-110s autocorrelation of
the valence target against short films. Two earlier metrics were rejected before the canonical
one:

1. **Pooled predicted-vs-true r under blocked CV** (the original PoC metric): grows more
   negative as Ridge alpha rises, the fingerprint of between-block drift dominating the pooled
   correlation.
2. **High-pass (1/128 Hz) drift control**: removes the drift but strips ~79% of the target
   variance (target autocorr ~82s vs a 128s filter period) and makes planted-signal recovery
   timescale-dependent. The metric distorts the answer.
3. **Canonical**: within-fold Pearson (single subject) and leave-one-subject-out Ridge
   (pooled). Only this scheme passes the planted-signal positive control, with a circular-shift
   null centered near zero.

## Method

- Mask: posterior MNI gray matter, y < -60 mm (occipital), 43683 voxels, in MNI space so it
  transfers identically across subjects.
- Target: PleasantOther consensus (annotation column 3), SPM double-gamma HRF, resampled to the
  TR (1.3 s) grid, placed at the events-derived film onset.
- Subjects: the five lowest-motion runs of the target film, **pre-registered by mriqc `fd_mean`
  before any decode**. sub-S07 (the earlier single-subject PoC subject) is excluded.
- Estimator: Ridge (alpha 10000; decode is alpha-stable). Pooled = leave-one-subject-out
  (train on 4, predict the 5th from their own occipital pattern, score within subject). This
  asks whether a cross-subject decoder generalises to a new person.
- Controls: a planted-signal positive control (inject a known timecourse, recover it) and a
  circular-shift null (shift the target, breaking the brain link while keeping its
  autocorrelation). No real number is reported until the positive control passes.

## Scope

Occipital mask means valence-CORRELATED VISUAL signal, not pure affect; one ROI, one target.
This is **not** a claim that the brain lacks valence information. A real test needs longer or
repeated stimuli, an affective-ROI mask (insula / vmPFC / amygdala), or many more subjects,
each with the planted-signal positive control attached. Significance is withheld project-wide;
descriptive effect sizes only.

## Reproduce

```bash
conda activate lang_brain_project   # numpy, pandas, nibabel, nilearn, scikit-learn, scipy, matplotlib

# 1. fetch + cache the occipital matrices (one bold at a time, dropped after extraction)
python src/analysis/fetch_extract_cache.py Payload      sub-S08 sub-S05 sub-S11 sub-S15 sub-S06
python src/analysis/fetch_extract_cache.py TearsOfSteel sub-S31 sub-S08 sub-S11 sub-S25 sub-S04

# 2. single-subject decode + positive control (documented to fail at the valence timescale)
python src/analysis/decode_brain.py single Payload sub-S08

# 3. pooled leave-one-subject-out + both gates; writes results/brain_decode_<film>.json
python src/analysis/decode_brain.py pooled Payload      sub-S08 sub-S05 sub-S11 sub-S15 sub-S06
python src/analysis/decode_brain.py pooled TearsOfSteel sub-S31 sub-S08 sub-S11 sub-S25 sub-S04

# 4. build the canonical two-film figure (and the single-subject self-correction contrast)
python src/analysis/make_brain_two_film_figure.py
python src/analysis/make_single_subject_contrast.py
```

Data: ds004892 (fMRI) and ds004872 (annotations) at `~/ds004892` and `~/ds004872` (datalad;
`get` one bold at a time, the full set is 131 GB). The cached `.npz` matrices are all the
pooled analyses and the figures need; the bolds can stay dropped.

## Note on older scripts

`decode_valence_poc.py` (single-subject PoC, June 7) and `decode_valence_multi.py`
(multi-subject driver) are kept for provenance, but their **raw pooled predicted-vs-true r is
superseded** (it is the drift-prone metric #1 above). Use `decode_brain.py` and its within-fold
/ leave-one-subject-out estimator with the positive control attached.
