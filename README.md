# emotion-label-validation-fmri

**Validating transformer-generated emotion labels against human consensus in naturalistic-film fMRI**

Brainhack School 2026 final project · in active development · not peer-reviewed
Author: Thach Thao Le (Jessica), PhD student, Graduate Institute of Linguistics, National Taiwan University

> Code and results are shared for course review and educational evaluation.
> Not licensed for reuse or redistribution pending publication. See NOTICE.

## Overview

Can transformer-generated emotion labels, derived from film dialogue, agree with the human
consensus annotation well enough to complement slow, costly hand annotation? This project
tests that on the **Emo-FilM** dataset (fMRI plus continuous emotion ratings collected while
participants watched short films), using a per-segment classifier (BERT) and an LLM
(Gemini 3.5 Flash), with a single-subject brain-decoding proof-of-concept.

## Findings (preliminary)

Across 12 naturalistic films, automatic sentiment labels agree with human valence only
weakly, and only where emotion is carried in explicit dialogue:

- **Per-segment BERT** reaches human-level agreement on one film only (Tears of Steel,
  r ≈ 0.36, the most lexically explicit); elsewhere near zero, with apparent negatives
  traceable to transcription artifacts (songs and credits mis-scored as dialogue).
- **A stronger model (Gemini 3.5 Flash)**, scoring the same isolated segments through the
  same pipeline, agrees somewhat more consistently (it shifts correlations positive and
  removes BERT's spurious negatives), but the gains are weak and absent on sparse-dialogue films.
- **A preliminary dialogue-context condition** (plus or minus three segments, four films)
  raises agreement on dialogue-driven films and stays flat on a dialogue-light one, a
  principled null. On Tears of Steel it reaches consensus-level agreement (r ≈ 0.59). This is
  a within-Gemini result (isolation vs context, same model); context was not run on BERT, so
  no context-based BERT-vs-Gemini claim is made.
- **Brain decoding (proof-of-concept):** human valence decodes from one subject's occipital
  BOLD (r ≈ 0.21), a valence-correlated visual signal. The fair human-versus-automatic
  comparison is underpowered, so nothing is claimed from it; the result is that the label and
  neural arms hit the same dialogue-density limit.

**Conclusion:** text-based sentiment, even with a frontier LLM, is not yet a robust substitute
for human emotion annotation in naturalistic film. It tracks human valence substantially only
where emotion is stated in the words. Results are reported as effect sizes with cross-model and
cross-film replication, not p-values: emotion annotations are so autocorrelated (~80 to 110 s)
that significance testing on single short films is infeasible.

Human reference levels are cited from Morgenroth et al. (PleasantOther inter-rater r ≈ 0.58,
the highest-agreement item; dataset-mean r ≈ 0.39, which the abstract rounds to 0.38). The
per-rater series are not in the public release, so these are cited from the paper, not
recomputed here. The 0.58 figure is pairwise inter-rater, a noisier yardstick than the averaged
consensus the models are scored against, so the context result above is "consensus-level on one
film," not "beating humans."

## Data

**Emo-FilM** (Morgenroth et al., 2025, *Scientific Data* 12:684), on OpenNeuro:
[ds004892](https://openneuro.org/datasets/ds004892) (fMRI) and
[ds004872](https://openneuro.org/datasets/ds004872) (annotations), both CC0. 14 short films,
30 participants, 44 annotators, 50 emotion items. The human valence signal is the appraisal
item *PleasantOther*, the highest-agreement item in the dataset.

## Method (brief)

Film dialogue is scored per segment (BERT P(positive); Gemini 0 to 1 valence), built into a
1 Hz valence signal, and correlated (Pearson r, +2 s offset) against the human *PleasantOther*
consensus. Each segment is scored on its own, so BERT and the LLM see identical input and the
only variable is the model; a context condition repeats this with a plus-or-minus-three-segment
window. A contamination check removes non-dialogue audio (songs, recited verse, credits) that
Whisper transcribes and sentiment models mis-score. A single-subject Ridge decode
(leave-one-block-out, contiguous CV) provides the brain proof-of-concept. Significance is
withheld by design; calibration shows the block bootstrap is far too liberal on single short
films given the annotation autocorrelation.

Full detail, expanded results, and statistics are reserved for the forthcoming paper.

## Repository layout
```
src/figures/        result and deck figures
src/signals/        per-segment BERT and Gemini valence signals, transcript prep
src/contamination/  find and remove transcription artifacts
src/analysis/       alignment, correlation, calibration, controls, brain decoding
data/               subtitles, processed transcripts, per-model 1 Hz signals
figures/ results/   generated figures, tabular outputs
```
Run scripts from the repo root, e.g. `python src/figures/make_delta_forest.py`. Large or
private data (brain images, film media) is gitignored.

## References
- Morgenroth, E., et al. (2025). Emo-FilM: A multimodal dataset for affective neuroscience
  using naturalistic stimuli. *Scientific Data, 12*, 684.
- Morgenroth, E., Moia, S., Vilaclara, L., Fournier, R., Muszynski, M., Ploumitsakou, M.,
  Almato-Bellavista, M., Vuilleumier, P., & Van De Ville, D. (2024). Emo-FilM: A multimodal
  dataset for affective neuroscience using naturalistic stimuli. *PsyArXiv*.
  https://doi.org/10.31234/osf.io/qzdbu

## Contact
Thach Thao Le (Jessica) · GitHub [@jessicathao](https://github.com/jessicathao) · Graduate
Institute of Linguistics, National Taiwan University
