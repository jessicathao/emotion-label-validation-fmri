# emotion-label-validation-fmri

**Validating transformer-generated emotion labels against human consensus in naturalistic film fMRI**

Status: In active development (Brainhack School 2026 final project)
Author: Thach Thao Le (Jessica), PhD student, Linguistics, National Taiwan University

## Project overview

This project tests whether emotion labels generated automatically by transformer
language models agree with human consensus annotations, using the **Emo-FilM**
dataset: fMRI and emotion ratings collected while participants watched short films.

The central question: **can an automatic NLP method label the emotional valence of
film dialogue well enough to complement, and help scale, slow and costly human
annotation?**

### Research questions
1. **Agreement (preliminary):** Do automatic sentiment labels, derived from film
   dialogue, correlate with the human consensus annotation? Tested for a
   per-segment classifier (BERT) and an LLM (Gemini 3.5 Flash)
   across 12 films, with three sentiment models on the cleanest film, plus a
   preliminary dialogue-context manipulation on four films.
2. **Brain decoding (controlled null):** Can those labels decode emotional
   valence from the fMRI BOLD signal? A powered, multi-subject occipital decode has been
   run on two films. It does not recover human valence above chance, and
   a planted-signal positive control confirms the estimator would have detected a
   signal if one were present, so the result is a controlled null, not a missing
   measurement.

## Findings so far

*Preliminary: Brainhack School 2026 project.*

Across 12 naturalistic films, automatic sentiment labels agree with human valence
only weakly, and only where emotion is carried in explicit dialogue:

- **Per-segment BERT** reaches human-level agreement on one film only (Tears of
  Steel, r ≈ 0.36, the most lexically explicit); elsewhere near zero, with apparent
  negatives traceable to transcription artifacts (songs and credits mis-scored as dialogue).
- **A stronger model (Gemini 3.5 Flash)**, scoring the same isolated segments through
  the same pipeline, agrees somewhat more consistently (it shifts correlations positive
  and removes BERT's spurious negatives), but the gains are weak and absent on
  sparse-dialogue films.
- **A preliminary dialogue-context condition** (plus or minus three segments, four films)
  raises agreement on dialogue-driven films and stays flat on a dialogue-light one, a
  principled null; on Tears of Steel it reaches consensus-level agreement (r ≈ 0.59). This
  is a within-Gemini result (isolation vs context, same model); context was not run on
  BERT, so no context-based BERT-vs-Gemini claim is made.
- **Reference levels** (cited from Morgenroth et al. 2025): PleasantOther inter-rater
  r ≈ 0.58 (the highest-agreement item), dataset-mean r ≈ 0.39. The 0.58 is pairwise
  inter-rater, a noisier yardstick than the averaged consensus the models are scored
  against, so the context result is "consensus-level on one film," not "beating humans."
- **Brain decoding (controlled null, two films):** a powered leave-one-subject-out
  decode of human PleasantOther valence from posterior occipital BOLD reads at chance
  on both a dialogue-dense film (Payload: real r = +0.033 vs a shift-null of
  +0.044 ± 0.013) and a dialogue-light one (Tears of Steel: real r = +0.089 vs a
  shift-null of +0.127 ± 0.042). A planted-signal positive control passes on both
  films (recovery r = +0.276 and +0.249 at the valence timescale), so the estimator
  can detect a signal it should detect; the real valence signal is simply not there.
  The earlier single-subject r ≈ 0.21 is retracted: it came from the highest-motion
  run in the dataset and reverses in sign (−0.16) on the cleanest subject of the same
  film. The elevated Tears of Steel chance level (≈ +0.13) is itself a finding:
  occipital BOLD is so visually structured that a naive test against zero would have
  called the real +0.089 a hit, which is why the decode is scored against a shift-null,
  not zero. Scope is bounded to one ROI (posterior occipital) and one target
  (PleasantOther); this is not a claim that the brain lacks valence information. The
  label and neural arms converge on the same ~80 to 110 s annotation-autocorrelation
  limit, reached here from the neural side.

Conclusion: text-based sentiment, even with a frontier LLM, is not yet a robust
substitute for human emotion annotation in naturalistic film. It tracks human valence
substantially only where emotion is stated in the words. Results are reported as effect
sizes with cross-model and cross-film replication, not p-values, because the annotation
autocorrelation (~80 to 110 s) makes significance testing on single short films infeasible.

## Why it matters
- **Label provenance:** any decoder is only as good as its labels; where automatic labels
  fail is understudied.
- **Scale and reach:** reliable automatic labels would open many unlabeled naturalistic
  datasets for affective neuroscience, and the method extends to cross-linguistic data,
  including under-represented languages such as Vietnamese.

This measures emotion in film *via language* (spoken dialogue), not language processing in
isolation, a deliberate and scoped first test.

## Data

**Emo-FilM** (Morgenroth et al., 2025, *Scientific Data* 12:684), on OpenNeuro:
- fMRI: [ds004892](https://openneuro.org/datasets/ds004892) (CC0)
- Annotations: [ds004872](https://openneuro.org/datasets/ds004872) (CC0)
- 14 short films, 30 participants (3 Tesla), 44 annotators, 50 emotion items.
- TR = 1.3 s. Preprocessed BOLD available in MNI space.

The **human valence signal** is taken from the appraisal item *PleasantOther* (the
highest-agreement item in the dataset, inter-rater r ≈ 0.58).

## Method

```
Film dialogue        per-segment              valence (1 Hz)
(.srt / Whisper) ──► sentiment score    ──►   NaN where silent ──┐
                     BERT  (P(pos))                              │
                     Gemini (0–1 valence)                        ▼
                                              Pearson r  ◄── Human consensus
                                              per film,       (PleasantOther, 1 Hz)
                                              +2 s offset,
                                              real-timeline
                                              block-bootstrap CI
                                                                 │
                                   (controlled null) ─────────────┘
                              decode valence from BOLD (leave-one-subject-out,
                              5 subjects/film, planted-signal positive control
                              + circular-shift null), two films
```

Each dialogue segment is scored **on its own** (without the surrounding dialogue
segments), so the per-segment classifier and the LLM see identical input and the only
variable is the model; a context condition repeats this with a plus-or-minus-three-segment
window as the only change. Agreement is reported as an effect size with a descriptive 95%
bootstrap CI; significance is withheld because calibration (`bootstrap_calibration.py`) shows
the block bootstrap is far too liberal on single short films, given that the annotation
autocorrelation time (~80 to 110 s) dwarfs any usable block size. Human reference levels are
the values reported in Morgenroth et al. 2025 (the per-rater series are not in the public
derivatives, so they are cited from the paper, not recomputed here). A contamination check
removes non-dialogue audio (songs, recited verse, credit tails) that Whisper transcribes and
sentiment models then mis-score; on Spaceman this collapses BERT's phantom negative toward zero.

For the brain arm, valence is decoded from posterior occipital BOLD with ridge
regression under leave-one-subject-out cross-validation across five subjects per film;
a planted-signal positive control verifies the estimator recovers a known signal at the
valence timescale, and a circular-shift null sets the chance level. Reported on two
films (Payload and Tears of Steel).

## Technical stack

| Component | Tool |
|---|---|
| fMRI data handling | nilearn, nibabel |
| Per-segment sentiment | Hugging Face Transformers (`phanerozoic/BERT-Sentiment-Classifier`; also DistilBERT/SST-2, SiEBERT/RoBERTa-large for robustness) |
| LLM (Gemini) | Gemini API (`gemini-3.5-flash`; `gemini-3.1-pro-preview` for a model-strength check) |
| Machine learning | scikit-learn (Ridge decoding; SVM from coursework) |
| Transcription | OpenAI Whisper (films without official subtitles) |
| Plotting | matplotlib |
| Environment | Conda (Python 3.11), macOS Apple Silicon |
| Version control | Git / GitHub |

## Repository layout
```
src/
  figures/        scripts that build the deck and result figures
  signals/        build the per-segment BERT and Gemini valence signals + transcript prep
  contamination/  find and remove transcription artifacts (songs, recited verse, credits)
  analysis/       alignment, correlation, calibration, controls, brain decoding
data/
  subs/          film subtitle files (.srt)
  transcripts/   processed transcripts + per-model 1 Hz signals (.json)
figures/         all generated figures (.png / .pdf)
results/ docs/ notebooks/   tabular outputs, notes, exploration
```
Run scripts from the repo root, e.g. `python src/figures/make_delta_forest.py`.
Large/private data (brain images, film media) and resume caches are gitignored.

## Progress
- [x] Project proposal and design (pitched June 2026)
- [x] Dataset selection and fMRI structure explored (MNI preprocessed, TR 1.3 s)
- [x] PleasantOther human valence signal extracted from ds004872
- [x] Subtitle/film-time alignment resolved (+2 s offset, 1 Hz grid)
- [x] Transcripts for all 12 dialogue films (official .srt where available, else Whisper)
- [x] BERT valence vs human consensus, all 12 films
- [x] Three-model robustness (BERT, DistilBERT, SiEBERT) on the clean film
- [x] Transcription-contamination check + Spaceman decontamination
- [x] Bootstrap calibration → significance withheld; effect-size reporting adopted
- [x] LLM (Gemini 3.5 Flash) probe across all 12 films
- [x] Model-strength check (Gemini 3.1 Pro) confirming the sparse-dialogue null
- [x] Result figures (forest plot vs the inter-rater reference; decontamination mechanism)
- [x] Context condition (dialogue-context window), preliminary, four films
- [x] Brain decoding, single-subject occipital run (superseded; retracted as a result)
- [x] Powered, multi-subject leave-one-subject-out decode on two films (Payload, Tears
      of Steel), with a planted-signal positive control: a controlled null
- [ ] Run the context condition on BERT too, for a symmetric context comparison
- [ ] Singapore symposium presentation (July 2026)

## Background
Builds on Brainhack School 2026 (NTU Taiwan): fMRI decoding with SVM (Haxby,
84.5%), MLP in PyTorch (82.8%), open neuroimaging data handling (ADHD 200,
ABIDE II), and reproducible-research practices (Git, FAIR, OSF).

## References
- Morgenroth, E., et al. (2025). Emo-FilM: A multimodal dataset for affective
  neuroscience using naturalistic stimuli. *Scientific Data, 12*, 684.
- Morgenroth, E., Moia, S., Vilaclara, L., Fournier, R., Muszynski, M.,
  Ploumitsakou, M., Almato-Bellavista, M., Vuilleumier, P., & Van De Ville, D.
  (2024). Emo-FilM: A multimodal dataset for affective neuroscience using
  naturalistic stimuli. *PsyArXiv*. https://doi.org/10.31234/osf.io/qzdbu

## Acknowledgments
Brainhack School 2026 organizers and TAs across the Taipei and Singapore hubs.

## Contact
GitHub: [@jessicathao](https://github.com/jessicathao)

---
Code and results are shared for course review and educational evaluation. Not licensed for reuse or redistribution pending publication. See NOTICE.
