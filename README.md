# emotion-label-validation-fmri

Validating transformer-generated emotion labels against human consensus in naturalistic film fMRI

Status: In active development (Brainhack School 2026 final project). Public repository.
Author: Thach Thao Le (Jessica), PhD student, Linguistics, National Taiwan University

**Live project page:** https://jessicathao.github.io/emotion-label-validation-fmri/

## Project overview

This project tests whether emotion labels generated automatically by transformer language models agree with human consensus annotations, using the Emo-FilM dataset: fMRI and emotion ratings collected while participants watched short films.

The central question: can an automatic NLP method label the emotional valence of film dialogue well enough to complement, and help scale, slow and costly human annotation?

## Research questions

1. **Agreement:** Do automatic sentiment labels, derived from film dialogue, correlate with the human consensus annotation? Tested for a per-segment classifier (BERT) and an LLM (Gemini 3.5 Flash) across 12 films, with three sentiment models on the cleanest film, plus a preliminary dialogue-context manipulation run on both models on four films.
2. **Brain decoding:** Can those labels decode emotional valence from the fMRI BOLD signal? A powered, multi-subject occipital decode, validated by a planted-signal positive control and a visual-feature decomposition, returns a controlled, mechanistic null on two films.

## Findings so far

Preliminary (Brainhack School 2026), reported qualitatively; quantitative results are reserved for a manuscript in preparation.

- **Automatic labels track human valence only where emotion is spoken.** Per-segment BERT agrees with human ratings on only the most lexically explicit film; an LLM (Gemini 3.5 Flash) is somewhat more consistent but its gains are modest, and absent on sparse-dialogue films.
- **Dialogue context helps the model that can use it.** In a preliminary four-film condition, context raises LLM agreement on dialogue-driven films but produces no coherent effect for per-segment BERT.
- **The brain arm is a controlled, mechanistic null.** Decoding human valence from posterior occipital BOLD reads at chance on two films; the same region tracks low-level visual features of the film, not valence. Bounded to one ROI and one target, this is not a claim that the brain lacks valence information.
- **Reporting standard.** Results are reported as effect sizes with cross-model and cross-film replication rather than p-values, because the emotion annotations are strongly autocorrelated.

**Conclusion:** text-based sentiment, even with a frontier LLM, is not yet a robust substitute for human emotion annotation in naturalistic film; the label and neural arms reach the same boundary from two directions.

## Why it matters

- Label provenance: any decoder is only as good as its labels; where automatic labels fail is understudied.
- Scale and reach: reliable automatic labels would open many unlabeled naturalistic datasets for affective neuroscience, and the method extends to cross-linguistic data, including under-represented languages such as Vietnamese.
- This measures emotion in film via language (spoken dialogue), not language processing in isolation, a deliberate and scoped first test.

## Data

Emo-FilM (Morgenroth et al., 2025, Scientific Data 12:684), on OpenNeuro:

- fMRI: ds004892 (CC0)
- Annotations: ds004872 (CC0)
- 14 short films, 30 participants (3 Tesla), 44 annotators, 50 emotion items.
- TR = 1.3 s. Preprocessed BOLD available in MNI space.
- The human valence signal is taken from the appraisal item PleasantOther, the highest-agreement item in the dataset.

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
                              + circular-shift null), two films; the same decoder
                              is then run on 1 Hz visual features (motion, luminance,
                              contrast, edges, cuts, saturation) to localize what the
                              occipital signal actually carries
```

Each dialogue segment is scored in isolation, so the classifier and the LLM see identical input, with the model as the only variable; the context condition adds a window of three segments on each side, on both models. Agreement is reported as an effect size with a descriptive 95% bootstrap CI; significance is withheld because the annotations are autocorrelated on a timescale comparable to film length, making the block bootstrap too liberal on single short films. Human reference levels come from Morgenroth et al. 2025 (the per-rater series are not in the public derivatives), and a contamination check removes non-dialogue audio (songs, recited verse, credits) that Whisper transcribes and sentiment models then mis-score.

For the brain arm, human valence is decoded from posterior occipital BOLD by ridge regression under leave-one-subject-out cross-validation across five subjects per film, gated by a planted-signal positive control and a circular-shift null. Rerunning the same pipeline on each 1 Hz visual feature (and on valence with the visual set regressed out) localizes the occipital signal as visual rather than affective. Reported on two films, Payload and Tears of Steel.

## Technical stack

| Component | Tool |
|---|---|
| fMRI data handling | nilearn, nibabel |
| Per-segment sentiment | Hugging Face Transformers (phanerozoic/BERT-Sentiment-Classifier; also DistilBERT/SST-2, SiEBERT/RoBERTa-large for robustness) |
| LLM (Gemini) | Gemini API (gemini-3.5-flash; gemini-3.1-pro-preview for a model-strength check) |
| Machine learning | scikit-learn (Ridge decoding; SVM from coursework) |
| Transcription | OpenAI Whisper (films without official subtitles) |
| Visual features (brain confound) | OpenCV (1 Hz luminance, contrast, motion, edges, cuts, saturation) |
| Plotting | matplotlib |
| Environment | Conda (Python 3.11), macOS Apple Silicon |
| Version control | Git / GitHub |

## Repository layout

```
src/
  figures/        scripts that build the deck and result figures
  signals/        build the per-segment BERT and Gemini valence signals + transcript prep
  contamination/  find and remove transcription artifacts (songs, recited verse, credits)
  analysis/       alignment, correlation, calibration, controls, brain decoding,
                  visual-feature regressors + decomposition
data/
  subs/          film subtitle files (.srt)
  transcripts/   processed transcripts + per-model 1 Hz signals (.json)
  visual/        1 Hz low-level visual features per film (.npz / .csv; reproducible from cuts)
figures/         all generated figures (.png / .pdf)
results/ docs/ notebooks/   tabular outputs, notes, exploration
```

Run scripts from the repo root, e.g. `python src/figures/make_delta_forest.py`. Large/private data (brain images, film media) and resume caches are gitignored.

## Progress

- [x] Project proposal and design (pitched June 2026)
- [x] Dataset selection and fMRI structure explored (MNI preprocessed, TR 1.3 s)
- [x] PleasantOther human valence signal extracted from ds004872
- [x] Subtitle/film-time alignment resolved (+2 s offset, 1 Hz grid)
- [x] Transcripts for all 12 dialogue films (official .srt where available, else Whisper)
- [x] BERT valence vs human consensus, all 12 films
- [x] Three-model robustness (BERT, DistilBERT, SiEBERT) on the clean film
- [x] Transcription-contamination check + decontamination
- [x] Bootstrap calibration → significance withheld; effect-size reporting adopted
- [x] LLM (Gemini 3.5 Flash) probe across all 12 films
- [x] Model-strength check (Gemini 3.1 Pro) confirming the sparse-dialogue null
- [x] Result figures (forest plot vs the inter-rater reference; decontamination mechanism)
- [x] Context condition run on both BERT and Gemini, four films; BERT shows no coherent effect (its one apparent rise is smoothing), Gemini gains on dialogue-driven films
- [x] Brain decoding, single-subject occipital run (superseded; retracted as a result)
- [x] Powered, multi-subject leave-one-subject-out decode on two films, with a planted-signal positive control: a controlled null
- [x] Visual-feature decomposition on both films: occipital decodes visual dynamics (motion strongest), not valence; the null is now mechanistic
- [ ] Extend the context condition beyond four films (deferred to the manuscript)
- [ ] Singapore symposium presentation (July 2026)

## Background

Builds on Brainhack School 2026 (NTU Taiwan): fMRI decoding with SVM (Haxby), MLP in PyTorch, open neuroimaging data handling (ADHD 200, ABIDE II), and reproducible-research practices (Git, FAIR, OSF).

## References

- Morgenroth, E., Moia, S., Vilaclara, L. et al. Emo-FilM: A multimodal dataset for affective neuroscience using naturalistic stimuli. Scientific Data 12, 684 (2025). [doi.org/10.1038/s41597-025-04803-5](https://doi.org/10.1038/s41597-025-04803-5)
- Morgenroth, E., Moia, S., Vilaclara, L., Fournier, R., Muszynski, M., Ploumitsakou, M., Almato-Bellavista, M., Vuilleumier, P., & Van De Ville, D. (2024). Emo-FilM: A multimodal dataset for affective neuroscience using naturalistic stimuli. PsyArXiv. https://doi.org/10.31234/osf.io/qzdbu

## Acknowledgments

Brainhack School 2026 organizers and TAs across the Taipei and Singapore hubs.

## Contact

GitHub: @jessicathao

Code and results are shared for course review and educational evaluation. Not licensed for reuse or redistribution pending publication. See NOTICE.
