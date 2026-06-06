# lang_brain_project

**Validating transformer-generated emotion labels against human consensus in naturalistic film fMRI**

Status: In active development (Brainhack School 2026 final project)
Author: Thach Thao Le (Jessica) — PhD student, Linguistics, National Taiwan University

## Project overview

This project tests whether emotion labels generated automatically by transformer
language models agree with human consensus annotations, using the **Emo-FilM**
dataset — fMRI and emotion ratings collected while participants watched short films.

The central question: **can an automatic NLP method label the emotional valence of
film dialogue well enough to substitute for slow, costly human annotation?**

### Research questions
1. **Agreement (done):** Do automatic sentiment labels, derived from film
   dialogue, correlate with the human consensus annotation? Tested for a
   word-level classifier (BERT) and a context-aware LLM (Gemini 3.5 Flash)
   across 12 films, with three sentiment models on the cleanest film.
2. **Brain decoding (planned):** Can those labels decode emotional valence from
   the fMRI BOLD signal as well as human labels can? (Designed, not yet run.)

## Findings so far

*Preliminary — Brainhack School 2026 project, not yet peer-reviewed.*

Across 12 naturalistic films, automatic sentiment labels agree with human
valence only weakly, and only where emotion is carried in explicit dialogue:

- **Word-level BERT** reaches human-level agreement on just one film (Tears of
  Steel, r ≈ 0.36), the most lexically explicit; elsewhere it clusters near zero,
  with apparent negatives traceable to transcription artifacts (songs and credits
  mis-scored as dialogue).
- **A context-aware LLM** (Gemini 3.5 Flash), scoring the same isolated dialogue
  segments through the same pipeline, agrees with humans somewhat more
  consistently (it shifts most correlations positive and removes BERT's spurious
  negatives), but the gains are weak, partly concentrated in
  transcription-contaminated films, and absent on sparse-dialogue films.
- On a sparse-dialogue film (Sintel) the null is **model-independent**: the
  strongest available model (Gemini 3.1 Pro) also recovers nothing.
- **No automatic label, BERT or LLM, reaches the human inter-rater ceiling**
  (PleasantOther r ≈ 0.58); even the LLM's best only reaches the dataset mean
  (r ≈ 0.39).

Conclusion: text-based sentiment, even with a frontier LLM, is not a robust
substitute for human emotion annotation in naturalistic film. It tracks human
valence substantially only when emotion is stated in the words. This is a
methodological as well as an empirical result: automatic text labels inherit
every weakness of the transcription step, and emotion annotations are so
autocorrelated that significance testing on single short films is infeasible —
so results are reported as effect sizes with cross-model and cross-film
replication rather than p-values.

## Why it matters
- **Label provenance:** any decoder is only as good as its labels; knowing where
  automatic labels fail is understudied.
- **Scale:** if automatic labels prove reliable, many unlabeled naturalistic datasets
  become usable for affective neuroscience.
- **Extensible:** the same method can later move to cross-linguistic data, including
  under-represented languages such as Vietnamese.

This measures emotion in film *via language* (spoken dialogue), not language
processing in isolation — a deliberate, scoped first test.

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
                                          (planned) ─────────────┘
                              decode valence from BOLD (leave-one-film-out CV),
                              comparing human vs automatic label sources
```

Each dialogue segment is scored **in isolation** (no surrounding context), so the
word-level classifier and the context-aware LLM see identical input and the only
variable is the model. Signals are built on a 1 Hz grid, aligned to the scan with
a fixed +2 s offset, and correlated against the human *PleasantOther* consensus.

Agreement is reported as an **effect size** with a descriptive 95% bootstrap CI.
Significance testing is deliberately withheld: calibration (`bootstrap_calibration.py`)
shows the block bootstrap is far too liberal on single short films, because the
annotation autocorrelation time (~80–110 s) dwarfs any usable block size. The
finding therefore rests on effect sizes plus cross-model and cross-film
replication. Human reference levels: dataset-mean inter-rater r ≈ 0.39;
PleasantOther (highest-agreement item) r ≈ 0.58.

A contamination check is part of the method: Whisper transcribes non-dialogue
audio (songs, recited verse, credit tails) as text, which sentiment models then
score. `mask_signal.py` removes such segments; on Spaceman this collapses BERT's
phantom negative toward zero while revealing a real positive for the LLM —
demonstrating that transcription noise corrupts text-based emotion labels
differently depending on the model.

## Technical stack

| Component | Tool |
|---|---|
| fMRI data handling | nilearn, nibabel |
| Word-level sentiment | Hugging Face Transformers (`phanerozoic/BERT-Sentiment-Classifier`; also DistilBERT/SST-2, SiEBERT/RoBERTa-large for robustness) |
| Context-aware LLM | Gemini API (`gemini-3.5-flash`; `gemini-3.1-pro-preview` for a model-strength check) |
| Machine learning | scikit-learn (SVM, planned decoding) |
| Transcription | OpenAI Whisper (films without official subtitles) |
| Plotting | matplotlib |
| Environment | Conda (Python 3.11), macOS Apple Silicon |
| Version control | Git / GitHub |

## Repository layout
```
make_gemini_signal.py     per-segment LLM valence probe (model-agnostic, cached)
batch_llm_probe.py        run the probe + correlation over all dialogue films
mask_signal.py            null out contaminated second-ranges from a signal
spaceman_decontam_all.py  apply one decontamination cut across all models
make_forest_bert_gemini.py   BERT-vs-LLM forest figure (vs human ceiling)
make_spaceman_mechanism.py   per-model decontamination shift figure
src/
  correlate_v3.py            alignment + correlation + real-timeline bootstrap CI
  bootstrap_calibration.py   calibration showing significance is not testable here
data/
  subs/          film subtitle files (.srt)
  transcripts/   processed transcripts + per-model 1 Hz signals (.json)
figures/         generated result figures (.png / .pdf)
docs/ notebooks/ results/   notes, exploration, outputs
```
Large/private data (brain images, film media) and resume caches are gitignored.

## Progress
- [x] Project proposal and design (pitched June 2026)
- [x] Dataset selection and fMRI structure explored (MNI preprocessed, TR 1.3 s)
- [x] PleasantOther human valence signal extracted from ds004872
- [x] Subtitle/film-time alignment resolved (+2 s offset, 1 Hz grid)
- [x] Transcripts for all 12 dialogue films (official .srt where available, else Whisper)
- [x] BERT valence vs human consensus — all 12 films
- [x] Three-model robustness (BERT, DistilBERT, SiEBERT) on the clean film
- [x] Transcription-contamination check + Spaceman decontamination
- [x] Bootstrap calibration → significance withheld; effect-size reporting adopted
- [x] Context-aware LLM (Gemini 3.5 Flash) probe across all 12 films
- [x] Model-strength check (Gemini 3.1 Pro) confirming the sparse-dialogue null
- [x] Result figures (forest plot vs human ceiling; decontamination mechanism)
- [ ] Context condition (dialogue + scene context) — designed, next experiment
- [ ] Brain decoding, comparing human vs automatic label sources — planned
- [ ] Singapore symposium presentation (June 2026)

## Background
Builds on Brainhack School 2026 (NTU Taiwan): fMRI decoding with SVM (Haxby,
84.5%), MLP in PyTorch (82.8%), open neuroimaging data handling (ADHD 200,
ABIDE II), and reproducible-research practices (Git, FAIR, OSF).

## References
- Morgenroth, E., et al. (2025). Emo-FilM: A multimodal dataset for affective
  neuroscience using naturalistic stimuli. *Scientific Data, 12*, 684.

## Acknowledgments
Brainhack School 2026 organizers and TAs across the Taipei and Singapore hubs.

## Contact
GitHub: [@jessicathao](https://github.com/jessicathao)
