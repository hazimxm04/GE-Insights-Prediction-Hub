# MYRamalan - Malaysia's Election Prediction Hub

A multi-phase Malaysian state election intelligence system: classical ML,
sentiment analysis, RAG, and MLOps automation in one connected pipeline.
Validated against actual 2026 state election results.

---

## Results (verified against real 2026 election outcomes)

Each seat gets one yes/no prediction: **does Pakatan Harapan (the opposition) win the seat, or does it go to BN / PN?**
Every test below trains only on earlier elections and is scored on a later one. The features use only the last
election before the one being predicted, and the one before that.

| State | Test | Seats | Model correct | "Never opposition" | "Same as last time" | Wrong (false alarm / missed) | OOD flagged |
|---|---|---|---|---|---|---|---|
| Johor | 2022 → 2026 | 56 | **48 (85.7%)** | 48 (85.7%) | 50 (89.3%) | 8 (7 / 1) | 41.1% |
| Negeri Sembilan | 2023 → 2026 | 36 | **31 (86.1%)** | 25 (69.4%) | 30 (83.3%) | 5 (5 / 0) | 80.6% |
| Selangor | pending | — | live forecast, not yet validated | — | — | — | — |
| Melaka | pending | — | live forecast, not yet validated | — | — | — | — |
| Perak | pending | — | live forecast, not yet validated | — | — | — | — |

- **"Never opposition"** always predicts that the opposition loses. **"Same as last time"** predicts that each seat is won by
  the same side as at the previous election. A model is only useful if it beats these simple rules.
- **Johor:** the model ties "never opposition" and is two seats behind "same as last time".
- **Negeri Sembilan:** the model beats both, but only by one seat over "same as last time".
- **12 of the 13 errors are false alarms:** the model predicted an opposition win in a seat that went to BN or PN.
- **Accuracy vs ranking.** AUC is high (0.93 Johor, 0.99 Negeri Sembilan): the model ranks seats well. Accuracy is lower
  because of the 0.5 cut-off. The training elections include the 2018 opposition wave, so the model learned that the
  opposition wins often, and over-predicts it in 2026. The cut-off has not been tuned on the 2026 results, because that
  would be testing on the answers.
- **Before the second leakage fix** (see below) the same real-2026 test scored 78.57% (44/56) for Johor and 83.33% (30/36)
  for Negeri Sembilan, which was *worse* than "never opposition" in Johor.

### Earlier elections (same method)

| State | Test | Seats | Model | "Never opposition" | "Same as last time" |
|---|---|---|---|---|---|
| Johor | 2018 → 2022 | 56 | 51 | 44 | 32 |
| Negeri Sembilan | 2018 → 2023 | 35 | 32 | 18 | 32 |
| Selangor | 2018 → 2023 | 56 | 37 | 24 | 37 |
| Melaka | 2018 → 2021 | 28 | 22 | 23 | 18 |
| Perak | 2018 → 2022 | 59 | 46 | 35 | 50 |

Each of these models trained on only two earlier elections, so the data is very thin.
Reproduce with `python scripts/evaluate_holdout.py` (writes `backend/models/holdout_results.json`).

The training score is not a performance number: it is 96–100% for every state, while the held-out scores above are 66–91%.
Training now saves the held-out score in `metadata.json` next to the training score.

Predicts binary outcome per seat: **Harapan wins** vs. **BN / PN win** (BN and PN are not separated). See
[docs/multiclass_analysis.md](docs/multiclass_analysis.md) for why this framing was chosen, and for an important open
finding below.

---

## Key finding: BN-PN coalition dynamics are election-specific, not fixed

In the 2026 election, BN and PN's relationship differed **by state**:
in Johor they competed as separate, opposing coalitions; in Negeri
Sembilan they appear to have operated under a seat-allocation pact
(in most seats, only one of BN or PN fielded a candidate). This is a
political arrangement specific to this election cycle — it is not a
permanent rule, and could look different next election.

The current model does not yet capture this. The proposed fix is a
**data-derived feature per seat** ("did both BN and PN field
candidates in this seat, this election?") rather than a hardcoded
assumption about how the coalitions relate — so the model adapts to
whatever the actual candidate lists say for each election, rather
than encoding today's specific alliance as permanent. This is the
single most promising lever identified for improving accuracy, ahead
of further hyperparameter tuning. Full writeup:
[docs/multiclass_analysis.md](docs/multiclass_analysis.md).

---

## Other findings

**Outcome leakage, found and fixed (second leak).** Separately from the sentiment leak below, some structural features
(majority change, turnout change, and similar) were computed from the election being predicted, so the model was partly
shown the answer. Now every feature uses only the last election before the predicted one, and the one before that
(`backend/core/features.py`), elections before 2004 are excluded because the seat counts differed, and seats are matched
on their number (N.07) rather than their name. A test (`scripts/tests/test_no_outcome_leakage.py`) rewrites the predicted
election and fails if any feature changes. The evaluation numbers above come from the fixed pipeline.

**Why accuracy is no better than the simple rules while AUC is high.** The model ranks seats well (AUC 0.93 in Johor,
0.99 in Negeri Sembilan), but its accuracy is no better than the simple rules "never opposition" and "same as last
time". Accuracy depends on the 0.5 probability cut-off, and because the training years include the 2018 opposition
wave, the model over-predicts opposition wins in 2026: 12 of the 13 errors were false alarms. Next, I will choose the
cut-off using only earlier elections, never the 2026 results, and add a per-seat feature for whether both BN and PN
field candidates.

**Pooled model across states, inconclusive.** One model trained on all five states (with a time-ordered test that never
uses the test year or later) got more seats right in Selangor, Perak and Negeri Sembilan, fewer in Melaka, and about the
same in Johor (`backend/models/pooled_results.json`). Seats in one election swing together, so there are far fewer
independent observations than seats; no improvement is claimed.

**Temporal leakage, found and fixed.** An earlier version of this
pipeline included prediction-period (Aug 2026) sentiment scores as a
*training* feature for transitions ending in 2013, 2018, and 2022 —
using information that did not exist yet at the time of those
elections. Fixed by splitting the feature set: training uses only
structural + ethnicity features; sentiment/economic features are
reserved for prediction time, where they are temporally valid.

**3-class coalition split investigated, and reverted.** Tested
splitting the target into BN / Pakatan / PN instead of binary. Found
it underperforms: PN only emerged as a coalition in 2020, so no
training transition has PN as a "previous" winner, and PN's seat
count in its one possible training transition varies sharply by
state (2 seats in Melaka vs. 26 in Perak) — a genuine data
limitation, not a tuning problem (confirmed via GridSearchCV, which
improved cross-validation scores but produced zero change in real
validation accuracy). Reverted to binary; full investigation
documented rather than discarded.

**Voter-roll demographics provide genuine seat-level signal.**
Ethnicity and age composition (`chinese_pct`, `young_malay_pct`,
`youth_pct`, etc.), sourced from anonymised voter rolls, are
consistently among the top features by importance across all 5
states — this is real, seat-varying signal that structural features
alone (majority change, turnout, incumbency) don't capture.

**Sentiment/narrative features are national, weighted by
demographics.** State-level sentiment scores are constant across all
seats in a state (zero tree importance on their own). Weighting
national narrative themes (e.g., "cost of living," "Islam threat")
by each seat's demographic composition produces genuine seat-level
variation — different seats respond differently to the same national
narrative. This is an idea that is **not in the current models**: it was
kept out of training to avoid the sentiment leakage described below, and
the live predictor currently holds these inputs at 0. It needs a retrain
with real seat-level values and a fair held-out test before any claim.

---

## Architecture

```
Data sources
  electiondata.my (ballots, stats, voter rolls)
  5 RSS news sources (FMT, Malaysiakini, Malay Mail, Utusan Malaysia, Bernama*)
  yfinance (KLCI, USD/MYR)
       |
       v
Training pipeline
  backend/core/features.py
    build_transition()       -- the single feature definition: uses only
                                election A and the one before it, B gives
                                the target only
    build_features()         -- same features for forecasting (no B needed)
  backend/core/states.py   -- registry of states and their elections
  scripts/state_data.py    -- one shared data loader (local copy first)
  scripts/train_models.py
    train_state()            -- RF + XGB ensemble, calibration,
                                OOD detector (EllipticEnvelope), held-out
                                score saved to metadata.json
  scripts/training_config.py -- per-state transitions, model settings
                                (shared), feature lists
       v
backend/models/{state}/*.pkl + metadata.json
       |
       +--> scripts/validation.py
       |      Scores the saved models on the real 2026 results
       |      (Johor, Neg Sembilan); refuses models built with old
       |      features or tested on an election inside their training
       |      window; writes validated_accuracy into metadata.json
       |
       +--> backend/core/models/state_predictor.py
       |      Live prediction: RF+XGB ensemble -> OOD check -> blend
       |      toward the seat's historical base rate if out-of-distribution.
       |      Features come from the same code as training; the base rate
       |      uses only elections up to the last known one.
       |
       +--> mlops/model_versioning.py
       |      Archives the previous model; a newly trained model is
       |      staged first and only replaces "latest" if its HELD-OUT
       |      score (same test election) is at least as good as the
       |      current model's. Models built with old features are
       |      always replaced.
       |
       +--> mlops/alerts.py
       |      Logs every training/validation event to a JSON audit
       |      trail; email/Slack alerts supported (best-effort)
       |
       +--> mlops/dags/ + mlops/scheduler.py
              Daily sentiment scrape+score, weekly economic update,
              manual-trigger drift check, monthly retraining --
              APScheduler (local) + a GitHub Actions scheduled
              workflow (cloud) that refreshes the data, runs the
              tests, retrains, validates, and opens a pull request

* Bernama RSS currently disabled -- intermittent Groq scoring hang
  not yet root-caused; see docs/multiclass_analysis.md
```

---

## Feature set

**Training features (15)** — structural + ethnicity only, no
sentiment/economic (avoids temporal leakage):

| Category | Features |
|---|---|
| Structural (6) | majority_change, turnout_change, incumbent_held, log_voters, majority_perc_change, n_candidates_b (candidates at the last election) |
| Ethnicity + age (8) | malay_pct, chinese_pct, indian_pct, young_malay_pct, young_chinese_pct, older_malay_pct, youth_pct, median_age |
| Placeholder (1) | tension_x_mixed (constant 0 in training) |

**Prediction-time features (24)** — sentiment, economic, and national-narrative
interaction features. **Not used by the current models or the live predictor**:
the models were trained without them (state-level values are the same for every
seat, and `tension_x_mixed` is a constant 0 in training, so it is also held at 0
when predicting). Re-enabling them needs a retrain with real seat-level values:

| Category | Features |
|---|---|
| Sentiment (4) | bn_sentiment, harapan_sentiment, pn_sentiment, racial_tension_index |
| Economic (1) | economic_pressure |
| Sentiment × ethnicity interactions (5) | bn_sent_x_malay, harapan_sent_x_chinese, pn_sent_x_young_malay, tension_x_mixed, economic_x_youth |
| National narrative (1) | narrative_pressure |

---

## Tech stack

| Component | Technology |
|---|---|
| ML models | scikit-learn (Random Forest, CalibratedClassifierCV, EllipticEnvelope), XGBoost |
| Hyperparameter search | Time-ordered grid search scored by Brier score (the real 2026 results are never used) |
| LLM / sentiment | Groq API (llama-3.1-8b-instant) |
| Vector store / RAG | ChromaDB, sentence-transformers |
| Deep learning (economic forecast) | PyTorch (LSTM) |
| Data | electiondata.my, yfinance |
| Testing | pytest-style regression + smoke tests |
| CI/CD | GitHub Actions (test suite on every push; scheduled monthly retraining workflow) |
| Scheduling (local) | APScheduler |

---

## Testing & CI

![Tests](https://github.com/hazimxm04/GE-Insights-Prediction-Hub/actions/workflows/ci.yml/badge.svg)

Run locally: `python -m pytest scripts/tests tests -k "not end_to_end"`

- `test_no_outcome_leakage.py`: rewrites the predicted election and checks that no feature changes, per feature, so a
  failure names the leaking feature. Also checks the features still respond to the last known election, that elections
  before 2004 are refused, and that renamed seats still match by number.
- `test_data_quality.py`: checks the data for every state (years, seat counts, usable rows).
- `test_training.py`: regression tests for bugs found in development (coalition-naming, config completeness, feature
  schema) plus an end-to-end training smoke test.
- `test_training_outputs.py`: the held-out test comes after the training years; validation refuses old or leaky models;
  the promotion gate compares held-out scores and keeps the better model.
- `test_predictor_inputs.py`: the live predictor builds features and base rates from the same code as training, uses
  nothing after the last known election, and matches `scripts/validation.py` on real data.

CI runs `scripts/tests` and `tests` on pushes to `main` and `fix/**` branches and on pull requests into `main`.

---

## MLOps

- **Model versioning** (`mlops/model_versioning.py`): a newly trained model is staged first. It replaces the current
  model only if its **held-out score** (same test election and same feature version) is at least as good; the previous
  model is archived, and a rejected candidate is kept for inspection. Training accuracy is never used for promotion: it
  is 96–100% for every state while held-out scores are 66–91%.
- **Event logging** (`mlops/alerts.py`): every training run logs to a JSON audit trail (`backend/logs/`).
- **Orchestration**: `mlops/dags/dag_training.py` wraps the training pipeline for local/scheduled use via
  `mlops/scheduler.py` (APScheduler). A parallel cloud path exists in `.github/workflows/scheduled-training.yml`: a
  monthly job that refreshes the election data, runs the tests, retrains, runs validation, and opens a **pull request**
  with the new models. It no longer pushes straight to `main`, so a person reviews each retrained model.
- **Known limitation**: the held-out gate decides between a new and the current model, but with only two or three
  elections per state its score is noisy. Planned improvement: a separate "published" tier that gates what
  `StatePredictor` actually serves. See [docs/multiclass_analysis.md](docs/multiclass_analysis.md).

---

## Project structure

```
scripts/
  training_config.py     -- transitions, shared model settings, feature lists
  train_models.py         -- unified training pipeline (all 5 states), held-out score
  validation.py            -- scores saved models on real results (johor, neg_sembilan)
  evaluate_holdout.py      -- time-ordered test vs simple baselines, all states
  evaluate_pooled.py       -- pooled vs per-state experiment
  state_data.py            -- one shared data loader
  check_state_years.py     -- prints years / seat counts per state
  tune_hyperparams.py      -- time-ordered settings comparison (current settings kept: gaps were within noise)
  tests/                   -- leakage, data-quality, training and predictor tests, run via CI

backend/
  core/features.py                 -- the single feature definition (no leakage)
  core/states.py                   -- registry of states and elections
  core/models/state_predictor.py   -- live prediction, OOD fallback
  models/{state}/                   -- trained models, versioned history, metadata
  scripts/download_data.py         -- election data, driven by the state registry
  scripts/download_ethnicity.py    -- voter-roll ethnicity files, driven by the registry
  scripts/add_ethnicity_features.py -- demographic-weighted narrative features

sentiment/
  scrapers/news_scraper.py         -- RSS scraping, 5 sources, per-source lean disclosed
  scoring/sentiment_scorer.py      -- Groq scoring, national narrative themes

economic/
  models/lstm_model.py              -- PyTorch LSTM, economic pressure score

chatbot/
  knowledge_base/builder.py        -- ChromaDB indexing
  chain/rag_chain.py                -- RAG pipeline (question -> retrieve -> Groq -> answer)

mlops/
  model_versioning.py    -- held-out promotion gate, archiving
  alerts.py                -- event logging + optional email/Slack
  scheduler.py             -- APScheduler entry point
  dags/
    dag_sentiment.py       -- daily: scrape -> score -> update -> rebuild RAG
    dag_economic.py        -- weekly: fetch -> LSTM -> pressure score
    dag_drift.py            -- manual: post-election accuracy + drift check
    dag_training.py         -- wraps train_models.py for scheduled/manual runs

.github/workflows/
  ci.yml                    -- leakage + training tests on push / pull request
  scheduled-training.yml    -- monthly: refresh data, test, retrain, validate, open a PR

docs/
  multiclass_analysis.md    -- full investigation log: binary vs. 3-class target,
                                sentiment-feature leakage analysis, BN-PN coalition
                                finding, known issues and future work

data/
  raw/                       -- ballots, stats, ethnicity, news (parquet/CSV)
  processed/                 -- sentiment scores, narrative scores, economic pressure
```

---

## Status and roadmap

Done: leak-free features, time-ordered evaluation against baselines, held-out promotion gate, tests and CI,
time-ordered tuning, one data layer driven by the state registry.

Next: release v1 and freeze dated forecasts (so they can be checked against the real result later); decide
whether a headline-theme classifier is worth using (DistilBERT vs the Groq zero-shot baseline, on a hand-labelled
set); LLM evaluation and agent work; per-coalition targets (BN / PH / PN) and 1-vs-1 versus multi-corner contests;
more states; finish moving the API and chatbot onto the same state list.

---

## Known limitations

- **The model does not clearly beat simple rules.** On the real 2026 results it ties "never opposition" in Johor and beats
  "same as last time" by one seat in Negeri Sembilan.
- **Very little data:** 28–59 seats per state and only two or three elections per state; seats in one election swing
  together, so the effective sample is much smaller than the seat count.
- **Base-rate shift:** the training years include the 2018 opposition wave, so the model over-predicts opposition wins
  (12 of 13 errors in 2026 are false alarms).
- **High OOD rate in 2026** (41% Johor, 81% Negeri Sembilan): the 2026 seats look different from the training elections.
- **Ethnicity is a single snapshot per state**, not matched to each election year.
- **Serving layer not finished:** the predictor and the chatbot's "what if" tool now use the same feature code as
  training, but the API lists only Johor, Negeri Sembilan and Melaka, the chatbot knows two states, and the drift check
  still reads older column names.
- **No public demo at the moment:** run the API locally with `uvicorn backend.app.main:app`.
- **Binary target only** (Harapan vs BN/PN); 1-vs-1 versus multi-corner contests are not modelled yet.
- Only Johor and Neg Sembilan have held their 2026 elections, so
  those are the only two states with a real-election score.
  Selangor, Melaka, and Perak produce live forecasts.
- BN-PN coalition dynamics are election- and state-specific (see Key
  Finding above) — not yet modeled.
- Sentiment-scoring pipeline (`sentiment_scorer.py`, via Groq)
  intermittently hangs beyond its configured timeout; not isolated
  to one news source. Affects only live-forecast sentiment features
  for pending states, not training or validated accuracy. Bernama
  RSS temporarily disabled as a partial mitigation.
- News scraping covers English/Malay sources only; Chinese-language
  media is not scraped, limiting sentiment signal for Chinese-
  majority seats.
- Scoped to Peninsular Malaysian states; Sabah/Sarawak use a
  different party system (GPS, GRS, WARISAN) not covered by the
  current BN/Harapan/PN framework.

---

## Quick start

```bash
git clone https://github.com/hazimxm04/GE-Insights-Prediction-Hub
cd GE-Insights-Prediction-Hub

pip install -r requirements.txt

# Add your Groq API key to backend/.env
# GROQ_API_KEY=your_key_here

# Download election data (local copies; add --refresh after a new election)
python backend/scripts/download_data.py

# Run the tests, including the no-leakage test
python -m pytest scripts/tests tests -k "not end_to_end"

# Train all 5 states (saves a held-out score per state)
python scripts/train_models.py

# Score on the real 2026 results (johor, neg_sembilan)
python scripts/validation.py

# Compare against simple baselines, all states
python scripts/evaluate_holdout.py
```

---

## Author

Built as an ML/AI engineering portfolio project demonstrating
end-to-end system design: data ingestion, feature engineering,
model training and evaluation, MLOps (versioning, CI/CD, scheduled
automation), and honest documentation of what works, what doesn't,
and why.

Built with AI assistance (Claude): design decisions, evaluation and testing
are mine; the AI helped write and review code.
