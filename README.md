# MYRamalan - Malaysia's Election Prediction Hub

A multi-phase Malaysian state election intelligence system: classical ML,
sentiment analysis, RAG, and MLOps automation in one connected pipeline.
Validated against actual 2026 state election results.

---

## Results (verified against real 2026 election outcomes)

| State | Election | Seats | Accuracy | OOD flagged |
|---|---|---|---|---|
| Johor | 2026 | 56 | **78.57%** (44/56) | 12.5% |
| Negeri Sembilan | 2026 | 36 | **83.33%** (30/36) | 52.8% |
| Selangor | pending | — | live forecast, not yet validated | — |
| Melaka | pending | — | live forecast, not yet validated | — |
| Perak | pending | — | live forecast, not yet validated | — |

Predicts binary outcome per seat: **BN** vs. **non-BN** (Harapan and PN
combined as "opposition"). See
[docs/multiclass_analysis.md](docs/multiclass_analysis.md) for why this
framing was chosen, and for an important open finding below.

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
narrative. Currently used only at prediction time for the 3 pending
states (Selangor, Melaka, Perak), not at training time (to avoid the
leakage described above).

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
  scripts/train_models.py
    build_transition()      -- structural + ethnicity features per
                                historical (year_a -> year_b) transition
    train_state()            -- RF + XGB ensemble, calibration,
                                OOD detector (EllipticEnvelope)
  scripts/training_config.py -- per-state hyperparameters (tuned via
                                GridSearchCV), transitions, feature lists
       |
       v
backend/models/{state}/*.pkl + metadata.json
       |
       +--> scripts/validation.py
       |      Regenerates predictions live for states with real 2026
       |      ground truth (Johor, Neg Sembilan); writes
       |      validated_accuracy back into metadata.json
       |
       +--> backend/core/models/state_predictor.py
       |      Live prediction: RF+XGB ensemble -> OOD check -> blend
       |      toward historical seat base-rate if out-of-distribution.
       |      Adds sentiment/narrative features (prediction-time only)
       |      via merge_ethnicity_into_features().
       |
       +--> mlops/model_versioning.py
       |      Archives every trained model with a timestamp; only
       |      promotes to "latest" if it beats the current model's
       |      VALIDATED accuracy (falls back to training accuracy
       |      with an explicit warning if no validated number exists)
       |
       +--> mlops/alerts.py
       |      Logs every training/validation event to a JSON audit
       |      trail; email/Slack alerts supported (best-effort)
       |
       +--> mlops/dags/ + mlops/scheduler.py
              Daily sentiment scrape+score, weekly economic update,
              manual-trigger drift check, monthly retraining --
              APScheduler (local) + a GitHub Actions scheduled
              workflow (cloud, genuinely automated) for retraining

* Bernama RSS currently disabled -- intermittent Groq scoring hang
  not yet root-caused; see docs/multiclass_analysis.md
```

---

## Feature set

**Training features (15)** — structural + ethnicity only, no
sentiment/economic (avoids temporal leakage):

| Category | Features |
|---|---|
| Structural (6) | majority_change, turnout_change, incumbent_held, log_voters, majority_perc_change, n_candidates_b |
| Ethnicity + age (8) | malay_pct, chinese_pct, indian_pct, young_malay_pct, young_chinese_pct, older_malay_pct, youth_pct, median_age |
| Interaction (1) | tension_x_mixed |

**Prediction-time features (24)** — adds sentiment, economic, and
national-narrative interaction features, valid only because they're
applied after training, not baked into historical rows:

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
| Hyperparameter search | GridSearchCV + StratifiedKFold |
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

`scripts/tests/test_training.py` contains regression tests written
for bugs actually found during development (a coalition-naming
standardization bug, a config-completeness check, a training/
inference feature-schema mismatch), plus one end-to-end training
smoke test — not exhaustive coverage, but targeted at real failure
modes. Runs automatically on every push via GitHub Actions.

---

## MLOps

- **Model versioning** (`mlops/model_versioning.py`): every training
  run is archived with a timestamp before being overwritten. A new
  model only gets promoted to "latest" if it beats the current
  model's *validated* accuracy — this distinction matters: a model
  showed 98%+ training accuracy while validating at only 78.57% on
  real 2026 results, so training accuracy alone is treated as an
  unreliable promotion signal.
- **Event logging** (`mlops/alerts.py`): every training run logs to
  a JSON audit trail (`backend/logs/`).
- **Orchestration**: `mlops/dags/dag_training.py` wraps the training
  pipeline for local/scheduled use via `mlops/scheduler.py`
  (APScheduler). A parallel, genuinely-automated path exists via
  `.github/workflows/scheduled-training.yml` — a monthly GitHub
  Actions cron job that retrains, versions, and commits updated
  models back to the repo, independent of any local machine
  (verified via a real scheduled run).
- **Known, accepted limitation**: scheduled retraining currently
  auto-promotes without a manual review gate. Given the training-vs-
  validated-accuracy finding above, this is a real, disclosed risk
  for a politically sensitive model. Planned improvement: a separate
  "published" tier so scheduled retraining can run freely while a
  manual review/approve step gates what `StatePredictor` actually
  serves. See [docs/multiclass_analysis.md](docs/multiclass_analysis.md).

---

## Project structure

```
scripts/
  training_config.py     -- per-state hyperparameters, transitions, feature lists
  train_models.py         -- unified training pipeline (all 5 states)
  validation.py            -- validates against real results (johor, neg_sembilan)
  tune_hyperparams.py      -- GridSearchCV + StratifiedKFold search
  tests/test_training.py   -- regression + smoke tests, run via CI

backend/
  core/models/state_predictor.py   -- live prediction, OOD fallback
  core/pipelines/state_pipeline.py -- feature engineering helpers
  models/{state}/                   -- trained models, versioned history, metadata
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
  model_versioning.py    -- promotion gates (validated accuracy preferred)
  alerts.py                -- event logging + optional email/Slack
  scheduler.py             -- APScheduler entry point
  dags/
    dag_sentiment.py       -- daily: scrape -> score -> update -> rebuild RAG
    dag_economic.py        -- weekly: fetch -> LSTM -> pressure score
    dag_drift.py            -- manual: post-election accuracy + drift check
    dag_training.py         -- wraps train_models.py for scheduled/manual runs

.github/workflows/
  ci.yml                    -- test suite on every push
  scheduled-training.yml    -- monthly automated retraining (GitHub Actions cron)

docs/
  multiclass_analysis.md    -- full investigation log: binary vs. 3-class target,
                                sentiment-feature leakage analysis, BN-PN coalition
                                finding, known issues and future work

data/
  raw/                       -- ballots, stats, ethnicity, news (parquet/CSV)
  processed/                 -- sentiment scores, narrative scores, economic pressure
```

---

## Known limitations

- Only Johor and Neg Sembilan have held their 2026 elections, so
  those are the only two states with real validated accuracy.
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
- Scheduled retraining auto-promotes without manual review (see
  MLOps section above).

---

## Quick start

```bash
git clone https://github.com/hazimxm04/GE-Insights-Prediction-Hub
cd GE-Insights-Prediction-Hub

pip install -r requirements.txt

# Add your Groq API key to backend/.env
# GROQ_API_KEY=your_key_here

# Train all 5 states
python scripts/train_models.py

# Validate against real 2026 results (johor, neg_sembilan)
python scripts/validation.py

# Run tests
python scripts/tests/test_training.py
```

---

## Author

Built as an ML/AI engineering portfolio project demonstrating
end-to-end system design: data ingestion, feature engineering,
model training and evaluation, MLOps (versioning, CI/CD, scheduled
automation), and honest documentation of what works, what doesn't,
and why.
