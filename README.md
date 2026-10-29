# GE-Insights Prediction Hub

A multi-phase Malaysian state election intelligence system, tested on the actual 2026 state election results. Combines classical ML, deep learning, LLM sentiment analysis, RAG, and MLOps in one connected pipeline.

> **Status:** the training, validation and evaluation pipeline is leak-free and tested (see [Results](#results)). The API, chatbot and dashboards are being moved onto the same feature code.

**Live API:** https://elegant-cooperation-production-67c5.up.railway.app

---

## Results

The model answers one question per seat: **does the opposition (Pakatan Harapan) win?** BN and PN count together as "no".
Every test trains only on earlier elections and tests on a later one. Features use only the last election before the one
being predicted, and the one before that.

### Real 2026 elections (never seen in training)

| State | Seats | Opposition won | Model correct | "Never opposition" | "Same as last time" | Wrong (false alarm / missed) | AUC |
|-------|-------|----------------|---------------|--------------------|---------------------|------------------------------|-----|
| Johor | 56 | 8 | 48 (85.7%) | 48 (85.7%) | 50 (89.3%) | 8 (7 / 1) | 0.93 |
| Negeri Sembilan | 36 | 11 | 31 (86.1%) | 25 (69.4%) | 30 (83.3%) | 5 (5 / 0) | 0.99 |

- **"Never opposition"** always predicts that the opposition loses. **"Same as last time"** predicts every seat is won by the same side as at the previous election. A model is only useful if it beats these.
- **Johor:** the model ties the first baseline and loses to the second.
- **Negeri Sembilan:** the model beats both, but only by one seat over "same as last time".
- **12 of the 13 errors are false alarms:** the model predicted opposition wins in seats that went to BN or PN.

**Accuracy vs AUC.** Accuracy is close to the simple baselines, but AUC is high (0.93 and 0.99). AUC asks whether the model
ranks seats correctly (which seats are *more* likely to go to the opposition), while accuracy depends on the 0.5 cut-off.
The training elections included the 2018 opposition wave, so the model learned that the opposition wins often and its
cut-off over-predicts opposition wins in 2026. The ranking is useful; the cut-off is not yet. I have not tuned the cut-off
on the 2026 results, because that would be testing on the answers.

### Earlier elections (same method)

| State | Test | Seats | Model | "Never opposition" | "Same as last time" |
|-------|------|-------|-------|--------------------|---------------------|
| Johor | 2018 -> 2022 | 56 | 51 | 44 | 32 |
| Negeri Sembilan | 2018 -> 2023 | 35 | 32 | 18 | 32 |
| Selangor | 2018 -> 2023 | 56 | 37 | 24 | 37 |
| Melaka | 2018 -> 2021 | 28 | 22 | 23 | 18 |
| Perak | 2018 -> 2022 | 59 | 46 | 35 | 50 |

Each of these models trained on only two earlier elections, so the data is very thin.
Reproduce: `python scripts/evaluate_holdout.py` (writes `backend/models/holdout_results.json`).

### Why the numbers in earlier versions of this README were higher

An earlier version reported 94.64% (Johor) and 91.67% (Negeri Sembilan). Those scores were inflated by **data leakage**:
some features were computed from the election being predicted, so the model was partly shown the answer. I found it by asking
which election each feature was built from, rewrote the features so they use only the election before the one being predicted
(and the one before that), and added a test (`scripts/tests/test_no_outcome_leakage.py`) that rewrites the predicted election
and fails if any feature changes. The scores above are the honest ones, and the leakage test runs in CI.

---

## Key Findings

**1. Leakage can make a weak model look strong.**
Fixing it dropped Johor from 94.64% to 85.71%, which turned out to be no better than the "never opposition" baseline.
Always compare against simple baselines, and always test on an election that comes after the training elections.

**2. Ranking is better than the yes/no call.**
AUC is 0.93 (Johor) and 0.99 (Negeri Sembilan), but the 0.5 cut-off produces false alarms because the training years had an
unusually strong opposition. A calibration method that survives a change in the base rate is the obvious next step.

**3. Pooling states gave a mixed result.**
One model trained on all five states (with a fair time-ordered test) got more seats right in Selangor, Perak and Negeri Sembilan,
fewer in Melaka, and about the same in Johor (`backend/models/pooled_results.json`). Seats in the same election swing together,
so there are far fewer independent observations than seats, and I do not claim an improvement.

**4. LSTM confirmed the efficient market hypothesis**
Daily KLCI and USD/MYR prediction achieved 0.52% MAPE (accurate in absolute terms) but only 52% directional accuracy, near-random.
Documented honestly as a finding rather than hidden.

---

## Architecture

```
Data sources
  electiondata.my (voter rolls, results)
  5 RSS news sources (FMT, Malaysiakini, Malay Mail, Utusan, Bernama)
  yfinance (KLCI, USD/MYR)
       |
       v
Phase 1: Election predictor
  backend/core/features.py → the single feature definition (no leakage)
  backend/core/states.py → registry of states and elections
  scripts/train_models.py → RF + XGB ensemble, OOD detector, held-out score
  state_predictor.py → predict_seat(), predict_all()
  FastAPI → /predict/seat/{state}, /predict/all/{state}
       |
Phase 2: Sentiment + demographics
  news_scraper.py → 5 politically diverse RSS sources
  sentiment_scorer.py → Groq/Llama 3.1, 9 scores per article
  voter rolls → ethnicity + age per DUN seat (3–5M rows)
  add_ethnicity_features.py → 13 new seat-level features
       |
Phase 3: RAG chatbot
  builder.py → ChromaDB vector store (325 documents)
  rag_chain.py → question → embed → retrieve → Groq → answer
  FastAPI → /chatbot/ask
       |
Phase 4: LSTM economic forecasting
  collector.py → yfinance (KLCI, USD/MYR, 4000+ rows each)
  preprocessor.py → 60-day sliding windows
  lstm_model.py → PyTorch, 2-layer LSTM, hidden_size=64
  evaluator.py → economic_pressure_score per election period
       |
Phase 6: MLOps automation
  dag_sentiment.py → daily 8am: scrape → score → update → rebuild RAG
  dag_economic.py → weekly Monday: fetch → LSTM → pressure scores
  dag_drift.py → post-election: load → accuracy → drift? → retrain
  scheduler.py → APScheduler (Airflow-compatible DAG design)
```

---

## Feature Set (15 features)

| Category | Features |
|----------|----------|
| Structural (6) | majority_change, turnout_change, incumbent_held, log_voters, majority_perc_change, n_candidates_b (candidates at the last election) |
| Ethnicity + age (8) | malay_pct, chinese_pct, indian_pct, young_malay_pct, young_chinese_pct, older_malay_pct, youth_pct, median_age |
| Placeholder (1) | tension_x_mixed (fixed at 0 in training) |

The structural features use only the last election before the one being predicted and the one before that. Ethnicity and age
come from one voter-roll snapshot per state, so they are the same for every election in that state.
Sentiment, economic and narrative features exist in the repository but are **not used by the current models**: state-level
values are the same for every seat, so tree models cannot split on them, and the seat-level narrative idea has not been
re-tested since the leakage fix.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML models | scikit-learn (RF, XGB, LR, CalibratedCV, EllipticEnvelope) |
| Deep learning | PyTorch (2-layer LSTM) |
| LLM / sentiment | Groq API (llama-3.1-8b-instant, free tier) |
| Vector store | ChromaDB + sentence-transformers (all-MiniLM-L6-v2) |
| RAG | LangChain-compatible pipeline |
| API | FastAPI, deployed on Railway |
| MLOps | APScheduler (Airflow-compatible DAG design) |
| Data | electiondata.my, yfinance, Bank Negara API |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info + available states |
| GET | `/health` | Health check |
| POST | `/predict/seat/{state}` | Predict single seat with custom features |
| GET | `/predict/all/{state}` | Forecast all seats from the latest election in the data |
| GET | `/analysis/metadata/{state}` | Model accuracy + feature importance |
| POST | `/chatbot/ask` | RAG chatbot — ask about predictions |
| GET | `/economic/forecast` | Latest LSTM economic forecast |

---

## Quick Start

```bash
git clone https://github.com/hazimxm04/GE-Insights-Prediction-Hub
cd GE-Insights-Prediction-Hub

pip install -r requirements.txt

# Set up environment
cp backend/.env.example backend/.env
# Add: GROQ_API_KEY=your_key

# Download voter roll data
python backend/scripts/download_ethnicity.py

# Download election data (local copies; --refresh after a new election)
python backend/scripts/download_data.py

# Run the tests (including the no-leakage test)
python -m pytest scripts/tests tests -k "not end_to_end"

# Train models (saves a held-out score per state)
python scripts/train_models.py

# Validate on the real 2026 results (Johor, Negeri Sembilan)
python scripts/validation.py

# Honest comparison against simple baselines
python scripts/evaluate_holdout.py

# Start API
python backend/app/main.py
# → http://localhost:8000
```

---

## MLOps Pipeline

Three automated DAGs run on schedule:

```bash
# Run all DAGs once (test mode)
python mlops/scheduler.py --test

# Start live scheduler
python mlops/scheduler.py
```

| DAG | Schedule | Tasks |
|-----|----------|-------|
| dag_sentiment | Daily 8am | scrape news → score sentiment → update state scores → rebuild RAG |
| dag_economic | Weekly Monday | fetch KLCI/MYR → LSTM forecast → update pressure scores |
| dag_drift | Manual (post-election) | load predictions → compute accuracy → detect drift → retrain |

---

## RAG Chatbot

```bash
python chatbot/chain/rag_chain.py
```

Example questions:
- "What is the predicted outcome for Melaka 2026?"
- "Why did the model get N.12 Bentayan wrong?"
- "What is the economic pressure score for Johor?"
- "Which seats in NS did the model predict incorrectly?"
- "What is the racial tension index right now?"

---

## Project Structure

```
GE-Insights-Prediction-Hub/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app + router setup
│   │   └── routes/
│   │       ├── predictions.py         # /predict endpoints
│   │       ├── analysis.py            # /analysis endpoints
│   │       └── chatbot.py             # /chatbot/ask endpoint
│   ├── core/
│   │   ├── models/
│   │   │   └── state_predictor.py     # Inference + OOD + fallback
│   │   ├── features.py                # The single feature definition
│   │   ├── states.py                  # State / election registry
│   │   └── pipelines/
│   │       └── state_pipeline.py      # Legacy pipeline (being retired)
│   ├── models/                        # Saved .pkl files per state
│   └── scripts/
│       ├── download_data.py           # Election data download / refresh
│       └── download_ethnicity.py      # Voter roll processing
├── scripts/
│   ├── train_models.py                # Train RF + XGB + OOD, held-out score
│   ├── validation.py                  # Score on real 2026 results
│   ├── evaluate_holdout.py            # Time-ordered test vs baselines
│   ├── evaluate_pooled.py             # Pooled vs per-state experiment
│   ├── state_data.py                  # One shared data loader
│   └── tests/                         # Leakage, data-quality, training tests
├── sentiment/
│   ├── scrapers/news_scraper.py       # 5 RSS sources
│   └── scoring/sentiment_scorer.py   # Groq/Llama, 9 scores
├── economic/
│   ├── data/                          # collector + preprocessor
│   ├── models/lstm_model.py           # PyTorch LSTM
│   └── evaluation/evaluator.py       # RMSE, MAE, directional acc
├── chatbot/
│   ├── knowledge_base/builder.py     # ChromaDB indexing
│   └── chain/rag_chain.py            # RAG pipeline
├── mlops/
│   ├── scheduler.py                   # APScheduler entry point
│   └── dags/
│       ├── dag_sentiment.py           # Daily pipeline
│       ├── dag_economic.py            # Weekly pipeline
│       └── dag_drift.py              # Drift detection
└── data/
    ├── raw/                           # Parquet files + voter rolls
    └── processed/                     # CSVs for model consumption
```

---

## Limitations

- **The model does not clearly beat simple rules.** On the real 2026 results it ties or barely beats "same as last time".
- **Very little data:** 28–59 seats per state and only two or three elections per state. Seats in one election swing together, so the effective sample is much smaller than the seat count.
- **Base-rate shift:** the training years include the 2018 opposition wave, so the model over-predicts opposition wins.
- **Binary target only:** BN and PN are not separated, and 1-vs-1 versus multi-corner contests are not modelled (planned).
- **Ethnicity is one snapshot per state**, not matched to each election year.
- **Serving layer in transition:** the chatbot "what if" tool and the legacy pipeline still use older feature code.
- **National indicators for state elections:** KLCI and USD/MYR are national signals applied to state predictions — a known granularity mismatch. State-level economic data (BNM GDP by state) is annual and too sparse for LSTM.
- **English/Malay news sources only:** Chinese-language media (Sin Chew, Guang Ming) is not scraped, which limits sentiment signal quality for Chinese-majority seats.
- **LSTM directional accuracy ~52%:** Consistent with efficient market hypothesis for daily price prediction. The economic_pressure_score is meaningful as a trend signal over 90-day election periods, not for daily trading.
- **Melaka (and Selangor, Perak) have no 2026 result yet.** Forecasts will be frozen and timestamped before the elections, then scored.

---

## Honest Negative Results

| Finding | Implication |
|---------|-------------|
| Accuracy was inflated by leakage (94.64% -> 85.71% for Johor) | Always test on a later election and compare with simple baselines |
| Model ties "never opposition" in Johor 2026 | Little skill beyond a trivial rule at the 0.5 cut-off |
| Pooling all states: mixed | Too few independent elections to conclude |
| State-level sentiment = 0% feature importance | Constant features can't be split by trees |
| LSTM directional accuracy = 52% | Daily price prediction near-random (EMH confirmed) |

---

## Data Sources

| Source | Usage | License |
|--------|-------|---------|
| electiondata.my | Election results + voter rolls | CC0 |
| yfinance | KLCI + USD/MYR prices | Yahoo Finance ToS |
| FMT, Malaysiakini, Malay Mail, Utusan, Bernama | Sentiment scoring | RSS public feeds |
| Bank Negara Malaysia API | OPR data | BNM open data |

---

## Author

Built as an AI/ML portfolio project demonstrating end-to-end system design: from raw data collection through feature engineering, model training, deployment, RAG integration, and MLOps automation.

Built with AI assistance (Claude): the design decisions, evaluation and testing are mine; the AI helped write and review code.

> Predicting who wins — and being honest about why the model fails — is more valuable than inflating accuracy numbers.
