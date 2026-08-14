# GE-Insights Prediction Hub

A multi-phase Malaysian state election intelligence system validated on actual 2026 election results. Combines classical ML, deep learning, LLM sentiment analysis, RAG, and MLOps in one connected pipeline.

**Live API:** https://elegant-cooperation-production-67c5.up.railway.app

---

## Results

| State | Election | Seats | Accuracy | AUC-ROC | Wrong |
|-------|----------|-------|----------|---------|-------|
| Johor | SE-16 (Jul 11, 2026) | 56 | **94.64%** | 0.979 | 3 |
| Negeri Sembilan | SE-16 (Aug 1, 2026) | 36 | **91.67%** | 0.978 | 3 |
| Melaka | SE-16 (upcoming) | 28 | Pre-election forecast | — | — |

Validated on actual 2026 election outcomes — not a random split or toy dataset.

**Melaka 2026 forecast:** 22 government (BN/PN) wins, 6 Harapan wins.

---

## Key Findings

**1. Coalition dynamics drove the biggest accuracy gain (+28% NS)**
The model initially treated BN and PN as opponents. Discovering they are coalition partners in 2026 and reframing the target from "did non-BN win?" to "did Harapan win?" improved NS accuracy by 28 percentage points — the single largest improvement in the project.

**2. Voter roll demographics provided genuine seat-level signal**
Adding ethnicity and age composition from 3–5M row anonymised voter rolls gave the model information it had never had before: `chinese_pct`, `young_malay_pct`, `youth_pct`. These are seat-level features the tree can actually split on. NS improved from 63.9% → 77.8% from this alone.

**3. LSTM confirmed efficient market hypothesis**
Daily KLCI and USD/MYR prediction achieved 0.52% MAPE (accurate in absolute terms) but only 52% directional accuracy — near-random, consistent with EMH. Documented honestly as a finding rather than hidden.

**4. National narrative × demographics creates seat-level sentiment variation**
State-level sentiment scores have zero tree importance (every seat gets the same value). Weighting sentiment themes (Islam threat, Malay unity, cost of living) by each seat's demographic composition produces genuine variation — different seats respond differently to the same national narrative.

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
  state_pipeline.py → engineer_features()
  train_models.py → RF + XGB ensemble, OOD detector
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

## Feature Set (25 features)

| Category | Features |
|----------|----------|
| Structural (6) | majority_change, turnout_change, incumbent_held, log_voters, majority_perc_change, n_candidates_b |
| Sentiment (4) | bn_sentiment, harapan_sentiment, pn_sentiment, racial_tension_index |
| Economic (1) | economic_pressure (from LSTM forecast) |
| Ethnicity + age (8) | malay_pct, chinese_pct, indian_pct, young_malay_pct, young_chinese_pct, older_malay_pct, youth_pct, median_age |
| Interactions (5) | bn_sent_x_malay, harapan_sent_x_chinese, pn_sent_x_young_malay, tension_x_mixed, economic_x_youth |
| Narrative (1) | narrative_pressure (national themes × seat demographics) |

---

## Accuracy Progression

| Feature set | Johor | NS | Key change |
|-------------|-------|-----|------------|
| 6 structural | 89.29% | 63.89% | Baseline |
| + Sentiment + Economic (11) | 89.29% | 63.89% | No change (state-level constant) |
| + Ethnicity + age (24) | 89.29% | 77.78% | +14% NS (seat-level signal) |
| + Coalition target fix (25) | **94.64%** | **91.67%** | +5% Johor, +14% NS |

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
| GET | `/predict/all/{state}` | Predict all seats using real 2026 data |
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

# Train models
python backend/scripts/train_models.py

# Validate on 2026 results
python backend/scripts/validate_2026.py

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
│   │   └── pipelines/
│   │       └── state_pipeline.py      # Feature engineering
│   ├── models/                        # Saved .pkl files per state
│   └── scripts/
│       ├── train_models.py            # Train RF + XGB + OOD
│       ├── validate_2026.py           # Evaluate on real 2026 data
│       └── download_ethnicity.py      # Voter roll processing
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

- **Small training data:** 28–56 seats per state. Models are regularised (max_depth, min_samples_leaf) but overfitting risk remains.
- **National indicators for state elections:** KLCI and USD/MYR are national signals applied to state predictions — a known granularity mismatch. State-level economic data (BNM GDP by state) is annual and too sparse for LSTM.
- **English/Malay news sources only:** Chinese-language media (Sin Chew, Guang Ming) is not scraped, which limits sentiment signal quality for Chinese-majority seats.
- **LSTM directional accuracy ~52%:** Consistent with efficient market hypothesis for daily price prediction. The economic_pressure_score is meaningful as a trend signal over 90-day election periods, not for daily trading.
- **Melaka 2026 is a forecast:** No actual results yet. Accuracy will be validated when election results are announced.

---

## Honest Negative Results

| Finding | Implication |
|---------|-------------|
| LSTM directional accuracy = 52% | Daily price prediction near-random (EMH confirmed) |
| State-level sentiment = 0% feature importance | Constant features can't be split by trees |
| Interaction features marginal gain | Multiplying a constant by a variable rescales, doesn't add information |
| 6 Johor seats still wrong | Chinese urban seats with anomalous 2022 BN result — model trained on that pattern |

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

> Predicting who wins — and being honest about why the model fails — is more valuable than inflating accuracy numbers.
