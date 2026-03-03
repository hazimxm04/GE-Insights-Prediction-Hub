# GE Insights — Election Prediction Hub 🗳️

> A full-stack election prediction system for Malaysian General Elections (GE12–GE14) featuring two AI pipelines — a Random Forest ML pipeline for predicting parliamentary seat outcomes, and an LLM pipeline using Gemini AI to generate contextual political analysis from model outputs Disclaimer: GEemini API is unavailable unless if theres a usable API key to be used as it needs to be purchased D:


---

## What It Does

GE Insights lets you explore and simulate outcomes across GE12, GE13, and GE14 using a trained Random Forest classifier. You can:

- **Predict** win probability for any parliamentary seat, state, or region under simulated conditions
- **Simulate** how changes occur using swing rate, turnout rate, and base voter rate to see how they affect coalition outcomes
- **Compare** coalition performance between two election years head-to-head — including which seats changed hands
- **Explain** every prediction with an AI-generated political analysis powered by Gemini 
- **Evaluate** model reliability through live accuracy, F1 score, confusion matrix, and cross-validation metrics on the dashboard

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React + TypeScript + Tailwind CSS |
| Backend | FastAPI (Python) |
| ML Model | scikit-learn Random Forest + Pipeline |
| Database | Supabase (PostgreSQL) |
| LLM Layer | Google Gemini AI |
| Data Pipeline | pandas, numpy, custom feature engineering |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   React Frontend                     │
│  Sidebar Controls → Dashboard → AI Analysis Card    │
└────────────────────┬────────────────────────────────┘
                     │ REST API
┌────────────────────▼────────────────────────────────┐
│                  FastAPI Backend                     │
│                                                      │
│  /predict-summary   → ML prediction engine          │
│  /api/compare       → cross-year coalition analysis │
│  /api/explain       → Gemini LLM explanation        │
│  /api/model-report  → evaluation metrics            │
│  /api/options       → dynamic filter options        │
└──────┬──────────────────────────┬───────────────────┘
       │                          │
┌──────▼──────┐          ┌────────▼────────┐
│  Supabase   │          │   Gemini API    │
│ (GE12-GE14  │          │  (explanation)  │
│  ~666 rows) │          └─────────────────┘
└─────────────┘
```

---

## ML Pipeline

The prediction model uses a **Random Forest Classifier** wrapped in a scikit-learn Pipeline with ordinal encoding for categorical features.

**Features used:**
- `relative_vote_margin` — engineered from historical vote share per seat
- `turnout_rate` — simulated based on slider input based on the actual total voters come to vote
- `log_total_voters` — log-scaled registered voter count
- `state`, `region` — geographic context
- `standardized_coalition` — normalized coalition names across GE datasets 

**Simulation flow:**
```
User Input (swing %, turnout %, base voter rate)
        ↓
Apply swing to selected coalition only
Opponents stay at historical vote share
        ↓
clean_dataset() recomputes relative_vote_margin
        ↓
RandomForest.predict_proba() → win probability
        ↓
Gemini AI generates plain-English political analysis
```

---

## Features

### Predict Mode
- Select scope: **Seat / State / Region**
- Choose target from dynamically loaded options (year-specific)
- Select coalition — options change per GE year (GE13 shows *Pakatan Rakyat*, GE14 shows *Harapan* etc.)
- Adjust **Swing Rate**, **Turnout Rate**, **Base Voter Rate** sliders
- Swing applied only to selected coalition — opponents stay historical
- Results show win probability (single seat) or average confidence (multi-seat)

### Compare Mode
- Select two GE years (Dataset A vs Dataset B)
- Choose a **different coalition per dataset** — accounts for coalition name changes across elections
- Filter by Seat / State / Region
- Shows head-to-head seat counts, win rates, delta, regional breakdown, and seats that changed hands

### Model Performance Card
- Always visible on the dashboard
- Shows Accuracy / Precision / Recall / F1
- 5-fold CV bar chart with mean ± std
- Confusion matrix (colour-coded)
- Updates when you switch dataset year
---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- Supabase account with election data loaded
- Google Gemini API key (free at aistudio.google.com)

### Backend Setup

```bash
cd backend
pip install -r requirements.txt

# Create your .env file
cp .env.example .env
# Fill in SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY

# Train models and generate evaluation reports
python model_training.py

# Start the API server
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```
---

## Environment Variables

Create a `.env` file in the `backend/` folder:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
GEMINI_API_KEY=your_gemini_api_key
```

See `.env.example` for the template.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict-summary` | Predict outcomes with simulation params |
| `GET` | `/api/compare` | Compare coalition performance between two GE years |
| `POST` | `/api/explain` | Generate Gemini AI analysis of a prediction |
| `GET` | `/api/options` | Fetch year-specific states, seats, regions, coalitions |
| `GET` | `/api/model-report` | Return saved ML evaluation metrics |

---

## Dataset

Historical results from **GE12 (2008), GE13 (2013), GE14 (2018)** — covering all 222 parliamentary seats per election stored in Supabase.

| Year | Coalitions |
|------|-----------|
| GE12 | Barisan, Pakatan Rakyat, Others |
| GE13 | Barisan, Pakatan Rakyat, Others |
| GE14 | Barisan, Harapan, PAS, Warisan, BERJASA, Others |

## Future Improvement
1. Include future and more past GE datasets (GE15, GE11 etc.)
2. Include State Level Seat Prediction
3. Include more feature sin predictions (Ethnicities, Income etc)
4. Implement other Multi-algorithm benchmarking (XGBoost, ensemble methods) to compare against the current Random Forest pipeline for better accuracies.

---

## Project Structure

```
GE-Insights-Prediction-Hub/
├── README.md
├── .gitignore
├── backend/
│   ├── main.py                 # FastAPI app — all endpoints
│   ├── analysis.py             # ML training + cross-year comparison
│   ├── predictor.py            # Model loading + prediction engine
│   ├── model_training.py       # Train models + save evaluation reports
│   ├── gedataset_pipeline.py   # Feature engineering + clean_dataset()
│   ├── database.py             # Supabase client + data fetching
│   └── .env.example            # Environment variable template
└── frontend/
    └── src/
        └── components/
            └── dashboard/
                ├── Dashboard.tsx
                ├── sidebar.tsx
                └── BellCurve.tsx
```

---

## Author

Thank you for reading till the end of this. I hope you will enjoy this reading as much as I enjoy developing this project in my previous Winter break :D

---
