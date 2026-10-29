"""
builder.py
==========
Builds the ChromaDB vector store from your project data:
  1. Election predictions (validation_2026.csv per state)
  2. Scored sentiment articles (scored_articles.csv)
  3. State sentiment scores (state_sentiment_scores.csv)
  4. Economic pressure scores (election_economic_pressure.csv)
  5. Melaka 2026 prediction (melaka_2026_prediction.csv)

Each piece of data becomes a "document" in ChromaDB.
Documents are embedded as vectors so semantic search works:
  "Why did BN win?" finds relevant BN prediction docs
  "What is racial tension?" finds relevant sentiment docs

Usage:
    python chatbot/knowledge_base/builder.py
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# ── Load env ────────────────────────────────────────────────────────

env_paths = [
    Path("backend/.env"),
    Path(__file__).resolve().parents[3] / "backend" / ".env",
]
for p in env_paths:
    if p.exists():
        load_dotenv(p)
        break

# ── Paths ───────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
CHROMA_DIR    = Path("chatbot/chroma_db")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── ChromaDB setup ──────────────────────────────────────────────────

def get_chroma_client():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client, embedding_fn


# ── Document builders ───────────────────────────────────────────────

def build_prediction_docs(state: str) -> list:
    path = Path(f"backend/models/{state}/validation_2026.csv")
    if not path.exists():
        print(f"  No validation data for {state}")
        return []

    df = pd.read_csv(path)
    docs = []

    for _, row in df.iterrows():
        seat    = row.get('seat', '')
        pred    = row.get('prediction', '')
        actual  = row.get('actual', '')
        prob    = row.get('probability', 0)
        is_ood  = row.get('is_ood', False)
        correct = str(pred).strip() == str(actual).strip()

        # Make wrong predictions very explicit
        if not correct:
            correctness = f"WRONG PREDICTION - model predicted {pred} but actual result was {actual}"
        else:
            correctness = f"Correct prediction - model predicted {pred}, actual was {actual}"

        text = f"""
Seat: {seat} ({state.replace('_', ' ').title()})
Model prediction: {pred} (probability: {prob:.2f})
Actual 2026 result: {actual}
{correctness}
OOD flagged: {'Yes - unusual pattern' if is_ood else 'No'}
State: {state}
        """.strip()

        docs.append({
            'text': text,
            'metadata': {
                'type':       'prediction',
                'state':      state,
                'seat':       seat,
                'prediction': str(pred),
                'actual':     str(actual),
                'correct':    str(correct),
                'is_ood':     str(is_ood),
            }
        })

    wrong_count = sum(1 for _, r in df.iterrows()
                      if str(r.get('prediction','')).strip() != str(r.get('actual','')).strip())
    print(f"  Built {len(docs)} prediction docs for {state} ({wrong_count} wrong)")
    return docs


def build_sentiment_docs() -> list:
    """
    Convert scored_articles.csv into documents.
    Each article becomes one document with its sentiment scores.
    """
    path = PROCESSED_DIR / "scored_articles.csv"
    if not path.exists():
        print("  No scored articles found")
        return []

    df = pd.read_csv(path).head(200)  # limit to 200 most recent
    docs = []

    for _, row in df.iterrows():
        title   = row.get('title', '')
        source  = row.get('source', '')
        bn      = row.get('bn_sentiment', 0)
        harapan = row.get('harapan_sentiment', 0)
        pn      = row.get('pn_sentiment', 0)
        tension = row.get('racial_tension', 0)
        snippet = row.get('text_snippet', '')[:300]

        text = f"""
News article: {title}
Source: {source}
BN sentiment: {bn:+.2f}
Harapan sentiment: {harapan:+.2f}
PN sentiment: {pn:+.2f}
Racial tension: {tension:.2f}
Content: {snippet}
        """.strip()

        docs.append({
            'text': text,
            'metadata': {
                'type': 'sentiment_article',
                'source': str(source),
                'title': str(title[:100]),
                'bn_sentiment': str(bn),
                'harapan_sentiment': str(harapan),
                'pn_sentiment': str(pn),
            }
        })

    print(f"  Built {len(docs)} sentiment article docs")
    return docs


def build_state_summary_docs() -> list:
    """
    Build high-level state summary documents combining
    prediction accuracy + sentiment + economic context.
    """
    docs = []

    # State sentiment scores
    sentiment_path = PROCESSED_DIR / "state_sentiment_scores.csv"
    economic_path  = PROCESSED_DIR / "election_economic_pressure.csv"

    sentiment_df = pd.read_csv(sentiment_path) if sentiment_path.exists() else pd.DataFrame()
    economic_df  = pd.read_csv(economic_path)  if economic_path.exists()  else pd.DataFrame()

    state_info = {
        'johor': {
            'election_date': 'July 11, 2026',
            'total_seats': 56,
            'accuracy': '94.64%',
            'wrong_seats': 3,
        },
        'neg_sembilan': {
            'election_date': 'August 1, 2026',
            'total_seats': 36,
            'accuracy': '91.67%',
            'wrong_seats': 3,
        },
        'melaka': {
            'election_date': 'Upcoming 2026',
            'total_seats': 28,
            'accuracy': 'Pre-election forecast',
            'wrong_seats': 'N/A',
        },
    }

    docs.append({
        'text': """
    Melaka 2026 election overall prediction summary:
    Total seats: 28
    Predicted government wins (BN/PN): 22 seats
    Predicted Harapan wins: 6 seats
    BN/PN expected majority: comfortable

    Predicted Harapan seats (Chinese-majority):
    N.22 Bandar Hilir (chinese=73%): Harapan P=1.00
    N.20 Kota Laksamana (chinese=77%): Harapan P=1.00
    N.19 Kesidang (chinese=56%): Harapan P=1.00
    N.16 Ayer Keroh (chinese=48%): Harapan P=1.00
    N.17 Bukit Katil (mixed): Harapan P=0.52
    N.11 Sungai Udang (Malay): Harapan P=0.52 OOD

    Predicted BN/PN seats: remaining 22 rural Malay seats
    Pattern: Chinese-majority seats go Harapan,
            Malay-majority seats go BN/PN.
    This forecast uses 2018->2021 patterns + 25 features
    including voter roll demographics and national sentiment.
        """.strip(),
        'metadata': {
            'type': 'melaka_summary',
            'state': 'melaka',
        }
    })

    for state, info in state_info.items():
        # Get sentiment for this state
        sent_row = sentiment_df[sentiment_df['state'] == state] if not sentiment_df.empty else pd.DataFrame()
        econ_key = f"{state}_2026"
        econ_row = economic_df[economic_df['state'] == econ_key] if not economic_df.empty else pd.DataFrame()

        sent_text = ""
        if not sent_row.empty:
            row = sent_row.iloc[0]
            sent_text = f"""
Current sentiment ({state}):
  BN sentiment: {row.get('bn_sentiment', 0):+.2f}
  Harapan sentiment: {row.get('harapan_sentiment', 0):+.2f}
  PN sentiment: {row.get('pn_sentiment', 0):+.2f}
  Racial tension index: {row.get('racial_tension_index', 0):.3f}
  Dominant coalition in media: {row.get('dominant_sentiment', 'Unknown')}"""

        econ_text = ""
        if not econ_row.empty:
            score = econ_row.iloc[0].get('economic_pressure_score', 0)
            level = "HIGH STRESS" if score < -0.3 else "MODERATE" if score < 0 else "POSITIVE"
            econ_text = f"\nEconomic pressure score: {score:.4f} ({level})"

        text = f"""
State: {state.replace('_', ' ').title()}
Election date: {info['election_date']}
Total DUN seats: {info['total_seats']}
Model accuracy on 2026 results: {info['accuracy']}
Wrong predictions: {info['wrong_seats']}
{sent_text}
{econ_text}

Key finding: The model uses 25 features including structural
voting patterns, voter roll demographics (3-5M rows),
LLM sentiment scoring, and PyTorch LSTM economic forecasting.
Coalition target: BN+PN = government bloc, Harapan = opposition.
        """.strip()

        docs.append({
            'text': text,
            'metadata': {
                'type': 'state_summary',
                'state': state,
            }
        })

    print(f"  Built {len(docs)} state summary docs")
    return docs


def build_melaka_prediction_docs() -> list:
    """Build documents from Melaka 2026 pre-election prediction."""
    path = PROCESSED_DIR / "melaka_2026_prediction.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    docs = []

    for _, row in df.iterrows():
        seat      = row.get('seat_name', '')
        pred      = row.get('prediction', '')
        prob      = row.get('probability', 0)
        is_ood    = row.get('is_ood', False)
        malay_pct = row.get('malay_pct', 0)
        chin_pct  = row.get('chinese_pct', 0)
        incumbent = row.get('incumbent', '')

        text = f"""
Melaka 2026 pre-election prediction:
Seat: {seat}
2021 incumbent: {incumbent}
2026 forecast: {pred} (probability: {prob:.2f})
Malay voter composition: {malay_pct:.1%}
Chinese voter composition: {chin_pct:.1%}
OOD flagged: {'Yes' if is_ood else 'No'}
Note: This is a forecast before the election. 
Accuracy will be validated when results are announced.
        """.strip()

        docs.append({
            'text': text,
            'metadata': {
                'type': 'melaka_forecast',
                'seat': str(seat),
                'prediction': str(pred),
                'probability': str(prob),
            }
        })

    print(f"  Built {len(docs)} Melaka forecast docs")
    return docs

def build_selangor_prediction_docs() -> list:
    path = PROCESSED_DIR / "selangor_2026_bluwave.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    docs = []

    for _, row in df.iterrows():
        seat        = row.get('seat_name', '')
        prob        = row.get('harapan_holds_probability', 0)
        vulnerability = row.get('vulnerability', '')
        is_ood      = row.get('is_ood', False)
        malay_pct   = row.get('malay_pct', 0)
        chin_pct    = row.get('chinese_pct', 0)
        winner_2023 = row.get('winner_2023', '')
        ph_2008     = row.get('ph_won_2008', 0)
        ph_2013     = row.get('ph_won_2013', 0)
        ph_2018     = row.get('ph_won_2018', 0)

        prediction = 'Harapan' if prob >= 0.5 else 'BN/PN'

        text = f"""
Selangor blue wave vulnerability analysis:
Seat: {seat}
2023 winner: {winner_2023}
Vulnerability category: {vulnerability}
Harapan holds probability: {prob:.2f} (prediction: {prediction})
Historical PH loyalty: won 2008={bool(ph_2008)}, 2013={bool(ph_2013)}, 2018={bool(ph_2018)}
Malay voter composition: {malay_pct:.1%}
Chinese voter composition: {chin_pct:.1%}
OOD flagged: {'Yes' if is_ood else 'No'}
Note: This is a blue wave scenario analysis, not tied to a specific
election date. Combines historical PH loyalty with Johor-model transfer
and current sentiment pressure.
        """.strip()

        docs.append({
            'text': text,
            'metadata': {
                'type': 'selangor_forecast',
                'seat': str(seat),
                'state': 'selangor',
                'prediction': prediction,
                'vulnerability': str(vulnerability),
            }
        })

    print(f"  Built {len(docs)} Selangor forecast docs")
    return docs

# ── Build ChromaDB ──────────────────────────────────────────────────

def build_knowledge_base():
    """Build complete ChromaDB knowledge base from all project data."""
    print("Building GE-Insights knowledge base...\n")
    print("="*55)

    client, embedding_fn = get_chroma_client()

    # Delete existing collection if rebuilding
    try:
        client.delete_collection("ge_insights")
        print("Deleted existing collection")
    except:
        pass

    collection = client.create_collection(
        name="ge_insights",
        embedding_function=embedding_fn,
        metadata={"description": "GE-Insights election prediction knowledge base"}
    )

    # Collect all documents
    all_docs = []

    print("\nBuilding documents:")
    for state in ['johor', 'neg_sembilan', 'melaka']:
        all_docs.extend(build_prediction_docs(state))

    all_docs.extend(build_sentiment_docs())
    all_docs.extend(build_state_summary_docs())
    all_docs.extend(build_melaka_prediction_docs())
    all_docs.extend(build_selangor_prediction_docs())

    # Add before collection.add()
    wrong_summary = """
    Wrong predictions summary:

    Johor 2026 (3 wrong seats):
    N.12 Bentayan: predicted BN, actual Harapan (chinese_pct=70%)
    N.13 Simpang Jeram: predicted BN, actual Harapan (chinese_pct=45%)
    N.41 Puteri Wangsa: predicted BN, actual Harapan (chinese_pct=52%)
    N.45 Stulang: predicted BN, actual Harapan (chinese_pct=53%)
    N.48 Skudai: predicted BN, actual Harapan (chinese_pct=61%)
    N.52 Senai: predicted BN, actual Harapan (chinese_pct=55%)
    Pattern: all Chinese-majority urban seats that flipped back to Harapan

    Negeri Sembilan 2026 (3 wrong seats):
    N.01 Chennah: predicted non-BN, actual BN
    N.18 Pilah: predicted non-BN, actual BN
    N.36 Repah: predicted non-BN, actual BN
    Pattern: Malay BN strongholds where model overcorrected

    Overall accuracy:
    Johor: 94.64% (53/56 correct)
    NS: 91.67% (33/36 correct)
    """

    all_docs.append({
        'text': wrong_summary.strip(),
        'metadata': {'type': 'wrong_predictions_summary', 'state': 'all'}
    })

    # Add to ChromaDB
    print(f"\nAdding {len(all_docs)} documents to ChromaDB...")

    texts    = [d['text'] for d in all_docs]
    metas    = [d['metadata'] for d in all_docs]
    ids      = [f"doc_{i}" for i in range(len(all_docs))]

    # Add in batches of 100
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        collection.add(
            documents=texts[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
            ids=ids[i:i+batch_size],
        )
        print(f"  Added batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

    print(f"\nKnowledge base built successfully!")
    print(f"  Total documents: {collection.count()}")
    print(f"  Stored in: {CHROMA_DIR}")

    # Test search
    print(f"\nTest search: 'Why did BN win Johor?'")
    results = collection.query(
        query_texts=["Why did BN win Johor?"],
        n_results=2
    )
    for doc in results['documents'][0]:
        print(f"  → {doc[:100]}...")


if __name__ == "__main__":
    build_knowledge_base()