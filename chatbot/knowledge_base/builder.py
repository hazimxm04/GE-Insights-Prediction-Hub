"""
chatbot/knowledge_base/builder.py
===================================
Builds the ChromaDB knowledge base for the GE-Insights RAG chatbot.

Reads LIVE data (metadata.json, validation_2026.csv, or a live
StatePredictor forecast) for all 5 states, rather than hardcoded
numbers -- so the knowledge base reflects whatever the current
trained models actually say, not a stale snapshot from an earlier
run.

Usage:
    python chatbot/knowledge_base/builder.py
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Load env ────────────────────────────────────────────────────────

env_paths = [
    Path("backend/.env"),
    ROOT / "backend" / ".env",
]
for p in env_paths:
    if p.exists():
        load_dotenv(p)
        break

# ── Paths ───────────────────────────────────────────────────────────

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "backend" / "models"
CHROMA_DIR    = ROOT / "chatbot" / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

ALL_STATES = ['johor', 'neg_sembilan', 'selangor', 'melaka', 'perak']

# States that have held their 2026 election (real ground truth exists)
VALIDATED_STATES = {'johor', 'neg_sembilan'}


# ── ChromaDB setup ──────────────────────────────────────────────────

def get_chroma_client():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client, embedding_fn


# ── Document builders ───────────────────────────────────────────────

def build_prediction_docs(state: str) -> list:
    """
    Build prediction documents for one state. Uses real validated
    results (validation_2026.csv) if the state's election has
    happened; otherwise falls back to a live forecast via
    StatePredictor, clearly labeled as unvalidated.
    """
    val_path = MODELS_DIR / state / 'validation_2026.csv'

    if val_path.exists():
        df = pd.read_csv(val_path)
        docs = []

        for _, row in df.iterrows():
            seat    = row.get('seat', '')
            pred    = row.get('predicted_non_bn', row.get('prediction', ''))
            actual  = row.get('actual_non_bn', row.get('actual', ''))
            prob    = row.get('probability', 0)
            is_ood  = row.get('is_ood', False)
            correct = row.get('correct', str(pred).strip() == str(actual).strip())

            correctness = (
                f"WRONG PREDICTION - model predicted {pred} but actual result was {actual}"
                if not correct else
                f"Correct prediction - model predicted {pred}, actual was {actual}"
            )

            text = f"""
Seat: {seat} ({state.replace('_', ' ').title()})
Model prediction: {pred} (probability: {float(prob):.2f})
Actual 2026 result: {actual}
{correctness}
OOD flagged: {'Yes - unusual pattern' if is_ood else 'No'}
State: {state}
Status: VALIDATED against real 2026 election results
            """.strip()

            docs.append({
                'text': text,
                'metadata': {
                    'type':       'prediction',
                    'state':      state,
                    'seat':       str(seat),
                    'prediction': str(pred),
                    'actual':     str(actual),
                    'correct':    str(correct),
                    'is_ood':     str(is_ood),
                    'validated':  'true',
                }
            })

        wrong_count = sum(1 for _, r in df.iterrows() if not r.get('correct', True))
        print(f"  Built {len(docs)} VALIDATED prediction docs for {state} ({wrong_count} wrong)")
        return docs

    else:
        try:
            from backend.core.models.state_predictor import StatePredictor

            predictor = StatePredictor(state)
            forecast_df = predictor.predict_all()

            if forecast_df.empty:
                print(f"  No forecast available for {state} (election data not ready)")
                return []

            docs = []
            for _, row in forecast_df.iterrows():
                seat   = row.get('seat_name', '')
                pred   = row.get('prediction', '')
                prob   = row.get('probability', 0)
                is_ood = row.get('is_ood', False)

                text = f"""
Seat: {seat} ({state.replace('_', ' ').title()})
Model forecast: {pred} (probability: {float(prob):.2f})
OOD flagged: {'Yes - unusual pattern' if is_ood else 'No'}
State: {state}
Status: LIVE FORECAST -- election has not happened yet, this is
NOT a validated result. Accuracy will only be known once the
{state.replace('_', ' ').title()} election takes place.
                """.strip()

                docs.append({
                    'text': text,
                    'metadata': {
                        'type':       'prediction',
                        'state':      state,
                        'seat':       str(seat),
                        'prediction': str(pred),
                        'is_ood':     str(is_ood),
                        'validated':  'false',
                    }
                })

            print(f"  Built {len(docs)} LIVE FORECAST docs for {state} (not yet validated)")
            return docs

        except Exception as e:
            print(f"  Could not build forecast docs for {state}: {e}")
            return []


def build_sentiment_docs() -> list:
    """Convert scored_articles.csv into documents, one per article."""
    path = PROCESSED_DIR / "scored_articles.csv"
    if not path.exists():
        print("  No scored articles found")
        return []

    df = pd.read_csv(path).head(200)
    docs = []

    for _, row in df.iterrows():
        title   = row.get('title', '')
        source  = row.get('source', '')
        bn      = row.get('bn_sentiment', 0)
        harapan = row.get('harapan_sentiment', 0)
        pn      = row.get('pn_sentiment', 0)
        tension = row.get('racial_tension', 0)
        snippet = str(row.get('text_snippet', ''))[:300]

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
                'title': str(title)[:100],
                'bn_sentiment': str(bn),
                'harapan_sentiment': str(harapan),
                'pn_sentiment': str(pn),
            }
        })

    print(f"  Built {len(docs)} sentiment article docs")
    return docs


def build_state_summary_docs() -> list:
    """
    Build high-level state summary documents combining prediction
    accuracy + sentiment + economic context. Reads accuracy/wrong-
    seat counts LIVE from each state's metadata.json and
    validation_2026.csv instead of hardcoding numbers.
    """
    docs = []

    sentiment_path = PROCESSED_DIR / "state_sentiment_scores.csv"
    economic_path  = PROCESSED_DIR / "election_economic_pressure.csv"

    sentiment_df = pd.read_csv(sentiment_path) if sentiment_path.exists() else pd.DataFrame()
    economic_df  = pd.read_csv(economic_path)  if economic_path.exists()  else pd.DataFrame()

    for state in ALL_STATES:
        election_status = "2026 (validated)" if state in VALIDATED_STATES else "pending"

        metadata_path = MODELS_DIR / state / 'metadata.json'
        seats = 'unknown'
        accuracy_text = "Not yet validated -- this state's election is pending."
        wrong_seats_text = "N/A (pending election)"

        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            seats = meta.get('seats', 'unknown')

            if 'validated_accuracy' in meta:
                acc = meta['validated_accuracy']
                accuracy_text = f"{acc:.2%} (validated against real 2026 results)"
            else:
                train_acc = meta.get('models', {}).get('ensemble', {}).get('accuracy')
                if train_acc is not None:
                    accuracy_text = (f"{train_acc:.2%} training accuracy "
                                     f"(NOT validated -- election pending or "
                                     f"validation.py not yet run for this state)")

        val_path = MODELS_DIR / state / 'validation_2026.csv'
        if val_path.exists():
            val_df = pd.read_csv(val_path)
            if 'correct' in val_df.columns:
                wrong_df = val_df[val_df['correct'] == False]
                wrong_names = wrong_df['seat'].tolist() if 'seat' in wrong_df.columns else []
                wrong_seats_text = (f"{len(wrong_names)} seats: {', '.join(wrong_names)}"
                                     if wrong_names else "0 (all correct)")

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
Election status: {election_status}
Total DUN seats: {seats}
Model accuracy: {accuracy_text}
Wrong predictions: {wrong_seats_text}
{sent_text}
{econ_text}

Model uses structural voting patterns and voter roll demographics
at training time. Sentiment and narrative features are not used by the current models.
Prediction target: does Pakatan Harapan (the opposition) win the seat, or does it go to BN/PN? BN-PN
coalition dynamics vary by state and election cycle -- see
docs/multiclass_analysis.md for the full methodology and known
limitations.
        """.strip()

        docs.append({
            'text': text,
            'metadata': {'type': 'state_summary', 'state': state}
        })

    print(f"  Built {len(docs)} state summary docs (live data)")
    return docs


def build_wrong_predictions_summary_doc() -> dict:
    """
    Build the 'wrong predictions summary' document dynamically from
    each validated state's real validation_2026.csv.
    """
    sections = []

    for state in sorted(VALIDATED_STATES):
        val_path = MODELS_DIR / state / 'validation_2026.csv'
        metadata_path = MODELS_DIR / state / 'metadata.json'

        if not val_path.exists():
            continue

        val_df = pd.read_csv(val_path)
        total = len(val_df)
        wrong_df = val_df[val_df['correct'] == False] if 'correct' in val_df.columns else pd.DataFrame()
        correct_count = total - len(wrong_df)

        lines = [f"{state.replace('_', ' ').title()} 2026 ({len(wrong_df)} wrong seats):"]
        for _, row in wrong_df.iterrows():
            seat = row.get('seat', 'unknown')
            pred = row.get('predicted_non_bn', row.get('prediction', 'unknown'))
            actual = row.get('actual_non_bn', row.get('actual', 'unknown'))
            lines.append(f"  {seat}: predicted={pred}, actual={actual}")

        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            acc = meta.get('validated_accuracy')
            if acc is not None:
                lines.append(f"Overall accuracy: {acc:.2%} ({correct_count}/{total} correct)")

        sections.append("\n".join(lines))

    text = "Wrong predictions summary (live, from validation_2026.csv):\n\n" + "\n\n".join(sections)

    return {
        'text': text.strip(),
        'metadata': {'type': 'wrong_predictions_summary', 'state': 'all'}
    }


# ── Build ChromaDB ──────────────────────────────────────────────────

def build_knowledge_base():
    """Build complete ChromaDB knowledge base from all project data."""
    print("Building GE-Insights knowledge base...\n")
    print("=" * 55)

    client, embedding_fn = get_chroma_client()

    try:
        client.delete_collection("ge_insights")
        print("Deleted existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name="ge_insights",
        embedding_function=embedding_fn,
        metadata={"description": "GE-Insights election prediction knowledge base"}
    )

    all_docs = []

    print("\nBuilding documents:")
    for state in ALL_STATES:
        all_docs.extend(build_prediction_docs(state))

    all_docs.extend(build_sentiment_docs())
    all_docs.extend(build_state_summary_docs())
    all_docs.append(build_wrong_predictions_summary_doc())

    print(f"\nAdding {len(all_docs)} documents to ChromaDB...")

    texts = [d['text'] for d in all_docs]
    metas = [d['metadata'] for d in all_docs]
    ids   = [f"doc_{i}" for i in range(len(all_docs))]

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        collection.add(
            documents=texts[i:i + batch_size],
            metadatas=metas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )
        print(f"  Added batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

    print(f"\nKnowledge base built successfully!")
    print(f"  Total documents: {collection.count()}")
    print(f"  Stored in: {CHROMA_DIR}")

    print(f"\nTest search: 'What is the accuracy for Johor?'")
    results = collection.query(
        query_texts=["What is the accuracy for Johor?"],
        n_results=2
    )
    for doc in results['documents'][0]:
        print(f"  -> {doc[:150]}...")


if __name__ == "__main__":
    build_knowledge_base()