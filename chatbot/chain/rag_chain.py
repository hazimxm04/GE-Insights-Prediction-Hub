"""
rag_chain.py
============
RAG chatbot + function calling for live "what if" scenario predictions.

Two modes:
  1. RAG retrieval (existing) — answers factual questions from
     the knowledge base (validation results, sentiment, etc.)
  2. Tool calling (new) — when user asks a "what if" scenario
     question, calls predict_seat() live with real seat baseline
     + user's requested changes, then explains the result.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# -- Load env ----------------------------------------------------------

for p in [Path("backend/.env"), Path(__file__).resolve().parents[2] / "backend/.env"]:
    if p.exists():
        load_dotenv(p)
        break

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in backend/.env")

CHROMA_DIR = Path(__file__).resolve().parents[2] / "chatbot/chroma_db"
MODEL      = "openai/gpt-oss-20b"
N_RESULTS  = 4

groq_client = Groq(api_key=GROQ_API_KEY)


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection(name="ge_insights", embedding_function=embedding_fn)


# ======================================================================
# TOOL: live scenario prediction
# ======================================================================

def get_seat_baseline(state: str, seat_name: str) -> dict:
    """Get a seat's REAL structural features as defaults."""
    from backend.core.pipelines.state_pipeline import StateElectionPipeline

    pipeline = StateElectionPipeline(state)
    year_from, year_to = (2022, 2026) if state == 'johor' else (2023, 2026)
    df = pipeline.engineer_features(year_from, year_to)

    row = df[df['seat'] == seat_name]
    if row.empty:
        return None

    return {
        'majority_change':      float(row.iloc[0]['majority_change']),
        'turnout_change':       float(row.iloc[0]['turnout_change']),
        'incumbent_held':       int(row.iloc[0]['incumbent_held']),
        'log_voters':           float(row.iloc[0]['log_voters']),
        'majority_perc_change': float(row.iloc[0]['majority_perc_change']),
        'n_candidates_b':       int(row.iloc[0]['n_candidates_b']),
    }


def predict_seat_tool(state: str, seat_name: str,
                       turnout_change: float = None,
                       majority_change: float = None,
                       majority_perc_change: float = None,
                       incumbent_held: int = None,
                       n_candidates_b: int = None) -> dict:
    """
    Tool function called by the LLM. Merges user-specified changes
    with the seat's real baseline for anything not specified,
    then calls the actual trained model.
    """
    from backend.core.models.state_predictor import StatePredictor

    baseline = get_seat_baseline(state, seat_name)
    if baseline is None:
        return {"error": f"Seat '{seat_name}' not found in {state}"}

    features = baseline.copy()
    if turnout_change is not None:
        features['turnout_change'] = turnout_change
    if majority_change is not None:
        features['majority_change'] = majority_change
    if majority_perc_change is not None:
        features['majority_perc_change'] = majority_perc_change
    if incumbent_held is not None:
        features['incumbent_held'] = incumbent_held
    if n_candidates_b is not None:
        features['n_candidates_b'] = n_candidates_b

    predictor = StatePredictor(state)
    result = predictor.predict_seat(seat_name, features)

    return {
        "seat": seat_name,
        "state": state,
        "prediction": "Harapan" if result['prediction'] == 'non-BN' else "BN/PN",
        "probability": round(result['probability'], 3),
        "is_ood": result['is_ood'],
        "baseline_turnout_change": round(baseline['turnout_change'], 2),
        "scenario_turnout_change": round(features['turnout_change'], 2),
    }


TOOLS = [{
    "type": "function",
    "function": {
        "name": "predict_seat_scenario",
        "description": (
            "Predicts whether Harapan or the BN+PN government coalition "
            "wins a specific Johor or Negeri Sembilan DUN seat, under a "
            "hypothetical 'what if' scenario. Use this whenever the user "
            "asks about a hypothetical change (turnout, margin, etc.) for "
            "a NAMED seat -- not for questions about actual historical results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "enum": ["johor", "neg_sembilan"],
                    "description": "Which state the seat is in"
                },
                "seat_name": {
                    "type": "string",
                    "description": "Exact seat name, e.g. 'N.01 Buloh Kasap'"
                },
                "turnout_change": {
                    "type": "number",
                    "description": "Percentage point change in voter turnout "
                                   "vs the real baseline, e.g. -10 for 10pp lower"
                },
                "majority_perc_change": {
                    "type": "number",
                    "description": "Change in winning margin as a fraction "
                                   "of total voters, e.g. -0.1 for margin "
                                   "shrinking by 10% of the electorate"
                }
            },
            "required": ["state", "seat_name"]
        }
    }
}]


def handle_tool_call(tool_call) -> dict:
    """Execute the tool call requested by the LLM."""
    args = json.loads(tool_call.function.arguments)
    return predict_seat_tool(
        state=args.get("state"),
        seat_name=args.get("seat_name"),
        turnout_change=args.get("turnout_change"),
        majority_perc_change=args.get("majority_perc_change"),
    )


def retrieve(question: str, collection, n: int = N_RESULTS) -> list:
    results = collection.query(query_texts=[question], n_results=n)
    return list(zip(results['documents'][0], results['metadatas'][0]))


def build_prompt(question: str, retrieved_docs: list) -> str:
    context_parts = []
    for i, (doc, meta) in enumerate(retrieved_docs, 1):
        doc_type = meta.get('type', 'document')
        context_parts.append(f"[Document {i} -- {doc_type}]\n{doc}")
    context = "\n\n".join(context_parts)

    return f"""You are an AI assistant for GE-Insights, a Malaysian election prediction system.
Answer questions about election predictions, sentiment analysis, and economic forecasting
using ONLY the provided context. Be specific and cite actual numbers from the context.
If the context doesn't contain enough information, say so honestly.

Context from the GE-Insights knowledge base:
{context}

Question: {question}

Answer (be specific, cite numbers from the context, keep it concise):"""


def is_scenario_question(question: str) -> bool:
    """Detect if this is a 'what if' scenario question vs factual RAG question."""
    q = question.lower()
    scenario_keywords = ['what if', 'if turnout', 'if the', 'suppose',
                         'hypothetically', 'what would happen']
    return any(k in q for k in scenario_keywords)


# ======================================================================
# Main ask() -- routes to tool calling OR RAG
# ======================================================================

def ask(question: str, collection) -> dict:
    """Full pipeline: route to tool calling (scenarios) or RAG (facts)."""

    # -- Route 1: Scenario question -> tool calling ----------------
    if is_scenario_question(question):
        messages = [{
            "role": "system",
            "content": (
                "You help users explore hypothetical election scenarios. "
                "When asked a 'what if' question about a specific seat, "
                "call predict_seat_scenario with the seat name and the "
                "changed parameter. State names are 'johor' or 'neg_sembilan' -- "
                "infer from context or ask if ambiguous."
            )
        }, {
            "role": "user",
            "content": question
        }]

        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=500,
            temperature=0.1,
        )

        msg = response.choices[0].message
        if msg.tool_calls:
            tool_result = handle_tool_call(msg.tool_calls[0])

            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": msg.tool_calls[0].id,
                "content": json.dumps(tool_result)
            })

            final_response = groq_client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=400,
                temperature=0.2,
            )

            return {
                'question':  question,
                'answer':    final_response.choices[0].message.content.strip(),
                'sources':   ['live_prediction'],
                'n_sources': 1,
                'tool_result': tool_result,
            }
        else:
            return {
                'question':  question,
                'answer':    msg.content or "I couldn't identify which seat and scenario you're asking about -- could you specify the seat name and state?",
                'sources':   [],
                'n_sources': 0,
            }

    # -- Route 2: Factual question -> RAG (existing logic) ---------
    q_lower = question.lower()

    is_wrong_query = any(k in q_lower for k in [
        'wrong', 'incorrect', 'mistake', 'error',
        'predict incorrectly', 'got wrong', 'failed'
    ])
    is_summary_query = any(k in q_lower for k in [
        'summary', 'overall', 'total', 'how many seats',
        'overall prediction', 'win in melaka', 'melaka prediction'
    ])

    state_filter = None
    if any(k in q_lower for k in ['ns', 'negeri sembilan', 'sembilan']):
        state_filter = 'neg_sembilan'
    elif any(k in q_lower for k in ['johor', 'jhr']):
        state_filter = 'johor'
    elif any(k in q_lower for k in ['melaka', 'malacca']):
        state_filter = 'melaka'
    elif any(k in q_lower for k in ['selangor']):
        state_filter = 'selangor'

    if is_wrong_query:
        summary_results = collection.query(
            query_texts=["wrong predictions incorrect seats Johor NS accuracy"],
            n_results=2, where={"type": {"$eq": "wrong_predictions_summary"}}
        )
        regular_results = collection.query(query_texts=[question], n_results=4)
        retrieved = []
        if summary_results['documents'][0]:
            retrieved = list(zip(summary_results['documents'][0], summary_results['metadatas'][0]))
        retrieved += list(zip(regular_results['documents'][0], regular_results['metadatas'][0]))[:4]

    elif is_summary_query:
        retrieved = retrieve(question, collection, n=6)
        summary_results = collection.query(
            query_texts=["Melaka 2026 overall prediction 22 seats BN Harapan summary"],
            n_results=2, where={"type": {"$in": ["melaka_summary", "state_summary"]}}
        )
        if summary_results['documents'][0]:
            summary_docs = list(zip(summary_results['documents'][0], summary_results['metadatas'][0]))
            retrieved = summary_docs + retrieved[:4]

    elif state_filter:
        results = collection.query(query_texts=[question], n_results=6,
                                   where={"state": {"$eq": state_filter}})
        retrieved = list(zip(results['documents'][0], results['metadatas'][0]))
        if len(retrieved) < 3:
            retrieved += retrieve(question, collection, n=3)
    else:
        retrieved = retrieve(question, collection, n=6)

    prompt = build_prompt(question, retrieved)
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.2,
    )

    return {
        'question':  question,
        'answer':    response.choices[0].message.content.strip(),
        'sources':   [meta.get('type', 'unknown') for _, meta in retrieved],
        'n_sources': len(retrieved),
    }


# -- Interactive CLI -----------------------------------------------------

def chat():
    print("\nGE-Insights RAG Chatbot (with live scenario predictions)")
    print("="*60)
    print("Ask factual questions OR 'what if' scenarios.")
    print("Type 'quit' to exit.\n")

    collection = get_collection()
    print(f"Knowledge base loaded: {collection.count()} documents\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ('quit', 'exit', 'q'):
            break

        result = ask(question, collection)
        print(f"\nAssistant: {result['answer']}")
        print(f"\n[Sources: {', '.join(result['sources']) if result['sources'] else 'live model call'}]")
        print("-"*60 + "\n")


if __name__ == "__main__":
    chat()