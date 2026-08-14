import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# ── Load env ────────────────────────────────────────────────────────

for p in [Path("backend/.env"), Path(__file__).resolve().parents[3] / "backend/.env"]:
    if p.exists():
        load_dotenv(p)
        break

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in backend/.env")

# ── Config ──────────────────────────────────────────────────────────

CHROMA_DIR   = Path("chatbot/chroma_db")
MODEL        = "llama-3.1-8b-instant"
N_RESULTS    = 4   # number of docs to retrieve per query

# ── Clients ─────────────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection(
        name="ge_insights",
        embedding_function=embedding_fn
    )

groq_client = Groq(api_key=GROQ_API_KEY)

# ── RAG pipeline ─────────────────────────────────────────────────────

def retrieve(question: str, collection, n: int = N_RESULTS) -> list:
    """Retrieve most relevant documents for the question."""
    results = collection.query(
        query_texts=[question],
        n_results=n
    )
    docs  = results['documents'][0]
    metas = results['metadatas'][0]
    return list(zip(docs, metas))


def build_prompt(question: str, retrieved_docs: list) -> str:
    """Assemble question + retrieved context into a prompt."""
    context_parts = []
    for i, (doc, meta) in enumerate(retrieved_docs, 1):
        doc_type = meta.get('type', 'document')
        context_parts.append(f"[Document {i} — {doc_type}]\n{doc}")

    context = "\n\n".join(context_parts)

    return f"""You are an AI assistant for GE-Insights, a Malaysian election prediction system.
Answer questions about election predictions, sentiment analysis, and economic forecasting
using ONLY the provided context. Be specific and cite actual numbers from the context.
If the context doesn't contain enough information, say so honestly.

Context from the GE-Insights knowledge base:
{context}

Question: {question}

Answer (be specific, cite numbers from the context, keep it concise):"""


def ask(question: str, collection) -> dict:
    """Full RAG pipeline: retrieve → build prompt → generate answer."""

    q_lower = question.lower()

    # Detect query type
    is_wrong_query = any(k in q_lower for k in [
        'wrong', 'incorrect', 'mistake', 'error',
        'predict incorrectly', 'got wrong', 'failed'
    ])

    is_summary_query = any(k in q_lower for k in [
        'summary', 'overall', 'total', 'how many seats',
        'overall prediction', 'win in melaka', 'melaka prediction'
    ])

    # Detect state
    state_filter = None
    if any(k in q_lower for k in ['ns', 'negeri sembilan', 'sembilan']):
        state_filter = 'neg_sembilan'
    elif any(k in q_lower for k in ['johor', 'jhr']):
        state_filter = 'johor'
    elif any(k in q_lower for k in ['melaka', 'malacca']):
        state_filter = 'melaka'

    # ── Wrong prediction queries ───────────────────────────────────
    if is_wrong_query:
        # Explicitly fetch wrong predictions summary doc first
        summary_results = collection.query(
            query_texts=["wrong predictions incorrect seats Johor NS accuracy"],
            n_results=2,
            where={"type": {"$eq": "wrong_predictions_summary"}}
        )
        # Then fetch regular results
        regular_results = collection.query(
            query_texts=[question],
            n_results=4
        )
        # Combine: summary first
        retrieved = []
        if summary_results['documents'][0]:
            retrieved = list(zip(
                summary_results['documents'][0],
                summary_results['metadatas'][0]
            ))
        retrieved += list(zip(
            regular_results['documents'][0],
            regular_results['metadatas'][0]
        ))[:4]

    # ── Summary queries ────────────────────────────────────────────
    elif is_summary_query:
        retrieved = retrieve(question, collection, n=6)
        summary_results = collection.query(
            query_texts=["Melaka 2026 overall prediction 22 seats BN Harapan summary"],
            n_results=2,
            where={"type": {"$in": ["melaka_summary", "state_summary"]}}
        )
        if summary_results['documents'][0]:
            summary_docs = list(zip(
                summary_results['documents'][0],
                summary_results['metadatas'][0]
            ))
            retrieved = summary_docs + retrieved[:4]

    # ── State-specific queries ─────────────────────────────────────
    elif state_filter:
        results = collection.query(
            query_texts=[question],
            n_results=6,
            where={"state": {"$eq": state_filter}}
        )
        retrieved = list(zip(
            results['documents'][0],
            results['metadatas'][0]
        ))
        # Top up if not enough results
        if len(retrieved) < 3:
            retrieved += retrieve(question, collection, n=3)

    # ── General queries ────────────────────────────────────────────
    else:
        retrieved = retrieve(question, collection, n=6)

    # Build prompt + generate answer
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
# ── Interactive CLI ──────────────────────────────────────────────────

def chat():
    print("\nGE-Insights RAG Chatbot")
    print("="*50)
    print("Ask anything about Malaysian election predictions.")
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
        print(f"\n[Sources: {', '.join(result['sources'])}]")
        print("-"*50 + "\n")


if __name__ == "__main__":
    chat()