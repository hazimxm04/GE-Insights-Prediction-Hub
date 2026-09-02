"""
2_Ask_Me!.py
============
RAG chatbot interface — redesigned to match dashboard's
consistent visual language (hero card, styled example prompts).
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.api import ask_chatbot

st.set_page_config(page_title="Ask Me! — MY Ramalan Politik", page_icon="🤖", layout="wide")

# ── Shared design system CSS ────────────────────────────────────

st.markdown("""
<style>
.chatbot-hero {
    background: linear-gradient(135deg, #1a0505 0%, #3d0d0d 50%, #1a0505 100%);
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 20px;
    border: 1px solid rgba(220, 38, 38, 0.2);
}
.chatbot-title {
    font-size: 28px;
    font-weight: 800;
    color: #FAFAFA;
    margin-bottom: 4px;
}
.chatbot-subtitle {
    font-size: 14px;
    color: #9CA3AF;
}
div[data-testid="stButton"] > button {
    background: #1A1D24;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    color: #D1D5DB;
    text-align: left;
    padding: 12px 16px;
    font-size: 13px;
    transition: border-color 0.15s ease, background 0.15s ease;
    width: 100%;
}
div[data-testid="stButton"] > button:hover {
    border-color: rgba(220, 38, 38, 0.4);
    background: #22262E;
    color: #FAFAFA;
}
</style>
""", unsafe_allow_html=True)

# ── Hero header ──────────────────────────────────────────────────

st.markdown("""
<div class="chatbot-hero">
    <div class="chatbot-title">🤖 Ask Me!</div>
    <div class="chatbot-subtitle">Ask anything about election predictions, accuracy, or sentiment analysis — or try a "what if" scenario.</div>
</div>
""", unsafe_allow_html=True)

# ── Example questions ─────────────────────────────────────────────

with st.expander("💡 Example questions", expanded=len(st.session_state.get('messages', [])) == 0):
    cols = st.columns(2)
    examples = [
        "What is the predicted outcome for Melaka 2026?",
        "Why did the model get N.12 Bentayan wrong?",
        "What is the accuracy on the 2026 Johor election?",
        "Which seats in NS did the model predict incorrectly?",
        "What is the racial tension index right now?",
        "What if turnout dropped 10% in N.01 Buloh Kasap?",
    ]
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, key=f"ex_{i}"):
                st.session_state['question'] = ex

st.divider()

# ── Chat interface ────────────────────────────────────────────────

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

question = st.chat_input("Ask about the predictions...")

if 'question' in st.session_state:
    question = st.session_state.pop('question')

if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = ask_chatbot(question)

        if 'error' in result:
            answer = f"Error: {result['error']}"
            sources = []
        else:
            answer = result.get('answer', 'No answer returned')
            sources = result.get('sources', [])

        st.markdown(answer)
        if sources:
            st.caption(f"Sources: {', '.join(set(sources))}")

    st.session_state.messages.append({'role': 'assistant', 'content': answer})