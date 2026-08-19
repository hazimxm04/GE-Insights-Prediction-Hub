"""
2_Chatbot.py
============
RAG chatbot interface.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.api import ask_chatbot

st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 GE-Insights Chatbot")
st.markdown("Ask anything about election predictions, accuracy, or sentiment analysis.")

# ── Example questions ─────────────────────────────────────────────

st.subheader("Example questions:")
cols = st.columns(2)
examples = [
    "What is the predicted outcome for Melaka 2026?",
    "Why did the model get N.12 Bentayan wrong?",
    "What is the accuracy on the 2026 Johor election?",
    "Which seats in NS did the model predict incorrectly?",
    "What is the racial tension index right now?",
    "What is the economic pressure score for Johor?",
]
for i, ex in enumerate(examples):
    with cols[i % 2]:
        if st.button(ex, key=f"ex_{i}"):
            st.session_state['question'] = ex

st.divider()

# ── Chat interface ────────────────────────────────────────────────

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Input
question = st.chat_input("Ask about the predictions...")

# Handle example button clicks
if 'question' in st.session_state:
    question = st.session_state.pop('question')

if question:
    # Add user message
    st.session_state.messages.append({
        'role': 'user',
        'content': question
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = ask_chatbot(question)

        if 'error' in result:
            answer = f"Error: {result['error']}"
        else:
            answer = result.get('answer', 'No answer returned')
            sources = result.get('sources', [])

        st.markdown(answer)
        if sources:
            st.caption(f"Sources: {', '.join(set(sources))}")

    st.session_state.messages.append({
        'role': 'assistant',
        'content': answer
    })