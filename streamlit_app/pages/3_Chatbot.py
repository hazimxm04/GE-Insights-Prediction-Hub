"""
3_Chatbot.py
============
Ask Me!: the analyst agent. The model chooses which data tool to call (forecast, validation, flips,
sentiment, what-if) and answers from the results. Each answer shows the tools it used.

The old keyword-routed RAG chatbot is still available from the sidebar for comparison.
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.set_page_config(page_title="Ask Me! — MY Ramalan Politik", page_icon="🤖", layout="wide")

from chatbot.agent import run_agent       # noqa: E402

HISTORY_TURNS = 6                          # earlier messages sent back to the agent


@st.cache_resource
def load_old_chatbot():
    """The previous keyword-routed chatbot, loaded only if someone picks it."""
    from chatbot.chain.rag_chain import ask, get_collection
    return ask, get_collection()

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
    <div class="chatbot-subtitle">An analyst agent: it looks up the forecasts, the validation against real results, flips and sentiment with its own tools, then answers from what it found.</div>
</div>
""", unsafe_allow_html=True)

mode = st.sidebar.radio("Chatbot", ["Agent (new)", "Keyword chatbot (old)"],
                        help="The agent picks its own tools. The old chatbot routes by keywords.")
show_plan_on = st.sidebar.toggle("Plan first", value=True,
                              help="The agent writes a short plan before it starts, and ticks steps off as it goes. "
                                   "Costs one extra model call per question.")
check_numbers = st.sidebar.toggle("Check numbers", value=True,
                                  help="Every number and year in the answer must appear in a tool result. "
                                       "One retry if not; otherwise the answer carries a warning.")
st.sidebar.caption("Forecasts exist only for states that have not voted. States that have voted are "
                   "checked against the real result. If one is missing for a state, the agent says so.")

# ── Example questions ─────────────────────────────────────────────

with st.expander("💡 Example questions", expanded=len(st.session_state.get('messages', [])) == 0):
    cols = st.columns(2)
    examples = [
        "How did the model do in Johor compared with simply repeating the last result?",
        "Which seats flipped in Negeri Sembilan, and did the model call them?",
        "What is the forecast for Melaka and which seats change side?",
        "Does Selangor have a validated model?",
        "What is the news sentiment right now, and how fresh is it?",
        "What if turnout dropped 10 points in N.01 Kuala Linggi (Melaka)?",
    ]
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, key=f"ex_{i}"):
                st.session_state['question'] = ex

st.divider()

# ── Chat interface ────────────────────────────────────────────────

if 'messages' not in st.session_state:
    st.session_state.messages = []


def show_plan(plan):
    if not plan:
        return
    done = sum(1 for p in plan if p["done"])
    with st.expander(f"📋 Plan ({done} of {len(plan)} steps done)"):
        for p in plan:
            tool = f" · `{p['tool']}`" if p["tool"] != "none" else ""
            st.markdown(f"{'✅' if p['done'] else '⬜'} {p['do']}{tool}")


def show_trace(steps):
    if not steps:
        st.caption("No tools used.")
        return
    with st.expander(f"🔧 What the agent did ({len(steps)} tool call{'s' if len(steps) != 1 else ''})"):
        for n, step in enumerate(steps, 1):
            st.markdown(f"**{n}. `{step['tool']}`**  `{step.get('arguments') or '{}'}`")
            st.code(json.dumps(step['result'], indent=2, ensure_ascii=False, default=str)[:1500], language="json")


for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg.get('steps') is not None:
            show_plan(msg.get('plan'))
            show_trace(msg['steps'])

question = st.chat_input("Ask about the predictions...")

if 'question' in st.session_state:
    question = st.session_state.pop('question')

if question:
    history = [{'role': m['role'], 'content': m['content']}
               for m in st.session_state.messages[-HISTORY_TURNS:]]
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        steps, plan = None, None
        if mode.startswith("Agent"):
            try:
                with st.status("Thinking...", expanded=True) as status:
                    def on_event(kind, payload):
                        if kind == "plan" and payload.get("steps"):
                            status.update(label="Plan")
                            for n, p in enumerate(payload["steps"], 1):
                                tool = f" · `{p['tool']}`" if p["tool"] != "none" else ""
                                st.markdown(f"**{n}.** {p['do']}{tool}")
                        elif kind == "tool":
                            status.update(label=f"Running {payload['tool']}...")
                            st.markdown(f"🔧 `{payload['tool']}` {payload.get('arguments') or ''}")

                    result = run_agent(question, history=history, plan=show_plan_on, verify=check_numbers,
                                       on_event=on_event)
                    status.update(label="Done", state="complete", expanded=False)
                answer, steps, plan = result['answer'], result['steps'], result.get('plan')
            except Exception as e:                       # e.g. no API key
                answer, steps = f"The agent could not start: {e}", []
            st.markdown(answer)
            show_plan(plan)
            show_trace(steps)
        else:
            sources = []
            with st.spinner("Searching knowledge base..."):
                try:
                    ask, collection = load_old_chatbot()
                    result = ask(question, collection)
                    answer = result.get('answer', 'No answer returned')
                    sources = result.get('sources', [])
                except Exception as e:
                    answer = f"The old chatbot could not start: {e}"
            st.markdown(answer)
            if sources:
                st.caption(f"Sources: {', '.join(set(sources))}")

    st.session_state.messages.append({'role': 'assistant', 'content': answer, 'steps': steps, 'plan': plan})