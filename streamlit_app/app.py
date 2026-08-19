"""
app.py
======
GE-Insights Streamlit Dashboard — Home page
"""

import streamlit as st
from utils.api import health_check

# ── Page config ───────────────────────────────────────────────────

st.set_page_config(
    page_title="GE-Insights Prediction Hub",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Header ────────────────────────────────────────────────────────

st.title("🗳️ GE-Insights Prediction Hub")
st.markdown("**Malaysian State Election Prediction System**")
st.markdown("Validated on actual 2026 election results")

# ── API status ────────────────────────────────────────────────────

with st.spinner("Checking API..."):
    api_ok = health_check()

if api_ok:
    st.success("✅ Live API connected (Railway)")
else:
    st.error("❌ API offline — showing cached data only")

st.divider()

# ── Accuracy results ──────────────────────────────────────────────

st.subheader("📊 Validated Accuracy on Real 2026 Elections")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Johor SE-16 (Jul 11, 2026)",
        value="94.64%",
        delta="53/56 seats correct"
    )

with col2:
    st.metric(
        label="Negeri Sembilan SE-16 (Aug 1, 2026)",
        value="91.67%",
        delta="33/36 seats correct"
    )

with col3:
    st.metric(
        label="Melaka SE-16 (Upcoming)",
        value="22 / 28",
        delta="Govt wins predicted"
    )

st.divider()

# ── Accuracy progression ──────────────────────────────────────────

st.subheader("📈 Accuracy Progression")

import pandas as pd
import plotly.graph_objects as go

progression = pd.DataFrame({
    'Feature Set': [
        '6 structural',
        '+ Sentiment + Economic (11)',
        '+ Ethnicity + Age (24)',
        '+ Coalition target fix (25)',
    ],
    'Johor': [89.29, 89.29, 89.29, 94.64],
    'NS':    [63.89, 63.89, 77.78, 91.67],
})

fig = go.Figure()
fig.add_trace(go.Bar(
    name='Johor',
    x=progression['Feature Set'],
    y=progression['Johor'],
    marker_color='#1f77b4',
    text=progression['Johor'].apply(lambda x: f"{x}%"),
    textposition='outside'
))
fig.add_trace(go.Bar(
    name='NS',
    x=progression['Feature Set'],
    y=progression['NS'],
    marker_color='#ff7f0e',
    text=progression['NS'].apply(lambda x: f"{x}%"),
    textposition='outside'
))
fig.update_layout(
    barmode='group',
    yaxis=dict(range=[50, 100], title='Accuracy (%)'),
    xaxis_title='Feature Set',
    legend=dict(orientation='h', y=1.1),
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Key findings ──────────────────────────────────────────────────

st.subheader("🔑 Key Findings")

col1, col2 = st.columns(2)

with col1:
    st.info("""
**Coalition target fix (+28% NS accuracy)**

Initially modeled 'BN vs rest' — wrong for 2026
where BN and PN are coalition partners.
Reframing as 'government bloc vs Harapan'
improved NS accuracy by 28 percentage points.
    """)

    st.info("""
**Voter roll demographics (seat-level signal)**

Added ethnicity + age composition from 3–5M
anonymised voter rolls. chinese_pct became the
#1 most important feature (importance: 0.200).
    """)

with col2:
    st.info("""
**LSTM confirmed efficient market hypothesis**

PyTorch LSTM achieved 0.53% MAPE but only
52% directional accuracy — near-random,
consistent with EMH for daily price prediction.
    """)

    st.info("""
**National narrative × demographics**

Weighted national themes (Islam threat, Malay
unity, cost of living) by seat demographics
produces genuine seat-level sentiment variation.
    """)

st.divider()

# ── Tech stack ────────────────────────────────────────────────────

st.subheader("🛠️ Tech Stack")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**ML**")
    st.markdown("scikit-learn\nRF + XGB\nOOD detection")
with col2:
    st.markdown("**Deep Learning**")
    st.markdown("PyTorch\nLSTM\nSliding windows")
with col3:
    st.markdown("**NLP / RAG**")
    st.markdown("Groq/Llama\nChromaDB\nLangChain")
with col4:
    st.markdown("**MLOps**")
    st.markdown("APScheduler\n3 DAGs\nDrift detection")

st.divider()

st.markdown("""
**Navigation:** Use the sidebar to explore predictions,
ask the chatbot, or see the Melaka 2026 forecast.

[GitHub](https://github.com/hazimxm04/GE-Insights-Prediction-Hub) |
[Live API](https://elegant-cooperation-production-67c5.up.railway.app/docs)
""")