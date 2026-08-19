"""
3_Melaka_2026.py
================
Melaka 2026 pre-election forecast visualization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Melaka 2026", page_icon="🗺️", layout="wide"
)

st.title("🗺️ Melaka 2026 Pre-Election Forecast")
st.warning("""
**Disclaimer:** This is a pre-election forecast based on
2018→2021 historical patterns + current sentiment and economic data.
Accuracy will be validated when Melaka 2026 results are announced.
""")

# ── Load Melaka prediction CSV ────────────────────────────────────

DATA_PATH = Path(__file__).resolve().parents[2] / \
            "data/processed/melaka_2026_prediction.csv"

if not DATA_PATH.exists():
    st.error("Melaka prediction file not found. Run predict_melaka.py first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# ── Summary ───────────────────────────────────────────────────────

govt_wins = (df['prediction'] == 'BN').sum()
harapan_wins = (df['prediction'] != 'BN').sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Seats", len(df))
col2.metric("Govt (BN/PN) Predicted", govt_wins)
col3.metric("Harapan Predicted", harapan_wins)

st.divider()

# ── Seat predictions table ────────────────────────────────────────

st.subheader("Seat-Level Predictions")

def color_pred(val):
    if val == 'BN':
        return 'background-color: #d0e8ff'
    return 'background-color: #ffd0d0'

display_cols = [c for c in [
    'seat_name', 'incumbent', 'prediction',
    'probability', 'malay_pct', 'chinese_pct', 'is_ood'
] if c in df.columns]

df_display = df[display_cols].copy()
if 'probability' in df_display.columns:
    df_display['probability'] = df_display['probability'].apply(
        lambda x: f"{float(x):.2f}"
    )
if 'malay_pct' in df_display.columns:
    df_display['malay_pct'] = df_display['malay_pct'].apply(
        lambda x: f"{float(x):.1%}"
    )
if 'chinese_pct' in df_display.columns:
    df_display['chinese_pct'] = df_display['chinese_pct'].apply(
        lambda x: f"{float(x):.1%}"
    )

st.dataframe(
    df_display.style.applymap(
        color_pred, subset=['prediction']
    ),
    use_container_width=True,
    height=500
)

st.divider()

# ── Ethnicity vs prediction chart ─────────────────────────────────

st.subheader("Prediction by Chinese Voter Composition")

if 'chinese_pct' in df.columns:
    df_plot = df.copy()
    df_plot['chinese_pct_num'] = df_plot['chinese_pct'].astype(float)
    df_plot['probability_num'] = df_plot['probability'].astype(float)

    fig = px.scatter(
        df_plot,
        x='chinese_pct_num',
        y='probability_num',
        color='prediction',
        hover_data=['seat_name'],
        color_discrete_map={'BN': '#1f77b4', 'non-BN': '#d62728'},
        labels={
            'chinese_pct_num': 'Chinese Voter % in Seat',
            'probability_num': 'Harapan Win Probability',
        },
        title='Higher Chinese composition → Higher Harapan probability'
    )
    fig.add_hline(y=0.5, line_dash="dash",
                  line_color="gray",
                  annotation_text="Decision boundary (50%)")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Key observations ──────────────────────────────────────────────

st.subheader("Key Observations")

col1, col2 = st.columns(2)
with col1:
    st.success("""
**Certain Harapan wins (P=1.00):**
- N.22 Bandar Hilir (Chinese 73%)
- N.20 Kota Laksamana (Chinese 77%)
- N.19 Kesidang (Chinese 56%)
- N.16 Ayer Keroh (Chinese 48%)
    """)

with col2:
    st.info("""
**Swing/uncertain seats (P≈0.50):**
- N.17 Bukit Katil (mixed seat)
- N.11 Sungai Udang (OOD flagged)

**Certain BN holds (P=0.00):**
- All rural Malay-majority seats
- N.02 Tanjung Bidara (Malay 93%)
    """)