"""
1_Predictions.py
================
Seat-level prediction viewer for Johor and NS.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.api import get_all_predictions

st.set_page_config(page_title="Predictions", page_icon="🗳️", layout="wide")

st.title("🗳️ Seat Predictions")
st.markdown("Real 2026 election predictions vs actual results")

# ── State selector ────────────────────────────────────────────────

state = st.selectbox(
    "Select state:",
    ["johor", "neg_sembilan"],
    format_func=lambda x: "Johor" if x == "johor" else "Negeri Sembilan"
)

# ── Load from validation CSV ──────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
csv_path = ROOT / f"backend/models/{state}/validation_2026.csv"

if not csv_path.exists():
    st.error(f"No validation data for {state}")
    st.stop()

df = pd.read_csv(csv_path)

# ── Summary metrics ───────────────────────────────────────────────

total    = len(df)
correct  = df['correct'].sum()
wrong    = total - correct
accuracy = correct / total

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Seats", total)
col2.metric("Accuracy", f"{accuracy:.2%}")
col3.metric("Correct", int(correct))
col4.metric("Wrong", int(wrong))

st.divider()

# ── Prediction vs Actual table ────────────────────────────────────

st.subheader("Prediction vs Actual Result")

df['prediction'] = df['predicted_non_bn'].apply(
    lambda x: 'Harapan' if x == 1 else 'BN/PN'
)
df['actual'] = df['actual_non_bn'].apply(
    lambda x: 'Harapan' if x == 1 else 'BN/PN'
)

display_cols = ['seat', 'prediction', 'actual',
                'correct', 'probability', 'is_ood']

df_display = df[display_cols].copy()
df_display['probability'] = df_display['probability'].apply(
    lambda x: f"{float(x):.2f}"
)
df_display['correct'] = df_display['correct'].apply(
    lambda x: "✅" if x else "❌"
)

st.dataframe(df_display, use_container_width=True, height=500)

st.divider()

# ── Wrong predictions ─────────────────────────────────────────────

st.subheader("❌ Wrong Predictions")

wrong_df = df[df['correct'] == False]

if wrong_df.empty:
    st.success("All predictions correct!")
else:
    for _, row in wrong_df.iterrows():
        st.error(
            f"**{row['seat']}** — "
            f"Predicted: `{row['prediction']}` | "
            f"Actual: `{row['actual']}` | "
            f"P={float(row['probability']):.2f} "
            f"{'⚠️ OOD' if row.get('is_ood') else ''}"
        )

st.divider()

# ── Correct vs Wrong chart ────────────────────────────────────────

st.subheader("Prediction Accuracy Breakdown")

df['result'] = df['correct'].apply(
    lambda x: "✅ Correct" if x else "❌ Wrong"
)

fig = px.histogram(
    df, x='prediction',
    color='result',
    barmode='group',
    color_discrete_map={
        '✅ Correct': '#2ecc71',
        '❌ Wrong':   '#e74c3c'
    },
    labels={'prediction': 'Predicted Coalition', 'count': 'Seats'},
    title=f'{state.replace("_"," ").title()} 2026 — Correct vs Wrong'
)
fig.update_layout(height=350)
st.plotly_chart(fig, use_container_width=True)