import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.core import sentiment_adjustment as sa
from backend.core.frozen_forecast import latest_frozen
from backend.core.states import STATES

st.set_page_config(page_title="Sentiment-informed prediction (experimental)", layout="wide")

LABELS = {'johor': 'Johor', 'neg_sembilan': 'Negeri Sembilan', 'selangor': 'Selangor',
          'melaka': 'Melaka', 'perak': 'Perak'}

st.title("Sentiment-informed prediction (experimental)")
st.warning(
    "The validated forecast is the frozen model: it uses seat history and demographics, and its "
    "accuracy was tested on held-out elections. This page is a separate research prototype. "
    "The sentiment effects below are **assumptions written by a person, not fitted to data**, "
    "because we have no past news paired with past results. No accuracy is claimed for these numbers. "
    "The frozen forecast is not changed by anything on this page."
)

with st.expander("Assumptions used (positive = toward Harapan, negative = toward BN/PN)", expanded=False):
    st.dataframe(sa.assumption_table(), use_container_width=True, hide_index=True)
    st.markdown(
        f"- Only the **change** in a theme between two news windows counts, not its level.\n"
        f"- At least **{sa.MIN_HEADLINES}** headlines are needed in each window.\n"
        f"- `islam_threat` is down-weighted (low-reliability label). `govt_performance` is not used.\n"
        f"- Cost of living and corruption push away from whoever held the seat last time, so both sides can lose.\n"
        f"- The shift is capped at {sa.MAX_ABS_SHIFT} on the logit scale. "
        f"Frozen probabilities of exactly 0 or 1 are treated as {sa.PROB_FLOOR} / {sa.PROB_CEIL}.\n"
        f"- Voter groups come from the voter-roll ethnicity and age data. There is no rural/urban split yet."
    )

windows = sa.load_windows()
if windows is None:
    st.info("No sentiment adjustment is shown yet: two scored news windows are needed "
            "(file `data/processed/narrative_windows.csv`). Nothing here is a prediction until then.")
    st.stop()

current, reference = windows
ok, message = sa.data_status(current, reference)
st.caption(f"Windows: reference {reference.get('label', reference.get('date'))} -> "
           f"current {current.get('label', current.get('date'))}.")
if not ok:
    st.info(message)
    st.stop()

fresh = sa.freshness(current, reference)
(st.warning if fresh['stale'] else st.success)(fresh['text'])
with st.expander("How this is refreshed", expanded=False):
    st.markdown(
        "The maintainer runs `python scripts/refresh_sentiment.py`. It scrapes the last 60 days of "
        "Malaysian political headlines with a fixed query list, scores only headlines not scored yet "
        "(capped per run), then compares the **latest 30 days** with the **30 days before**. "
        "The page does not update itself: the date above is the newest news it has used."
    )

chg = sa.theme_change(current, reference)
st.subheader("Change in news themes")
st.dataframe(pd.DataFrame([{k: round(v, 3) for k, v in chg.items()}]), use_container_width=True, hide_index=True)

state = st.radio("State", list(LABELS), format_func=LABELS.get, horizontal=True)
forecast, info = latest_frozen(state)
if forecast is None:
    st.info("This state has no frozen forecast yet.")
    st.stop()

out, _ = sa.adjust_forecast(forecast, sa.load_demographics(state), current, reference)
out['change'] = (out['sentiment_probability'] - out['probability']).round(3)

c1, c2, c3 = st.columns(3)
c1.metric("Seats with a changed call", int(out['sentiment_flips_call'].sum()))
c2.metric("Largest move (points)", f"{out['change'].abs().max() * 100:.1f}")
c3.metric("Seats moved by more than 1 point", int((out['change'].abs() > 0.01).sum()))

fig = go.Figure()
fig.add_bar(x=out['seat_name'], y=out['probability'], name="Frozen forecast")
fig.add_bar(x=out['seat_name'], y=out['sentiment_probability'], name="Sentiment-informed (assumed)")
fig.add_hline(y=0.5, line_dash="dot")
fig.update_layout(barmode='group', height=420, yaxis_title="P(Harapan wins)", xaxis_tickangle=-60,
                  margin=dict(t=20))
st.plotly_chart(fig, use_container_width=True)

show = out[['seat_name', 'probability', 'sentiment_probability', 'change', 'sentiment_flips_call']]
show = show.rename(columns={'probability': 'frozen', 'sentiment_probability': 'sentiment-informed',
                            'sentiment_flips_call': 'call changes'})
st.dataframe(show.sort_values('change', key=abs, ascending=False), use_container_width=True, hide_index=True)
