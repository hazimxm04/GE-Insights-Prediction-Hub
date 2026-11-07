"""
2_Sentiment_Analysis.py
=======================
The INPUTS of the experimental sentiment layer: where the latest news leans, what it is about,
what it does to each state, and the scored headlines behind it. The seat-by-seat outcome lives on
the Prediction Analysis page (switch "Prediction view" to "Prediction (Model + Sentiment)").

Sentiment is scored on NATIONAL news and applied to every state through each seat's voter mix.
There are no state-specific headline scores.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.core import sentiment_adjustment as sa
from backend.core.frozen_forecast import latest_frozen
from sentiment.scoring.score_quality import usable, SCORE_FIELDS

st.set_page_config(page_title="Sentiment Scoring Analysis", layout="wide")

STATES = {'johor': 'Johor', 'neg_sembilan': 'Negeri Sembilan', 'selangor': 'Selangor',
          'melaka': 'Melaka', 'perak': 'Perak'}
THEME_NAMES = {'islam_threat': 'Islam threat', 'malay_unity': 'Malay unity', 'cost_living': 'Cost of living',
               'corruption': 'Corruption', 'govt_performance': 'Government performance (not used)'}
HARAPAN_RED, GOVT_BLUE, GREY = '#DC2626', '#2563EB', '#6B7280'
BN_NAVY, PN_DARK_NAVY = '#1E3A8A', '#0A1A4F'
POOL_FILE = ROOT / 'data/processed/scored_pool.csv'

st.title("Sentiment Scoring Analysis (experimental)")

windows = sa.load_windows()
if windows is None:
    st.info("No sentiment data yet: two scored news windows are needed "
            "(`data/processed/narrative_windows.csv`, built by `scripts/refresh_sentiment.py`).")
    st.stop()

current, reference = windows
fresh = sa.freshness(current, reference)
ok, message = sa.data_status(current, reference)
if not ok:
    st.info(message)
    st.stop()

# ── Current media sentiment ─────────────────────────────────────────────────────
# "Last updated" is the date of the newest headline scored, read from the data (nothing typed by hand).
_days = fresh['days_old']
_ago = "today" if _days == 0 else f"{_days} day{'s' if _days != 1 else ''} ago"
st.markdown(
    "<div style='display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;"
    "background:linear-gradient(145deg,#334155,#0F172A);color:#fff;border-radius:16px;"
    "padding:14px 22px;box-shadow:0 6px 18px rgba(0,0,0,.18);margin-bottom:6px'>"
    "<div style='font-size:.85rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;opacity:.85'>"
    "Last updated</div>"
    f"<div style='font-size:1.6rem;font-weight:800'>{pd.Timestamp(fresh['updated']):%d %b %Y}</div>"
    f"<div style='font-size:.9rem;opacity:.85'>{_ago}</div></div>",
    unsafe_allow_html=True)
if fresh['stale']:
    st.warning(f"This is more than {sa.STALE_AFTER_DAYS} days old, so treat it as out of date.")

st.subheader("Current Media Sentiment")
BOXES = [('harapan_sentiment', 'Harapan', HARAPAN_RED, '#7F1D1D'),
         ('bn_sentiment', 'BN', BN_NAVY, '#172554'),
         ('pn_sentiment', 'PN', PN_DARK_NAVY, '#050B26')]
CARD_CSS = """
<style>
.sc{border-radius:16px;padding:20px 22px 16px;color:#fff;box-shadow:0 6px 18px rgba(0,0,0,.18)}
.sc .nm{font-size:.85rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;opacity:.85}
.sc .val{font-size:3rem;font-weight:800;line-height:1.1;margin:4px 0 2px;font-variant-numeric:tabular-nums}
.sc .wd{font-size:.9rem;opacity:.9;margin-bottom:14px}
.sc .tr{position:relative;height:8px;border-radius:99px;background:rgba(255,255,255,.28)}
.sc .mid{position:absolute;left:50%;top:-4px;width:2px;height:16px;background:rgba(255,255,255,.7)}
.sc .dot{position:absolute;top:-5px;width:18px;height:18px;margin-left:-9px;border-radius:50%;
         background:#fff;border:3px solid rgba(0,0,0,.35)}
.sc .ax{display:flex;justify-content:space-between;font-size:.72rem;opacity:.75;margin-top:8px}
</style>
"""


def sentiment_word(v: float) -> str:
    """Plain-language label for the card. Display only: the cut-offs are not used by the model."""
    if v <= -0.5: return "Very negative"
    if v <= -0.15: return "Negative"
    if v < 0.15: return "Mixed / neutral"
    if v < 0.5: return "Positive"
    return "Very positive"


st.markdown(CARD_CSS, unsafe_allow_html=True)
for col, (field, name, top, bottom) in zip(st.columns(3), BOXES):
    if field in current and pd.notna(current.get(field)):
        v = float(current[field])
        pos = min(max((v + 1) / 2 * 100, 0), 100)
        body = (f"<div class='val'>{v:+.2f}</div><div class='wd'>{sentiment_word(v)}</div>"
                f"<div class='tr'><div class='mid'></div><div class='dot' style='left:{pos:.1f}%'></div></div>"
                f"<div class='ax'><span>-1</span><span>0</span><span>+1</span></div>")
    else:
        body = "<div class='val'>n/a</div>"
    col.markdown(f"<div class='sc' style='background:linear-gradient(145deg,{top},{bottom})'>"
                 f"<div class='nm'>{name}</div>{body}</div>", unsafe_allow_html=True)
st.caption(f"Latest window: {current.get('label')}. Compared with: {reference.get('label')}. "
           "Scores run from -1 (very negative) to +1 (very positive).")

# ── 2. What the news is about ───────────────────────────────────────────────────
st.subheader("1. What the news is about")
change = sa.theme_change(current, reference)
names = [THEME_NAMES[t] for t in sa.THEMES]
fig = go.Figure()
fig.add_bar(x=names, y=[float(reference[t]) for t in sa.THEMES], name="Previous 30 days", marker_color=GREY)
fig.add_bar(x=names, y=[float(current[t]) for t in sa.THEMES], name="Latest 30 days", marker_color=HARAPAN_RED)
fig.update_layout(barmode='group', height=340, yaxis_title="Average theme score (0 to 1)", margin=dict(t=10))
st.plotly_chart(fig, use_container_width=True)
st.dataframe(pd.DataFrame({'theme': names,
                           'previous': [round(float(reference[t]), 3) for t in sa.THEMES],
                           'latest': [round(float(current[t]), 3) for t in sa.THEMES],
                           'change': [round(change[t], 3) for t in sa.THEMES]}),
             use_container_width=True, hide_index=True)

# ── 3. What it does to each state ───────────────────────────────────────────────
st.subheader("2. What it does to each state")
st.caption("The same national news is applied to every state through each seat's voter mix "
           "(race and age). Positive move = toward Harapan.")
rows, detail, voted = [], {}, []
for key, label in STATES.items():
    forecast, _ = latest_frozen(key)
    if forecast is None:
        voted.append(label)
        continue
    out, _ = sa.adjust_forecast(forecast, sa.load_demographics(key), current, reference)
    out['move_points'] = (out['sentiment_probability'] - out['probability']) * 100
    detail[key] = out
    rows.append({'state': label,
                 'seats': len(out),
                 'Harapan-side (model)': int((out['probability'] >= 0.5).sum()),
                 'Harapan-side (with sentiment)': int((out['sentiment_probability'] >= 0.5).sum()),
                 'calls changed': int(out['sentiment_flips_call'].sum()),
                 'avg move (pts)': round(float(out['move_points'].mean()), 2),
                 'largest move (pts)': round(float(out['move_points'].abs().max()), 1)})
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
if voted:
    st.caption(f"{' and '.join(voted)} have already voted. They are validated against the real result, "
               "so there is no forecast to adjust here.")

if detail:
    pick = st.radio("See the seats that move most in:", list(detail), format_func=STATES.get, horizontal=True)
    out = detail[pick]
    top = out.reindex(out['move_points'].abs().sort_values(ascending=False).index).head(10)
    st.dataframe(top[['seat_name', 'probability', 'sentiment_probability', 'move_points']].rename(
        columns={'seat_name': 'seat', 'probability': 'model P(Harapan)',
                 'sentiment_probability': 'with sentiment P(Harapan)', 'move_points': 'move (points)'}).round(3),
        use_container_width=True, hide_index=True)
    st.caption("The full seat list for every state is on the Prediction Analysis page.")

# ── 4. The scored headlines ─────────────────────────────────────────────────────
st.subheader("3. The scored headlines")
st.caption("Each score is a language model's reading of one headline. Check a few yourself.")
if not POOL_FILE.exists():
    st.info("The scored headlines are not published in this copy of the app.")
else:
    pool = usable(pd.read_csv(POOL_FILE))
    pool['published'] = pd.to_datetime(pool['published'], errors='coerce', utc=True)
    pool = pool.sort_values('published', ascending=False)
    shown = [c for c in ['published', 'title', 'source'] + SCORE_FIELDS if c in pool.columns]
    sort_by = st.selectbox("Sort by", ['newest'] + [f for f in SCORE_FIELDS if f in pool.columns])
    view = pool if sort_by == 'newest' else pool.sort_values(sort_by, ascending=False)
    st.dataframe(view[shown].head(100).round({f: 2 for f in SCORE_FIELDS if f in shown}),
                 use_container_width=True, hide_index=True)
    st.download_button("Download all scored headlines (CSV)", pool[shown].to_csv(index=False),
                       file_name="scored_headlines.csv", mime="text/csv")

# ── Assumptions and refresh ─────────────────────────────────────────────────────
with st.expander("Assumptions used (positive = toward Harapan, negative = toward BN/PN)", expanded=False):
    st.dataframe(sa.assumption_table(), use_container_width=True, hide_index=True)
    st.markdown(
        f"- Only the **change** in a theme between two news windows counts, not its level.\n"
        f"- At least **{sa.MIN_HEADLINES}** headlines are needed in each window.\n"
        f"- `islam_threat` is down-weighted (low-reliability label). `govt_performance` is not used.\n"
        f"- Cost of living and corruption push away from whoever held the seat last time, so both sides can lose.\n"
        f"- The shift is capped at {sa.MAX_ABS_SHIFT} on the logit scale. "
        f"Probabilities of exactly 0 or 1 are treated as {sa.PROB_FLOOR} / {sa.PROB_CEIL}.\n"
        f"- Voter groups come from the voter-roll ethnicity and age data. There is no rural/urban split yet."
    )
with st.expander("How this is refreshed", expanded=False):
    st.markdown(
        "The maintainer runs `python scripts/refresh_sentiment.py`. It scrapes the last 60 days of Malaysian "
        "political headlines with a fixed query list, scores only headlines not scored yet (capped per run), "
        "then compares the **latest 30 days** with the **30 days before**. The page does not update itself: "
        "the date at the top is the newest news it has used."
    )