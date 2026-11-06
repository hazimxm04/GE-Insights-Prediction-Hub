"""
app.py  (home page)
===================
Structure:
  1. Title + LIVE/OFFLINE badge (API health) + level toggle (State DUN / Parliament)
  2. Headline numbers (validated accuracy read from the validation CSVs, never typed in)
  3. Map of state seats: FORECAST ONLY. Map view = "Prediction" or "Prediction + sentiment".
     States without a forecast yet (today Johor, Negeri Sembilan) stay grey ("No forecast yet").
  4. State-by-state summary: option "Validated" (states that have voted) or "Forecast" (all five;
     a state without a forecast shows "not available yet").
  5. Disclaimer and navigation

A state has a forecast if a frozen forecast folder exists for it (latest_frozen returns data). If not, the page
shows "not available yet". Nothing is hard-coded per state: add a forecast folder and it appears.
"""

import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core import sentiment_adjustment as sa
from backend.core.frozen_forecast import latest_frozen
from utils.api import health_check

st.set_page_config(page_title="MYRamalan", page_icon="🗳️", layout="wide", initial_sidebar_state="expanded")

# ── The states ──────────────────────────────────────────────────────────────────
# key -> label (the label must match the 'state' column of data/processed/map_data.geojson)
STATES = {'johor': 'Johor', 'neg_sembilan': 'Negeri Sembilan',
          'melaka': 'Melaka', 'selangor': 'Selangor', 'perak': 'Perak'}
# states that have voted = states with a validation file (checked against the real 2026 result)
VALIDATED = [k for k in STATES if (ROOT / f"backend/models/{k}/validation_2026.csv").exists()]

GEOJSON_PATH = ROOT / "data/processed/map_data.geojson"

# ── CSS ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.hero-container {
    background: linear-gradient(135deg, #1a0505 0%, #3d0d0d 50%, #1a0505 100%);
    border-radius: 16px;
    padding: 40px 30px;
    margin-bottom: 24px;
    border: 1px solid rgba(220, 38, 38, 0.2);
    position: relative;
}
.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: #FAFAFA;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 15px;
    color: #9CA3AF;
    margin-bottom: 0;
}


.status-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    float: right;
}
.status-live { background-color: #16A34A; color: white; }
.status-offline { background-color: #DC2626; color: white; }

/* Stat cards with hover lift effect */
.stat-card {
    background: #1A1D24;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    min-height: 76px;
    transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    cursor: default;
}
.stat-card:hover {
    transform: translateY(-3px);
    border-color: rgba(220, 38, 38, 0.4);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}
.stat-card .value {
    font-size: 24px;
    font-weight: 800;
    color: #FAFAFA;
}
.stat-card .label {
    font-size: 11px;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Dominant hero stat (map seat count) — bigger, distinct */
.hero-stat-card {
    background: linear-gradient(135deg, #1A1D24 0%, #252932 100%);
    border: 1px solid rgba(220, 38, 38, 0.3);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    cursor: default;
}
.hero-stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(220, 38, 38, 0.15);
}
.hero-stat-card .value {
    font-size: 40px;
    font-weight: 900;
    color: #FAFAFA;
    line-height: 1;
}
.hero-stat-card .label {
    font-size: 12px;
    color: #F87171;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 6px;
}

/* Legend styling */
.legend-row {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 6px;
    font-size: 13px;
}
.legend-label {
    color: #9CA3AF;
    font-weight: 600;
    min-width: 90px;
}
.legend-swatch {
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 3px;
    margin-right: 5px;
    vertical-align: middle;
}
.legend-swatch-dashed {
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 3px;
    margin-right: 5px;
    vertical-align: middle;
    border: 2px dashed;
    background: transparent;
}
</style>
""", unsafe_allow_html=True)


# ── Data helpers (one source of truth, the same files the Prediction Analysis page reads) ──────────

@st.cache_data(ttl=300)
def validation_summary(key: str) -> dict:
    """Johor / Negeri Sembilan: seats, correct, wrong, accuracy, from backend/models/<state>/validation_2026.csv."""
    df = pd.read_csv(ROOT / f"backend/models/{key}/validation_2026.csv")
    correct, total = int(df['correct'].sum()), len(df)
    return {'total': total, 'correct': correct, 'wrong': total - correct, 'accuracy': correct / total}


def forecast_frame(key: str, with_sentiment: bool, windows=None):
    """Seat probabilities for an upcoming state from its FROZEN forecast. Returns (DataFrame, calls_changed).
    Columns: state, seat_name, probability. With sentiment, the probability is the experimental adjusted one."""
    forecast, _ = latest_frozen(key)
    if forecast is None:
        return None, 0
    if with_sentiment and windows is not None:
        current, reference = windows
        out, _ = sa.adjust_forecast(forecast, sa.load_demographics(key), current, reference)
        frame = pd.DataFrame({'seat_name': out['seat_name'], 'probability': out['sentiment_probability']})
        changed = int(out['sentiment_flips_call'].sum())
    else:
        frame = pd.DataFrame({'seat_name': forecast['seat_name'], 'probability': forecast['probability']})
        changed = 0
    frame['state'] = STATES[key]
    return frame, changed


def seat_key(name) -> str:
    """'N.39 Kota Damansara' and 'Kota Damansara' both become 'kotadamansara', so map and forecast can be joined."""
    name = re.sub(r'^\s*N\.?\s*\d+\s*', '', str(name), flags=re.I)
    return re.sub(r'[^a-z0-9]', '', name.lower())


def swing_category(prob) -> str:
    if pd.isna(prob):
        return 'No forecast yet'
    if prob >= 0.70:
        return 'Harapan (strong)'
    if prob >= 0.55:
        return 'Harapan (lean)'
    if prob >= 0.45:
        return 'Toss-up'
    if prob >= 0.30:
        return 'BN/PN (lean)'
    return 'BN/PN (strong)'


COLOR_MAP = {
    'Harapan (strong)': '#DC2626', 'Harapan (lean)': '#F87171', 'Toss-up': '#A78BFA',
    'BN/PN (lean)': '#60A5FA', 'BN/PN (strong)': '#2563EB', 'No forecast yet': '#4B5563',
}


def change_vs_last_election(key: str):
    """TODO (you): how many seats change side compared with the last election, for the summary card.
    Hint: pages/1_Prediction_Analysis.py already does `df['last_side'] = last_side(df)` with
    `from backend.core.frozen_forecast import latest_frozen, last_side`. Return e.g. {'harapan': +3, 'govt': -3},
    or None while this is not written. The card shows a dash for None."""
    return None


# ── 1. Title and level ──────────────────────────────────────────────────────────

api_ok = health_check()      # LIVE once the API is deployed and reachable, OFFLINE until then
status_class = "status-live" if api_ok else "status-offline"
status_text = "🟢 LIVE" if api_ok else "🔴 OFFLINE"

st.markdown(f"""
<div class="hero-container">
    <span class="status-badge {status_class}">{status_text}</span>
    <div class="hero-title">🗳️ MYRamalan</div>
    <div class="hero-subtitle" style="margin-top:4px; font-style:italic; color:#6B7280;">Data-driven. Real predictions.</div>
    <div class="hero-subtitle">Malaysian State Assembly (DUN) election prediction, validated on actual 2026 results.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stRadio"] > div {
    flex-direction: row;
    gap: 10px;
    background: #1A1D24;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    width: fit-content;
}
div[data-testid="stRadio"] label {
    background: transparent;
    padding: 8px 20px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    transition: background 0.15s ease, color 0.15s ease;
    cursor: pointer;
}
div[data-testid="stRadio"] label:hover {
    background: rgba(220, 38, 38, 0.1);
}
div[data-testid="stRadio"] input:checked + div {
    color: #FAFAFA;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none;
}
</style>
""", unsafe_allow_html=True)

level = st.radio("Level:", ["State (DUN)", "Parliament"], horizontal=True, label_visibility="collapsed")
if "Parliament" in level:
    st.markdown("""
    <div class="hero-container" style="padding:60px 30px; text-align:center; margin-top:16px;">
        <div style="font-size:48px; margin-bottom:12px;">🔒</div>
        <div class="hero-title" style="font-size:28px;">Coming Soon</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not GEOJSON_PATH.exists():
    st.error("Run build_map_data.py first to generate the map.")
    st.stop()
gdf = gpd.read_file(GEOJSON_PATH)
gdf['id'] = gdf['dun']

# ── 2. Headline numbers ─────────────────────────────────────────────────────────

st.write("")
johor, ns = validation_summary('johor'), validation_summary('neg_sembilan')
hc1, hc2 = st.columns([1.3, 2.7])
with hc1:
    st.markdown(f"""<div class="hero-stat-card">
        <div class="value">{len(gdf)}</div><div class="label">DUN Seats Analysed</div></div>""", unsafe_allow_html=True)
with hc2:
    c1, c2, c3 = st.columns(3)
    for col, value, label in [(c1, f"{johor['accuracy']:.2%}", "Johor accuracy (validated)"),
                              (c2, f"{ns['accuracy']:.2%}", "NS accuracy (validated)"),
                              (c3, str(len(STATES)), "States covered")]:
        with col:
            st.markdown(f"""<div class="stat-card"><div class="value">{value}</div>
                <div class="label">{label}</div></div>""", unsafe_allow_html=True)
st.caption("Validated accuracy is the model's pre-election prediction against the real 2026 result, "
           "on 56 Johor seats and 36 Negeri Sembilan seats. See Methodology for the baselines it is compared with.")
st.divider()

# ── 3. Map: forecast only ───────────────────────────────────────────────────────

st.subheader("🗺️ Map of State Legislative Seats (DUN): forecast")
map_view = st.radio("Map view:", ["Prediction (Model)", "Prediction (Model + Sentiment)"], horizontal=True)
use_sentiment = map_view.startswith("Prediction +")

windows = None
if use_sentiment:
    windows = sa.load_windows()
    ok, message = (False, "No sentiment data yet.") if windows is None else sa.data_status(*windows)
    if not ok:
        st.warning(f"{message} Showing the prediction without sentiment.")
        use_sentiment, windows = False, None
    else:
        st.caption(sa.freshness(*windows)['text'] + " Experimental: the effect sizes are assumptions, and no "
                   "accuracy is claimed for this view.")

frames, changed_total, has_forecast = [], 0, []
for key in STATES:
    frame, changed = forecast_frame(key, use_sentiment, windows)
    if frame is not None:          # no frozen forecast -> the state stays grey on the map
        frames.append(frame)
        has_forecast.append(STATES[key])
        changed_total += changed
probs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['state', 'seat_name', 'probability'])
probs['key'] = probs['seat_name'].map(seat_key)
probs = probs.drop_duplicates(['state', 'key'])

gdf['key'] = gdf['dun'].map(seat_key)
gdf = gdf.drop(columns=[c for c in ['probability'] if c in gdf.columns]).merge(
    probs[['state', 'key', 'probability']], on=['state', 'key'], how='left')
gdf['color_label'] = gdf['probability'].map(swing_category)

with st.expander("🔍 Filter map by state"):
    view_states = st.multiselect("Show only:", list(STATES.values()), default=list(STATES.values()))
gdf_view = gdf[gdf['state'].isin(view_states)] if view_states else gdf

fig = px.choropleth_map(
    gdf_view, geojson=json.loads(gdf_view.to_json()), locations='id', featureidkey='properties.id',
    color='color_label', color_discrete_map=COLOR_MAP, map_style="carto-positron",
    zoom=7.3, center={"lat": 2.6, "lon": 102.7}, opacity=0.80, hover_name='dun',
    hover_data={'state': True, 'probability': ':.2f', 'id': False, 'color_label': False},
    labels={'color_label': 'Forecast'},
)
fig.update_traces(marker_line_width=1.0, marker_line_color='rgba(255,255,255,0.4)')
fig.update_layout(height=680, margin={"r": 0, "t": 0, "l": 0, "b": 0}, showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style="background:#1A1D24; border-radius:10px; padding:14px 18px; margin-top:-8px;">
    <div class="legend-row" style="margin-bottom:0;">
        <span class="legend-label">Forecast:</span>
        <span><span class="legend-swatch" style="background:#2563EB;"></span>BN/PN strong</span>
        <span><span class="legend-swatch" style="background:#60A5FA;"></span>BN/PN lean</span>
        <span><span class="legend-swatch" style="background:#A78BFA;"></span>Toss-up</span>
        <span><span class="legend-swatch" style="background:#F87171;"></span>Harapan lean</span>
        <span><span class="legend-swatch" style="background:#DC2626;"></span>Harapan strong</span>
        <span><span class="legend-swatch" style="background:#4B5563;"></span>No forecast yet</span>
    </div>
</div>
""", unsafe_allow_html=True)

caption = ("Hover a seat for its probability. Many seats use the model's fallback to the seat's own history "
           "(see Methodology, limitations), so 'strong' means the forecast is far from 50%, not that it is certain.")
if use_sentiment:
    caption += f" Seats whose call changes with sentiment: {changed_total}."
st.caption(caption)

unmatched = int(gdf.loc[gdf['state'].isin(has_forecast), 'probability'].isna().sum())
if unmatched:
    st.warning(f"{unmatched} forecast-state seats on the map could not be matched to a forecast (seat names differ). "
               "Check seat_key() against the 'dun' column of map_data.geojson.")
st.divider()

# ── 4. State-by-state summary ───────────────────────────────────────────────────

st.subheader("📊 State-by-State Summary")
summary_view = st.radio("Summary:", ["Validated", "Upcoming Forecast"], horizontal=True,
                        help="Validated: states that have voted, checked against the real result. "
                             "Forecast: predictions for states that have not voted yet.")


def validated_card(key: str):
    s = validation_summary(key)
    st.markdown(f"**{STATES[key]}**")
    st.markdown("🟢 Validated")
    st.markdown(f"""
    <div class="stat-card" style="text-align:left; padding:14px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span style="color:#4ADE80; font-size:12px;">Accuracy</span>
            <span style="font-weight:700;">{s['accuracy']:.2%}</span></div>
        <div style="font-size:11px; color:#6B7280; margin-bottom:8px;">{s['correct']}/{s['total']} seats correct</div>
        <div style="display:flex; justify-content:space-between; background:#450A0A; border-radius:6px; padding:6px 8px;">
            <span style="color:#FCA5A5; font-size:12px;">Wrong predictions</span>
            <span style="font-weight:700; color:#FCA5A5; font-size:12px;">{s['wrong']} ({s['wrong'] / s['total']:.0%})</span></div>
    </div>""", unsafe_allow_html=True)


def forecast_card(key: str):
    st.markdown(f"**{STATES[key]}**")
    forecast, _ = latest_frozen(key)
    if forecast is None:
        st.markdown("⚪ Not available yet")
        st.markdown("""<div class="stat-card" style="text-align:left; padding:14px; color:#9CA3AF; font-size:12px;">
            Forecast data not available yet.</div>""", unsafe_allow_html=True)
        return
    harapan = int((forecast['probability'] >= 0.5).sum())
    total = len(forecast)
    govt = total - harapan
    delta = change_vs_last_election(key)          # TODO (you), see the function above
    d_h = f"{delta['harapan']:+d} vs last election" if delta else "change vs last election: not computed yet"
    d_g = f"{delta['govt']:+d} vs last election" if delta else "change vs last election: not computed yet"
    st.markdown("🟡 Forecast")
    st.markdown(f"""
    <div class="stat-card" style="text-align:left; padding:14px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span style="color:#93C5FD; font-size:12px;">BN+PN</span>
            <span style="font-weight:700;">{govt} ({govt / total:.0%})</span></div>
        <div style="font-size:11px; color:#6B7280; margin-bottom:8px;">{d_g}</div>
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span style="color:#FCA5A5; font-size:12px;">Harapan</span>
            <span style="font-weight:700;">{harapan} ({harapan / total:.0%})</span></div>
        <div style="font-size:11px; color:#6B7280;">{d_h}</div>
    </div>""", unsafe_allow_html=True)


keys = VALIDATED if summary_view == "Validated" else list(STATES)
for col, key in zip(st.columns(len(keys)), keys):
    with col:
        (validated_card if summary_view == "Validated" else forecast_card)(key)
        st.write("")
        if st.button("→ View details", key=f"detail_{summary_view}_{key}", use_container_width=True):
            st.switch_page("pages/1_Prediction_Analysis.py")

# ── 5. Disclaimer and navigation ────────────────────────────────────────────────

st.divider()
st.caption("⚠️ **Personal portfolio project** demonstrating data science methodology. "
           "Not an official or authoritative election prediction service.")
st.markdown("**Pages:** **Prediction Analysis** (seat tables, validation and forecasts) · "
            "**Sentiment Analysis** (the experimental news layer) · **Chatbot** (ask questions) · "
            "**Methodology** (models, baselines and limitations)")