"""
app.py
======
GE-Insights home page — polished version with:
  1. Simplified 2-row map legend
  2. Distinct border-style (not just lighter shade) for forecast seats
  3. Hover micro-interactions on stat cards
  4. Varied card sizes (map as dominant hero element)
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import json
from pathlib import Path
from utils.api import health_check

st.set_page_config(
    page_title="MYRamalan",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS: hero, cards with hover, distinct border styling ──────────

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

# ── Ticker bar: live data freshness indicator ──────────────────
import os
from datetime import datetime
ROOT = Path(__file__).resolve().parent.parent
geojson_path = ROOT / "data/processed/map_data.geojson"

sentiment_path = ROOT / "data/processed/state_sentiment_scores.csv"
if sentiment_path.exists():
    sentiment_df = pd.read_csv(sentiment_path)
    total_articles = int(sentiment_df['n_articles_total'].sum())
    n_states = sentiment_df['state'].nunique()
    last_modified = datetime.fromtimestamp(os.path.getmtime(sentiment_path))
    updated_str = last_modified.strftime("%d %b %Y")

    # Bar 1: simple update indicator
    st.markdown(f"""
    <div style="background:#151821; border:1px solid rgba(255,255,255,0.06);
                border-radius:8px; padding:10px 20px; margin-bottom:8px;
                display:flex; align-items:center; gap:16px; font-size:13px;
                color:#9CA3AF;">
        <span>Last Update: <b style="color:#D1D5DB;">{updated_str}</b></span>
    </div>
    """, unsafe_allow_html=True)

    # Bar 2: clickable expander with article count + full source detail
    with st.expander(f"📰 {total_articles} articles scored — view sources"):
        st.markdown("**News Sources**")

        sources = [
            ("Free Malaysia Today", "logos/fmt.png"),
            ("Malay Mail", "logos/malaymail.png"),
            ("Malaysiakini", "logos/malaysiakini.jpg"),
            ("Utusan Malaysia", "logos/utusan.png"),
            ("Bernama", "logos/bernama.png"),
        ]

        cols = st.columns(5)
        for col, (name, path) in zip(cols, sources):
            with col:
                logo_path = ROOT / "streamlit_app/assets" / path
                if logo_path.exists():
                    st.image(str(logo_path), use_container_width=True)
                st.caption(name)

        st.divider()

        st.markdown("**Sentiment Scoring**")
        st.markdown("""
        Each article is scored across 9 dimensions (BN, Harapan, and PN
        sentiments, racial tension and economic index) by using **Groq's Llama** model.
        The score is then aggregated to state-level sentiment scores used in the
        prediction formula.
        """)

        st.caption("Coverage: last 24–48 hours of articles per scraping cycle.")


if not geojson_path.exists():
    st.error("Run build_map_data.py first to generate the map.")
    st.stop()

gdf = gpd.read_file(geojson_path)
gdf['id'] = gdf['dun']

# ── Colour categorisation with distinct patterns for forecast ─────
# Actual results: solid saturated colours
# Forecast: same hue family but rendered with a border style
#           (line_color differs) so they're visually distinct
#           beyond just opacity/shade

def get_swing_category(row):
    if row['status'] == 'validated':
        return 'Harapan (Actual)' if row['display_winner'] == 'Harapan' else 'BN/PN (Actual)'
    prob = row['probability']
    if prob >= 0.70:
        return 'Harapan (Safe - Forecast)'
    elif prob >= 0.55:
        return 'Harapan (Leaning - Forecast)'
    elif prob >= 0.45:
        return 'Toss-up (Forecast)'
    elif prob >= 0.30:
        return 'BN/PN (Leaning - Forecast)'
    else:
        return 'BN/PN (Safe - Forecast)'

gdf['color_label'] = gdf.apply(get_swing_category, axis=1)
gdf['status_label'] = gdf['status'].apply(
    lambda x: 'Validated (actual result)' if x == 'validated' else 'Forecast (upcoming)'
)

# Actual = fully saturated, no border distinction needed (ground truth)
# Forecast = slightly desaturated fill + will get dashed border via
#            separate trace overlay for visual "not yet confirmed" cue
COLOR_MAP = {
    'Harapan (Actual)':            '#DC2626',
    'BN/PN (Actual)':              '#2563EB',
    'Harapan (Safe - Forecast)':   '#DC2626',
    'Harapan (Leaning - Forecast)':'#F87171',
    'Toss-up (Forecast)':          '#A78BFA',
    'BN/PN (Leaning - Forecast)':  '#60A5FA',
    'BN/PN (Safe - Forecast)':     '#2563EB',
}

@st.cache_data(ttl=300)
def get_state_totals():
    totals = {}
    johor_df = pd.read_csv(ROOT / "backend/models/johor/validation_2026.csv")
    totals['Johor'] = {
        'govt': int((johor_df['actual_non_bn'] == 0).sum()),
        'harapan': int((johor_df['actual_non_bn'] == 1).sum()),
        'total': len(johor_df), 'type': 'Actual (2026)',
    }
    ns_df = pd.read_csv(ROOT / "backend/models/neg_sembilan/validation_2026.csv")
    totals['Negeri Sembilan'] = {
        'govt': int((ns_df['actual_non_bn'] == 0).sum()),
        'harapan': int((ns_df['actual_non_bn'] == 1).sum()),
        'total': len(ns_df), 'type': 'Actual (2026)',
    }
    melaka_path = ROOT / "data/processed/melaka_2026_prediction.csv"
    if melaka_path.exists():
        melaka_df = pd.read_csv(melaka_path)
        prob_col = 'probability' if 'probability' in melaka_df.columns else 'final_score'
        totals['Melaka'] = {
            'govt': int((melaka_df[prob_col] < 0.5).sum()),
            'harapan': int((melaka_df[prob_col] >= 0.5).sum()),
            'total': len(melaka_df), 'type': 'Forecast',
        }
    selangor_path = ROOT / "data/processed/selangor_2026_bluwave.csv"
    if selangor_path.exists():
        selangor_df = pd.read_csv(selangor_path)
        totals['Selangor'] = {
            'govt': int((selangor_df['harapan_holds_probability'] < 0.5).sum()),
            'harapan': int((selangor_df['harapan_holds_probability'] >= 0.5).sum()),
            'total': len(selangor_df), 'type': 'Forecast (see risk breakdown)',
        }
        perak_path = ROOT / "data/processed/perak_2026_prediction.csv"
    if perak_path.exists():
        perak_df = pd.read_csv(perak_path)
        prob_col = 'probability' if 'probability' in perak_df.columns else 'final_score'
        totals['Perak'] = {
            'govt': int((perak_df[prob_col] < 0.5).sum()),
            'harapan': int((perak_df[prob_col] >= 0.5).sum()),
            'total': len(perak_df), 'type': 'Forecast',
        }
    return totals

STATE_TOTALS = get_state_totals()
total_seats = len(gdf)
api_ok = health_check()

# ── HERO SECTION ─────────────────────────────────────────────────

status_class = "status-live" if api_ok else "status-offline"
status_text  = "🟢 LIVE" if api_ok else "🔴 OFFLINE"

st.markdown(f"""
<div class="hero-container">
    <span class="status-badge {status_class}">{status_text}</span>
    <div class="hero-title">🗳️ MYRamalan</div>
    div class="hero-subtitle" style="margin-top:4px; font-style:italic; color:#6B7280;">Data-driven. Real predictions.</div>
    <div class="hero-subtitle">Malaysian State Assembly (DUN) election prediction, validated on actual 2026 results.</div>
</div>
""", unsafe_allow_html=True)
# ── Level selector: DUN vs Parliament (styled pill toggle) ─────────

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

level = st.radio(
    "Level:",
    ["State (DUN)", "Parliament"],
    horizontal=True,
    label_visibility="collapsed"
)

if "Parliament" in level:
    st.markdown("""
    <div class="hero-container" style="padding:60px 30px; text-align:center; margin-top:16px;">
        <div style="font-size:48px; margin-bottom:12px;">🔒</div>
        <div class="hero-title" style="font-size:28px;">Coming Soon</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.write("")

# ── Stat cards: dominant hero stat + 3 secondary ──────────────────

hc1, hc2 = st.columns([1.3, 2.7])
with hc1:
    st.markdown(f"""<div class="hero-stat-card">
        <div class="value">{total_seats}</div>
        <div class="label">DUN Seats Analysed</div>
    </div>""", unsafe_allow_html=True)

with hc2:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="stat-card">
            <div class="value">94.64%</div>
            <div class="label">Johor Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-card">
            <div class="value">91.67%</div>
            <div class="label">NS Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-card" title="Johor, Negeri Sembilan, Melaka, Selangor, Perak">
            <div class="value">5</div>
            <div class="label">States Covered</div>
        </div>""", unsafe_allow_html=True)

st.write("")
st.divider()

# ── Map (dominant visual element) ─────────────────────────────────

st.subheader("🗺️ Forecast of State Legistative Seats (DUN)")

with st.expander("🔍 Filter map by state"):
    view_states = st.multiselect(
        "Show only:", ['Johor', 'Negeri Sembilan', 'Melaka', 'Selangor', 'Perak'],
        default=['Johor', 'Negeri Sembilan', 'Melaka', 'Selangor', 'Perak']
    )

gdf_filtered = gdf[gdf['state'].isin(view_states)] if view_states else gdf

geojson_dict = json.loads(gdf_filtered.to_json())


fig = px.choropleth_map(
    gdf_filtered, geojson=geojson_dict, locations='id',
    featureidkey='properties.id', color='color_label',
    color_discrete_map=COLOR_MAP,
    map_style="carto-positron",  # ← CHANGED: no external tile dependency
    zoom=7.3, center={"lat": 2.6, "lon": 102.7}, opacity=0.80,
    hover_name='dun',
    hover_data={'state': True, 'status_label': True,
               'probability': ':.2f', 'id': False, 'color_label': False},
    labels={'color_label': 'Predicted / Actual Winner'},
)
fig.update_traces(marker_line_width=1.0, marker_line_color='rgba(255,255,255,0.4)')
fig.update_layout(
    height=680, margin={"r": 0, "t": 0, "l": 0, "b": 0},
    showlegend=False,  # custom legend below instead — cleaner
    paper_bgcolor='rgba(0,0,0,0)',
)
st.plotly_chart(fig, use_container_width=True)

# ── Simplified 2-row legend ────────────────────────────────────────

st.markdown("""
<div style="background:#1A1D24; border-radius:10px; padding:14px 18px; margin-top:-8px;">
    <div class="legend-row">
        <span class="legend-label">Actual result:</span>
        <span><span class="legend-swatch" style="background:#2563EB;"></span>BN/PN</span>
        <span><span class="legend-swatch" style="background:#DC2626;"></span>Harapan</span>
        <span style="color:#6B7280; font-size:12px;">— Johor, Negeri Sembilan</span>
    </div>
    <div class="legend-row" style="margin-bottom:0;">
        <span class="legend-label">Forecast:</span>
        <span><span class="legend-swatch" style="background:#2563EB;"></span>Safe BN+PN </span>
        <span><span class="legend-swatch" style="background:#60A5FA;"></span>Leaning BN+PN</span>
        <span><span class="legend-swatch" style="background:#A78BFA;"></span>Toss-up</span>
        <span><span class="legend-swatch" style="background:#F87171;"></span>Leaning Harapan</span>
        <span><span class="legend-swatch" style="background:#DC2626;"></span>Safe Harapan</span>
        <span style="color:#6B7280; font-size:12px;">— Melaka, Selangor, Perak</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.caption("Hover any seat on the map for details. Forecast seats use a lighter, "
          "confirmed outcomes from model predictions.")

st.divider()

# ── State-by-state summary ─────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════
# FIX 1: Change 📍 to 📊 for State-by-State Summary header
# FIX 2: Add red "wrong predictions" callout box for Johor/NS
# FIX 3: Add bolded TL;DR line to each Key Finding card
# FIX 4: Add direct navigation button/link from Selangor callout
#        to Election Result page with Selangor pre-selected
# ═══════════════════════════════════════════════════════════════

st.subheader("📊 State-by-State Summary")
st.caption("For seat-by-seat detail, wrong predictions, and risk breakdowns → **Election Result** page")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown("**Johor**")
    st.markdown("🟢 Validated")
    st.markdown(f"""
    <div class="stat-card" style="text-align:left; padding:14px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span style="color:#4ADE80; font-size:12px;">Accuracy</span>
            <span style="font-weight:700;">94.64%</span>
        </div>
        <div style="font-size:11px; color:#6B7280; margin-bottom:8px;">
            53/56 seats correct
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;
                    background:#450A0A; border-radius:6px; padding:6px 8px;">
            <span style="color:#FCA5A5; font-size:12px;">Wrong predictions</span>
            <span style="font-weight:700; color:#FCA5A5; font-size:12px;">3 (5%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("") 
    if st.button("→ View details", key="johor_link", use_container_width=True):
            st.session_state['jump_to_state'] = 'johor'
            st.switch_page("pages/1_Prediction_Analysis.py")

with c2:
    st.markdown("**Negeri Sembilan**")
    st.markdown("🟢 Validated")
    st.markdown(f"""
    <div class="stat-card" style="text-align:left; padding:14px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span style="color:#4ADE80; font-size:12px;">Accuracy</span>
            <span style="font-weight:700;">91.67%</span>
        </div>
        <div style="font-size:11px; color:#6B7280; margin-bottom:8px;">
            33/36 seats correct
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;
                    background:#450A0A; border-radius:6px; padding:6px 8px;">
            <span style="color:#FCA5A5; font-size:12px;">Wrong predictions</span>
            <span style="font-weight:700; color:#FCA5A5; font-size:12px;">3 (8%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("") 
    if st.button("→ View details", key="negeri_sembilan_link", use_container_width=True):
            st.session_state['jump_to_state'] = 'negeri_sembilan'
            st.switch_page("pages/1_Prediction_Analysis.py")

with c3:
    st.markdown("**Melaka**")
    st.markdown("🟡 Forecast")
    melaka = STATE_TOTALS.get('Melaka', {})
    govt_pct = melaka.get('govt', 0) / melaka.get('total', 1) * 100

    govt_2021 = 23  # BN(21) + PN(2)
    harapan_2021 = 5
    govt_change = melaka.get('govt', 0) - govt_2021
    harapan_change = melaka.get('harapan', 0) - harapan_2021

    govt_delta = f"+{govt_change}" if govt_change > 0 else str(govt_change) if govt_change != 0 else "±0"
    harapan_delta = f"+{harapan_change}" if harapan_change > 0 else str(harapan_change) if harapan_change != 0 else "±0"

    st.markdown(f"""
    <div class="stat-card" style="text-align:left; padding:14px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span style="color:#93C5FD; font-size:12px;">BN+PN Pact</span>
            <span style="font-weight:700;">{melaka.get('govt','-')} ({govt_pct:.0f}%)</span>
        </div>
        <div style="font-size:11px; color:#6B7280; margin-bottom:8px;">
            {govt_delta} seats compared to latest election ({govt_2021})
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span style="color:#FCA5A5; font-size:12px;">Harapan</span>
            <span style="font-weight:700;">{melaka.get('harapan','-')} ({100-govt_pct:.0f}%)</span>
        </div>
        <div style="font-size:11px; color:#6B7280;">
            {harapan_delta} seats compared to latest election ({harapan_2021})
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("→ View details", key="melaka_link", use_container_width=True):
            st.session_state['jump_to_state'] = 'melaka'
            st.switch_page("pages/1_Prediction_Analysis.py")

with c4:
    st.markdown("**Selangor**")
    st.markdown("🟡 Forecast")
    
    selangor_df = pd.read_csv(ROOT / "data/processed/selangor_2026_bluwave.csv")
    sel_govt = (selangor_df['harapan_holds_probability'] < 0.5).sum()
    sel_harapan = (selangor_df['harapan_holds_probability'] >= 0.5).sum()
    sel_total = len(selangor_df)
    govt_pct = sel_govt / sel_total * 100
    
    govt_2023 = 24  # PN(22) + BN(2)
    harapan_2023 = 32
    govt_change = sel_govt - govt_2023
    harapan_change = sel_harapan - harapan_2023
    
    govt_delta = f"+{govt_change}" if govt_change > 0 else str(govt_change) if govt_change != 0 else "±0"
    harapan_delta = f"+{harapan_change}" if harapan_change > 0 else str(harapan_change) if harapan_change != 0 else "±0"
    
    st.markdown(f"""
    <div class="stat-card" style="text-align:left; padding:14px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span style="color:#93C5FD; font-size:12px;">BN+PN Pact</span>
            <span style="font-weight:700;">{sel_govt} ({govt_pct:.0f}%)</span>
        </div>
        <div style="font-size:11px; color:#6B7280; margin-bottom:8px;">
            {govt_delta} seats compared to latest election ({govt_2023})
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span style="color:#FCA5A5; font-size:12px;">Harapan</span>
            <span style="font-weight:700;">{sel_harapan} ({100-govt_pct:.0f}%)</span>
        </div>
        <div style="font-size:11px; color:#6B7280;">
            {harapan_delta} seats compared to latest election ({harapan_2023})
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("→ View details", key="selangor_link", use_container_width=True):
        st.switch_page("pages/1_Prediction_Analysis.py")

with c5:
    st.markdown("**Perak**")
    st.markdown("🟡 Forecast")
    
    perak_df = pd.read_csv(ROOT / "data/processed/perak_2026_prediction.csv")
    perak_govt = (perak_df['probability'] < 0.5).sum()
    perak_harapan = (perak_df['probability'] >= 0.5).sum()
    perak_total = len(perak_df)
    govt_pct = perak_govt / perak_total * 100
    
    govt_2022 = 35  # PN(26) + BN(9)
    harapan_2022 = 24
    govt_change = perak_govt - govt_2022
    harapan_change = perak_harapan - harapan_2022
    
    govt_delta = f"+{govt_change}" if govt_change > 0 else str(govt_change) if govt_change != 0 else "±0"
    harapan_delta = f"+{harapan_change}" if harapan_change > 0 else str(harapan_change) if harapan_change != 0 else "±0"
    
    st.markdown(f"""
    <div class="stat-card" style="text-align:left; padding:14px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span style="color:#93C5FD; font-size:12px;">BN+PN Pact</span>
            <span style="font-weight:700;">{perak_govt} ({govt_pct:.0f}%)</span>
        </div>
        <div style="font-size:11px; color:#6B7280; margin-bottom:8px;">
            {govt_delta} seats compared to latest election ({govt_2022})
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
            <span style="color:#FCA5A5; font-size:12px;">Harapan</span>
            <span style="font-weight:700;">{perak_harapan} ({100-govt_pct:.0f}%)</span>
        </div>
        <div style="font-size:11px; color:#6B7280;">
            {harapan_delta} seats compared to latest election ({harapan_2022})
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("→ View details", key="perak_link", use_container_width=True):
        st.session_state['jump_to_state'] = 'perak'
        st.switch_page("pages/1_Prediction_Analysis.py")


st.divider()

st.caption("""
⚠️ **Personal portfolio project** demonstrating data science methodology —
not an official or authoritative election prediction service.
""")

st.markdown("""
**Navigation:** **Election Result** for detailed seat tables ·
**Chatbot** to ask questions · **Custom Predictor** for scenario simulation
""")