import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="MYRamalan", page_icon="🗳️", layout="wide")

# ── Shared design system CSS (same as home page) ──────────────────

st.markdown("""
<style>
.hero-container-sm {
    background: linear-gradient(135deg, #1a0505 0%, #3d0d0d 50%, #1a0505 100%);
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 20px;
    border: 1px solid rgba(220, 38, 38, 0.2);
}
.hero-title-sm {
    font-size: 32px;
    font-weight: 800;
    color: #FAFAFA;
    margin-bottom: 4px;
}
.stat-card {
    background: #1A1D24;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    min-height: 90px;
}
.stat-card .value {
    font-size: 30px;
    font-weight: 800;
    color: #FAFAFA;
}
.stat-card .label {
    font-size: 12px;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Pill-style radio toggle (matches home page) */
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
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-container-sm">
    <div class="hero-title-sm">🗳️ MYRamalan - Prediction Analysis</div>
</div>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parents[2]

STATE_STATUS = {
    'johor': 'validated', 'neg_sembilan': 'validated',
    'melaka': 'forecast', 'selangor': 'forecast', 'perak': 'forecast',
}
STATE_LABELS = {
    'johor': 'Johor', 'neg_sembilan': 'Negeri Sembilan',
    'melaka': 'Melaka', 'selangor': 'Selangor', 'perak': 'Perak',
}

def stat_card(value, label, color="#FAFAFA"):
    st.markdown(f"""<div class="stat-card">
        <div class="value" style="color:{color};">{value}</div>
        <div class="label">{label}</div>
    </div>""", unsafe_allow_html=True)

# ── State selector ────────────────────────────────────────────────

category = st.radio(
    "Category:",
    ["Validated (Recent Elections from 2026)", "Upcoming Forecasts"],
    horizontal=True
)

available_states = [s for s, v in STATE_STATUS.items()
                    if v == ('validated' if 'Validated' in category else 'forecast')]

state = st.selectbox("Select state:", available_states,
                     format_func=lambda x: STATE_LABELS[x])
status = STATE_STATUS[state]

# ══════════════════════════════════════════════════════════════════
# VALIDATED: Johor, Negeri Sembilan
# ══════════════════════════════════════════════════════════════════

if status == 'validated':

    st.markdown("**Prediction Forecast vs 2026 DUN Result**")

    csv_path = ROOT / f"backend/models/{state}/validation_2026.csv"
    if not csv_path.exists():
        st.error(f"No validation data for {state}")
        st.stop()

    df = pd.read_csv(csv_path)
    total    = len(df)
    correct  = df['correct'].sum()
    wrong    = total - correct
    accuracy = correct / total

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card(total, "Total Seats")
    with c2: stat_card(f"{accuracy:.2%}", "Accuracy")
    with c3: stat_card(int(correct), "Correct", "#4ADE80")
    with c4: stat_card(int(wrong), "Wrong", "#F87171")

    st.write("")

    bn_pn_actual   = (df['actual_non_bn'] == 0).sum()
    harapan_actual = (df['actual_non_bn'] == 1).sum()

    c1, c2 = st.columns(2)
    with c1: stat_card(int(bn_pn_actual), "BN/PN Won (Actual)", "#93C5FD")
    with c2: stat_card(int(harapan_actual), "Harapan Won (Actual)", "#FCA5A5")

    st.divider()

    df['prediction_label'] = df['predicted_non_bn'].apply(
        lambda x: 'Harapan' if x == 1 else 'BN/PN'
    )
    df['actual_label'] = df['actual_non_bn'].apply(
        lambda x: 'Harapan' if x == 1 else 'BN/PN'
    )

    st.subheader("Model's False Predictions")

    wrong_df = df[df['correct'] == False]

    if wrong_df.empty:
        st.success("All predictions are correct.")
    else:
        def coalition_badge(label):
            if label == "BN/PN":
                return ('<span style="background:#1E3A5F; color:#93C5FD; '
                        'padding:3px 10px; border-radius:5px; font-size:12px; '
                        'font-weight:700; letter-spacing:0.3px;">BN/PN</span>')
            else:
                return ('<span style="background:#5F1E1E; color:#FCA5A5; '
                        'padding:3px 10px; border-radius:5px; font-size:12px; '
                        'font-weight:700; letter-spacing:0.3px;">HARAPAN</span>')

        st.markdown("""
        <style>
        .false-pred-row {
            background: #1A1D24;
            border: 1px solid rgba(255,255,255,0.08);
            border-left: 3px solid #F87171;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 10px;
            transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
        }
        .false-pred-row:hover {
            transform: translateX(4px);
            border-left-color: #DC2626;
            box-shadow: 0 4px 14px rgba(220, 38, 38, 0.12);
        }
        .false-pred-seat {
            font-size: 15px;
            font-weight: 700;
            color: #FAFAFA;
        }
        .false-pred-meta {
            font-size: 12px;
            color: #9CA3AF;
            margin-top: 2px;
        }
        .false-pred-prob {
            font-size: 16px;
            font-weight: 800;
            color: #FAFAFA;
        }
        </style>
        """, unsafe_allow_html=True)

        for _, row in wrong_df.iterrows():
            pred_badge   = coalition_badge(row['prediction_label'])
            actual_badge = coalition_badge(row['actual_label'])
            st.markdown(f"""
            <div class="false-pred-row">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="false-pred-seat">{row['seat']}</div>
                        <div class="false-pred-meta">
                            Predicted {pred_badge} &nbsp;→&nbsp; Actual {actual_badge}
                        </div>
                    </div>
                    <div class="false-pred-prob">P={row['probability']:.2f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Pattern insight, only if demographic data is available
        if 'chinese_pct' in wrong_df.columns:
            avg_chinese_wrong = wrong_df['chinese_pct'].mean()
            st.info(f"""
            **Pattern:** {len(wrong_df)} wrong predictions, average Chinese
            composition {avg_chinese_wrong:.0%} — concentrated in
            Chinese-majority seats where demographic signal was weaker.
            """)
        else:
            st.info(f"""
            **{len(wrong_df)} of {len(df)} seats** ({len(wrong_df)/len(df):.1%})
            were mispredicted — see the confidence chart below for the full
            distribution of prediction certainty across all seats.
            """)

    st.divider()

    # ── Toggle: Validated view vs Predicted (formula-driven) view ──

    chart_view = st.radio(
        "Chart view:",
        ["✅ Validated (vs actual result)", "📊 Predicted (seat composition)"],
        horizontal=True
    )

    if chart_view == "✅ Validated (vs actual result)":
        st.subheader("Prediction Confidence vs Outcome")
        st.caption("Sorted left (BN+PN Pact confident) to right (Harapan confident) — X marks show wrong predictions")

        df_sorted = df.sort_values('probability').reset_index(drop=True)

        fig = px.scatter(
            df_sorted, x='probability', y='seat',
            color='correct',
            color_discrete_map={True: '#4ADE80', False: '#F87171'},
            symbol='correct',
            symbol_map={True: 'circle', False: 'x'},
            labels={'probability': 'Prediction (0 = BN+PN Pact wins, 1 = Harapan wins)', 'seat': ''},
        )
        fig.update_traces(marker=dict(size=11, line=dict(width=1, color='rgba(255,255,255,0.3)')))
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray",
                     annotation_text="50/50", annotation_position="top")
        fig.add_annotation(x=0.05, y=1.05, xref="paper", yref="paper",
                          text="← BN+PN Pact confident", showarrow=False,
                          font=dict(color="#93C5FD", size=11))
        fig.add_annotation(x=0.95, y=1.05, xref="paper", yref="paper",
                          text="Harapan confident →", showarrow=False,
                          font=dict(color="#FCA5A5", size=11))
        fig.update_yaxes(categoryorder='array', categoryarray=df_sorted['seat'].tolist(), tickfont=dict(size=10))
        fig.update_layout(
            height=max(500, len(df_sorted) * 13),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=10, r=10, t=60, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader("Behind the Prediction")
        st.caption("Left of dashed line (BN+PN Pact favoured) vs right (Harapan favoured) — "
                "shows how seat composition drives the forecast")

        df_plot = df.copy()
        df_plot['ethnic_margin'] = df_plot['chinese_pct'] - df_plot['malay_pct']
        df_plot = df_plot.sort_values('probability').reset_index(drop=True)

        fig = px.scatter(
            df_plot, x='ethnic_margin', y='probability',
            color='probability',
            color_continuous_scale=[[0, '#2563EB'], [0.5, '#A78BFA'], [1, '#DC2626']],
            hover_data=['seat'],
            labels={
                'chinese_pct': 'Seat Ethnic Composition (Chinese %)',
                'probability': 'Prediction (0 = BN+PN Pact, 1 = Harapan)'
            },
        )
        fig.update_traces(marker=dict(size=13, line=dict(width=1, color='rgba(255,255,255,0.3)')))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                    annotation_text="50/50", annotation_position="right")
        fig.add_annotation(x=0.02, y=0.05, xref="paper", yref="paper",
                        text="↓ BN+PN Pact favoured", showarrow=False,
                        font=dict(color="#93C5FD", size=11))
        fig.add_annotation(x=0.02, y=0.95, xref="paper", yref="paper",
                        text="↑ Harapan favoured", showarrow=False,
                        font=dict(color="#FCA5A5", size=11))
        fig.update_layout(
            height=450,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_colorbar=dict(title="Prediction", tickformat='.1f'),
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Pattern:** Seat-level demographic composition tracks closely with 
        predicted outcome — consistent with Malaysia's well-documented coalition voting history,
        not a new or original claim.
        """)

    st.divider()

    display_cols = ['seat', 'prediction_label', 'actual_label',
                    'correct', 'probability', 'is_ood']
    df_display = df[display_cols].copy()
    df_display.columns = ['Seat', 'Predicted', 'Actual', 'Correct', 'Probability', 'OOD']
    df_display['Probability'] = df_display['Probability'].apply(lambda x: f"{float(x):.2f}")
    df_display['Correct'] = df_display['Correct'].apply(lambda x: "✅" if x else "❌")

    with st.expander(f"📋 Full seat-by-seat table ({len(df)} seats)"):
        st.dataframe(df_display, use_container_width=True, height=400)

# ══════════════════════════════════════════════════════════════════
# FORECAST: Melaka, Selangor
# ══════════════════════════════════════════════════════════════════

else:
    if state == 'melaka':
        #placed under methodology: st.warning("Based on 2018→2021 patterns + current sentiment/economic data.")
        csv_path = ROOT / "data/processed/melaka_2026_prediction.csv"
        prob_col = 'probability'
        seat_col = 'seat_name'
        last_winner_col = 'winner_2021'     
        winner_col_label = "2021" 

    elif state == 'perak':
        csv_path = ROOT / "data/processed/perak_2026_prediction.csv"
        prob_col = 'probability'
        seat_col = 'seat_name'
        last_winner_col = 'winner_2022'
        winner_col_label = "2022"
    else:
        csv_path = ROOT / "data/processed/selangor_2026_bluwave.csv"
        prob_col = 'harapan_holds_probability'
        seat_col = 'seat_name'
        last_winner_col = 'winner_2023'  
        winner_col_label = "2023" 

    if not csv_path.exists():
        st.error(f"Run predict_{state}.py first to generate data.")
        st.stop()

    df = pd.read_csv(csv_path)

    govt_wins    = (df[prob_col] < 0.45).sum()
    toss_ups     = ((df[prob_col] >= 0.45) & (df[prob_col] < 0.55)).sum()
    harapan_wins = (df[prob_col] >= 0.55).sum()

    total = len(df)
    govt_pct = govt_wins / total * 100
    toss_pct = toss_ups / total * 100
    harapan_pct = harapan_wins / total * 100

    st.markdown(f"""
    <div style="background:#1A1D24; border:1px solid rgba(255,255,255,0.08);
                border-radius:12px; padding:24px; margin-bottom:20px;">
        <div style="display:flex; height:10px; border-radius:5px; overflow:hidden; margin-bottom:20px;">
            <div style="width:{govt_pct}%; background:#2563EB;"></div>
            <div style="width:{toss_pct}%; background:#A78BFA;"></div>
            <div style="width:{harapan_pct}%; background:#DC2626;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
            <div>
                <div style="font-size:12px; color:#93C5FD; text-transform:uppercase; letter-spacing:0.5px;">
                    ● Govt (BN/PN) Predicted
                </div>
                <div style="font-size:32px; font-weight:800; color:#FAFAFA;">{govt_wins}
                    <span style="font-size:14px; font-weight:400; color:#6B7280;"> / {total}</span>
                </div>
            </div>
            <div style="width:1px; height:44px; background:rgba(255,255,255,0.1);"></div>
            <div>
                <div style="font-size:12px; color:#C4B5FD; text-transform:uppercase; letter-spacing:0.5px;">
                    ● Toss-up
                </div>
                <div style="font-size:32px; font-weight:800; color:#FAFAFA;">{toss_ups}
                    <span style="font-size:14px; font-weight:400; color:#6B7280;"> / {total}</span>
                </div>
            </div>
            <div style="width:1px; height:44px; background:rgba(255,255,255,0.1);"></div>
            <div>
                <div style="font-size:12px; color:#FCA5A5; text-transform:uppercase; letter-spacing:0.5px;">
                    ● Harapan Predicted
                </div>
                <div style="font-size:32px; font-weight:800; color:#FAFAFA;">{harapan_wins}
                    <span style="font-size:14px; font-weight:400; color:#6B7280;"> / {total}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    

    safe_govt   = df[df[prob_col] < 0.20].sort_values(prob_col)
    likely_govt = df[(df[prob_col] >= 0.20) & (df[prob_col] < 0.35)].sort_values(prob_col)
    lean_govt   = df[(df[prob_col] >= 0.35) & (df[prob_col] < 0.45)].sort_values(prob_col)
    tossup      = df[(df[prob_col] >= 0.45) & (df[prob_col] < 0.55)].sort_values(prob_col)
    lean_harapan   = df[(df[prob_col] >= 0.55) & (df[prob_col] < 0.65)].sort_values(prob_col, ascending=False)
    likely_harapan = df[(df[prob_col] >= 0.65) & (df[prob_col] < 0.80)].sort_values(prob_col, ascending=False)
    safe_harapan   = df[df[prob_col] >= 0.80].sort_values(prob_col, ascending=False)

    def show_seat_list(subset):
        if subset.empty:
            st.info("No seats in this category.")
            return

        st.markdown("""
        <style>
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #1A1D24;
        }
        </style>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            hcol1, hcol2, hcol3 = st.columns([3, 1.3, 1])
            hcol1.markdown("**Seat**")
            hcol2.markdown("**Trend**")
            hcol3.markdown("**Probability**")
            st.divider()

            for _, row in subset.iterrows():
                trend = row.get('trend', 0)
                if trend > 0.02:
                    trend_color = "#4ADE80"
                    trend_arrow = "↑"
                    trend_label = "Harapan gaining"
                elif trend < -0.02:
                    trend_color = "#F87171"
                    trend_arrow = "↓"
                    trend_label = "Harapan losing"
                else:
                    trend_color = "#6B7280"
                    trend_arrow = "→"
                    trend_label = "Stable"

                c1, c2, c3 = st.columns([3, 1.3, 1])
                c1.markdown(f"**{row[seat_col]}**")
                c2.markdown(f"<span style='color:{trend_color};'>{trend_arrow} {trend:+.2f}</span>", unsafe_allow_html=True)
                c3.markdown(f"`{row[prob_col]:.2f}`")


    st.markdown("""
    <div style="background:#1A1D24; border:1px solid rgba(255,255,255,0.08);
                border-radius:8px; padding:10px 20px; margin-bottom:16px;
                display:flex; align-items:center; gap:20px; font-size:13px;">
        <span style="color:#9CA3AF; font-weight:600;">Trend legend:</span>
        <span style="color:#F87171;">↓ Red — trending toward BN+PN</span>
        <span style="color:#4ADE80;">↑ Green — trending toward Harapan</span>
        <span style="color:#6B7280;">→ Grey — stable, no clear direction</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1A1D24; border:1px solid rgba(255,255,255,0.08);
                border-radius:8px; padding:10px 20px; margin-bottom:16px;
                font-size:12px; color:#9CA3AF;">
        <span style="font-weight:700; color:#D1D5DB;">Trend strength:</span>
        ±0.02 or less = stable &nbsp;·&nbsp;
        ±0.03–0.10 = mild &nbsp;·&nbsp;
        ±0.11–0.20 = moderate &nbsp;·&nbsp;
        ±0.21 and beyond = strong historical shift
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    f"🔴 Safe BN+PN ({len(safe_govt)})",
    f"🟠 Likely BN+PN ({len(likely_govt)})",
    f"🟡 Lean BN+PN ({len(lean_govt)})",
    f"⚪ Toss-up ({len(tossup)})",
    f"🟡 Lean Harapan ({len(lean_harapan)})",
    f"🟢 Likely Harapan ({len(likely_harapan)})",
    f"🟢 Safe Harapan ({len(safe_harapan)})",
])
    with tab1:
        show_seat_list(safe_govt)
    with tab2:
        show_seat_list(likely_govt)
    with tab3:
        st.caption("Competitive Seat")
        show_seat_list(lean_govt)
    with tab4:
        show_seat_list(tossup)
    with tab5:
        show_seat_list(lean_harapan)
    with tab6:
        show_seat_list(likely_harapan)
    with tab7:
        show_seat_list(safe_harapan)

    st.markdown("""
    <div style="margin: 32px 0; border-top: 2px solid rgba(220, 38, 38, 0.15);"></div>
    """, unsafe_allow_html=True)

    st.subheader("🔄 Predicted Seat Flips vs Last Election")
    st.caption(f"Comparing predicted winner to the {winner_col_label} result")

    df_flip = df.copy()
    df_flip['predicted_winner'] = df_flip[prob_col].apply(
        lambda x: 'Harapan' if x >= 0.5 else 'BN/PN'
    )
    df_flip['actual_side'] = df_flip[last_winner_col].apply(
        lambda x: 'Harapan' if x in ['PH', 'Harapan'] else 'BN/PN'
    )

    flips = df_flip[df_flip['predicted_winner'] != df_flip['actual_side']]
    harapan_gains = flips[flips['actual_side'] == 'BN/PN']    # BN/PN -> Harapan (Harapan gains)
    harapan_losses = flips[flips['actual_side'] == 'Harapan']  # Harapan -> BN/PN (BN/PN gains)

    net_change = len(harapan_gains) - len(harapan_losses)
    net_label = f"Harapan +{net_change}" if net_change > 0 else (f"BN/PN +{-net_change}" if net_change < 0 else "±0")
    net_color = "#F87171" if net_change > 0 else ("#93C5FD" if net_change < 0 else "#9CA3AF")

    # ── Summary strip ────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#1A1D24; border:1px solid rgba(255,255,255,0.08);
                border-radius:12px; padding:20px 24px; margin-bottom:20px;
                display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
        <div>
            <div style="font-size:12px; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.5px;">
                Total Flips
            </div>
            <div style="font-size:32px; font-weight:800; color:#FAFAFA;">{len(flips)}</div>
        </div>
        <div style="width:1px; height:40px; background:rgba(255,255,255,0.1);"></div>
        <div>
            <div style="font-size:12px; color:#F87171; text-transform:uppercase; letter-spacing:0.5px;">
                ↑ Harapan Gains
            </div>
            <div style="font-size:32px; font-weight:800; color:#F87171;">{len(harapan_gains)}</div>
        </div>
        <div style="width:1px; height:40px; background:rgba(255,255,255,0.1);"></div>
        <div>
            <div style="font-size:12px; color:#93C5FD; text-transform:uppercase; letter-spacing:0.5px;">
                ↑ BN/PN Gains
            </div>
            <div style="font-size:32px; font-weight:800; color:#93C5FD;">{len(harapan_losses)}</div>
        </div>
        <div style="width:1px; height:40px; background:rgba(255,255,255,0.1);"></div>
        <div>
            <div style="font-size:12px; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.5px;">
                Net Change
            </div>
            <div style="font-size:24px; font-weight:800; color:{net_color};">{net_label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(flips) == 0:
        st.success("No seats predicted to flip — all seats match their last election result.")
    else:
        fcol1, fcol2 = st.columns(2)

        with fcol1:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="font-size:18px;">📈</span>
                <span style="font-weight:700; font-size:15px; color:#F87171;">Harapan Gains</span>
                <span style="font-size:12px; color:#6B7280;">(BN/PN → Harapan)</span>
            </div>
            """, unsafe_allow_html=True)
            if harapan_gains.empty:
                st.markdown("""
                <div style="color:#6B7280; font-size:13px; padding:12px 0;">
                    No seats predicted to flip toward Harapan.
                </div>
                """, unsafe_allow_html=True)
            for _, row in harapan_gains.sort_values(prob_col, ascending=False).iterrows():
                st.markdown(f"""
                <div style="background:#052E16; border-left:3px solid #4ADE80;
                            border-radius:6px; padding:10px 14px; margin-bottom:8px;
                            display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:600; color:#FAFAFA; font-size:14px;">{row[seat_col]}</span>
                    <span style="font-weight:700; color:#4ADE80; font-size:14px;">P={row[prob_col]:.2f}</span>
                </div>
                """, unsafe_allow_html=True)

        with fcol2:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="font-size:18px;">📊</span>
                <span style="font-weight:700; font-size:15px; color:#93C5FD;">BN/PN Gains</span>
                <span style="font-size:12px; color:#6B7280;">(Harapan → BN/PN)</span>
            </div>
            """, unsafe_allow_html=True)
            if harapan_losses.empty:
                st.markdown("""
                <div style="color:#6B7280; font-size:13px; padding:12px 0;">
                    No seats predicted to flip toward BN/PN.
                </div>
                """, unsafe_allow_html=True)
            for _, row in harapan_losses.sort_values(prob_col).iterrows():
                st.markdown(f"""
                <div style="background:#1E293B; border-left:3px solid #93C5FD;
                            border-radius:6px; padding:10px 14px; margin-bottom:8px;
                            display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:600; color:#FAFAFA; font-size:14px;">{row[seat_col]}</span>
                    <span style="font-weight:700; color:#93C5FD; font-size:14px;">P={row[prob_col]:.2f}</span>
                </div>
                """, unsafe_allow_html=True)

    st.divider()


    if 'chinese_pct' in df.columns:
            st.subheader("Behind the Prediction")
            st.caption("Filled circles = model prediction. Hollow diamonds = "
                      "actual most recent election result. Large gaps between "
                      "them flag seats where the model disagrees with real history.")

            df_plot = df.copy()
            df_plot['ethnic_margin'] = df_plot['chinese_pct'] - df_plot['malay_pct']
            df_plot = df_plot.sort_values(prob_col).reset_index(drop=True)

            # Determine the actual-history column per state
            actual_col = None
            if 'ph_vote_share_2023' in df_plot.columns:
                actual_col = 'ph_vote_share_2023'
            elif 'ph_vote_share_2021' in df_plot.columns:
                actual_col = 'ph_vote_share_2021'

            fig = go.Figure()

            # Model prediction markers
            fig.add_trace(go.Scatter(
                x=df_plot['ethnic_margin'], y=df_plot[prob_col],
                mode='markers', name='Model Prediction',
                marker=dict(
                    size=13,
                    color=df_plot[prob_col],
                    colorscale=[[0, '#2563EB'], [0.5, '#A78BFA'], [1, '#DC2626']],
                    line=dict(width=1, color='rgba(255,255,255,0.3)'),
                    showscale=True,
                    colorbar=dict(title="Prediction", tickformat='.1f'),
                ),
                text=df_plot[seat_col],
                hovertemplate="<b>%{text}</b><br>Prediction: %{y:.2f}<br>Ethnic margin: %{x:.2f}<extra></extra>",
            ))

            # Actual historical result markers (if available)
            if actual_col:
                fig.add_trace(go.Scatter(
                    x=df_plot['ethnic_margin'], y=df_plot[actual_col] / 100,
                    mode='markers', name='Actual Last Election',
                    marker=dict(
                        size=9, color='rgba(255,255,255,0.9)',
                        symbol='diamond-open', line=dict(width=2, color='white'),
                    ),
                    text=df_plot[seat_col],
                    hovertemplate="<b>%{text}</b><br>Actual result: %{y:.2f}<extra></extra>",
                ))

                # Connect prediction to actual with a line for large gaps
                for _, row in df_plot.iterrows():
                    gap = abs(row[prob_col] - row[actual_col] / 100)
                    if gap > 0.25:  # flag significant disagreements
                        fig.add_trace(go.Scatter(
                            x=[row['ethnic_margin'], row['ethnic_margin']],
                            y=[row[prob_col], row[actual_col] / 100],
                            mode='lines',
                            line=dict(color='rgba(251, 191, 36, 0.6)', width=2, dash='dot'),
                            showlegend=False,
                            hoverinfo='skip',
                        ))

            fig.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                         annotation_text="Balanced seat", annotation_position="top")
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                         annotation_text="50/50", annotation_position="right")
            fig.add_annotation(x=0.02, y=0.05, xref="paper", yref="paper",
                              text="↓ BN+PN Pact favoured", showarrow=False,
                              font=dict(color="#93C5FD", size=11))
            fig.add_annotation(x=0.02, y=0.95, xref="paper", yref="paper",
                              text="↑ Harapan favoured", showarrow=False,
                              font=dict(color="#FCA5A5", size=11))
            fig.update_layout(
                height=480,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Ethnic Margin (Chinese % − Malay %)",
                yaxis_title="Probability (0 = BN+PN Pact, 1 = Harapan)",
                legend=dict(orientation='h', yanchor='bottom', y=1.08, x=0.3),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

            if actual_col:
                large_gaps = df_plot[
                    abs(df_plot[prob_col] - df_plot[actual_col] / 100) > 0.25
                ]
                if len(large_gaps) > 0:
                    st.warning(f"""
                    **⚠️ {len(large_gaps)} seat(s) show a large gap (>25pp) between
                    model prediction and actual last-election result** — dotted
                    yellow lines connect these. This typically happens for seats
                    affected by 2018 redelineation, where pre-2018 historical
                    data is unavailable, weakening the loyalty correction.
                    """)
            else:
                st.info("""
                **Pattern:** Seats near x=0 (genuinely mixed, no clear ethnic
                plurality) show the WEAKEST calibration. Ethnic composition
                is a strong signal but not universal — some Malay-majority
                seats are historically strong Harapan (PKR/AMANAH) territory,
                which the model may underweight without recent-history data.
                """)

st.divider()
st.info("📊 See the full **Methodology & Key Findings** page in the sidebar for detailed analysis.")
