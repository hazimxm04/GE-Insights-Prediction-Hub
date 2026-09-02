"""
3_Methodology.py
=================
Restructured: one headline, three key findings up front,
architecture/validation/limitations collapsed into expanders
for readers who want depth without forcing everyone to scroll past it.
"""

import streamlit as st

st.set_page_config(page_title="Methodology & Key Findings", layout="wide")

st.markdown("""
<style>
.finding-card {
    background:#1A1D24; border:1px solid rgba(255,255,255,0.08);
    border-radius:12px; padding:20px; height:100%;
}
.headline-card {
    background: linear-gradient(135deg, #1a0505 0%, #3d0d0d 50%, #1a0505 100%);
    border: 1px solid rgba(220, 38, 38, 0.2);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
}
.headline-number {
    font-size: 48px;
    font-weight: 900;
    color: #FAFAFA;
    line-height: 1;
}
.headline-label {
    font-size: 14px;
    color: #9CA3AF;
    margin-top: 8px;
}
.limitation-card {
    background:#1E293B; border-left:3px solid #93C5FD;
    border-radius:8px; padding:16px 20px; margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HEADLINE — single strongest fact, no scrolling needed to see it
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="headline-card">
    <div class="headline-number">94.64% / 91.67%</div>
    <div class="headline-label">
        Validated against actual 2026 Malaysian state election results —
        not a holdout set, the real outcome.
    </div>
</div>
""", unsafe_allow_html=True)

st.title("Methodology & Key Findings")
st.caption("How this system was built, validated, and where it's honestly still limited.")

st.divider()

# ══════════════════════════════════════════════════════════════════
# THREE FINDINGS — consolidated from 6, each with a real number
# ══════════════════════════════════════════════════════════════════

st.subheader("What Actually Moved The Needle")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="finding-card">
        <div style="color:#4ADE80; font-weight:700; font-size:16px; margin-bottom:10px;">
             Redefining Label Beats Feature Engineering
        </div>
        <div style="color:#FAFAFA; font-size:14px; font-weight:700; margin-bottom:8px;">
            One reframe beat every feature we added.
        </div>
        <div style="color:#D1D5DB; font-size:13px; line-height:1.5;">
            Redefining the target from "BN vs everyone" to "government
            bloc (BN+PN) vs Harapan" — matching the real 2026 political
            alignment — added 28 points on its own.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="finding-card">
        <div style="color:#F87171; font-weight:700; font-size:16px; margin-bottom:10px;">
            Cross-State Transfer Failed
        </div>
        <div style="color:#FAFAFA; font-size:14px; font-weight:700; margin-bottom:8px;">
            Johor's model will not work on other states.
        </div>
        <div style="color:#D1D5DB; font-size:13px; line-height:1.5;">
            Selangor's Malay voters (Pakatan Harapan's heartland since 2008)
            behave oppositely to Johor's. Required native, state-specific
            models — and a backtest confirming when the simpler original
            model (94.64%) beats the "improved" one (91.23% on the
            same test).
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="finding-card">
        <div style="color:#FBBF24; font-weight:700; font-size:16px; margin-bottom:10px;">
            💰 Stock Prediction Is Nearly Impossible (Expected)
        </div>
        <div style="color:#FAFAFA; font-size:14px; font-weight:700; margin-bottom:8px;">
            Validated against real world economic theory
        </div>
        <div style="color:#D1D5DB; font-size:13px; line-height:1.5;">
            An LSTM was built to forecast future KLCI (stock index) and
            USD/MYR values, but it only predicted the correct direction
            52% of the time — barely better than a coin flip. This
            confirms the Efficient Market Hypothesis: prices already
            reflect known information, so daily movement is genuinely
            unpredictable. The forecast still feeds the election model
            as one feature, weighted low to match its actual signal strength.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# EVERYTHING ELSE — collapsed, available on demand
# ══════════════════════════════════════════════════════════════════

with st.expander("📥 Data Pipeline"):
    st.markdown("""
    **Historical elections:** [electiondata.my](https://electiondata.my) (ballots + stats parquet, from 2008-2026)

    **Voter demographics:** GE-15 voter roll (3–5M rows), aggregated to
    seat-level ethnicity/age composition

    **Sentiment:** FMT + Malay Mail RSS scraped daily, scored via Groq/Llama
    (9 dimensions: BN/Harapan/PN sentiment, racial tension, per article)

    **Economic:** yfinance (KLCI, USD/MYR) + BNM API, forecasted via PyTorch LSTM

    **Boundaries:** electiondata.my DUN delimitation GeoJSON (445 Peninsular seats)
    """)

with st.expander("🧠 Model Architecture — Validated States (Johor, Negeri Sembilan)"):
    st.markdown("""
    Single-transition training (most recent completed election pair) →
    25 engineered features (structural, sentiment, demographic, interaction
    terms) → Random Forest + XGBoost ensemble → isotonic calibration →
    EllipticEnvelope OOD detection with base-rate fallback blending.

    **Result:** 94.64% (Johor) / 91.67% (NS) accuracy against actual
    2026 results.
    """)

with st.expander("🧠 Model Architecture — Forecast States (Melaka, Selangor, Perak)"):
    st.markdown("""
    Multi-transition training (2013→2018 + 2018→recent, combined) with
    stronger regularization → native RF+XGB ensemble per state → blended
    50/50 with a weighted-recency historical loyalty score:

    ```
    loyalty_score = weighted_avg_share + trend_adjustment − blue_wave_pressure
    ```

    Recency weights: 2008=10%, 2013=20%, 2018=30%, most recent=40%.

    Seats matched across election years by **number** (e.g. "N.21"), not
    name, to survive redelineation-driven name changes — N.21 "Chempaka"
    (2008/2013) became "Pandan Indah" (2018+); matching by name alone
    silently dropped its historical loyalty and understated a genuinely
    safe seat.
    """)

with st.expander("✅ Validation Strategy"):
    st.markdown("""
    **Direct validation (Johor, NS):** trained on year N, predicted year
    N+1, compared against the real, already-occurred result.

    **Indirect validation (forecast states):** no election has happened
    yet to check against. Instead, applied the SAME forecast methodology
    to Johor's known 2026 result as a backtest — 91.23% accuracy confirmed
    the general approach is sound, even though it underperforms Johor's
    own tuned model.

    **Overconfidence diagnostic:** checked every validated prediction for
    cases where the model was >90% confident but wrong. Johor: 0 such
    cases. NS: 2 (both genuinely mixed-demographic seats with no clear
    ethnic plurality).
    """)

with st.expander("🤖 RAG Chatbot + Function Calling"):
    st.markdown("""
    ChromaDB vector store (325+ documents: predictions, sentiment
    articles, state summaries) with semantic retrieval for factual
    questions.

    For "what if" scenario questions, routes to Groq tool-calling: the
    LLM extracts seat + hypothetical change, calls the live prediction
    model with the seat's real baseline features overridden, and
    explains the result in natural language.
    """)

with st.expander("⚠️ Known Limitations"):
    limitations = [
        "**Blend weights are hand-selected, not empirically tuned.** The "
        "50/50 split between native model and historical loyalty (and the "
        "blue-wave pressure multiplier) were chosen based on observed "
        "behavior, not a grid search or cross-validated fit.",

        "**Coalition target treats BN and PN as one \"non-Harapan\" bloc**, "
        "regardless of whether they contested the same seat together "
        "(Selangor 2023-style) or independently (Melaka/Perak 3-way "
        "contests). This simplification doesn't distinguish how \"hard\" "
        "a win against 1 vs. 2 opponents actually was.",

        "**Forecast states are genuinely unvalidated.** Melaka, Selangor, "
        "and Perak's next elections haven't happened — these are "
        "forecasts, not validated predictions, however sound the "
        "backtested methodology appears.",

        "**No formal train/validation split for forecast-state models.** "
        "They train on all available historical transitions; the Johor "
        "backtest substitutes for held-out validation but isn't the same "
        "as a true validation set.",

        "**Small training samples** (28–60 seats per state) mean "
        "individual hyperparameter choices (tree depth, regularization) "
        "can meaningfully shift results — chosen based on diagnosed "
        "overconfidence symptoms, not systematic tuning.",
    ]
    for lim in limitations:
        st.markdown(f"""<div class="limitation-card">{lim}</div>""",
                    unsafe_allow_html=True)

st.divider()
st.caption("""
⚠️ **Personal portfolio project** demonstrating data science methodology —
not an official or authoritative election prediction service.
""")