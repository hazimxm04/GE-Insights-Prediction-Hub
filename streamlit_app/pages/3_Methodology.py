"""
4_Methodology.py
=================
Full methodology, architecture, and key findings page.
"""

import streamlit as st

st.set_page_config(page_title="Methodology & Findings", page_icon="🔑", layout="wide")

st.markdown("""
<style>
.finding-card {
    background:#1A1D24; border:1px solid rgba(255,255,255,0.08);
    border-radius:12px; padding:20px; height:100%;
}
.method-card {
    background:#1A1D24; border:1px solid rgba(255,255,255,0.08);
    border-radius:12px; padding:20px; margin-bottom:16px;
}
.method-card h4 { color:#F87171; margin-bottom:8px; font-size:15px; }
.limitation-card {
    background:#1E293B; border-left:3px solid #93C5FD;
    border-radius:8px; padding:16px 20px; margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🔑 Methodology & Key Findings")
st.caption("How this system was built, validated, and where it's honestly still limited.")

# ══════════════════════════════════════════════════════════════════
# KEY FINDINGS
# ══════════════════════════════════════════════════════════════════

st.subheader("Key Findings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="finding-card">
        <div style="color:#4ADE80; font-weight:700; font-size:15px; margin-bottom:6px;">
            📈 Coalition Reframing: 63.89% → 91.67%
        </div>
        <div style="color:#FAFAFA; font-size:13px; font-weight:700; margin-bottom:6px;">
            TL;DR: Redefining "opposition" mattered more than any single feature.
        </div>
        <div style="color:#D1D5DB; font-size:13px; line-height:1.5;">
            NS accuracy jumped 28 points after reframing the target
            from "BN vs everyone" to "government bloc (BN+PN) vs
            Harapan opposition" — matching the real 2026 political
            alignment rather than historical rivalry.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("""
    <div class="finding-card">
        <div style="color:#93C5FD; font-weight:700; font-size:15px; margin-bottom:6px;">
            🗳️ Seat Demographics Drive Predictions
        </div>
        <div style="color:#FAFAFA; font-size:13px; font-weight:700; margin-bottom:6px;">
            TL;DR: Who lives in a seat predicts the winner better than sentiment or economy.
        </div>
        <div style="color:#D1D5DB; font-size:13px; line-height:1.5;">
            Seat-level ethnic composition (from 3-5M row voter roll
            data) became the model's strongest predictor — reflecting
            Malaysia's well-documented coalition voting patterns more
            strongly than sentiment, economic indicators, or historical
            margin alone.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("""
    <div class="finding-card">
        <div style="color:#F87171; font-weight:700; font-size:15px; margin-bottom:6px;">
            🧭 One Model Doesn't Fit All States
        </div>
        <div style="color:#FAFAFA; font-size:13px; font-weight:700; margin-bottom:6px;">
            TL;DR: Johor's model failed on Selangor — different states, different politics.
        </div>
        <div style="color:#D1D5DB; font-size:13px; line-height:1.5;">
            Applying Johor's model directly to Selangor failed —
            Selangor's Malay voters (PKR/AMANAH heartland since 2008)
            behave oppositely to Johor's (BN heartland). Required
            building native, state-specific models blended with
            historical loyalty scoring.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="finding-card">
        <div style="color:#FBBF24; font-weight:700; font-size:15px; margin-bottom:6px;">
            📊 Markets Are Efficient — Even in Malaysia
        </div>
        <div style="color:#FAFAFA; font-size:13px; font-weight:700; margin-bottom:6px;">
            TL;DR: Predicting stock prices daily is nearly impossible — and that's expected.
        </div>
        <div style="color:#D1D5DB; font-size:13px; line-height:1.5;">
            LSTM economic forecasting achieved only 52% directional
            accuracy on KLCI/MYR — barely better than a coin flip.
            This empirically confirms the Efficient Market Hypothesis
            rather than suggesting a modeling failure.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("""
    <div class="finding-card">
        <div style="color:#A78BFA; font-weight:700; font-size:15px; margin-bottom:6px;">
            🔄 Simpler Beats Complex — When Data Is Already Good
        </div>
        <div style="color:#FAFAFA; font-size:13px; font-weight:700; margin-bottom:6px;">
            TL;DR: Backtesting the "improved" method on Johor made it worse, not better.
        </div>
        <div style="color:#D1D5DB; font-size:13px; line-height:1.5;">
            Applying the weighted-recency + loyalty-blend method
            (built for Selangor/Melaka/Perak) to Johor's known 2026
            result scored 91.23% — 3.5 points below Johor's own
            single-transition model (94.64%). Extra complexity only
            helps when the simpler model was already unreliable.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("""
    <div class="finding-card">
        <div style="color:#4ADE80; font-weight:700; font-size:15px; margin-bottom:6px;">
            🔍 Seat Identity Bugs Are Real, Not Rare
        </div>
        <div style="color:#FAFAFA; font-size:13px; font-weight:700; margin-bottom:6px;">
            TL;DR: Seat names change between elections — matching by number, not name, fixed it.
        </div>
        <div style="color:#D1D5DB; font-size:13px; line-height:1.5;">
            N.21 "Chempaka" (2008/2013) became "Pandan Indah" (2018+)
            under redelineation — matching by full seat name silently
            dropped its historical loyalty, understating a genuinely
            safe seat. Fixed by matching on seat NUMBER instead.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

st.subheader("System Architecture")

st.markdown("""
<div class="method-card">
    <h4>📥 Data Pipeline</h4>
    <div style="color:#D1D5DB; font-size:13px; line-height:1.6;">
        <b>Historical elections:</b> electiondata.my (ballots + stats parquet, 2008–2026)<br>
        <b>Voter demographics:</b> GE-15 voter roll (3–5M rows), aggregated to seat-level
        ethnicity/age composition<br>
        <b>Sentiment:</b> FMT + Malay Mail RSS scraped daily, scored via Groq/Llama
        (9 dimensions: BN/Harapan/PN sentiment, racial tension, per article)<br>
        <b>Economic:</b> yfinance (KLCI, USD/MYR) + BNM API, forecasted via PyTorch LSTM<br>
        <b>Boundaries:</b> electiondata.my DUN delimitation GeoJSON (445 Peninsular seats)
    </div>
</div>

<div class="method-card">
    <h4>🧠 Model Architecture — Validated States (Johor, Negeri Sembilan)</h4>
    <div style="color:#D1D5DB; font-size:13px; line-height:1.6;">
        Single-transition training (most recent completed election pair) →
        25 engineered features (structural, sentiment, demographic, interaction terms) →
        Random Forest + XGBoost ensemble → isotonic calibration →
        EllipticEnvelope OOD detection with base-rate fallback blending.<br><br>
        <b>Result:</b> 94.64% (Johor) / 91.67% (NS) accuracy against actual 2026 results.
    </div>
</div>

<div class="method-card">
    <h4>🧠 Model Architecture — Forecast States (Melaka, Selangor, Perak)</h4>
    <div style="color:#D1D5DB; font-size:13px; line-height:1.6;">
        Multi-transition training (2013→2018 + 2018→recent, combined) with stronger
        regularization → native RF+XGB ensemble per state → blended 50/50 with a
        weighted-recency historical loyalty score:<br><br>
        <code>loyalty_score = weighted_avg_share + trend_adjustment − blue_wave_pressure</code><br>
        Recency weights: 2008=10%, 2013=20%, 2018=30%, most recent=40%.<br>
        Seats matched across election years by <b>number</b> (e.g. "N.21"), not name,
        to survive redelineation-driven name changes.
    </div>
</div>

<div class="method-card">
    <h4>✅ Validation Strategy</h4>
    <div style="color:#D1D5DB; font-size:13px; line-height:1.6;">
        <b>Direct validation (Johor, NS):</b> trained on year N, predicted year N+1,
        compared against the real, already-occurred result.<br>
        <b>Indirect validation (forecast states):</b> no election has happened yet to
        check against. Instead, applied the SAME forecast methodology to Johor's
        known 2026 result as a backtest — 91.23% accuracy confirmed the general
        approach is sound, even though it underperforms Johor's own tuned model.<br>
        <b>Overconfidence diagnostic:</b> checked every validated prediction for
        cases where the model was >90% confident but wrong. Johor: 0 such cases.
        NS: 2 (both genuinely mixed-demographic seats with no clear ethnic plurality).
    </div>
</div>

<div class="method-card">
    <h4>🤖 RAG Chatbot + Function Calling</h4>
    <div style="color:#D1D5DB; font-size:13px; line-height:1.6;">
        ChromaDB vector store (325+ documents: predictions, sentiment articles,
        state summaries) with semantic retrieval for factual questions.<br>
        For "what if" scenario questions, routes to Groq tool-calling: the LLM
        extracts seat + hypothetical change, calls the live prediction model with
        the seat's real baseline features overridden, and explains the result
        in natural language.
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# KNOWN LIMITATIONS
# ══════════════════════════════════════════════════════════════════

st.subheader("Known Limitations")

limitations = [
    "**Blend weights are hand-selected, not empirically tuned.** The 50/50 split "
    "between native model and historical loyalty (and the blue-wave pressure "
    "multiplier) were chosen based on observed behavior, not a grid search or "
    "cross-validated fit.",

    "**Coalition target treats BN and PN as one \"non-Harapan\" bloc**, regardless "
    "of whether they contested the same seat together (Selangor 2023-style) or "
    "independently (Melaka/Perak 3-way contests). This simplification doesn't "
    "distinguish how \"hard\" a win against 1 vs. 2 opponents actually was.",

    "**Forecast states are genuinely unvalidated.** Melaka, Selangor, and Perak's "
    "next elections haven't happened — these are forecasts, not validated "
    "predictions, however sound the backtested methodology appears.",

    "**No formal train/validation split for forecast-state models.** They train "
    "on all available historical transitions; the Johor backtest substitutes for "
    "held-out validation but isn't the same as a true validation set.",

    "**Small training samples** (28–60 seats per state) mean individual "
    "hyperparameter choices (tree depth, regularization) can meaningfully shift "
    "results — chosen based on diagnosed overconfidence symptoms, not "
    "systematic tuning.",
]

for lim in limitations:
    st.markdown(f"""<div class="limitation-card">{lim}</div>""", unsafe_allow_html=True)

st.divider()
st.caption("""
⚠️ **Personal portfolio project** demonstrating data science methodology —
not an official or authoritative election prediction service.
""")