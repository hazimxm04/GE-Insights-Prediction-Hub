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
    <div class="headline-number">85.71% / 86.11%</div>
    <div style="font-size:13px; color:#F87171; text-transform:uppercase; letter-spacing:1px; margin-top:4px; font-weight:600;">
        Accuracy Rate
    </div>
    <div class="headline-label" style="margin-top:8px;">
        Johor / Negeri Sembilan, checked against the actual 2026 results
        with leakage-free features. Simple rules score 48&ndash;50 of 56 (Johor)
        and 25&ndash;30 of 36 (NS), so read the accuracy next to those baselines.
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
             Defining The Target To Match Reality
        </div>
        <div style="color:#FAFAFA; font-size:14px; font-weight:700; margin-bottom:8px;">
            The target is the government bloc against Harapan.
        </div>
        <div style="color:#D1D5DB; font-size:13px; line-height:1.5;">
            Redefining the target from "BN vs everyone" to "government
            bloc (BN+PN) vs Harapan" — matching the real 2026 political
            alignment. It is also a known simplification (see Known Limitations).
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
            models.
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
            unpredictable.
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

    **Sentiment:** FMT + Malay Mail RSS scraped daily, scored via Groq
    (9 dimensions: BN/Harapan/PN sentiment, racial tension, per article)

    **Economic:** yfinance (KLCI, USD/MYR) + BNM API, forecasted via PyTorch LSTM

    **Boundaries:** electiondata.my DUN delimitation GeoJSON (445 Peninsular seats)
    """)

with st.expander("🧠 Model Architecture — Validated States (Johor, Negeri Sembilan)"):
    st.markdown("""
    Features come from one shared function (`backend/core/features.py`,
    version `a_only_v1`): structural and ethnicity features built only
    from information known before the election, so nothing from the
    result leaks in. Model: Random Forest + XGBoost, isotonic
    calibration, then outlier detection with a historical base-rate
    blend for seats unlike the training data.

    **Result:** 85.71% (Johor, 48/56) and 86.11% (NS, 31/36) against the
    actual 2026 results. The simple rules "never opposition" and "same
    as last time" score 48 and 50 of 56 in Johor, and 25 and 30 of 36 in
    NS, so the model clearly beats them in NS but only ties or trails
    them on accuracy in Johor.
    """)

with st.expander("🧠 Model Architecture — Forecast States (Melaka, Selangor, Perak)"):
    st.markdown("""
    The same pipeline and the same feature code are used for each
    state, trained on its own history, and the forecast is made for
    the next election.

    Seats are matched across election years by **number** (e.g. "N.21"),
    not name, to survive redelineation-driven name changes.

    **Frozen forecasts:** on 30 Oct 2026 the forecasts for Selangor (56
    seats), Melaka (28) and Perak (59) were saved with checksums and
    tagged `forecasts-2026-10-30` on GitHub, before any of these
    elections. They will be scored against the real results when they
    happen.
    """)


with st.expander("✅ Validation Strategy"):
    st.markdown("""
    **Direct validation (Johor, NS):** trained on earlier elections,
    predicted the 2026 result, compared against the real outcome,
    always against the simple baselines.

    **Forecast states:** no result exists yet. Instead the forecasts
    were frozen in advance (see above), so the later score is believable.
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
        "**Johor does not clearly beat simple rules on accuracy.** "
        "48/56 matches the \"never opposition\" rule and trails \"same as "
        "last time\" (50/56). The model's value there is ranking seats "
        "(AUC), not raw accuracy.",

        "**Confidence labels are not calibrated.** The HIGH/MEDIUM/LOW "
        "cut-offs do not fit the scale of the outlier scores, so every "
        "seat currently shows LOW confidence.",

        "**Coalition target treats BN and PN as one \"non-Harapan\" bloc**, "
        "regardless of whether they contested the same seat together "
        "(Selangor 2023-style) or independently (Melaka/Perak 3-way "
        "contests). This simplification doesn't distinguish how \"hard\" "
        "a win against 1 vs. 2 opponents actually was.",

        "**Forecast states are genuinely unvalidated.** Melaka, Selangor, "
        "and Perak's next elections haven't happened — these are "
        "forecasts, not validated predictions, however sound the "
        "backtested methodology appears.",

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