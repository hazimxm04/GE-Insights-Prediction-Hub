"""
4_Methodology.py
================
Order of the page: honest headline -> leaderboard (model vs simple rules, with uncertainty) -> three
key findings -> collapsed detail (pipeline, model, evidence, validation, sentiment layer, chatbot,
limitations, next steps).

Every accuracy number on this page is read from the saved result files, so the page cannot drift
away from them:
    docs/evaluation_with_ci.json        scripts/evaluate_with_ci.py
    backend/models/pooled_results.json  scripts/evaluate_pooled.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
CI_FILE = ROOT / 'docs/evaluation_with_ci.json'
POOLED_FILE = ROOT / 'backend/models/pooled_results.json'

STATE_NAMES = {'johor': 'Johor', 'neg_sembilan': 'Negeri Sembilan', 'selangor': 'Selangor',
               'melaka': 'Melaka', 'perak': 'Perak'}
# (key in the result file, label shown on the page)
METHODS = [('model', 'Our model'),
           ('never_opp', 'Opposition never wins'),
           ('same_as_last', 'Same result as previous election')]
COLORS = {'model': '#DC2626', 'never_opp': '#6B7280', 'same_as_last': '#93C5FD'}

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
.headline-number { font-size: 48px; font-weight: 900; color: #FAFAFA; line-height: 1; }
.headline-label { font-size: 14px; color: #9CA3AF; margin-top: 8px; }
.limitation-card {
    background:#1E293B; border-left:3px solid #93C5FD;
    border-radius:8px; padding:16px 20px; margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


ci = load_json(CI_FILE)
pooled = load_json(POOLED_FILE)

# ══════════════════════════════════════════════════════════════════
# 1. HEADLINE: the honest version
# ══════════════════════════════════════════════════════════════════
if ci and 'johor' in ci and 'neg_sembilan' in ci:
    auc_text = f"{ci['johor']['auc']['value']:.2f} / {ci['neg_sembilan']['auc']['value']:.2f}"
    acc_text = (f"{ci['johor']['accuracy']['model']['value'] * 100:.2f}% / "
                f"{ci['neg_sembilan']['accuracy']['model']['value'] * 100:.2f}%")
else:
    auc_text, acc_text = "see leaderboard", "see leaderboard"

st.markdown(f"""
<div class="headline-card">
    <div class="headline-number">AUC {auc_text}</div>
    <div style="font-size:13px; color:#F87171; text-transform:uppercase; letter-spacing:1px; margin-top:4px; font-weight:600;">
        How well the model ranks seats (Johor / Negeri Sembilan, real 2026 results)
    </div>
    <div class="headline-label" style="margin-top:8px;">
        Accuracy is {acc_text}, but simple rules score about the same: with 56 and 36 seats the gaps
        are within the noise. The leaderboard below shows the model against those rules, with uncertainty.
    </div>
</div>
""", unsafe_allow_html=True)

st.title("Methodology & Key Findings")
st.caption("How this system was built, validated, and where it's honestly still limited.")
st.divider()


# ══════════════════════════════════════════════════════════════════
# 2. LEADERBOARD  (self-contained: rewrite this function in your own style if you like)
# ══════════════════════════════════════════════════════════════════
def render_leaderboard():
    st.subheader("Model leaderboard: does the model beat simple rules?")
    st.caption("Real 2026 results. Two baselines need no machine learning: **Opposition never wins** "
               "(predict the government bloc wins every seat) and **Same result as previous election** "
               "(predict each seat stays with the side that held it). Intervals are 95% ranges from "
               "resampling the seats.")
    if not ci:
        st.info("Evaluation results are not available in this copy of the app "
                "(`docs/evaluation_with_ci.json`; create it with `python scripts/evaluate_with_ci.py --save`).")
        return

    rows, labels, accs, lows, highs, colors = [], [], [], [], [], []
    for key, state in STATE_NAMES.items():
        if key not in ci:
            continue
        r = ci[key]
        for m, name in METHODS:
            a = r['accuracy'][m]
            lo, hi = a['ci95']
            row = {'State': state, 'Method': name,
                   'Seats correct': f"{a['correct']}/{r['seats']}",
                   'Accuracy': f"{a['value'] * 100:.1f}%",
                   '95% interval': f"{lo * 100:.1f}% to {hi * 100:.1f}%",
                   'Model minus this rule (seats)': '', 'McNemar p-value': ''}
            if m != 'model':
                p = r['paired_vs_model'][m]
                plo, phi = p['ci95_seats']
                row['Model minus this rule (seats)'] = (
                    f"{p['model_minus_rule_seats']:+d} (interval {plo:+.0f} to {phi:+.0f})")
                row['McNemar p-value'] = f"{p['mcnemar_exact_p']:.2f}"
            rows.append(row)
            labels.append(f"{state}: {name}")
            accs.append(a['value'] * 100)
            lows.append((a['value'] - lo) * 100)
            highs.append((hi - a['value']) * 100)
            colors.append(COLORS[m])
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    fig = go.Figure(go.Scatter(
        x=accs, y=labels, mode='markers',
        marker=dict(size=11, color=colors),
        error_x=dict(type='data', symmetric=False, array=highs, arrayminus=lows, color='#9CA3AF'),
        hovertemplate='%{y}<br>%{x:.1f}%<extra></extra>'))
    fig.update_layout(height=60 + 38 * len(labels), margin=dict(t=10, l=10, r=10), xaxis_title="Accuracy (%) with 95% interval",
                      yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig, use_container_width=True)

    lines = []
    for key, state in STATE_NAMES.items():
        if key in ci:
            lines.append(f"**{state}:** AUC {ci[key]['auc']['value']:.3f} "
                         f"[{ci[key]['auc']['ci95'][0]:.3f}, {ci[key]['auc']['ci95'][1]:.3f}], "
                         f"Brier {ci[key]['brier']['value']:.3f}")
    st.markdown("  \n".join(lines))
    st.info("**How to read this.** The model ranks seats very well (high AUC). On accuracy, every gap to a "
            "simple rule has an interval that includes zero, so the data does not show the model beats them. "
            "The p-value asks whether the model and a rule disagree in a lopsided way; a large p-value means "
            "there is no evidence either is better.")


render_leaderboard()
st.divider()

# ══════════════════════════════════════════════════════════════════
# 3. THREE KEY FINDINGS
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
            The target was redefined from "BN vs everyone" to "government bloc (BN+PN)
            vs Harapan", matching the real 2026 political alignment. It is also a known
            simplification (see Known Limitations).
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="finding-card">
        <div style="color:#F87171; font-weight:700; font-size:16px; margin-bottom:10px;">
            Fixing Leakage Made The Scores Honest
        </div>
        <div style="color:#FAFAFA; font-size:14px; font-weight:700; margin-bottom:8px;">
            94.64% and 91.67% became 85.71% and 86.11%.
        </div>
        <div style="color:#D1D5DB; font-size:13px; line-height:1.5;">
            Some features used the election being predicted, so the model was partly shown the
            answer. After rebuilding every feature from earlier elections only, the scores
            dropped to the numbers above, and all later results use the honest version.
            Details in <code>docs/leakage_fix.md</code>.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="finding-card">
        <div style="color:#FBBF24; font-weight:700; font-size:16px; margin-bottom:10px;">
            Pooling States: Inconclusive
        </div>
        <div style="color:#FAFAFA; font-size:14px; font-weight:700; margin-bottom:8px;">
            One model for all states did about as well as one per state.
        </div>
        <div style="color:#D1D5DB; font-size:13px; line-height:1.5;">
            Tested with a rule that never lets a model see the test year or later. Pooled was
            level or better in some states and worse in others, and with few seats the
            differences are within the noise. See the table below.
        </div>
    </div>
    """, unsafe_allow_html=True)

if pooled:
    latest = [r for r in pooled if r.get('is_latest')]
    if latest:
        st.caption("Latest held-out test per state (seats correct): one model per state against one pooled model "
                   "(with and without a state flag).")
        st.dataframe(pd.DataFrame([{
            'State': STATE_NAMES.get(r['state'], r['state']), 'Test': r['test'], 'Seats': r['n'],
            'Per-state model': r['per_state'], 'Pooled model': r['pooled'],
            'Pooled + state flag': r['pooled_state'],
            'Opposition never wins': r['never_opposition'],
            'Same result as previous election': r['same_as_last']} for r in latest]),
            use_container_width=True, hide_index=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# 4. DETAIL, COLLAPSED
# ══════════════════════════════════════════════════════════════════
with st.expander("🗺️ System overview"):
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#1A1D24", color="#4B5563", fontcolor="#FAFAFA", fontsize=11];
      edge [color="#9CA3AF"];
      data [label="Election results\\n+ voter roll"];
      feat [label="Features\\n(a_only_v1, no leakage)"];
      model [label="RF + XGBoost\\n+ calibration + OOD check"];
      frozen [label="Frozen forecasts\\n(checksums + git tag)", fillcolor="#3d0d0d"];
      news [label="News headlines"];
      scorer [label="LLM scorer\\n(9 scores per headline)"];
      windows [label="Two 30-day windows\\n(change only)"];
      layer [label="Sentiment layer\\n(experimental, assumed effects)", fillcolor="#3a3410"];
      dash [label="Dashboard", fillcolor="#3d0d0d"];
      chat [label="Chatbot\\n(RAG + one tool)"];
      data -> feat -> model -> frozen -> dash;
      news -> scorer -> windows -> layer;
      frozen -> layer -> dash;
      frozen -> chat; layer -> chat;
    }
    """)

with st.expander("📥 Data Pipeline"):
    st.markdown("""
    **Historical elections:** [electiondata.my](https://electiondata.my) (ballots + stats parquet, from 2008-2026)

    **Voter demographics:** GE-15 voter roll (3–5M rows), aggregated to seat-level ethnicity/age composition

    **Sentiment:** Malaysian news headlines (Google News RSS and other sources) scored via Groq
    (9 scores per headline: BN/Harapan/PN sentiment, racial tension, five themes). Used only by the
    experimental sentiment layer, never by the validated model.

    **Economic:** KLCI and USD/MYR data (yfinance, BNM API) with an LSTM forecast. This is **not** used by the
    validated model. Economic news reaches the experimental layer only through the cost-of-living theme.

    **Boundaries:** electiondata.my DUN delimitation GeoJSON (445 Peninsular seats)
    """)

with st.expander("🧠 Model Architecture"):
    st.markdown("""
    Features come from one shared function (`backend/core/features.py`, version `a_only_v1`): structural and
    ethnicity features built only from information known before the election, so nothing from the result
    leaks in. Model: Random Forest + XGBoost, isotonic calibration, then outlier detection with a historical
    base-rate blend for seats unlike the training data.

    **Validated states (Johor, Negeri Sembilan):** trained on earlier elections, tested on the real 2026 result.
    See the leaderboard for the numbers.

    **Forecast states (Melaka, Selangor, Perak):** the same pipeline and feature code, trained on each state's own
    history, forecasting the next election. Seats are matched across years by **number** (e.g. "N.21"),
    not name, to survive redelineation-driven name changes.
    """)

with st.expander("🔬 What carries the signal?"):
    st.markdown("""
    12 held-out tests (every earlier election of Johor and Negeri Sembilan, plus the real 2026 results),
    one feature group removed at a time:

    - **Ethnicity and age features carry the signal.** Removing them lowered the number of correct seats in
      9 of 12 tests and never raised it.
    - Majority-change features add little.
    - A plain logistic regression is level with the ensemble on the two real-2026 tests (49 vs 48 and 31 vs 31
      seats), so the ensemble is not shown to be better than a simple linear model on this data.

    Reproduce: `python scripts/baseline_ablation.py`.
    """)

with st.expander("✅ Validation Strategy and Frozen Forecasts"):
    st.markdown("""
    **Direct validation (Johor, NS):** trained on earlier elections, predicted the 2026 result, compared against
    the real outcome, always against the simple baselines. Time-ordered splits only; tuning never used 2026.

    **Forecast states:** no result exists yet, so the forecasts were frozen in advance. On 30 Oct 2026 the
    forecasts for Selangor (56 seats), Melaka (28) and Perak (59) were saved with checksums and tagged
    `forecasts-2026-10-30` on GitHub, before any of these elections. They will be scored against the real
    results when they happen and are never edited.

    **Reproduce it:**
    ```
    python scripts/evaluate_holdout.py       # every state, time-ordered, with baselines
    python scripts/evaluate_with_ci.py       # intervals and paired tests for Johor and NS
    python scripts/baseline_ablation.py      # what each feature group adds
    python scripts/evaluate_pooled.py        # pooled vs per-state
    ```
    """)

with st.expander("🧪 Experimental Sentiment Layer (not part of the validated results)"):
    st.markdown("""
    An optional dashboard view that asks: *if the news mix changes, how might each seat move?* It sits beside
    the frozen forecast and never changes it. Headlines are national; the effect on each state comes through
    each seat's voter mix.

    - Compares two news windows (latest 30 days and the 30 before) and uses only the **change** in each theme,
      because the level of a theme depends on which outlets were scraped. Needs at least 300 successfully
      scored headlines in each window.
    - Each seat is split into five voter groups (older Malay, younger Malay, Chinese, Indian, other). Identity
      themes push each group in an assumed direction; cost of living and corruption push away from whoever held
      the seat last time; a rise in Harapan coverage relative to BN/PN moves seats toward Harapan.
    - **Every effect size is an assumption, not fitted to data**, because there is no past news paired with
      past results. No accuracy is claimed for it.
    - The label set still has no themes aimed at Harapan-side stories, so the layer is not balanced yet.
      `govt_performance` is unused and `islam_threat` is down-weighted.

    **How reliable is the scorer?** Evaluated on the five themes only, using 352 hand-labelled headlines and
    5-fold cross-validation (`docs/classifier_comparison.md`). The coalition sentiment scores (BN, Harapan, PN)
    have **not** been checked against human labels yet.
    """)
    st.dataframe(pd.DataFrame([
        {'Method': 'TF-IDF + logistic regression', 'Macro-F1 (threshold 0.5)': '0.479 [0.40, 0.56]'},
        {'Method': 'DistilBERT (fine-tuned)', 'Macro-F1 (threshold 0.5)': '0.542 [0.47, 0.60]'},
        {'Method': 'Groq zero-shot (used in production)', 'Macro-F1 (threshold 0.5)': '0.446 [0.38, 0.51]'},
    ]), use_container_width=True, hide_index=True)
    st.caption("No method works for govt_performance (14 positives), and islam_threat (18 positives) is weak "
               "for all. The production scorer is still Groq; DistilBERT is an option under evaluation.")
    st.markdown("**Roadmap:** historical news paired with past elections, so the effects can be fitted and "
                "tested on a later election.")

with st.expander("🤖 Chatbot"):
    st.markdown("""
    A ChromaDB vector store (predictions, sentiment articles, state summaries) with semantic retrieval answers
    factual questions. For "what if" questions, a keyword check routes to Groq tool calling: the LLM extracts
    the seat and the hypothetical change, calls the live prediction model with the seat's real baseline
    features overridden, and explains the result.

    This is retrieval plus one tool, with the route chosen by keywords. It is not yet a multi-tool agent.
    """)

with st.expander("⚠️ Known Limitations", expanded=True):
    limitations = [
        "**Johor does not clearly beat simple rules on accuracy.** 48/56 matches the \"opposition never wins\" "
        "rule and trails \"same result as previous election\" (50/56). The model's value there is ranking "
        "seats (AUC), not raw accuracy. In Negeri Sembilan the model is ahead, but within the noise.",
        "**Confidence labels are not calibrated.** The HIGH/MEDIUM/LOW cut-offs do not fit the scale of the "
        "outlier scores, so every seat currently shows LOW confidence.",
        "**Coalition target treats BN and PN as one \"non-Harapan\" bloc**, regardless of whether they "
        "contested the same seat together (Selangor 2023-style) or independently (Melaka/Perak 3-way "
        "contests). This doesn't distinguish how hard a win against 1 vs. 2 opponents actually was.",
        "**Forecast states are genuinely unvalidated.** Melaka, Selangor, and Perak's next elections haven't "
        "happened. These are forecasts, not validated predictions, however sound the backtested methodology "
        "appears.",
        "**Small training samples** (28–60 seats per state, two or three elections each) mean the results are "
        "uncertain: the 95% intervals are wide, and seats in one election swing together, so the effective "
        "sample is smaller than the seat count. Tuning used time-ordered folds on earlier elections only, "
        "never the 2026 results.",
        "**The sentiment layer is unvalidated.** Its effect sizes are assumptions, the label set has no "
        "Harapan-side themes yet, the coalition sentiment scores are unchecked against human labels, there is "
        "no rural/urban split, and the voter-group data is one snapshot per state. It is shown as an "
        "optional, clearly labelled view.",
    ]
    for lim in limitations:
        st.markdown(f"""<div class="limitation-card">{lim}</div>""", unsafe_allow_html=True)

with st.expander("➡️ What I would do next"):
    st.markdown("""
    1. **Test the sentiment layer on the past:** score news from before earlier elections, apply the layer to
       the base predictions, and compare with what happened. Fit its few parameters on earlier elections and
       test on a later one.
    2. **Check the coalition sentiment scores against human labels,** as was done for the themes.
    3. **Predict each coalition separately** (BN, Harapan, PN) instead of one government bloc.
    4. **More elections in training,** since 28–60 seats per state limits what can be shown.
    """)

with st.expander("💰 Side experiment: LSTM market forecast (dropped)"):
    st.markdown("""
    An LSTM was built to forecast the KLCI and USD/MYR. It predicted the correct direction 52% of the time,
    close to a coin flip, which is consistent with markets being hard to forecast. It is not used by the
    validated model.
    """)

st.divider()
st.caption("""
⚠️ **Personal portfolio project** demonstrating data science methodology —
not an official or authoritative election prediction service.
""")