"""
sentiment_scorer.py
===================
Scores Malaysian political news articles using Groq API.

KEY INSIGHT: Political sentiment works at NATIONAL level.
  "Islam terancam" affects Malay seats everywhere,
  not just articles mentioning a specific state.
  "Kos sara hidup" affects Chinese/urban seats everywhere.

So we score ALL national articles for:
  1. Coalition sentiment (BN/Harapan/PN)
  2. Narrative themes (Islam threat, Malay unity,
     cost of living, corruption, govt performance)

Then apply themes to each seat weighted by demographics
from voter roll (malay_pct, chinese_pct, youth_pct).
This gives GENUINE seat-level variation from national news.

Groq free tier: 30 requests/minute, 14,400 requests/day.

Usage:
    python sentiment_scorer.py                   -> auto-picks latest file
    python sentiment_scorer.py melaka_historical -> scores that file
    python sentiment_scorer.py ns_historical     -> scores that file

Output:
  data/processed/scored_articles.csv
  data/processed/state_sentiment_scores.csv
  data/processed/national_narrative_scores.csv   <- NEW
  data/processed/<state>_weekly_trend.csv
"""

import os
import re
import sys
import time
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# ── Load environment ───────────────────────────────────────────────

env_paths = [
    Path("backend/.env"),
    Path(__file__).resolve().parents[2] / "backend" / ".env",
    Path(__file__).resolve().parents[3] / "backend" / ".env",
    Path(".env"),
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")
        break

# ── Configure Groq ─────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Get a free key at: https://console.groq.com/keys\n"
        "Then add to backend/.env:\nGROQ_API_KEY=gsk_your_key_here"
    )

client = Groq(api_key=GROQ_API_KEY)
MODEL  = "llama-3.1-8b-instant"

print(f"Groq ready - model: {MODEL}")
print(f"Free tier: 30 req/min, 14,400 req/day\n")

# ── Paths ──────────────────────────────────────────────────────────

DATA_RAW       = Path("data/raw/news")
DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

INDEPENDENT_LEANS = {
    'independent_left', 'centrist', 'establishment_right',
    'mainstream_malay', 'business_neutral',
}

STATE_COLS = {
    'johor':        'mentions_johor',
    'neg_sembilan': 'mentions_neg_sembilan',
    'melaka':       'mentions_melaka',
}

# ── Core scoring ────────────────────────────────────────────────────

def score_article_all(text: str) -> dict:
    """
    Score one article for:
      - Coalition sentiment (BN/Harapan/PN)
      - Racial tension
      - National narrative themes (NEW):
          Islam_threat:    "Islam/Melayu terancam" discourse
          Malay_unity:     "Perpaduan Melayu" / BN-PN unity narrative
          Cost_living:     Economic hardship / kos sara hidup
          Corruption:      Anti-rasuah / corruption discourse
          Govt_performance: Government delivery / performance

    WHY THEMES MATTER:
      "Islam terancam" affects ALL Malay seats even if the
      article never mentions a specific state. By scoring
      these themes, we can weight their effect per seat
      based on demographic composition (malay_pct, chinese_pct).
    """
    prompt = f"""Analyze this Malaysian political news text carefully.

Text: {text[:500]}

Rate coalition sentiment (-1.0 = very negative, 0.0 = neutral, +1.0 = very positive):
BN: [number]
Harapan: [number]
PN: [number]

Rate racial/religious tension (0.0 = none, 1.0 = very high):
Tension: [number]

Rate these national narrative themes (0.0 = not present, 1.0 = very strong):
Islam_threat: [number]     <- "Islam/Melayu terancam" narrative strength
Malay_unity: [number]      <- Malay unity / BN-PN coalition narrative
Cost_living: [number]      <- Cost of living / economic hardship narrative
Corruption: [number]       <- Corruption / rasuah narrative
Govt_performance: [number] <- Government performance / delivery narrative

Return ONLY these 9 lines with numbers, nothing else."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.2,
        )
        result = response.choices[0].message.content.strip()

        def extract(label: str, default: float = 0.0) -> float:
            match = re.search(
                rf'{label}:\s*(-?\d+\.?\d*)',
                result, re.IGNORECASE
            )
            return float(match.group(1)) if match else default

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        return {
            # Coalition sentiment
            'bn_sentiment':       clamp(extract('BN'),        -1.0, 1.0),
            'harapan_sentiment':  clamp(extract('Harapan'),   -1.0, 1.0),
            'pn_sentiment':       clamp(extract('PN'),         -1.0, 1.0),
            'racial_tension':     clamp(extract('Tension'),    0.0,  1.0),
            # National narrative themes (NEW)
            'islam_threat':       clamp(extract('Islam_threat'),    0.0, 1.0),
            'malay_unity':        clamp(extract('Malay_unity'),     0.0, 1.0),
            'cost_living':        clamp(extract('Cost_living'),     0.0, 1.0),
            'corruption':         clamp(extract('Corruption'),      0.0, 1.0),
            'govt_performance':   clamp(extract('Govt_performance'), 0.0, 1.0),
        }

    except Exception as e:
        print(f"  Scoring failed: {e}")
        return {
            'bn_sentiment': 0.0, 'harapan_sentiment': 0.0,
            'pn_sentiment': 0.0, 'racial_tension': 0.0,
            'islam_threat': 0.0, 'malay_unity': 0.0,
            'cost_living': 0.0, 'corruption': 0.0,
            'govt_performance': 0.0,
        }


# ── Score all articles ─────────────────────────────────────────────

def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Score all articles. Note: no longer filtered by state mention
    because national narratives affect all states."""
    results = []
    total   = len(df)

    print(f"Scoring {total} articles with Groq ({MODEL})...")
    print("(No state filter -- national narratives affect all states)\n")

    for i, row in df.iterrows():
        text = str(row.get('full_text', row.get('title', '')))
        print(f"  [{i+1}/{total}] {text[:55]}...")

        scores = score_article_all(text)
        time.sleep(2)

        print(f"    BN={scores['bn_sentiment']:+.2f} | "
              f"Harapan={scores['harapan_sentiment']:+.2f} | "
              f"PN={scores['pn_sentiment']:+.2f} | "
              f"Tension={scores['racial_tension']:.2f} | "
              f"Islam={scores['islam_threat']:.2f} | "
              f"Cost={scores['cost_living']:.2f}")

        results.append({
            'title':               row.get('title', ''),
            'source':              row.get('source', ''),
            'source_lean':         row.get('source_lean', 'unknown'),
            'published':           row.get('published', ''),
            'mentions_johor':      row.get('mentions_johor', False),
            'mentions_neg_sembilan': row.get('mentions_neg_sembilan', False),
            'mentions_melaka':     row.get('mentions_melaka', False),
            # Coalition sentiment
            'bn_sentiment':        scores['bn_sentiment'],
            'harapan_sentiment':   scores['harapan_sentiment'],
            'pn_sentiment':        scores['pn_sentiment'],
            'racial_tension':      scores['racial_tension'],
            # National narrative themes
            'islam_threat':        scores['islam_threat'],
            'malay_unity':         scores['malay_unity'],
            'cost_living':         scores['cost_living'],
            'corruption':          scores['corruption'],
            'govt_performance':    scores['govt_performance'],
            'text_snippet':        text[:200],
        })

    return pd.DataFrame(results)


# ── National narrative aggregation (all articles) ───────────────────

def compute_national_narratives(df_scored: pd.DataFrame) -> dict:
    """
    Average theme scores across ALL articles (no state filter).
    These are national-level signals that affect all states.

    Returns dict used to weight sentiment per seat via demographics.
    """
    narratives = {
        'islam_threat':     df_scored['islam_threat'].mean(),
        'malay_unity':      df_scored['malay_unity'].mean(),
        'cost_living':      df_scored['cost_living'].mean(),
        'corruption':       df_scored['corruption'].mean(),
        'govt_performance': df_scored['govt_performance'].mean(),
        'n_articles':       len(df_scored),
    }

    print(f"\nNational narrative scores ({len(df_scored)} articles):")
    print(f"  Islam threat:     {narratives['islam_threat']:.3f}")
    print(f"  Malay unity:      {narratives['malay_unity']:.3f}")
    print(f"  Cost of living:   {narratives['cost_living']:.3f}")
    print(f"  Corruption:       {narratives['corruption']:.3f}")
    print(f"  Govt performance: {narratives['govt_performance']:.3f}")

    return narratives


def apply_narratives_to_seat(narratives: dict,
                              malay_pct: float,
                              chinese_pct: float,
                              youth_pct: float) -> float:
    """
    Convert national narrative themes into seat-specific pressure score.
    Same article/theme, different impact per seat based on demographics.

    Negative = more pressure on incumbent (harder to win)
    Positive = less pressure on incumbent (easier to win)

    Logic:
      Islam_threat strong + Malay-majority seat
        -> Malays consolidate toward PN, hurts Harapan/BN incumbents
      Cost_living strong + Chinese/youth-heavy seat
        -> Anti-incumbent sentiment, hurts whoever is in power
      Malay_unity strong + Malay-majority seat
        -> BN-PN coalition benefits, helps government bloc
      Corruption strong (any seat)
        -> Hurts incumbent universally
    """
    # Islam threat -> pushes Malay seats toward PN
    # (bad for Harapan, ambiguous for BN depending on seat)
    islam_effect = narratives['islam_threat'] * malay_pct * -0.5

    # Malay unity narrative -> consolidates Malay votes for govt bloc
    # (good for BN-PN in Malay seats)
    unity_effect = narratives['malay_unity'] * malay_pct * 0.3

    # Cost of living -> hurts incumbents in Chinese + youth seats
    cost_effect = narratives['cost_living'] * (chinese_pct + youth_pct) * -0.4

    # Corruption -> hurts incumbents everywhere
    corruption_effect = narratives['corruption'] * -0.3

    # Govt performance -> helps incumbents everywhere
    performance_effect = narratives['govt_performance'] * 0.2

    narrative_pressure = (
        islam_effect +
        unity_effect +
        cost_effect +
        corruption_effect +
        performance_effect
    )

    return round(float(narrative_pressure), 4)


# ── State-level aggregation ─────────────────────────────────────────

def compute_state_scores(df_scored: pd.DataFrame,
                          narratives: dict) -> pd.DataFrame:
    """
    Aggregate per-article scores into per-state scores.
    Reports both independent-only and all-sources versions.
    Includes national narrative summary per state.
    """
    state_scores = []
    has_lean = 'source_lean' in df_scored.columns

    for state, col in STATE_COLS.items():
        if col not in df_scored.columns:
            print(f"Column '{col}' not found, skipping {state}")
            continue

        state_df = df_scored[df_scored[col] == True]

        if len(state_df) == 0:
            print(f"No articles mentioning {state} -- using national average")
            state_df = df_scored   # fall back to ALL articles

        bn_all      = state_df['bn_sentiment'].mean()
        harapan_all = state_df['harapan_sentiment'].mean()
        pn_all      = state_df['pn_sentiment'].mean()
        tension_all = state_df['racial_tension'].mean()

        row = {
            'state':                state,
            'n_articles_total':     len(state_df),
            'bn_sentiment_all':     round(bn_all,      4),
            'harapan_sentiment_all': round(harapan_all, 4),
            'pn_sentiment_all':     round(pn_all,      4),
            'racial_tension_all':   round(tension_all, 4),
            'bn_sentiment_std':     round(state_df['bn_sentiment'].std() or 0.0, 4),
            # National narrative averages (same for all states,
            # but weighted differently per seat via demographics)
            'islam_threat':         round(narratives['islam_threat'],     4),
            'malay_unity':          round(narratives['malay_unity'],      4),
            'cost_living':          round(narratives['cost_living'],      4),
            'corruption':           round(narratives['corruption'],       4),
            'govt_performance':     round(narratives['govt_performance'], 4),
        }

        if has_lean:
            indep_df = state_df[state_df['source_lean'].isin(INDEPENDENT_LEANS)]
            if len(indep_df) > 0:
                row['n_articles_independent']   = len(indep_df)
                row['bn_sentiment_independent'] = round(indep_df['bn_sentiment'].mean(), 4)
                row['harapan_sentiment_indep']  = round(indep_df['harapan_sentiment'].mean(), 4)
                row['pn_sentiment_independent'] = round(indep_df['pn_sentiment'].mean(), 4)
            else:
                row['n_articles_independent']   = 0
                row['bn_sentiment_independent'] = row['bn_sentiment_all']
                row['harapan_sentiment_indep']  = row['harapan_sentiment_all']
                row['pn_sentiment_independent'] = row['pn_sentiment_all']

        row['bn_sentiment']         = row.get('bn_sentiment_independent', bn_all)
        row['harapan_sentiment']    = row.get('harapan_sentiment_indep', harapan_all)
        row['pn_sentiment']         = row.get('pn_sentiment_independent', pn_all)
        row['racial_tension_index'] = tension_all

        scores_dict = {
            'BN': row['bn_sentiment'],
            'Harapan': row['harapan_sentiment'],
            'PN': row['pn_sentiment'],
        }
        row['dominant_sentiment'] = max(scores_dict, key=scores_dict.get)

        state_scores.append(row)

    return pd.DataFrame(state_scores)


# ── Weekly trend ────────────────────────────────────────────────────

def compute_weekly_trend(df_scored: pd.DataFrame,
                          state: str) -> pd.DataFrame:
    """Group sentiment + themes by week."""
    col = f'mentions_{state}'

    if col in df_scored.columns:
        state_df = df_scored[df_scored[col] == True].copy()
    else:
        state_df = df_scored.copy()   # use all if no state filter

    if state_df.empty:
        return pd.DataFrame()

    state_df['published_dt'] = pd.to_datetime(
        state_df['published'], errors='coerce', utc=True
    )
    state_df = state_df.dropna(subset=['published_dt'])
    if state_df.empty:
        return pd.DataFrame()

    state_df['week'] = state_df['published_dt'].dt.to_period('W').astype(str)

    agg_cols = {
        'n_articles':        ('title', 'count'),
        'bn_sentiment':      ('bn_sentiment', 'mean'),
        'harapan_sentiment': ('harapan_sentiment', 'mean'),
        'pn_sentiment':      ('pn_sentiment', 'mean'),
        'racial_tension':    ('racial_tension', 'mean'),
        'islam_threat':      ('islam_threat', 'mean'),
        'malay_unity':       ('malay_unity', 'mean'),
        'cost_living':       ('cost_living', 'mean'),
        'corruption':        ('corruption', 'mean'),
    }

    weekly = state_df.groupby('week').agg(**agg_cols).round(4).reset_index()

    weekly['bn_momentum']      = weekly['bn_sentiment'].diff().round(4)
    weekly['pn_momentum']      = weekly['pn_sentiment'].diff().round(4)
    weekly['islam_momentum']   = weekly['islam_threat'].diff().round(4)
    weekly['cost_momentum']    = weekly['cost_living'].diff().round(4)

    return weekly


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) > 1:
        pattern = sys.argv[1]
        matched = sorted(glob.glob(str(DATA_RAW / f"{pattern}*.csv")))
        if not matched:
            print(f"No files matching '{pattern}*.csv' in {DATA_RAW}")
            exit(1)
        latest = matched[-1]
        is_historical = True
        print(f"Using specified file: {latest}")
    else:
        historical_files = sorted(glob.glob(str(DATA_RAW / "*_historical_*.csv")))
        news_files        = sorted(glob.glob(str(DATA_RAW / "news_*.csv")))

        if historical_files:
            latest = historical_files[-1]
            is_historical = True
            print(f"Using historical data: {latest}")
        elif news_files:
            latest = news_files[-1]
            is_historical = False
            print(f"Loading latest news scrape: {latest}")
        else:
            print("No news files found.")
            print("Run: python sentiment/scrapers/news_scraper.py first")
            exit(1)

    df = pd.read_csv(latest)
    print(f"   Articles: {len(df)}\n")

    if df.empty:
        print("No articles to score")
        exit(1)

    filename = Path(latest).stem
    detected_state = None
    for state in STATE_COLS:
        if state in filename:
            detected_state = state
            break

    # Score all articles (no state filter)
    df_scored = score_dataframe(df)

    # Save scored articles
    scored_path = DATA_PROCESSED / "scored_articles.csv"
    df_scored.to_csv(scored_path, index=False)
    print(f"\nScored articles saved: {scored_path}")

    # Compute national narrative scores
    print(f"\n{'='*60}")
    print(f"  NATIONAL NARRATIVE SCORES")
    print(f"{'='*60}")
    narratives = compute_national_narratives(df_scored)

    # Save narratives separately
    narrative_path = DATA_PROCESSED / "national_narrative_scores.csv"
    pd.DataFrame([narratives]).to_csv(narrative_path, index=False)
    print(f"\n  Saved: {narrative_path}")
    print(f"  These are applied per seat weighted by demographics")

    # State-level scores
    state_scores = compute_state_scores(df_scored, narratives)

    if state_scores.empty:
        print("\nNo state scores computed")
        exit(1)

    state_path = DATA_PROCESSED / "state_sentiment_scores.csv"
    state_scores.to_csv(state_path, index=False)
    print(f"\nState scores saved: {state_path}")

    # Print results
    print(f"\n{'='*60}")
    print(f"  STATE SENTIMENT SCORES (independent sources)")
    print(f"{'='*60}")
    print(f"  {'State':<16} {'BN':>7} {'Harapan':>9} {'PN':>7} "
          f"{'Tension':>9} {'Dominant':>10}")
    print(f"  {'-'*16} {'-'*7} {'-'*9} {'-'*7} {'-'*9} {'-'*10}")
    for _, row in state_scores.iterrows():
        print(
            f"  {row['state']:<16} "
            f"{row['bn_sentiment']:>+7.2f} "
            f"{row['harapan_sentiment']:>+9.2f} "
            f"{row['pn_sentiment']:>+7.2f} "
            f"{row['racial_tension_index']:>9.3f} "
            f"{row['dominant_sentiment']:>10}"
        )

    print(f"\n{'='*60}")
    print(f"  NATIONAL NARRATIVE THEMES")
    print(f"{'='*60}")
    print(f"  Islam threat (Malay seats):    {narratives['islam_threat']:.3f}")
    print(f"  Malay unity (BN-PN bloc):      {narratives['malay_unity']:.3f}")
    print(f"  Cost of living (urban/youth):  {narratives['cost_living']:.3f}")
    print(f"  Corruption (anti-incumbent):   {narratives['corruption']:.3f}")
    print(f"  Govt performance:              {narratives['govt_performance']:.3f}")

    print(f"\n  Example seat-level narrative pressure:")
    examples = [
        ('Rural Malay seat', 0.75, 0.10, 0.20),
        ('Chinese urban seat', 0.25, 0.65, 0.35),
        ('Mixed seat', 0.50, 0.40, 0.28),
    ]
    for label, m, c, y in examples:
        pressure = apply_narratives_to_seat(narratives, m, c, y)
        print(f"    {label:<25}: {pressure:+.4f}")

    # Weekly trend
    if is_historical and detected_state:
        print(f"\n{'='*60}")
        print(f"  {detected_state.upper()} WEEKLY TREND")
        print(f"{'='*60}")

        weekly = compute_weekly_trend(df_scored, detected_state)

        if not weekly.empty:
            trend_path = DATA_PROCESSED / f"{detected_state}_weekly_trend.csv"
            weekly.to_csv(trend_path, index=False)

            print(f"  {'Week':<24} {'N':>4} {'BN':>7} {'PN':>7} "
                  f"{'Islam':>7} {'Cost':>7}")
            print(f"  {'-'*24} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
            for _, row in weekly.iterrows():
                print(
                    f"  {row['week']:<24} "
                    f"{row['n_articles']:>4.0f} "
                    f"{row['bn_sentiment']:>+7.2f} "
                    f"{row['pn_sentiment']:>+7.2f} "
                    f"{row['islam_threat']:>7.3f} "
                    f"{row['cost_living']:>7.3f}"
                )
            print(f"\n  Weekly trend saved: {trend_path}")

    print(f"\nPhase 2 scoring complete!")
    print(f"\nNext:")
    print(f"  1. Update add_ethnicity_features.py to use narrative themes")
    print(f"  2. Retrain: python backend/scripts/train_models.py")
    print(f"  3. Validate: python backend/scripts/validate_2026.py")