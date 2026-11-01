import argparse
import pandas as pd
import numpy as np
from pathlib import Path
 
# ── Config ────────────────────────────────────────────────────────────
 
SCORE_COLS = [
    'bn_sentiment', 'harapan_sentiment', 'pn_sentiment',
    'racial_tension', 'islam_threat', 'malay_unity',
    'cost_living', 'corruption', 'govt_performance',
]
 
THEME_COLS = [
    'islam_threat', 'malay_unity', 'cost_living',
    'corruption', 'govt_performance',
]
 
DEFAULT_INPUT  = Path("data/processed/scored_articles.csv")
DEFAULT_OUTPUT = Path("eval/eval_set.csv")
DEFAULT_SIZE   = 100
RANDOM_SEED    = 42
 
 
# ── Helpers ───────────────────────────────────────────────────────────
 
def is_all_zero(row: pd.Series) -> bool:
    """Returns True if Groq returned all zeros (timeout/failure)."""
    return (row[SCORE_COLS] == 0).all()
 
 
def score_quality(row: pd.Series) -> str:
    """
    Rough quality label for an article's Groq scores.
    Useful to see in the eval set at a glance.
 
    high    — at least one strong signal (|score| >= 0.5)
    medium  — some signal but weak
    zero    — all zeros (Groq failure, filtered out before this runs)
    """
    max_abs = row[SCORE_COLS].abs().max()
    if max_abs >= 0.5:
        return "high"
    elif max_abs > 0:
        return "medium"
    else:
        return "zero"
 
 
def dominant_themes(row: pd.Series, threshold: float = 0.3) -> str:
    """
    Returns comma-separated list of themes Groq scored above threshold.
    Empty string if none. Used as a quick label in the eval set.
    """
    active = [t for t in THEME_COLS if row[t] >= threshold]
    return ",".join(active) if active else "none"
 
 
# ── Main ──────────────────────────────────────────────────────────────
 
def sample_eval_set(
    input_path: Path,
    output_path: Path,
    size: int = DEFAULT_SIZE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
 
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Total articles: {len(df)}")
 
    # ── Step 1: filter out all-zero rows (Groq failures) ─────────────
    zero_mask = df.apply(is_all_zero, axis=1)
    df_scored = df[~zero_mask].copy()
    print(f"  Groq failures (all-zero) removed: {zero_mask.sum()}")
    print(f"  Usable articles: {len(df_scored)}")
 
    if len(df_scored) < size:
        print(f"\n  WARNING: only {len(df_scored)} usable articles, "
              f"reducing eval size to {len(df_scored)}")
        size = len(df_scored)
 
    # ── Step 2: stratify by source_lean ──────────────────────────────
    # We want the eval set to reflect the source mix in the full data,
    # not oversample one outlet. Stratification handles this.
    lean_counts = df_scored['source_lean'].value_counts()
    print(f"\n  Source lean distribution (usable articles):")
    for lean, count in lean_counts.items():
        print(f"    {lean:<25} {count}")
 
    # Sample proportionally from each lean group
    sampled_parts = []
    for lean, count in lean_counts.items():
        group     = df_scored[df_scored['source_lean'] == lean]
        n_sample  = max(1, round(size * count / len(df_scored)))
        n_sample  = min(n_sample, len(group))
        sampled_parts.append(
            group.sample(n=n_sample, random_state=seed)
        )
 
    eval_df = pd.concat(sampled_parts).drop_duplicates()
 
    # If we're slightly over/under due to rounding, trim or top up
    if len(eval_df) > size:
        eval_df = eval_df.sample(n=size, random_state=seed)
    elif len(eval_df) < size:
        # fill remaining from whatever's left
        remaining = df_scored[~df_scored.index.isin(eval_df.index)]
        extra     = remaining.sample(
            n=min(size - len(eval_df), len(remaining)),
            random_state=seed,
        )
        eval_df = pd.concat([eval_df, extra])
 
    eval_df = eval_df.reset_index(drop=True)
 
    # ── Step 3: add helper columns ────────────────────────────────────
    eval_df['groq_score_quality'] = eval_df.apply(score_quality, axis=1)
    eval_df['groq_themes']        = eval_df.apply(dominant_themes, axis=1)
 
    # human_label columns — EMPTY for now, filled by label_articles.py
    # one binary column per theme: 0 = not present, 1 = present
    for theme in THEME_COLS:
        eval_df[f'human_{theme}'] = ""   # empty string = not yet labeled
 
    eval_df['label_notes'] = ""   # free-text notes during labeling
 
    # ── Step 4: save ─────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output_path, index=False)
 
    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  EVAL SET CREATED")
    print(f"{'='*55}")
    print(f"  Articles:       {len(eval_df)}")
    print(f"  Saved to:       {output_path}")
    print(f"\n  Score quality breakdown:")
    print(eval_df['groq_score_quality'].value_counts().to_string())
    print(f"\n  Theme distribution (Groq scores ≥ 0.3):")
    for theme in THEME_COLS:
        active = (eval_df[theme] >= 0.3).sum()
        print(f"    {theme:<20} {active:>3} articles")
    print(f"\n  Source lean breakdown:")
    print(eval_df['source_lean'].value_counts().to_string())
    print(f"\n  Next step: python eval/label_articles.py")
 
    return eval_df
 
 
# ── Entry point ───────────────────────────────────────────────────────
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Freeze a stratified eval set from scored_articles.csv"
    )
    parser.add_argument(
        "--input",  type=Path, default=DEFAULT_INPUT,
        help=f"Path to scored_articles.csv (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Where to save eval_set.csv (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--size",   type=int,  default=DEFAULT_SIZE,
        help=f"Number of articles in eval set (default: {DEFAULT_SIZE})"
    )
    args = parser.parse_args()
 
    sample_eval_set(args.input, args.output, args.size)