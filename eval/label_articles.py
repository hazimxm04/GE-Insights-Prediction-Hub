"""
eval/label_articles.py
=======================
CLI tool to manually label eval_set.csv articles with ground truth.

WHY THIS EXISTS:
  Groq's scores are our baseline — but we don't know if they're right.
  This tool lets you read each article and decide: which narrative
  themes are actually present? Your human labels become ground truth.

  Fine-tuned model vs Groq baseline is then measured against YOUR
  labels, not against each other — so the comparison is honest.

HOW IT WORKS:
  Shows you one article at a time. You type 0 or 1 for each theme.
  Progress is saved after every article so you can quit and resume.
  Aim to label at least 80 articles for a meaningful eval.

THEMES YOU'RE LABELING:
  islam_threat     — "Islam/Melayu terancam" narrative present?
  malay_unity      — BN-PN unity / Malay consolidation narrative?
  cost_living      — cost of living / economic hardship narrative?
  corruption       — corruption / rasuah narrative?
  govt_performance — government delivery / performance narrative?

  Label 1 if clearly present, 0 if absent or ambiguous.
  When in doubt, go with your gut — you know this domain.

Usage:
    python eval/label_articles.py
    python eval/label_articles.py --input eval/eval_set.csv
    python eval/label_articles.py --start 50   (resume from article 50)
"""

import argparse
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────

THEME_COLS = [
    'islam_threat',
    'malay_unity',
    'cost_living',
    'corruption',
    'govt_performance',
]

THEME_DESCRIPTIONS = {
    'islam_threat':     '"Islam/Melayu terancam" or religious threat narrative',
    'malay_unity':      'BN-PN unity / Malay consolidation / perpaduan Melayu',
    'cost_living':      'Cost of living / kos sara hidup / economic hardship',
    'corruption':       'Corruption / rasuah / abuse of power',
    'govt_performance': 'Government delivery / performance / policy outcomes',
}

DEFAULT_INPUT = Path("eval/eval_set.csv")


# ── Display helpers ───────────────────────────────────────────────────

def clear_line():
    print("\033[K", end="")


def print_header(current: int, total: int, labeled_so_far: int):
    print("\n" + "="*60)
    print(f"  ARTICLE {current}/{total}   |   labeled so far: {labeled_so_far}")
    print("="*60)


def print_groq_scores(row: pd.Series):
    """Show Groq's scores alongside so you can compare."""
    print("\n  Groq scored this as:")
    for theme in THEME_COLS:
        score = row[theme]
        bar   = "█" * int(score * 10)
        print(f"    {theme:<20} {score:.2f}  {bar}")


def print_article(row: pd.Series):
    print(f"\n  Title:   {row['title']}")
    print(f"  Source:  {row['source']}  ({row['source_lean']})")
    print(f"  Date:    {row['published']}")
    if 'text_snippet' in row and row['text_snippet'] != row['title']:
        print(f"  Text:    {str(row['text_snippet'])[:300]}")


# ── Label one article ─────────────────────────────────────────────────

def label_one(row: pd.Series) -> dict | None:
    """
    Interactive prompt for one article.
    Returns dict of {theme: 0/1} or None if user wants to skip/quit.
    """
    labels = {}

    print("\n  Label each theme (1 = present, 0 = absent, s = skip, q = quit):\n")

    for theme in THEME_COLS:
        desc = THEME_DESCRIPTIONS[theme]
        while True:
            raw = input(f"  {theme:<20} [{desc}]: ").strip().lower()
            if raw == 'q':
                return None          # signal to quit
            elif raw == 's':
                return "skip"        # signal to skip this article
            elif raw in ('0', '1'):
                labels[theme] = int(raw)
                break
            else:
                print("    → enter 0, 1, s (skip), or q (quit)")

    # Optional notes
    notes = input("\n  Notes (optional, press enter to skip): ").strip()
    labels['label_notes'] = notes

    return labels


# ── Main ──────────────────────────────────────────────────────────────

def label_articles(input_path: Path, start_from: int = 0):

    print(f"\nLoading: {input_path}")
    df = pd.read_csv(input_path)

    # figure out which articles still need labeling
    # handle both empty string and NaN (pandas reads blank CSV cells as NaN)
    human_col = f'human_{THEME_COLS[0]}'
    df[human_col] = df[human_col].fillna("").astype(str).str.strip()
    unlabeled_idx = df[df[human_col] == ""].index.tolist()
    already_done  = len(df) - len(unlabeled_idx)

    print(f"  Total articles: {len(df)}")
    print(f"  Already labeled: {already_done}")
    print(f"  Remaining: {len(unlabeled_idx)}")

    if not unlabeled_idx:
        print("\n  All articles labeled! Run eval/baseline_eval.py next.")
        return

    print("\n  Controls: 1=present  0=absent  s=skip  q=save&quit")
    print("  Progress is saved after every article.\n")
    input("  Press Enter to start labeling...")

    labeled_this_session = 0

    for i, idx in enumerate(unlabeled_idx):
        if i < start_from:
            continue

        row = df.loc[idx]
        print_header(
            current       = already_done + i + 1,
            total         = len(df),
            labeled_so_far= already_done + labeled_this_session,
        )
        print_article(row)
        print_groq_scores(row)

        result = label_one(row)

        if result is None:
            # user typed q — save and quit
            print(f"\n  Saving progress and quitting...")
            df.to_csv(input_path, index=False)
            print(f"  Saved. {already_done + labeled_this_session} articles labeled total.")
            print(f"  Resume with: python eval/label_articles.py")
            return

        if result == "skip":
            print("  Skipped.")
            continue

        # write labels back into dataframe
        for theme in THEME_COLS:
            df.at[idx, f'human_{theme}'] = result[theme]
        df.at[idx, 'label_notes'] = result.get('label_notes', '')

        labeled_this_session += 1

        # save after every article so progress isn't lost
        df.to_csv(input_path, index=False)
        print(f"\n  ✓ Saved. ({already_done + labeled_this_session} labeled total)")

    print(f"\n{'='*60}")
    print(f"  LABELING COMPLETE")
    print(f"{'='*60}")
    print(f"  Labeled this session: {labeled_this_session}")
    print(f"  Total labeled:        {already_done + labeled_this_session} / {len(df)}")
    print(f"\n  Next step: python eval/baseline_eval.py")


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manually label eval_set.csv articles with ground truth"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Path to eval_set.csv (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Skip the first N unlabeled articles (resume offset)"
    )
    args = parser.parse_args()

    label_articles(args.input, args.start)