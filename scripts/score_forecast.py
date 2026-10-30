"""
scripts/score_forecast.py
=========================
Score a FROZEN forecast against the real election result.

Run this after a state has voted and its new results are in data/raw/:

    python backend/scripts/download_data.py --refresh        # get the new election
    python scripts/score_forecast.py selangor                # latest frozen folder
    python scripts/score_forecast.py selangor --date 2026-10-30 --save

What it does:
  1. checks the frozen folder is unchanged (checksum from its manifest)
  2. finds the election that came after the one the forecast was made for
  3. matches seats on their number (N.07), not their name
  4. counts correct calls for the model and for the two simple rules saved
     in the frozen file ("never opposition", "same as last time"),
     plus false alarms, missed wins, Brier score and AUC

--save writes results_<year>.json next to the frozen files (a new file; the
frozen files themselves are never touched).

A model is only useful if it beats the simple rules, so they are always shown.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from backend.core.features import OPPOSITION, COALITION_RENAME, seat_no
import freeze_forecasts as ff


def actual_results(ballots: pd.DataFrame, year: int) -> pd.DataFrame:
    """One row per seat: seat_no, winning coalition, actual_opposition (1 = opposition won)."""
    b = ballots.copy()
    b['year'] = pd.to_datetime(b['date'], errors='coerce').dt.year
    w = b[(b['year'] == year) & b['result'].isin(['won', 'won_uncontested'])].copy()
    w['seat_no'] = w['seat'].map(seat_no)
    w['coalition'] = w['coalition'].replace(COALITION_RENAME)
    w = w.drop_duplicates('seat_no')
    w['actual_opposition'] = w['coalition'].isin(OPPOSITION).astype(int)
    return w[['seat_no', 'seat', 'coalition', 'actual_opposition']]


def result_year(ballots: pd.DataFrame, after_year: int):
    """The first election in the data after the one the forecast was made for (or None)."""
    years = sorted(y for y in pd.to_datetime(ballots['date'], errors='coerce').dt.year.dropna().unique()
                   if y > after_year)
    return int(years[0]) if years else None


def score_frozen(folder, ballots: pd.DataFrame) -> dict:
    """Score one frozen forecast folder against the real result in `ballots`."""
    folder = Path(folder)
    problems = ff.verify_folder(folder)
    if problems:
        raise SystemExit('The frozen folder fails its checks, so it is not scored:\n  ' + '\n  '.join(problems))

    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    fc = pd.read_csv(folder / 'forecast.csv')
    year_b = result_year(ballots, manifest['forecasting_election_after'])
    if year_b is None:
        raise SystemExit(f'No election after {manifest["forecasting_election_after"]} in the data yet, '
                         f'so there is nothing to score. Refresh the data after the election.')

    actual = actual_results(ballots, year_b)
    m = fc.merge(actual, on='seat_no', how='inner')
    unmatched_forecast = sorted(set(fc['seat_no']) - set(actual['seat_no']))
    unmatched_result = sorted(set(actual['seat_no']) - set(fc['seat_no']))
    if m.empty:
        raise SystemExit('No seats matched between the forecast and the result.')

    y = m['actual_opposition']
    pred = m['predicted_opposition']
    n = int(len(m))
    out = {
        'state': manifest['state'],
        'frozen_folder': folder.name,
        'frozen_at_utc': manifest['frozen_at_utc'],
        'forecast_made_after': manifest['forecasting_election_after'],
        'result_year': year_b,
        'seats_scored': n,
        'opposition_won': int(y.sum()),
        'model_correct': int((pred == y).sum()),
        'never_opposition_correct': int((m['baseline_never_opposition'] == y).sum()),
        'same_as_last_correct': int((m['baseline_same_as_last'] == y).sum()),
        'false_alarms': int(((pred == 1) & (y == 0)).sum()),     # said opposition, was not
        'missed_wins': int(((pred == 0) & (y == 1)).sum()),      # missed an opposition win
        'brier': round(float(brier_score_loss(y, m['probability'])), 4),
        'auc': round(float(roc_auc_score(y, m['probability'])), 3) if y.nunique() == 2 else None,
        'forecast_seats_without_result': unmatched_forecast,
        'result_seats_without_forecast': unmatched_result,
    }
    return out


def print_report(r: dict):
    n = r['seats_scored']
    print(f"\n{r['state']}: forecast frozen {r['frozen_at_utc']} (made after the {r['forecast_made_after']} election)")
    print(f"scored against the {r['result_year']} result: {n} seats, opposition won {r['opposition_won']}\n")
    print(f"  model               {r['model_correct']:>3}/{n}  ({r['model_correct'] / n:.1%})")
    print(f"  never opposition    {r['never_opposition_correct']:>3}/{n}  ({r['never_opposition_correct'] / n:.1%})")
    print(f"  same as last time   {r['same_as_last_correct']:>3}/{n}  ({r['same_as_last_correct'] / n:.1%})")
    print(f"\n  false alarms {r['false_alarms']}, missed wins {r['missed_wins']}, "
          f"Brier {r['brier']}, AUC {r['auc']}")
    if r['forecast_seats_without_result'] or r['result_seats_without_forecast']:
        print(f"  not matched: forecast seats without a result {r['forecast_seats_without_result']}, "
              f"result seats without a forecast {r['result_seats_without_forecast']}")
    best_rule = max(r['never_opposition_correct'], r['same_as_last_correct'])
    verdict = ('beats' if r['model_correct'] > best_rule else
               'ties' if r['model_correct'] == best_rule else 'is behind')
    print(f"\n  The model {verdict} the best simple rule ({best_rule}/{n}).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('state')
    ap.add_argument('--date', help='frozen folder date (YYYY-MM-DD); default: the latest one')
    ap.add_argument('--save', action='store_true', help='write results_<year>.json next to the frozen files')
    args = ap.parse_args()

    base = ff.FORECAST_DIR / args.state
    dates = sorted(p.name for p in base.glob('*') if (p / 'manifest.json').exists()) if base.exists() else []
    if not dates:
        raise SystemExit(f'No frozen forecasts for {args.state} in {base}')
    date = args.date or dates[-1]
    if date not in dates:
        raise SystemExit(f'No frozen folder {date} for {args.state}. Available: {", ".join(dates)}')

    from state_data import load_state_data
    ballots, _ = load_state_data(args.state)
    r = score_frozen(base / date, ballots)
    print_report(r)
    if args.save:
        path = base / date / f'results_{r["result_year"]}.json'
        path.write_text(json.dumps(r, indent=2) + '\n', encoding='utf-8', newline='\n')
        print(f'\nsaved {path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
