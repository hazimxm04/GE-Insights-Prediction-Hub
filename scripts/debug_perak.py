# debug_class_balance.py
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from train_models import build_transition

STATE_URLS = {
    'johor': 'jhr', 'neg_sembilan': 'nsn', 'selangor': 'sgr',
    'melaka': 'mlk', 'perak': 'prk',
}

TRANSITIONS = {
    'johor': [(2008,2013),(2013,2018),(2018,2022)],
    'neg_sembilan': [(2008,2013),(2013,2018),(2018,2023)],
    'selangor': [(2008,2013),(2013,2018),(2018,2023)],
    'melaka': [(2013,2018),(2018,2021)],
    'perak': [(2013,2018),(2018,2022)],
}

for state, prefix in STATE_URLS.items():
    print(f"\n{'='*50}\n{state.upper()}\n{'='*50}")
    ballots = pd.read_parquet(f'https://lake.electiondata.my/results_headline/headline_ballots_state_{prefix}.parquet')
    stats = pd.read_parquet(f'https://lake.electiondata.my/results_headline/headline_stats_state_{prefix}.parquet')
    
    for year_a, year_b in TRANSITIONS[state]:
        try:
            df = build_transition(ballots, stats, year_a, year_b)
            counts = df['target_coalition'].value_counts().sort_index()
            print(f"  {year_a}->{year_b}: {dict(counts)}")
        except Exception as e:
            print(f"  {year_a}->{year_b}: ERROR - {e}")