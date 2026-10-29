"""Print, for each state, the election years from 2000 on, seat counts, and how many
seats (by number, e.g. N.07) carry over from one election to the next.
Usage: python scripts/check_state_years.py [state ...]   (default: all five)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from evaluate_holdout import load_state
from train_models import seat_no, build_transition
from training_config import TRAINING_CONFIGS

for st in (sys.argv[1:] or list(TRAINING_CONFIGS)):
    try:
        bal, sta = load_state(st)
    except Exception as e:
        print(f"{st}: could not load ({type(e).__name__}: {e})"); continue
    sta = sta.copy(); sta["year"] = pd.to_datetime(sta["date"]).dt.year
    sta["no"] = sta["seat"].map(seat_no)
    years = [y for y in sorted(sta.year.unique()) if y >= 2000]
    print(f"== {st}   config transitions: {TRAINING_CONFIGS[st]['transitions']}")
    for a, b in zip(years, years[1:]):
        sa, sb = set(sta[sta.year == a].no), set(sta[sta.year == b].no)
        ok = ""
        try:
            n = len(build_transition(bal, sta.drop(columns=["year", "no"]), a, b)); ok = f"usable rows: {n}"
        except ValueError as e:
            ok = "not usable (needs an earlier 2004+ election)"
        print(f"  {a}->{b}: seats {len(sa)}->{len(sb)}, shared by number {len(sa & sb)}   {ok}")
