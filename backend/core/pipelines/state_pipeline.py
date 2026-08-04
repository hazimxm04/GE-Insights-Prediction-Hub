import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\hazim\Downloads\GE Insights Predictor")
RAW  = BASE / "data/raw"

# ── Coalition mapping ─────────────────────────────────────────────
# Map all coalition codes → 3 groups: Harapan, BN, PN, Others
COALITION_MAP = {
    'PH':        'Harapan',   # Pakatan Harapan
    'PR':        'Harapan',   # Pakatan Rakyat (predecessor)
    'BN':        'BN',        # Barisan Nasional
    'PERIKATAN': 'BN',        # Old Perikatan = BN era
    'PN':        'PN',        # Perikatan Nasional (new)
    'GS':        'PN',        # Gabungan Sejahtera (PN-aligned)
    'BERSATU+':  'PN',        # PN-aligned
    'SF':        'Others',
    'HAK':       'Others',
    'APU':       'Others',
    'GR':        'Others',
    'BA':        'Others',
    'ALONE':     'Others',    # Independent
}

# ── Election year config per state ───────────────────────────────
STATE_CONFIG = {
    'johor': {
        'ballots_file': 'johor_ballots.parquet',
        'stats_file':   'johor_stats.parquet',
        'train_year':   2018,
        'test_year':    2022,
        'val_year':     2026,
    },
    'neg_sembilan': {
        'ballots_file': 'ns_ballots.parquet',
        'stats_file':   'ns_stats.parquet',
        'train_year':   2018,
        'test_year':    2023,   # NS had election in 2023!
        'val_year':     2026,
    },
    'melaka': {
        'ballots_file': 'melaka_ballots.parquet',
        'stats_file':   'melaka_stats.parquet',
        'train_year':   2018,
        'test_year':    2021,   # Latest Melaka data
        'val_year':     None,   # No 2026 yet
    },
}


class StateElectionPipeline:
    """
    Loads + cleans + engineers features for one state.
    
    Usage:
        pipeline = StateElectionPipeline('johor')
        df_train, df_test = pipeline.get_train_test()
        df_val = pipeline.get_validation()
    """

    def __init__(self, state: str):
        if state not in STATE_CONFIG:
            raise ValueError(f"Unknown state: {state}. Choose from {list(STATE_CONFIG.keys())}")

        self.state  = state
        self.config = STATE_CONFIG[state]

        # Load raw files
        self.ballots = pd.read_parquet(RAW / self.config['ballots_file'])
        self.stats   = pd.read_parquet(RAW / self.config['stats_file'])

        # Parse year
        self.ballots['year'] = pd.to_datetime(self.ballots['date'], errors='coerce').dt.year.astype('Int64')
        self.stats['year']   = pd.to_datetime(self.stats['date'],   errors='coerce').dt.year.astype('Int64')

        print(f"✅ Loaded {state}: {len(self.ballots)} ballot rows, {len(self.stats)} stat rows")

    # ── Step 1: Get winner per seat per year ─────────────────────

    def get_winners(self, year: int) -> pd.DataFrame:
        """
        From ballots, get ONE row per seat = the winner only.
        Returns: seat, winner_coalition, winner_votes, total_candidates
        """
        df = self.ballots[self.ballots['year'] == year].copy()

        if df.empty:
            print(f"⚠️  No ballot data for {self.state} {year}")
            return pd.DataFrame()

        # Winner = rank 1 or result == won/won_uncontested
        winners = df[df['result'].isin(['won', 'won_uncontested'])].copy()

        # Map coalition to simplified group
        winners['coalition_group'] = winners['coalition'].map(COALITION_MAP).fillna('Others')

        # Keep only what we need
        winners = winners[['seat', 'coalition', 'coalition_group', 'votes', 'votes_perc']].copy()
        winners.columns = ['seat', 'winner_coalition_raw', 'winner_coalition', 'winner_votes', 'winner_votes_perc']

        return winners.reset_index(drop=True)

    # ── Step 2: Get seat stats per year ──────────────────────────

    def get_stats(self, year: int) -> pd.DataFrame:
        """
        From stats, get seat-level stats for one year.
        Returns: seat, voters_total, voter_turnout, majority, votes_valid
        """
        df = self.stats[self.stats['year'] == year].copy()

        if df.empty:
            print(f"⚠️  No stats data for {self.state} {year}")
            return pd.DataFrame()

        return df[['seat', 'voters_total', 'votes_valid', 'majority',
                   'voter_turnout', 'majority_perc', 'n_candidates']].reset_index(drop=True)

    # ── Step 3: Combine winners + stats for one year ─────────────

    def get_year_data(self, year: int) -> pd.DataFrame:
        """Merge winners + stats for one year"""
        winners = self.get_winners(year)
        stats   = self.get_stats(year)

        if winners.empty or stats.empty:
            return pd.DataFrame()

        merged = winners.merge(stats, on='seat', how='inner')
        merged['year'] = year
        merged['state'] = self.state

        print(f"  {self.state} {year}: {len(merged)} seats")

        return merged

    # ── Step 4: Engineer features (train + test pair) ────────────

    def engineer_features(self, year_a: int, year_b: int) -> pd.DataFrame:
        """
        Create feature set from TWO election years.
        year_a = earlier (used as baseline)
        year_b = later   (used as target)

        Features are CHANGES from year_a → year_b.
        Target: did opposition (non-BN) win in year_b?
        """
        df_a = self.get_year_data(year_a)
        df_b = self.get_year_data(year_b)

        if df_a.empty or df_b.empty:
            return pd.DataFrame()

        # Merge on seat
        merged = df_b.merge(
            df_a[['seat', 'winner_coalition', 'winner_votes',
                  'voter_turnout', 'majority', 'votes_valid']],
            on='seat',
            suffixes=('_b', '_a'),
            how='inner'
        )

        print(f"  Matched seats ({year_a}→{year_b}): {len(merged)}")

        # ── Engineer features ────────────────────────────────────

        # 1. Did same coalition win both elections?
        merged['incumbent_held'] = (
            merged['winner_coalition_b'] == merged['winner_coalition_a']
        ).astype(int)

        # 2. Change in majority (vote margin)
        merged['majority_change'] = merged['majority_b'] - merged['majority_a']

        # 3. Change in turnout
        merged['turnout_change'] = merged['voter_turnout_b'] - merged['voter_turnout_a']

        # 4. Log of total voters (scale)
        merged['log_voters'] = np.log1p(merged['voters_total'].astype(float))

        # 5. Majority as % of valid votes (strength of win)
        merged['majority_perc_b'] = np.where(
            merged['votes_valid_b'] > 0,
            merged['majority_b'] / merged['votes_valid_b'],
            0
        )
        merged['majority_perc_a'] = np.where(
            merged['votes_valid_a'] > 0,
            merged['majority_a'] / merged['votes_valid_a'],
            0
        )

        # 6. Change in win strength
        merged['majority_perc_change'] = merged['majority_perc_b'] - merged['majority_perc_a']

        # 7. Number of candidates (more = more fragmented)
        merged['n_candidates_b'] = merged['n_candidates']

        # ── Target variable ───────────────────────────────────────
        # 1 = opposition won (PH or PN), 0 = BN won
        # We predict: did NON-BN win?
        merged['target_non_bn_won'] = (merged['winner_coalition_b'] != 'BN').astype(int)

        # Also useful: did Harapan specifically win?
        merged['target_harapan_won'] = (merged['winner_coalition_b'] == 'Harapan').astype(int)

        # ── Final feature columns ─────────────────────────────────
        merged['year_from'] = year_a
        merged['year_to']   = year_b

        return merged.reset_index(drop=True)

    # ── Step 5: Get train/test split ─────────────────────────────

    def get_train_test(self):
        """
        Returns engineered features for:
          train: (train_year → test_year) transition
          
        This is the main dataset for model training.
        """
        train_year = self.config['train_year']
        test_year  = self.config['test_year']

        print(f"\n[{self.state.upper()}] Engineering features {train_year} → {test_year}")
        df = self.engineer_features(train_year, test_year)

        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Feature columns for model
        FEATURE_COLS = [
            'majority_change',
            'turnout_change',
            'incumbent_held',
            'log_voters',
            'majority_perc_change',
            'n_candidates_b',
        ]

        X = df[FEATURE_COLS].fillna(0)
        y = df['target_non_bn_won']  # 1 = non-BN won, 0 = BN won

        print(f"  X shape: {X.shape}")
        print(f"  y distribution: {y.value_counts().to_dict()}")

        return X, y, df

    # ── Step 6: Get 2026 validation data ─────────────────────────

    def get_validation(self):
        """
        Returns engineered features for validation year.
        Uses (test_year → val_year) transition.
        """
        val_year  = self.config['val_year']
        test_year = self.config['test_year']

        if val_year is None:
            print(f"⚠️  No validation year for {self.state}")
            return pd.DataFrame()

        print(f"\n[{self.state.upper()}] Validation features {test_year} → {val_year}")
        return self.engineer_features(test_year, val_year)


# ── Quick test ────────────────────────────────────────────────────

if __name__ == "__main__":

    for state in ['johor', 'neg_sembilan', 'melaka']:
        print(f"\n{'='*55}")
        print(f"  TESTING: {state.upper()}")
        print(f"{'='*55}")

        pipeline = StateElectionPipeline(state)
        X, y, df = pipeline.get_train_test()

        if not X.empty:
            print(f"\n  Features:")
            print(X.describe().round(3).to_string())
            print(f"\n  Target (non-BN won):")
            print(f"  0 (BN won):     {(y==0).sum()} seats")
            print(f"  1 (non-BN won): {(y==1).sum()} seats")

        val = pipeline.get_validation()
        if not val.empty:
            print(f"\n  Validation ({pipeline.config['val_year']}): {len(val)} seats")