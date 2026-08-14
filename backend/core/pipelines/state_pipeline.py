import pandas as pd
import numpy as np
from pathlib import Path
from backend.scripts.add_ethnicity_features import merge_ethnicity_into_features

BASE = Path(__file__).resolve().parents[3]
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
        'val_year':     2026,   # No 2026 yet
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
        OPPOSITION = {'Harapan'}
        merged['target_non_bn_won'] = (
            merged['winner_coalition_b'].isin(OPPOSITION)
        ).astype(int)
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

        df = self.engineer_features(test_year, val_year)
        if df.empty:
            return pd.DataFrame()

        sentiment = self.load_sentiment_features()
        df['bn_sentiment']         = sentiment['bn_sentiment']
        df['harapan_sentiment']    = sentiment['harapan_sentiment']
        df['pn_sentiment']         = sentiment['pn_sentiment']
        df['racial_tension_index'] = sentiment['racial_tension_index']

        economic_pressure = self.load_economic_features()
        df['economic_pressure'] = economic_pressure

        df = merge_ethnicity_into_features(
            df_features=df,
            state=self.state,
            year_b=val_year,
            sentiment=sentiment,
            economic_pressure=economic_pressure
        )

        return df
    def load_sentiment_features(self) -> dict:
        """Load Phase 2 sentiment scores for this state."""
        sentiment_path = Path("data/processed/state_sentiment_scores.csv")

        default = {
            'bn_sentiment':         0.0,
            'harapan_sentiment':    0.0,
            'pn_sentiment':         0.0,
            'racial_tension_index': 0.0,
        }

        if not sentiment_path.exists():
            print(f"  No sentiment data, using zeros")
            return default

        df = pd.read_csv(sentiment_path)
        row = df[df['state'] == self.state]

        if row.empty:
            print(f"  No sentiment for {self.state}, using zeros")
            return default

        return {
            'bn_sentiment':         float(row['bn_sentiment'].values[0]),
            'harapan_sentiment':    float(row['harapan_sentiment'].values[0]),
            'pn_sentiment':         float(row['pn_sentiment'].values[0]),
            'racial_tension_index': float(row['racial_tension_index'].values[0]),
        }

    def load_economic_features(self) -> float:
        """Load Phase 4 economic pressure score for this state."""
        economic_path = Path("data/processed/election_economic_pressure.csv")

        if not economic_path.exists():
            print(f"  No economic pressure data, using 0.0")
            return 0.0

        df = pd.read_csv(economic_path)

        state_key_map = {
            'johor':        'johor_2026',
            'neg_sembilan': 'neg_sembilan_2026',
            'melaka':       'melaka_2026',
        }

        key = state_key_map.get(self.state)
        if key is None:
            return 0.0

        row = df[df['state'] == key]
        if row.empty:
            print(f"  No economic pressure for {self.state}")
            return 0.0

        score = float(row['economic_pressure_score'].values[0])
        print(f"  Economic pressure ({self.state}): {score:+.4f}")
        return score

    def get_train_test_with_features(self):
        """6 structural + 4 sentiment + 1 economic = 11 features"""
        train_year = self.config['train_year']
        test_year  = self.config['test_year']

        print(f"\n[{self.state.upper()}] Engineering features "
              f"{train_year} -> {test_year} (11 features)")

        df = self.engineer_features(train_year, test_year)

        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        sentiment = self.load_sentiment_features()
        df['bn_sentiment']         = sentiment['bn_sentiment']
        df['harapan_sentiment']    = sentiment['harapan_sentiment']
        df['pn_sentiment']         = sentiment['pn_sentiment']
        df['racial_tension_index'] = sentiment['racial_tension_index']

        df['economic_pressure'] = self.load_economic_features()

        FEATURE_COLS = [
            'majority_change', 'turnout_change', 'incumbent_held',
            'log_voters', 'majority_perc_change', 'n_candidates_b',
            'bn_sentiment', 'harapan_sentiment', 'pn_sentiment',
            'racial_tension_index', 'economic_pressure',
        ]

        X = df[FEATURE_COLS].fillna(0)
        y = df['target_non_bn_won']

        print(f"  Features: {len(FEATURE_COLS)} total")
        print(f"  X shape: {X.shape}")

        return X, y, df
    def load_ethnicity_features(self, year: int) -> pd.DataFrame:
    
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from backend.scripts.add_ethnicity_features import load_ethnicity_for_state
        return load_ethnicity_for_state(self.state, year)


    def get_train_test_with_features(self):
        """
        Full feature set:
        6 structural + 4 sentiment + 1 economic
        + 8 ethnicity + 5 interactions = 24 features
        """
        from backend.scripts.add_ethnicity_features import merge_ethnicity_into_features

        train_year = self.config['train_year']
        test_year  = self.config['test_year']

        print(f"\n[{self.state.upper()}] Engineering features "
            f"{train_year} -> {test_year} (full feature set)")

        df = self.engineer_features(train_year, test_year)
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Phase 2: sentiment
        sentiment = self.load_sentiment_features()
        df['bn_sentiment']         = sentiment['bn_sentiment']
        df['harapan_sentiment']    = sentiment['harapan_sentiment']
        df['pn_sentiment']         = sentiment['pn_sentiment']
        df['racial_tension_index'] = sentiment['racial_tension_index']

        # Phase 4: economic
        economic_pressure = self.load_economic_features()
        df['economic_pressure'] = economic_pressure

        # Phase 5: ethnicity + age + interactions
        df = merge_ethnicity_into_features(
            df_features=df,
            state=self.state,
            year_b=test_year,
            sentiment=sentiment,
            economic_pressure=economic_pressure
        )

        FEATURE_COLS = [
            # Structural (6)
            'majority_change', 'turnout_change', 'incumbent_held',
            'log_voters', 'majority_perc_change', 'n_candidates_b',
            # Sentiment (4)
            'bn_sentiment', 'harapan_sentiment',
            'pn_sentiment', 'racial_tension_index',
            # Economic (1)
            'economic_pressure',
            # Ethnicity + age (8) ← SEAT-LEVEL, tree CAN split
            'malay_pct', 'chinese_pct', 'indian_pct',
            'young_malay_pct', 'young_chinese_pct',
            'older_malay_pct', 'youth_pct', 'median_age',
            # Interactions (5) ← sentiment × ethnicity
            'bn_sent_x_malay', 'harapan_sent_x_chinese',
            'pn_sent_x_young_malay', 'tension_x_mixed',
            'economic_x_youth',
            'narrative_pressure', 
        ]

        X = df[FEATURE_COLS].fillna(0)
        y = df['target_non_bn_won']

        print(f"  Features: {len(FEATURE_COLS)} total")
        print(f"  X shape: {X.shape}")

        return X, y, df


if __name__ == "__main__":

    for state in ['johor', 'neg_sembilan', 'melaka']:
        print(f"\n{'='*55}")
        pipeline = StateElectionPipeline(state)
        X, y, df = pipeline.get_train_test()
        if not X.empty:
            print(f"  X shape: {X.shape}")


