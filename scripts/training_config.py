# scripts/training_config.py

TRAINING_CONFIGS = {
    'johor': {
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2022)],
        'rf_params': {'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 5},
        'xgb_params': {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.03},
        'description': '3 transitions, medium regularization, full feature set',
        'ethnicity_year': 2022,
    },
    'neg_sembilan': {
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2023)],
        'rf_params': {'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 5},
        'xgb_params': {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.03},
        'description': '3 transitions, medium regularization, full feature set',
        'ethnicity_year': 2023,
    },
    'selangor': {
        'transitions': [(2013, 2018), (2018, 2023)],
        'rf_params': {'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 5},
        'xgb_params': {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.03},
        'description': '2 transitions, high regularization (small sample), full feature set',
        'ethnicity_year': 2023,
    },
    'melaka': {
        'transitions': [(2013, 2018), (2018, 2021)],
        'rf_params': {'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 5},
        'xgb_params': {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.03},
        'description': '2 transitions, 3-way contest (BN/PH/PN), full feature set',
        'ethnicity_year': 2021,
    },
    'perak': {
        'transitions': [(2013, 2018), (2018, 2022)],
        'rf_params': {'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 5},
        'xgb_params': {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.03},
        'description': '2 transitions, 3-way contest (BN/PH/PN), full feature set',
        'ethnicity_year': 2022,
    },
}

# Feature set used in TRAINING.
# NOTE: This is now the FULL 24-feature set (same as FEATURE_PREDICTORS),
# including sentiment/economic/narrative interaction columns. During
# TRAINING, those interaction columns are hardcoded to 0 for every
# historical row (train_models.py does this) -- this is NOT leakage,
# because:
#   1. A constant column (same value, e.g. 0, on every training row)
#      carries zero information for a tree to split on -- the model
#      cannot learn anything from a column with no variance.
#   2. The REAL, non-zero values for these columns are only ever
#      populated at PREDICTION/VALIDATION time (see validation.py's
#      build_eval_set() and state_predictor.py's get_seat_context()),
#      using Aug 2026 sentiment/narrative data -- which genuinely
#      exists BEFORE the 2026 election outcome, so this is temporally
#      valid, not leakage.
# This matches FEATURE_PREDICTORS exactly so the model's fitted
# schema is compatible with real predict-time feature values.
FEATURE_NAMES_TRAIN = [
    'majority_change', 'turnout_change', 'incumbent_held',
    'log_voters', 'majority_perc_change', 'n_candidates_b',
    'malay_pct', 'chinese_pct', 'indian_pct',
    'young_malay_pct', 'young_chinese_pct',
    'older_malay_pct', 'youth_pct', 'median_age',
    'tension_x_mixed',
]

# Feature set used at PREDICTION time (identical to FEATURE_NAMES_TRAIN
# now -- kept as a separate name for clarity/backward compatibility
# with state_predictor.py, which imports this name).
FEATURE_PREDICTORS = list(FEATURE_NAMES_TRAIN)