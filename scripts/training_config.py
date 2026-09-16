TRAINING_CONFIGS = {
    'johor': {
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2022)],
        'rf_params': {'n_estimators': 150, 'max_depth': 5, 'min_samples_leaf': 3},
        'xgb_params': {'n_estimators': 200, 'max_depth': 2, 'learning_rate': 0.05},
        'description': '3 transitions, tuned via GridSearchCV, binary (BN vs Pakatan)',
        'ethnicity_year': 2022,
    },
    'neg_sembilan': {
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2023)],
        'rf_params': {'n_estimators': 200, 'max_depth': 3, 'min_samples_leaf': 8},
        'xgb_params': {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.01},
        'description': '3 transitions, tuned via GridSearchCV, binary (BN vs Pakatan)',
        'ethnicity_year': 2023,
    },
    'selangor': {
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2023)],
        'rf_params': {'n_estimators': 100, 'max_depth': 5, 'min_samples_leaf': 3},
        'xgb_params': {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.1},
        'description': '3 transitions, tuned via GridSearchCV, binary (BN vs Pakatan)',
        'ethnicity_year': 2023,
    },
    'melaka': {
        'transitions': [(2013, 2018), (2018, 2021)],
        'rf_params': {'n_estimators': 100, 'max_depth': 3, 'min_samples_leaf': 8},
        'xgb_params': {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.01},
        'description': '2 transitions, tuned via GridSearchCV, binary (BN vs Pakatan)',
        'ethnicity_year': 2021,
    },
    'perak': {
        'transitions': [(2013, 2018), (2018, 2022)],
        'rf_params': {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 3},
        'xgb_params': {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.03},
        'description': '2 transitions, tuned via GridSearchCV, binary (BN vs Pakatan)',
        'ethnicity_year': 2022,
    },
}

# Feature set used in TRAINING (no sentiment/economic -- applied only at prediction time)
FEATURE_NAMES_TRAIN = [
    # Structural (6)
    'majority_change', 'turnout_change', 'incumbent_held',
    'log_voters', 'majority_perc_change', 'n_candidates_b',
    # Ethnicity + age (8)
    'malay_pct', 'chinese_pct', 'indian_pct',
    'young_malay_pct', 'young_chinese_pct',
    'older_malay_pct', 'youth_pct', 'median_age',
    # Ethnicity interaction (1)
    'tension_x_mixed',
]

# Feature set used at PREDICTION time (includes sentiment/economic, applied post-training)
FEATURE_PREDICTORS = [
    # Structural (6)
    'majority_change', 'turnout_change', 'incumbent_held',
    'log_voters', 'majority_perc_change', 'n_candidates_b',
    # Sentiment (4)
    'bn_sentiment', 'harapan_sentiment', 'pn_sentiment',
    'racial_tension_index',
    # Economic (1)
    'economic_pressure',
    # Ethnicity + age (8)
    'malay_pct', 'chinese_pct', 'indian_pct',
    'young_malay_pct', 'young_chinese_pct',
    'older_malay_pct', 'youth_pct', 'median_age',
    # Sentiment x Ethnicity interactions (5)
    'bn_sent_x_malay', 'harapan_sent_x_chinese',
    'pn_sent_x_young_malay', 'tension_x_mixed',
    'economic_x_youth',
    # National narrative (1)
    'narrative_pressure',
]