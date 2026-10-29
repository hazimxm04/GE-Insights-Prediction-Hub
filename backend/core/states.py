"""
The ONE list of states and their data settings.
Add a state here (and nowhere else) to make training, validation, evaluation and
data loading know about it.

transitions : (election A, election B) pairs used for training. A must have an
              earlier election in MIN_YEAR or later (see features.py).
final_test  : the latest real election, used only for scoring (never trained on).
voter_rolls : {year: election code} for the voter-roll files used to compute ethnicity
              features. The URL is built by voter_roll_url(); ethnicity_year (the roll the
              models are trained with) must be one of these years.
"""
STATES = {
    'johor': {
        'url_prefix': 'jhr', 'file_stem': 'johor', 'ethnicity_year': 2022,
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2022)],
        'final_test': (2022, 2026),
        'voter_rolls': {2022: 'se15', 2026: 'se16'},
    },
    'neg_sembilan': {
        'url_prefix': 'nsn', 'file_stem': 'ns', 'ethnicity_year': 2023,
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2023)],
        'final_test': (2023, 2026),
        'voter_rolls': {2023: 'se15', 2026: 'se16'},
    },
    'selangor': {
        'url_prefix': 'sgr', 'file_stem': 'selangor', 'ethnicity_year': 2023,
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2023)],
        'final_test': None,
        'voter_rolls': {2023: 'se15', 2026: 'se16'},
    },
    'melaka': {
        'url_prefix': 'mlk', 'file_stem': 'melaka', 'ethnicity_year': 2021,
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2021)],
        'final_test': None,
        'voter_rolls': {2021: 'se14', 2026: 'se16'},
    },
    'perak': {
        'url_prefix': 'prk', 'file_stem': 'perak', 'ethnicity_year': 2022,
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2022)],
        'final_test': None,
        'voter_rolls': {2022: 'se15', 2026: 'se16'},
    },
}

VOTER_ROLL_BASE = 'https://lake.electiondata.my/voter_rolls'


def voter_roll_url(state, year):
    """URL of a state's voter-roll file for a given election year."""
    info = STATES[state]
    return f"{VOTER_ROLL_BASE}/{info['url_prefix']}_{info['voter_rolls'][year]}_{year}.parquet"