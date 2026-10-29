"""
The ONE list of states and their data settings.
Add a state here (and nowhere else) to make training, validation, evaluation and
data loading know about it.

transitions : (election A, election B) pairs used for training. A must have an
              earlier election in MIN_YEAR or later (see features.py).
final_test  : the latest real election, used only for scoring (never trained on).
"""
STATES = {
    'johor': {
        'url_prefix': 'jhr', 'file_stem': 'johor', 'ethnicity_year': 2022,
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2022)],
        'final_test': (2022, 2026),
    },
    'neg_sembilan': {
        'url_prefix': 'nsn', 'file_stem': 'ns', 'ethnicity_year': 2023,
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2023)],
        'final_test': (2023, 2026),
    },
    'selangor': {
        'url_prefix': 'sgr', 'file_stem': 'selangor', 'ethnicity_year': 2023,
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2023)],
        'final_test': None,
    },
    'melaka': {
        'url_prefix': 'mlk', 'file_stem': 'melaka', 'ethnicity_year': 2021,
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2021)],
        'final_test': None,
    },
    'perak': {
        'url_prefix': 'prk', 'file_stem': 'perak', 'ethnicity_year': 2022,
        'transitions': [(2008, 2013), (2013, 2018), (2018, 2022)],
        'final_test': None,
    },
}