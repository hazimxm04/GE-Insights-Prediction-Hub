"""
Tests for chatbot/tools.py. No LLM, no API key. They use the repo's real frozen forecasts and validation
files for the numbers we already know (so a tool and the dashboard cannot disagree), and small temporary
files for the edge cases.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from chatbot import tools as T


# ── real data: the numbers the dashboard shows ────────────────────────────────

def test_johor_validation_matches_the_known_result():
    r = T.get_validation('johor')
    assert (r['seats'], r['correct']) == (56, 48) and r['accuracy'] == pytest.approx(0.8571, abs=1e-4)
    assert set(r['baselines_correct']) == {'Opposition never wins', 'Same result as previous election'}
    assert r['harapan_side_calls'] == {'made': 14, 'right': 7}
    assert len(r['wrong_seats']) == 8


def test_negeri_sembilan_validation_matches_the_known_result():
    r = T.get_validation('Negeri Sembilan')
    assert (r['seats'], r['correct']) == (36, 31)
    assert r['baselines_correct']['Same result as previous election'] == 30      # the model is one seat above it


def test_real_2026_change_in_seats_for_the_two_voted_states():
    j = T.compare_last_election('johor')
    assert j['last_election'] == {'Harapan': 12, 'BN/PN': 44} and j['now'] == {'Harapan': 8, 'BN/PN': 48}
    n = T.compare_last_election('ns')
    assert n['last_election'] == {'Harapan': 17, 'BN/PN': 19} and n['now'] == {'Harapan': 11, 'BN/PN': 25}


def test_forecast_states_report_changes_against_the_last_election():
    m = T.get_state_forecast('melaka')
    assert m['available'] and m['seats'] == 28
    assert m['last_election'] == {'Harapan': 5, 'BN/PN': 23} and m['forecast'] == {'Harapan': 4, 'BN/PN': 24}
    assert T.get_state_forecast('selangor')['seats_changing_side'] == 0
    p = T.list_flips('perak')
    assert p['count'] == 2 and (p['to_Harapan'], p['to_BN_PN']) == (1, 1)
    assert {s['seat'].split(' ', 1)[1] for s in p['seats']} == {'Kuala Kurau', 'Hulu Kinta'}


def test_a_state_that_has_voted_has_no_forecast_and_says_why():
    r = T.get_state_forecast('johor')
    assert r['available'] is False and 'no forecast model yet' in r['reason']
    assert T.get_validation('perak')['available'] is False


def test_flips_for_a_voted_state_say_whether_the_model_called_them():
    r = T.list_flips('johor')
    assert r['count'] == r['to_Harapan'] + r['to_BN_PN'] and r['to_Harapan'] == 1 and r['to_BN_PN'] == 5
    assert all('model_called_it' in s for s in r['seats'])


def test_one_seat_lookups():
    s = T.get_seat('perak', 'Kuala Kurau')
    assert s['kind'] == 'forecast' and s['changes_side'] is True and s['forecast_side'] == 'Harapan'
    j = T.get_seat('johor', 'N.02')
    assert j['seat'] == 'N.02 Jementah' and j['last_election_winner'] == 'PH' and j['actual_2026_winner'] == 'BN'
    assert j['model_correct'] is False and j['changed_side'] is True
    assert T.get_seat('johor', '2')['seat'] == 'N.02 Jementah'


def test_confidence_labels_are_never_returned():
    for r in (T.get_seat('melaka', 'N.01'), T.get_state_forecast('melaka'), T.get_seat('johor', 'N.01')):
        assert 'confidence' not in json.dumps(r).lower()
    assert 'uses_fallback' in T.get_seat('melaka', 'N.01')


# ── errors the model can read and recover from ───────────────────────────────

def test_bad_inputs_come_back_as_errors_not_exceptions():
    assert 'Unknown state' in T.get_state_forecast('narnia')['error']
    assert 'No seat matches' in T.get_seat('melaka', 'Atlantis')['error']
    assert 'Give a seat' in T.get_seat('melaka', '')['error']
    assert 'candidates' in T.get_seat('selangor', 'Sungai')            # several seats share the word, or none do
    assert 'error' in T.call_tool('no_such_tool', {})
    assert 'error' in T.call_tool('get_seat', '{not json')
    assert 'does not take' in T.call_tool('get_seat', {'state': 'melaka', 'seat': 'N.01', 'colour': 'red'})['error']


def test_call_tool_accepts_a_json_string_and_everything_is_json_safe():
    out = T.call_tool('list_flips', '{"state": "melaka"}')
    json.dumps(out)
    assert out['count'] == 1 and out['to_BN_PN'] == 1
    for name in ('get_state_forecast', 'list_flips', 'compare_last_election'):
        json.dumps(T.call_tool(name, {'state': 'selangor'}))


def test_every_schema_has_a_function_and_a_description():
    names = [s['function']['name'] for s in T.TOOL_SCHEMAS]
    assert sorted(names) == sorted(T.TOOL_FUNCS)
    assert all(len(s['function']['description']) > 40 for s in T.TOOL_SCHEMAS)


# ── sentiment tool, with a temporary windows file ────────────────────────────

def _windows(tmp_path, n=400):
    th = {'islam_threat': .2, 'malay_unity': .3, 'cost_living': .4, 'corruption': .1, 'govt_performance': .2}
    rows = [dict(date='2026-10-06', label='Previous 30 days', start='2026-09-07', n_articles=n,
                 harapan_sentiment=-.1, bn_sentiment=-.25, pn_sentiment=-.3, **th),
            dict(date='2026-11-05', label='Latest 30 days', start='2026-10-07', n_articles=n,
                 harapan_sentiment=-.12, bn_sentiment=-.2, pn_sentiment=-.31, **th)]
    path = tmp_path / 'w.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_sentiment_tool_reads_the_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(T, 'WINDOWS_PATH', _windows(tmp_path))
    r = T.get_sentiment()
    assert r['available'] and r['enough_headlines'] and r['news_up_to'] == '2026-11-05'
    assert r['coalition_sentiment_latest']['harapan'] == pytest.approx(-0.12)
    assert 'lean_change' in r and 'assumptions' in r['note']


def test_sentiment_tool_flags_too_few_headlines_and_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(T, 'WINDOWS_PATH', _windows(tmp_path, n=120))
    assert T.get_sentiment()['enough_headlines'] is False
    monkeypatch.setattr(T, 'WINDOWS_PATH', tmp_path / 'missing.csv')
    assert T.get_sentiment()['available'] is False


# ── a state with no frozen forecast and no validation file ───────────────────

def test_a_state_with_neither_file_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(T, 'FORECAST_BASE', tmp_path)
    monkeypatch.setattr(T, 'MODELS_DIR', tmp_path)
    r = T.get_state_forecast('perak')
    assert r['available'] is False and 'no forecast model yet' in r['reason']


# ── a state that has voted AND later gets a forecast built on its 2026 result ──

def _freeze_fake_johor(base):
    folder = Path(base) / 'johor' / '2026-12-01'
    folder.mkdir(parents=True)
    pd.DataFrame({'seat_no': ['N.01', 'N.02', 'N.03'],
                  'seat_name': ['N.01 Buloh Kasap', 'N.02 Jementah', 'N.03 Pemanis'],
                  'probability': [0.2, 0.7, 0.4], 'baseline_same_as_last': [0, 0, 0],
                  'model_used': ['calibrated_rf', 'calibrated_rf', 'ensemble+base_rate(alpha=1.00)']}
                 ).to_csv(folder / 'forecast.csv', index=False)
    (folder / 'manifest.json').write_text(json.dumps(
        {'state': 'johor', 'frozen_at_utc': '2026-12-01T00:00:00Z', 'code_commit': 'abcdef1234',
         'forecasting_election_after': 2026, 'n_seats': 3}), encoding='utf-8')


def test_new_forecast_for_a_voted_state_is_found_without_code_changes(tmp_path, monkeypatch):
    _freeze_fake_johor(tmp_path)
    monkeypatch.setattr(T, 'FORECAST_BASE', tmp_path)
    r = T.get_state_forecast('johor')
    assert r['available'] and r['seats'] == 3 and r['last_election_before_forecast'] == 2026
    assert r['forecast'] == {'Harapan': 1, 'BN/PN': 2}


def test_view_chooses_between_forecast_and_the_real_result(tmp_path, monkeypatch):
    _freeze_fake_johor(tmp_path)
    monkeypatch.setattr(T, 'FORECAST_BASE', tmp_path)
    assert T.get_seat('johor', 'N.02')['kind'] == 'forecast'                              # default: forecast
    v = T.get_seat('johor', 'N.02', view='validated')
    assert v['kind'].startswith('validated') and v['actual_2026_winner'] == 'BN'
    assert T.list_flips('johor', view='validated')['count'] == 6                           # the real 2026 flips
    assert T.list_flips('johor')['count'] == 1                                             # the fake forecast: N.02 flips
    assert T.compare_last_election('johor', view='validated')['now_means'] == 'real 2026 result'
    assert 'error' in T.get_seat('johor', 'N.02', view='sideways')


def test_asking_for_the_wrong_view_explains_itself():
    assert 'no validated model yet' in T.get_seat('perak', 'N.09', view='validated')['reason']      # Perak has not voted
    assert T.get_seat('melaka', 'N.01', view='forecast')['kind'] == 'forecast'


def test_view_is_offered_to_the_model_on_the_seat_tools():
    names = {s['function']['name']: s['function']['parameters']['properties'] for s in T.TOOL_SCHEMAS}
    for tool in ('get_seat', 'list_flips', 'compare_last_election'):
        assert 'view' in names[tool]
    assert 'Melaka' not in json.dumps(T.TOOL_SCHEMAS)                                     # no hard-coded state list


def test_every_view_says_which_election_last_time_was():
    assert T.get_validation('johor')['previous_election_year'] == 2022
    assert T.get_validation('neg_sembilan')['previous_election_year'] == 2023
    assert T.list_flips('johor')['previous_election_year'] == 2022
    assert T.get_seat('johor', 'N.12')['previous_election_year'] == 2022
    assert T.compare_last_election('johor')['previous_election_year'] == 2022
    assert T.compare_last_election('johor', 'N.12')['previous_election_year'] == 2022
    assert T.get_state_forecast('melaka')['last_election_before_forecast'] == 2021
    assert T.list_flips('melaka')['previous_election_year'] == 2021
    assert T.get_seat('perak', 'N.09')['previous_election_year'] == 2022