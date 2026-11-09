"""
chatbot/tools.py
================
The tools the analyst agent can call. Plain Python: no LLM, no Groq, no ChromaDB, so every tool can be
tested without an API key and a wrong number can be blamed on a tool, not on a model.

Every tool reads the SAME files as the dashboard:
  forecasts/<state>/<date>/forecast.csv      frozen forecasts (any state that has a frozen folder)
  backend/models/<state>/validation_2026.csv model against the real 2026 result (states that have voted)
  data/processed/narrative_windows.csv       the national news windows behind the sentiment layer

Conventions
  * "Harapan" = Pakatan Harapan wins the seat. BN and PN are one group, "BN/PN", the same as everywhere else.
  * A tool never raises. A problem comes back as {"error": "..."} so the model can read it and recover.
  * Results are small and JSON-safe. The model pays for every token it reads.
  * The 'confidence' column is NOT returned: it is always LOW (see the OOD note in the project docs).
    What IS returned is `uses_fallback`: the seat was out of distribution and its probability is the seat's own history.

A state with BOTH files (a state that has voted and later gets a forecast built on its 2026 result) answers
from the forecast by default; pass view='validated' to see the model against the real result instead.
No tool names a state list: when a new frozen forecast folder appears, the tools see it with no code change.

Tools: get_state_forecast, get_seat, list_flips, compare_last_election, get_sentiment, get_validation.
`call_tool(name, arguments)` runs one by name (arguments may be a dict or a JSON string).
"""
import inspect
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.features import OPPOSITION                       # noqa: E402
from backend.core.frozen_forecast import latest_frozen              # noqa: E402
from backend.core.states import STATES                              # noqa: E402
from chatbot.chain.scenario_utils import detect_state               # noqa: E402

MODELS_DIR = ROOT / 'backend' / 'models'
FORECAST_BASE = None            # None = the repo's forecasts/ folder; tests point this at a temp folder
WINDOWS_PATH = None             # None = data/processed/narrative_windows.csv
MAX_ROWS = 30                   # most seats a list tool returns

STATE_NAMES = {'johor': 'Johor', 'neg_sembilan': 'Negeri Sembilan', 'selangor': 'Selangor',
               'melaka': 'Melaka', 'perak': 'Perak'}
TARGET_NOTE = "Harapan = Pakatan Harapan wins the seat. BN and PN are counted together as BN/PN."


# ── helpers ─────────────────────────────────────────────────────────────────────

def _clean(x):
    """Make a value JSON-safe (numpy numbers, NaN, nested containers)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, 'item') and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float):
        return None if x != x else round(x, 4)
    return x


def _side(winner: str) -> str:
    return 'Harapan' if str(winner) in OPPOSITION else 'BN/PN'


def normalise_state(text):
    """State key for 'ns', 'Negeri Sembilan', 'johor', ... or None."""
    if text is None:
        return None
    t = str(text).strip().lower().replace(' ', '_')
    if t in STATES:
        return t
    return detect_state(str(text))


def _bad_state(text) -> dict:
    return {'error': f"Unknown state {text!r}. Use one of: {', '.join(STATES)}."}


def _validation(state: str):
    """The validation frame (Johor, NS) with sides added, or None if the state has no validation file."""
    path = MODELS_DIR / state / 'validation_2026.csv'
    if not path.exists():
        return None
    v = pd.read_csv(path)
    v['seat_no'] = v['seat'].astype(str).str.split().str[0]
    v['seat_name'] = v['seat'].astype(str)
    v['last_side'] = v['winner_coalition_a'].map(_side)           # a = the previous election's winner
    v['now_side'] = v['actual_non_bn'].map({1: 'Harapan', 0: 'BN/PN'})   # b = the 2026 result
    v['model_side'] = v['predicted_non_bn'].map({1: 'Harapan', 0: 'BN/PN'})
    return v


def _forecast(state: str):
    """(frame with sides added, info) for a state's newest frozen forecast, or (None, None)."""
    f, info = latest_frozen(state, FORECAST_BASE)
    if f is None:
        return None, None
    f = f.copy()
    f['last_side'] = f['baseline_same_as_last'].map({1: 'Harapan', 0: 'BN/PN'})
    f['now_side'] = (f['probability'] >= 0.5).map({True: 'Harapan', False: 'BN/PN'})
    return f, info


def _find_seat(frame: pd.DataFrame, query):
    """(row, None) for one seat, or (None, error dict). Accepts 'N.12', '12', 'Bentayan', 'N.12 Bentayan'."""
    q = str(query or '').strip()
    if not q:
        return None, {'error': 'Give a seat, for example "N.12 Bentayan", "N.12" or "Bentayan".'}
    m = re.match(r'^(?:n\.?\s*)?0*(\d+)(?:\s+.*)?$', q, re.IGNORECASE)
    if m:
        hit = frame[frame['seat_no'] == f'N.{int(m.group(1)):02d}']
    else:
        hit = frame[frame['seat_name'].str.contains(re.escape(q), case=False, regex=True)]
    if len(hit) == 1:
        return hit.iloc[0], None
    if hit.empty:
        return None, {'error': f"No seat matches {q!r} in this state. Seats are named like 'N.12 Bentayan'."}
    return None, {'error': f"{q!r} matches several seats; be more specific.",
                  'candidates': hit['seat_name'].head(5).tolist()}


def _pick_view(key: str, view):
    """(frame, 'forecast' | 'validated', error).
    Two separate things: the forecast model (a state that has not voted) and the validated model (a state
    that has voted, scored against the real result). A view that does not exist for the state is
    unavailable. It never silently switches to the other one. Only view=None picks for you:
    forecast if there is one, else validated, else unavailable."""
    if view not in (None, 'forecast', 'validated'):
        return None, None, {'error': "view must be 'forecast' or 'validated'."}
    if view in (None, 'forecast'):
        f, _ = _forecast(key)
        if f is not None:
            return f, 'forecast', None
    if view in (None, 'validated'):
        v = _validation(key)
        if v is not None:
            return v, 'validated', None
    return None, None, _unavailable(key, view)


def _prev_year(key: str, kind: str):
    """Year of the election 'last time' means: the one the model was trained up to (validated), or the last
    election before a frozen forecast. None if unknown."""
    try:
        if kind == 'forecast':
            _, info = _forecast(key)
            return int(info['forecasting_election_after']) if info else None
        meta = json.loads((MODELS_DIR / key / 'metadata.json').read_text(encoding='utf-8'))
        return int(meta['training_cutoff_year'])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def _count(frame, col, value) -> int:
    return int((frame[col] == value).sum())


def _unavailable(state: str, view: str = None) -> dict:
    name = STATE_NAMES[state]
    if view == 'forecast':
        reason = f"{name} has no forecast model yet."
    elif view == 'validated':
        reason = f"{name} has no validated model yet (no real result to score against)."
    else:
        reason = f"{name} has neither a forecast model nor a validated model yet."
    return {'state': name, 'available': False, 'reason': reason}


# ── tools ───────────────────────────────────────────────────────────────────────

def get_state_forecast(state: str) -> dict:
    """Seat totals of the newest frozen forecast, against who held the seats at the last election."""
    key = normalise_state(state)
    if key is None:
        return _bad_state(state)
    f, info = _forecast(key)
    if f is None:
        return _unavailable(key, "forecast")
    return _clean({
        'state': STATE_NAMES[key], 'available': True, 'seats': len(f),
        'frozen_date': info['folder_date'], 'last_election_before_forecast': info['forecasting_election_after'],
        'forecast': {'Harapan': _count(f, 'now_side', 'Harapan'), 'BN/PN': _count(f, 'now_side', 'BN/PN')},
        'last_election': {'Harapan': _count(f, 'last_side', 'Harapan'), 'BN/PN': _count(f, 'last_side', 'BN/PN')},
        'seats_changing_side': int((f['now_side'] != f['last_side']).sum()),
        'seats_on_fallback': int(f['model_used'].astype(str).str.contains('base_rate').sum()),
        'note': TARGET_NOTE + " Seats on fallback use the seat's own history, not the model.",
    })


def get_seat(state: str, seat: str, view: str = None) -> dict:
    """Forecast (or, for a state that has voted, model against the real result) for one seat."""
    key = normalise_state(state)
    if key is None:
        return _bad_state(state)
    frame, kind, err = _pick_view(key, view)
    if err:
        return err
    row, err = _find_seat(frame, seat)
    if err:
        return err
    if kind == 'forecast':
        return _clean({
            'state': STATE_NAMES[key], 'seat': row['seat_name'], 'kind': 'forecast',
            'p_harapan': row['probability'], 'forecast_side': row['now_side'],
            'last_election_side': row['last_side'], 'previous_election_year': _prev_year(key, 'forecast'),
            'changes_side': bool(row['now_side'] != row['last_side']),
            'uses_fallback': 'base_rate' in str(row['model_used']), 'note': TARGET_NOTE,
        })
    return _clean({
        'state': STATE_NAMES[key], 'seat': row['seat_name'], 'kind': 'validated against the real 2026 result',
        'previous_election_year': _prev_year(key, 'validated'),
        'last_election_winner': row['winner_coalition_a'], 'actual_2026_winner': row['winner_coalition_b'],
        'model_side': row['model_side'], 'p_harapan': row['probability'], 'model_correct': bool(row['correct']),
        'changed_side': bool(row['now_side'] != row['last_side']),
        'uses_fallback': bool(row['is_ood']), 'note': TARGET_NOTE,
    })


def list_flips(state: str, view: str = None) -> dict:
    """Seats that change side. Forecast: predicted side vs the last election.
    Validated (a state that has voted): the real 2026 result vs the previous election, and whether the model called it."""
    key = normalise_state(state)
    if key is None:
        return _bad_state(state)
    frame, kind, err = _pick_view(key, view)
    if err:
        return err
    ch = frame[frame['now_side'] != frame['last_side']]
    out = {'state': STATE_NAMES[key], 'count': len(ch), 'to_Harapan': int((ch['now_side'] == 'Harapan').sum()),
           'to_BN_PN': int((ch['now_side'] == 'BN/PN').sum()), 'truncated': len(ch) > MAX_ROWS,
           'previous_election_year': _prev_year(key, kind)}
    if kind == 'forecast':
        out['kind'] = 'forecast'
        out['seats'] = [{'seat': r['seat_name'], 'from': r['last_side'], 'to': r['now_side'], 'p_harapan': r['probability']}
                        for _, r in ch.head(MAX_ROWS).iterrows()]
    else:
        out['kind'] = 'real 2026 result against the previous election'
        out['model_called_correctly'] = int((ch['model_side'] == ch['now_side']).sum())
        out['seats'] = [{'seat': r['seat_name'], 'from': r['last_side'], 'to': r['now_side'],
                         'model_called_it': bool(r['model_side'] == r['now_side'])} for _, r in ch.head(MAX_ROWS).iterrows()]
    return _clean(out)


def compare_last_election(state: str, seat: str = None, view: str = None) -> dict:
    """Last election against now, for the whole state or one seat. 'Now' is the forecast for a state that
    has not voted and the real 2026 result for a validated view."""
    key = normalise_state(state)
    if key is None:
        return _bad_state(state)
    frame, kind, err = _pick_view(key, view)
    if err:
        return err
    now_label = 'forecast' if kind == 'forecast' else 'real 2026 result'
    if seat:
        row, err = _find_seat(frame, seat)
        if err:
            return err
        return _clean({'state': STATE_NAMES[key], 'seat': row['seat_name'], 'last_election': row['last_side'],
                       'now': row['now_side'], 'now_means': now_label,
                       'previous_election_year': _prev_year(key, kind),
                       'changed_side': bool(row['now_side'] != row['last_side'])})
    return _clean({'state': STATE_NAMES[key], 'now_means': now_label, 'seats': len(frame),
                   'previous_election_year': _prev_year(key, kind),
                   'last_election': {'Harapan': _count(frame, 'last_side', 'Harapan'),
                                     'BN/PN': _count(frame, 'last_side', 'BN/PN')},
                   'now': {'Harapan': _count(frame, 'now_side', 'Harapan'), 'BN/PN': _count(frame, 'now_side', 'BN/PN')},
                   'seats_changing_side': int((frame['now_side'] != frame['last_side']).sum()), 'note': TARGET_NOTE})


def _window_days(window: dict):
    """Length of the latest news window in days (30 by design), from its own start and end dates."""
    try:
        return int((pd.Timestamp(window['date']) - pd.Timestamp(window['start'])).days)
    except (KeyError, TypeError, ValueError):
        return 30


def get_sentiment() -> dict:
    """The national news sentiment behind the experimental sentiment layer (latest 30 days vs the 30 before)."""
    from backend.core import sentiment_adjustment as sa
    windows = sa.load_windows(WINDOWS_PATH)
    if windows is None:
        return {'available': False, 'reason': 'No two scored news windows exist yet.'}
    cur, ref = windows
    ok, message = sa.data_status(cur, ref)
    fresh = sa.freshness(cur, ref)
    out = {'available': True, 'enough_headlines': ok, 'status': message,
           'news_up_to': fresh['updated'], 'days_old': fresh['days_old'], 'stale': fresh['stale'],
           'window_days': _window_days(cur),
           'headlines': {'latest': fresh['n_current'], 'previous': fresh['n_reference']},
           'coalition_sentiment_latest': {k: cur.get(f'{k}_sentiment') for k in ('harapan', 'bn', 'pn')},
           'coalition_sentiment_previous': {k: ref.get(f'{k}_sentiment') for k in ('harapan', 'bn', 'pn')},
           'themes_latest': {t: cur.get(t) for t in sa.THEMES},
           'themes_previous': {t: ref.get(t) for t in sa.THEMES},
           'note': ("EXPERIMENTAL. National news scored by a language model, applied to every state through each seat's voter mix. "
                    "Effect sizes on seats are assumptions, not fitted. Scores run -1 to +1; themes 0 to 1.")}
    lean_now, lean_before = sa.coalition_lean(cur), sa.coalition_lean(ref)
    if lean_now is not None and lean_before is not None:
        out['lean_latest'] = lean_now
        out['lean_change'] = lean_now - lean_before
    return _clean(out)


def get_validation(state: str) -> dict:
    """How the model did against the real 2026 result, with the two baselines it has to beat."""
    key = normalise_state(state)
    if key is None:
        return _bad_state(state)
    v = _validation(key)
    if v is None:
        return _unavailable(key, 'validated')
    n = len(v)
    last_int = (v['last_side'] == 'Harapan').astype(int)
    h_calls = v[v['predicted_non_bn'] == 1]
    wrong = v[~v['correct'].astype(bool)]
    return _clean({
        'state': STATE_NAMES[key], 'previous_election_year': _prev_year(key, 'validated'), 'seats': n, 'correct': int(v['correct'].astype(bool).sum()),
        'accuracy': float(v['correct'].astype(bool).mean()),
        'baselines_correct': {'Opposition never wins': int((v['actual_non_bn'] == 0).sum()),
                              'Same result as previous election': int((last_int == v['actual_non_bn']).sum())},
        'harapan_side_calls': {'made': len(h_calls), 'right': int(h_calls['correct'].astype(bool).sum())},
        'seats_on_fallback': int(v['is_ood'].astype(bool).sum()),
        'wrong_seats': [{'seat': r['seat_name'], 'model': r['model_side'], 'actual': r['now_side']}
                        for _, r in wrong.iterrows()],
        'note': TARGET_NOTE + " The model is compared with two simple baselines; see them before calling it good.",
    })


# ── schemas and dispatcher ──────────────────────────────────────────────────────

_STATE_PARAM = {'type': 'string', 'enum': list(STATES), 'description': 'State key'}
_VIEW_PARAM = {'type': 'string', 'enum': ['forecast', 'validated'],
               'description': ("Optional. 'forecast' = the forward forecast; 'validated' = the model against the real "
                               "2026 result (states that have voted). Leave out to get the forecast if one exists, "
                               "otherwise the validated result.")}
_SEAT_PARAM = {'type': 'string', 'description': "A seat: 'N.12 Bentayan', 'N.12', '12' or just the name"}

TOOL_SCHEMAS = [
    {'type': 'function', 'function': {
        'name': 'get_state_forecast',
        'description': ("Seat totals of the newest frozen forecast for a state (only states that have a forecast): "
                        "how many seats Harapan vs BN/PN are forecast to win, how many held at the last election, "
                        "and how many change side. Use for 'how many seats', 'who is ahead'."),
        'parameters': {'type': 'object', 'properties': {'state': _STATE_PARAM}, 'required': ['state']}}},
    {'type': 'function', 'function': {
        'name': 'get_seat',
        'description': ("One seat: the model's probability that Harapan wins, the predicted side, who held it at the last "
                        "election, and whether it changes. For Johor and Negeri Sembilan it shows the model against the "
                        "real 2026 winner instead."),
        'parameters': {'type': 'object', 'properties': {'state': _STATE_PARAM, 'seat': _SEAT_PARAM, 'view': _VIEW_PARAM},
                       'required': ['state', 'seat']}}},
    {'type': 'function', 'function': {
        'name': 'list_flips',
        'description': ("The seats that CHANGE SIDE compared with the previous election, with direction. For forecast states: "
                        "predicted vs the last election. For Johor and Negeri Sembilan: the real 2026 result vs the "
                        "previous election, and whether the model called each one. This is NOT the list of seats the "
                        "model got wrong; for that use get_validation."),
        'parameters': {'type': 'object', 'properties': {'state': _STATE_PARAM, 'view': _VIEW_PARAM},
                       'required': ['state']}}},
    {'type': 'function', 'function': {
        'name': 'compare_last_election',
        'description': ("Last election against now, for a whole state (counts) or one seat (give seat). 'Now' is the "
                        "forecast for states yet to vote and the real 2026 result for Johor and Negeri Sembilan."),
        'parameters': {'type': 'object', 'properties': {'state': _STATE_PARAM, 'seat': _SEAT_PARAM, 'view': _VIEW_PARAM},
                       'required': ['state']}}},
    {'type': 'function', 'function': {
        'name': 'get_sentiment',
        'description': ("The national news sentiment for Harapan, BN and PN and the five themes (latest 30 days vs the "
                        "30 before), when it was last updated, and whether there are enough headlines. Experimental."),
        'parameters': {'type': 'object', 'properties': {}}}},
    {'type': 'function', 'function': {
        'name': 'get_validation',
        'description': ("How the model did against the REAL 2026 result: accuracy, the two baselines (Opposition never "
                        "wins; Same result as previous election), and wrong_seats, the exact seats the model got wrong. "
                        "Use it for 'how accurate', 'did it beat the baseline' and 'which seats did the model get wrong or "
                        "mis-call'. A state that has not voted comes back as unavailable."),
        'parameters': {'type': 'object', 'properties': {'state': _STATE_PARAM},
                       'required': ['state']}}},
]

TOOL_FUNCS = {'get_state_forecast': get_state_forecast, 'get_seat': get_seat, 'list_flips': list_flips,
              'compare_last_election': compare_last_election, 'get_sentiment': get_sentiment,
              'get_validation': get_validation}


def call_tool(name: str, arguments=None) -> dict:
    """Run one tool by name. Arguments may be a dict or a JSON string. Never raises."""
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return {'error': f"Unknown tool {name!r}. Tools: {', '.join(TOOL_FUNCS)}."}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return {'error': 'The arguments were not valid JSON.'}
    arguments = arguments or {}
    allowed = set(inspect.signature(fn).parameters)
    extra = set(arguments) - allowed
    if extra:
        return {'error': f"{name} does not take {sorted(extra)}. It takes {sorted(allowed) or 'no arguments'}."}
    try:
        return fn(**arguments)
    except TypeError as e:
        return {'error': f'{name} was called with the wrong arguments: {e}'}
    except Exception as e:                                          # a tool bug must not crash the agent loop
        return {'error': f'{name} failed: {type(e).__name__}: {e}'}