# Leakage fix: what was wrong, how it was found, how it is checked

## What was wrong
To predict election B, the model should only know about election A (the latest election before B) and the one
before it. Some features (majority change, turnout change and similar) were computed using election B itself, and
the seat base rate used B as well. The model was partly shown the answer, so the scores were inflated
(Johor 94.64%, Negeri Sembilan 91.67%).

## How it was found
Reading the feature code with one question: "for each feature, which election's numbers go into it?" Features that
needed B's numbers were leaking. A second pass found the same problem in the validation script, the evaluation
script, the live predictor and the chatbot baseline.

## The fix
- One feature definition, `backend/core/features.py`, used by training, evaluation, the predictor and the chatbot.
- `build_transition` (training) and `build_features` (forecasting, B not needed) share the same code.
- Base rates only use elections up to A. Elections before 2004 are excluded. Seats are matched by number (`N.07`).
- Evaluation is time-ordered and compared with two simple rules ("never opposition", "same as last time").

## How it is checked
`scripts/tests/test_no_outcome_leakage.py` rewrites election B (random results) and fails if any feature changes.
Training saves its held-out score in `metadata.json`; validation refuses models built with the old features.

## Result
Real 2026 elections: Johor 48/56 (85.7%), Negeri Sembilan 31/36 (86.1%). Before this fix the same real-2026 test
gave 78.57% and 83.33%, and the old 94.64% / 91.67% were not honest numbers. The model only just beats the simple
rules, and that is stated in the README.
