# Modeling Investigation Log: Coalition Target, Hyperparameters, and Feature Scope

This document records three investigations conducted during model
development, why each decision was made, and what was ruled out.
Referenced from code comments in `train_models.py` and
`state_predictor.py`.

## 1. Binary target vs. 3-class (BN / Pakatan / PN)

**What was tried:** Splitting the target from binary (BN vs.
non-BN) into three classes: BN, Pakatan (PR+PH combined, since PR
dissolved into PH in 2015 and never competed against it in the same
election), and PN.

**Result:** Multi-class accuracy on real 2026 validation data was
substantially lower than binary (Johor: 78.57% vs. binary's ~94%
from an earlier iteration; Neg Sembilan: 69.44% vs. higher binary
baseline).

**Root cause, confirmed empirically:** PN only emerged as a
coalition in 2020. This creates a structural (not random) data
limitation:
- No training transition has PN as a "previous" winner (the earliest
  possible PN win falls in the *most recent* transition per state),
  so any coalition-history feature (`prev_was_pn`) is a constant
  column of zeros across 100% of training data.
- PN's seat count in its one possible training transition varies
  wildly by state due to real political geography (Melaka: 2 seats,
  Perak: 26 seats) — not a sampling artifact.
- GridSearchCV hyperparameter tuning (StratifiedKFold, f1_macro
  scoring) improved cross-validation scores on the training data but
  produced **zero change** in real 2026 validation accuracy,
  confirming the ceiling was a genuine information gap, not a
  tuning problem.

**Decision:** Reverted to binary (BN vs. non-BN, where non-BN =
Pakatan or PN combined). This is not "avoiding a harder question" —
for 2 of the 3 historical transitions used in training, PN did not
exist as a possible answer at all, so binary and 3-class framing
were mathematically equivalent for ~65% of training data. The binary
model answers "Pakatan vs. establishment," a well-posed question
given available data; BN-vs-PN specifically remains unresolved.

## 2. Restoring sentiment/narrative interaction features

**Context:** An earlier iteration of this pipeline used
`merge_ethnicity_into_features()`, which computes demographic-weighted
interaction features (e.g., `bn_sent_x_malay`, `narrative_pressure` —
national sentiment themes weighted by seat-level ethnicity/age
composition) and reportedly achieved higher validation accuracy.

**What was investigated:** Whether these features could be safely
reintroduced into the current pipeline without reintroducing the
temporal leakage that was fixed earlier (Aug 2026 sentiment baked
into 2008-2022 training rows).

**Finding:** Populating these columns with real sentiment/narrative
values ONLY at prediction/validation time (keeping them as a
constant 0 during training, since real historical sentiment per
election year is not available) is leakage-safe, but **functionally
inert**: a decision tree cannot learn to use a feature with zero
variance across all training rows. Feeding real values at prediction
time to a model that never saw that feature vary during training has
no effect on the resulting prediction — it does not recover the
higher accuracy.

**Open question, not resolved:** Whether the original 94%+ result
had genuine, varying sentiment/narrative signal across historical
training rows (which would require real per-election-year sentiment
data, not a single current snapshot applied uniformly) — and if so,
whether that data existed validly for each historical year or was
itself a form of leakage. This was not verified, and rather than
ship an unaudited number, the simpler, fully-verified binary model
was retained.

**Decision:** Do not restore these features until historical
(2008-2022), per-election-year sentiment/narrative data can be
sourced and validated as non-leaky. Documented as scoped future
work.

## 3. Current production model (as of this writeup)

- Target: binary (BN vs. non-BN)
- Features: 15 structural + ethnicity columns (no sentiment/economic
  at training time; see `training_config.py` FEATURE_NAMES_TRAIN)
- Hyperparameters: `max_depth=4, min_samples_leaf=5` (RF),
  `max_depth=3, learning_rate=0.03` (XGB) — moderate regularization,
  chosen for small per-state training sets (53-160 rows)
- Validation (against actual 2026 results, the only two states with
  held elections so far):
  - Johor: 78.57% (44/56 seats)
  - Neg Sembilan: 83.33% (30/36 seats)
- Selangor, Melaka, Perak: models trained, but 2026 elections pending
  — no ground truth available yet to validate against