# Headline theme classifier: fine-tuned DistilBERT vs Groq zero-shot

## Question
Which method should tag Malaysian political headlines with five themes (islam_threat, malay_unity,
cost_living, corruption, govt_performance): a fine-tuned DistilBERT, a TF-IDF + logistic regression
baseline, or Groq's LLM used zero-shot?

## Data and method
- 352 headlines labelled by hand (headline only, 0/1 per theme). The sheets were shuffled before
  labelling, and a headline that appears in two sheets is kept once.
- Positives per theme: malay_unity 108, cost_living 29, corruption 28, islam_threat 18,
  govt_performance 14.
- 5-fold cross-validation. Every headline is scored by a model that never trained on it
  (out-of-fold). Groq is not trained, so all headlines count for it.
- Groq here is `openai/gpt-oss-20b` (low reasoning effort), scored once on every headline with the
  production prompt. The earlier baseline model (`llama-3.1-8b-instant`) was retired by Groq, so
  these numbers are not comparable with the earlier Llama scores.
- Reproduce: `python finetune/build_pool.py`, `python finetune/score_pool_groq.py`,
  `python finetune/cross_validate.py --eval-csv eval/pool_v3_scored.csv`.

## Results

Macro-F1 over the five themes (95% CI from resampling headlines):

| Method | threshold 0.5 | threshold 0.3 |
|---|---|---|
| TF-IDF + logistic regression | 0.479 [0.40, 0.56] | 0.374 [0.33, 0.42] |
| DistilBERT | **0.542** [0.47, 0.60] | **0.510** [0.45, 0.56] |
| Groq zero-shot | 0.446 [0.38, 0.51] | 0.347 [0.30, 0.39] |
| always "yes" | 0.189 | 0.189 |

Paired difference, DistilBERT minus Groq: +0.098 [+0.018, +0.179] at 0.5 and +0.162 [+0.101, +0.223]
at 0.3, so the interval excludes zero both times. DistilBERT minus TF-IDF: no clear difference at 0.5
(+0.065 [-0.011, +0.138]), better at 0.3.

Average precision (ranking quality, no threshold), per theme:

| Method | islam_threat | malay_unity | cost_living | corruption | govt_performance | macro |
|---|---|---|---|---|---|---|
| TF-IDF | 0.293 | 0.912 | 0.717 | 0.516 | 0.099 | 0.507 |
| DistilBERT | 0.420 | 0.896 | 0.883 | 0.753 | 0.051 | 0.600 |
| Groq | 0.362 | 0.713 | 0.728 | 0.706 | 0.127 | 0.527 |
| chance (true share) | 0.051 | 0.307 | 0.082 | 0.080 | 0.040 | 0.112 |

## What the numbers say
- DistilBERT beats Groq zero-shot on macro-F1 at both thresholds. The gain comes from malay_unity,
  cost_living and corruption.
- On ranking quality (average precision) DistilBERT leads (0.600 vs 0.527), but the intervals overlap
  ([0.54, 0.66] vs [0.47, 0.60]), so this measure alone does not separate them.
- Groq over-flags: at 0.5 it marks corruption positive on 19% of headlines (true share 8%) and
  govt_performance on 16% (true 4%).
- For malay_unity, TF-IDF is as good as DistilBERT (0.912 vs 0.896), so a simple model is enough there.
- **No method works for govt_performance** (14 positives): every average precision is near chance.
  DistilBERT never fires on it at 0.5. islam_threat (18 positives) is weak for all methods.

## Limits
- 352 headlines, one run, one random seed. The two rare themes have too few positives to judge.
- Headline text only, no article body.
- Groq was used zero-shot with its default thresholds; DistilBERT learned from in-domain labels, which
  is the point of the comparison, but it is not a like-for-like "same information" test.
- The cut-offs were not tuned on these results.

## Decision
On this labelled set DistilBERT is the better theme classifier for malay_unity, cost_living and
corruption, and no method is reliable for islam_threat or govt_performance. Groq stays the running
scorer until DistilBERT is wired into the pipeline. The theme scores are not used by the current
election models (held at 0), so this choice does not change any forecast.
Next if continued: label more headlines for the two rare themes before drawing any conclusion on them.
