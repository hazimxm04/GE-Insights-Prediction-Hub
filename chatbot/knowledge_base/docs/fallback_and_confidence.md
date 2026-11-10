# Fallback seats and confidence

## What a fallback seat is
For each seat the model checks whether it looks like the seats it was trained on (an out-of-distribution, or OOD, check). If a seat is outside what the model has seen, the probability is blended with the seat's own historical base rate: how often the opposition won there in earlier elections. A seat handled this way is a fallback seat (uses_fallback is true in the tools). For these seats the model had no reliable basis, so the answer is mostly the seat's history, not a model judgement.

## Many seats fall back
In several states most seats are flagged, because training data is thin. This is reported in the tools as seats_on_fallback.

## No confidence level
The system does not give a confidence level. An earlier confidence label (HIGH, MEDIUM, LOW) was found to read LOW for every seat in the frozen forecasts, because the OOD score is on a different scale from the thresholds. That is a known weakness. It has been recorded, not hidden, and the chatbot tools do not return it. Do not describe any seat as high or low confidence.

## Probability versus the call
A probability of 0.5 or more is a Harapan call. The cut-off has not been tuned on the 2026 results, because that would be testing on the answers.
