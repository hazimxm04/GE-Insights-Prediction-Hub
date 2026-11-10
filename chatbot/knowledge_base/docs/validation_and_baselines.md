# How the model is judged

## Two simple baselines
"Opposition never wins" always predicts that Harapan loses. "Same result as the previous election" predicts each seat is won by the same side as last time. A model is only useful if it beats these. The get_validation tool reports how many seats each gets right.

## Honest reading
In the 2026 tests the model ranks seats well (high AUC) but is not shown to beat the simple rules on accuracy. In Johor it ties "opposition never wins" and is behind "same result as the previous election". In Negeri Sembilan it is ahead of both, but by a small margin that is inside the uncertainty. Confidence intervals from resampling seats are wide with only 36 or 56 seats.

## Why accuracy is lower than ranking
Accuracy depends on the 0.5 cut-off. The training years include the 2018 opposition wave, so the model learned that the opposition wins often and over-predicts Harapan in 2026. Most errors are false alarms: a Harapan call in a seat that went to BN/PN.

## Real flips
A flip is a seat whose winning side changed from the previous election. For a state that has voted, list_flips shows the real flips and whether the model called them. The model called few of them. Predicting that the previous result repeats is hard to beat in a year with few flips.

## Frozen forecasts
Forecasts for states that have not voted were frozen on a date with a checksum, so they become a true out-of-sample test when the results arrive.
