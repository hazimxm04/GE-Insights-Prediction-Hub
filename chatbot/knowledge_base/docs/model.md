# How the model works

## Inputs
For each seat the model uses only information from the last election before the one being predicted, and the one before that. Structural features: change in winning majority, change in turnout, whether the incumbent side held the seat, number of voters (log), and number of candidates. Demographic features from anonymised voter rolls: ethnic and age mix of the seat (for example the share of Chinese voters, young Malay voters, youth, median age).

## What carries the signal
In the ablation tests, ethnicity and age features carried the signal: removing them lowered the number of correct seats in most tests and never raised it. Majority-change features added little. A plain logistic regression scored about level with the ensemble, so the ensemble is not shown to be better than a simple linear model on this data.

## Models
A random forest and an XGBoost model are averaged. A calibrated random forest gives the probability for seats that look like the training data. The target is binary: Harapan wins or not.

## Time-ordered testing
Every test trains only on earlier elections and is scored on a later one. Training uses very few elections per state (often two transitions), so the data is thin.

## Leakage fixes
Two leaks were found and fixed. Sentiment scores from the prediction period had been used as a training feature, which used information that did not exist at the time. Some structural features had been computed from the election being predicted. Both are fixed and a test rewrites the predicted election and fails if any feature changes.
