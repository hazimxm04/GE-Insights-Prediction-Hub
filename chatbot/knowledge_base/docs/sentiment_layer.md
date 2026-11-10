# The experimental sentiment layer

## What it is
A separate layer, not part of the validated model, asks: if the mix of news changes, how might each seat's probability move? It never changes the frozen forecast. It is shown as an optional view and is experimental.

## How headlines are scored
Political headlines from Malaysian news (English and Malay) are collected for two 30-day windows. A language model scores each headline on nine fields: sentiment toward BN, Harapan and PN, racial tension, and five themes (Islam threat, Malay unity, cost of living, corruption, government performance). The scores are by a model, not by people, and have not been checked against hand labels, especially for Malay.

## National, not state-specific
Headlines are national. The same news is applied to every state through each seat's voter mix (race and age).

## Only the change counts
Only the change between the two windows is used, never the level. At least a minimum number of headlines is needed in each window or the layer says there is no data. The get_sentiment tool reports whether data exists and how fresh it is.

## The effects are assumptions
How much a theme moves a seat is an assumption, not fitted to data. Islam threat is down-weighted and government performance is not used. Cost of living and corruption push away from whoever held the seat. The shift is capped. No accuracy is claimed for this layer.
