# What MYRamalan is

## The question the model answers
For every state-assembly (DUN) seat the model answers one yes/no question: does Pakatan Harapan (PH) win the seat, or does it go to BN or PN? BN and PN are counted together as one group, written "BN/PN". "Harapan" in the tools and in answers means Pakatan Harapan.

## Seat names
Seats are written like "N.12 Bentayan": N, the seat number inside the state, then the name. Seats are matched on the number, not the name, because names change between elections.

## Forecast and validation are different things
A forecast is made for a state that has not voted yet. The forecasts are frozen (saved with a checksum and the code version) so they cannot be edited after the result is known. A validation is for a state that has voted: the model, trained only on earlier elections, is compared with the real result. A state has one or the other for a given election. If a state has no forecast yet, or no validation yet, it is unavailable. It is not filled in from the other.

## Where the numbers live
Figures such as seat counts, accuracy and flips come from the data tools (get_state_forecast, get_validation, list_flips, get_seat, compare_last_election). These documents explain how and why, and carry no state figures.
