# Known limitations

- Very little training data per state, so many seats are outside what the model has seen and fall back to history.
- On the 2026 results the model is not shown to beat simple rules; the Johor model is behind "same result as the previous election".
- No confidence level is reported because the old label was unreliable.
- BN and PN are counted together. In the 2026 election their relationship differed by state (separate in Johor, a seat pact in parts of Negeri Sembilan), and the model does not capture this yet.
- The sentiment layer is experimental: model-scored headlines, assumed effects, national news only.
- The chatbot answers from tools. If a tool says something is unavailable, the answer is that it is unavailable.
- Seats and states outside the five covered (Johor, Negeri Sembilan, Selangor, Melaka, Perak) are not available.
