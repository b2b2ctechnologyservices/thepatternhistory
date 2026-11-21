# ThePatternHistory

Type a ticker and see which chart patterns are firing today, and how the same
patterns played out every time they fired over the last ~25 years: sample size,
win rate, average/median/worst returns at 1, 5, 20 and 60 days, and every past
occurrence including the losers.

- `rules.py`: the pattern detectors.
- `pattern_history/`: price data (yfinance, adjusted) and the historical replay.
- `web/streamlit_app.py`: the one-page site.

Run locally: `pip install -r web/requirements.txt && streamlit run web/streamlit_app.py`

Educational only. Not investment advice. Past patterns do not guarantee future results.
