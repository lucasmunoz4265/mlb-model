# MLB Model — personal sports-betting model

Elo + pitcher rating model for MLB, with a live dashboard for tonight's slate.

## Local use

```
./venv/bin/streamlit run dashboard.py
```

Requires `.env` with `ODDS_API_KEY=...` (free key from the-odds-api.com).

## Files

- `fetch_games.py` — pulls historical games from MLB-StatsAPI
- `elo.py` / `pitcher_elo.py` — team + pitcher rating systems
- `backtest.py` — model vs historical sportsbook lines
- `features.py` / `gbm_model.py` — extended feature set + GBM comparison
- `kelly.py` — bet sizing simulator
- `tonight.py` — CLI: tonight's recommendations + log to tracker
- `tracker.py` — log/update/summarize bet performance
- `dashboard.py` — Streamlit dashboard

Backtest summary: pure team+pitcher Elo at edge ≥ 15% with 0.05x Kelly grew $1000 → $2017 across 4 simulated MLB seasons.
