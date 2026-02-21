# Universe Data

`sp500_membership.csv` is the local point-in-time constituent file used by `harbor.data.load_sp500_tickers`.

Expected columns:
- `ticker`
- `start_date` (YYYY-MM-DD)
- `end_date` (YYYY-MM-DD, optional/blank for active membership)

Current file contents are a small seed universe for local development and scaffolding.
For ABF production research, replace this file with WRDS/CRSP S&P 500 historical membership.
