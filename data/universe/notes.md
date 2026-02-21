# `data/universe` Notes

Files:
- `sp500_membership.csv`

## Explanation
This folder stores point-in-time index membership data used to construct valid historical universes.

Required schema:
- `ticker`: symbol.
- `start_date`: first inclusion date.
- `end_date`: last inclusion date (blank means currently active).

How it functions:
- For a chosen date `t`, loader selects tickers satisfying:
  - `start_date <= t`
  - `end_date is null OR end_date >= t`

Why this math/filtering matters:
- Prevents look-ahead and survivorship bias in historical experiments.

Current state:
- Seed dataset is present for scaffold runs.
- Replace with WRDS/CRSP-grade full history for research-grade inference.

## Your Notes
- 
