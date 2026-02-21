# `data/` Notes

Subfolders:
- `universe/`
- `cache/`

## Explanation
`data/` is the repository's local data workspace.

Role split:
- `universe/`: curated structural input data (high importance, affects validity).
- `cache/`: generated retrieval artifacts (low importance, disposable).

How this functions operationally:
1. Universe membership defines which assets can appear at each point in time.
2. Loader pulls prices/rates and writes cache artifacts for speed.
3. Backtests consume cleaned time series; cache can always be regenerated.

Critical validity point:
- Universe data quality is first-order for survivorship-bias control.

## Your Notes
- 
