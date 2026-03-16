# Massive API Integration — Design Spec

**Date:** 2026-03-12
**Status:** Approved
**Goal:** Replace yfinance seed data with Massive API as the primary data source for survivorship-bias-free S&P 500 historical prices, achieving CRSP-equivalent data quality without institutional credentials.

---

## Context

Harbor currently uses yfinance with a 23-stock seed universe. A `load_crsp_prices()` stub exists but raises `NotImplementedError` (requires WRDS institutional access). The Massive REST API provides comprehensive historical OHLC bars, dividends, splits, and ticker metadata — sufficient to replace CRSP for development and research purposes.

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Separate `massive.py` module (Approach B) | `load_prices.py` is already ~930 lines; clean separation enables future Massive endpoints |
| S&P 500 membership | Hybrid: Wikipedia historical scraper + Massive for market data | Massive has no index membership endpoint; Wikipedia tracks all additions/removals |
| Fallback behavior | Massive-first, yfinance fallback per-chunk | Keeps pipeline resilient; no silent full-degradation |
| Consumer API | Transparent swap via `MASSIVE_API_KEY` env var | Zero changes to experiments, notebooks, or tests |

---

## 1. `harbor/data/massive.py` — Massive API Client

### Class: `MassiveClient`

Standalone HTTP client wrapping the Massive REST API. Uses `requests` with exponential-backoff retry.

**Constructor:**
```python
MassiveClient(
    api_key: str | None = None,  # Falls back to MASSIVE_API_KEY env var
    base_url: str = "https://api.massive.com",
    max_retries: int = 5,
    backoff_base: float = 2.0,
    timeout: float = 30.0,
    max_workers: int = 8,
)
```

**Public methods:**

#### `fetch_bars(ticker, start, end, adjusted=True) -> pd.DataFrame`
- Endpoint: `GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}`
- Query params: `adjusted`, `sort=asc`, `limit=50000`
- Follows `next_url` for pagination
- Returns: DataFrame indexed by date with columns `[open, high, low, close, volume, vwap]`

#### `fetch_bars_bulk(tickers, start, end, adjusted=True) -> pd.DataFrame`
- Concurrent fetching across tickers via `ThreadPoolExecutor(max_workers)`
- Extracts `close` column from each ticker's bars
- Returns: Wide DataFrame — `DatetimeIndex` rows x ticker columns (close prices only)
- Matches `_download_price_panel` output format for drop-in compatibility

#### `fetch_dividends(ticker, start, end) -> pd.DataFrame`
- Endpoint: `GET /stocks/v1/dividends`
- Query params: `ticker`, `ex_dividend_date.gte`, `ex_dividend_date.lte`, `limit=5000`
- Follows `next_url` for pagination
- Returns: DataFrame with `[ex_dividend_date, cash_amount, frequency, distribution_type]`

#### `fetch_splits(ticker, start, end) -> pd.DataFrame`
- Endpoint: `GET /stocks/v1/splits`
- Query params: `ticker`, `execution_date.gte`, `execution_date.lte`, `limit=5000`
- Follows `next_url` for pagination
- Returns: DataFrame with `[execution_date, split_from, split_to, adjustment_type]`

#### `list_tickers(market="stocks", active=True, limit=1000) -> list[dict]`
- Endpoint: `GET /v3/reference/tickers`
- Follows `next_url` for pagination
- Returns: List of ticker metadata dicts

**Internal methods:**

#### `_request(method, path, params=None) -> dict`
- Adds `Authorization: Bearer {api_key}` header
- Exponential backoff on 429/503 status codes
- Raises `MassiveAPIError` on non-retryable failures (401, 403, 404)

#### `_paginate(initial_response) -> list[dict]`
- Follows `next_url` links, accumulates all `results` arrays

**Exceptions:**
- `MassiveAPIError(Exception)` — base error with status code and message
- `MassiveAuthError(MassiveAPIError)` — missing or invalid API key

**Estimated size:** ~200-250 lines

---

## 2. `harbor/data/universe.py` — S&P 500 Historical Membership Builder

### Function: `build_sp500_membership() -> pd.DataFrame`

Scrapes Wikipedia's S&P 500 page to reconstruct full historical constituent membership.

**Data sources:**
- Table 0: Current constituents (columns: `Symbol`, `Date added`, ...)
- Table 1: Historical changes (columns: `Date`, `Added Ticker`, `Added Name`, `Removed Ticker`, `Removed Name`)

**Algorithm:**
1. Fetch current constituents from Table 0 → these are the "still active" members
2. Fetch historical changes from Table 1 (additions/removals back to ~2000)
3. Walk changes in reverse chronological order:
   - Each "Added" entry → `start_date` for that ticker
   - Each "Removed" entry → `end_date` for that ticker
4. For current constituents without a recorded addition date, default `start_date` to the earliest date in the changes table
5. Output: DataFrame with `[ticker, start_date, end_date]` where `end_date` is `NaT` for current members

### Function: `save_sp500_membership(path=None) -> Path`

Calls `build_sp500_membership()` and writes the result to `data/universe/sp500_membership.csv`. Idempotent — overwrites the existing seed file. Returns the path written.

### Function: `_scrape_sp500_tables() -> tuple[pd.DataFrame, pd.DataFrame]`

Internal helper. Fetches both Wikipedia tables with error handling. Raises `RuntimeError` if page structure has changed (missing expected columns).

**Expected output:** ~700-800 rows covering ~500 current + ~200-300 historical constituents with point-in-time membership dates.

**Estimated size:** ~150-200 lines

---

## 3. Integration into `PriceLoader`

### Provider Routing in `_fetch_chunk`

```python
# In PriceLoader.__init__:
self._massive_client: MassiveClient | None = None
api_key = os.environ.get("MASSIVE_API_KEY")
if api_key:
    from harbor.data.massive import MassiveClient
    self._massive_client = MassiveClient(api_key=api_key)

# In PriceLoader._fetch_chunk:
def _fetch_chunk(self, tickers, start, end, adjusted):
    if self._massive_client is not None:
        return self._massive_client.fetch_bars_bulk(
            tickers, str(start.date()), str(end.date()), adjusted=adjusted
        )
    return _download_price_panel(tickers, start, end, adjusted=adjusted)
```

### Per-Chunk Fallback

In `_fetch_with_retry`, after all Massive retries exhaust:

```python
except RuntimeError:
    if self._massive_client is not None:
        warnings.warn(
            f"Massive API unavailable for chunk {tickers[:3]}...; "
            f"falling back to yfinance"
        )
        return _download_price_panel(tickers, start, end, adjusted=adjusted)
    raise
```

### Cache Key Differentiation

Add provider to cache key payload:

```python
def _cache_paths(self, tickers, start, end, adjusted):
    provider = "massive" if self._massive_client is not None else "yfinance"
    key = _cache_key({
        "tickers": sorted(tickers),
        "start": str(start.date()),
        "end": str(end.date()),
        "adjusted": adjusted,
        "provider": provider,
    })
    ...
```

### Changes to `__init__.py`

Export `MassiveClient` and `build_sp500_membership`:

```python
from harbor.data.massive import MassiveClient
from harbor.data.universe import build_sp500_membership, save_sp500_membership
```

---

## 4. Testing Strategy

### `tests/test_massive.py` (~200 lines)

| Test | What it verifies |
|------|-----------------|
| `test_fetch_bars_single_ticker` | Parses Massive OHLC response, correct DataFrame shape and dtypes |
| `test_fetch_bars_pagination` | Follows `next_url`, concatenates results |
| `test_fetch_bars_bulk` | Multiple tickers assembled into wide close-price DataFrame |
| `test_fetch_bars_adjusted` | `adjusted` param passed through correctly |
| `test_fetch_dividends` | Parses dividend response fields |
| `test_fetch_splits` | Parses split response fields |
| `test_list_tickers` | Pagination and filtering |
| `test_retry_on_429` | Backoff triggered, succeeds on retry |
| `test_retry_exhausted` | `MassiveAPIError` raised after max retries |
| `test_missing_api_key` | `MassiveAuthError` raised with clear message |
| `test_auth_header` | `Authorization: Bearer ...` header present on requests |

All tests mock `requests.get` — no live API calls.

### `tests/test_universe.py` (~100 lines)

| Test | What it verifies |
|------|-----------------|
| `test_build_membership_from_tables` | Correct `ticker,start_date,end_date` rows from fixture data |
| `test_point_in_time_filtering` | Added-2015/removed-2020 stock appears for `as_of=2018` but not `as_of=2021` |
| `test_current_members_have_nat_end_date` | Active constituents have `end_date=NaT` |
| `test_save_writes_csv` | `save_sp500_membership()` creates valid CSV file |
| `test_malformed_table_raises` | `RuntimeError` if Wikipedia table structure changes |

All tests mock `pd.read_html`.

### Existing tests unchanged

- `tests/test_risk.py`, `tests/test_risk_h2.py`, `tests/test_abf_q1.py` — same `load_sp500_prices` interface, no env var set in CI = yfinance path.

### Provider fallback tests (in `test_massive.py`)

| Test | What it verifies |
|------|-----------------|
| `test_fallback_on_massive_failure` | yfinance used when Massive fails, warning emitted |
| `test_partial_fallback` | Some chunks via Massive, failed chunk via yfinance |

**Total new tests:** ~20-25

---

## 5. File Changes Summary

### New Files

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `harbor/data/massive.py` | Massive API client | ~200-250 |
| `harbor/data/universe.py` | S&P 500 membership builder | ~150-200 |
| `tests/test_massive.py` | Client unit tests | ~200 |
| `tests/test_universe.py` | Universe builder tests | ~100 |

### Modified Files

| File | Change | Est. Lines Changed |
|------|--------|-------------------|
| `harbor/data/load_prices.py` | Provider routing in `_fetch_chunk`, fallback in `_fetch_with_retry`, provider in cache key | ~25 |
| `harbor/data/__init__.py` | Export new symbols | ~6 |
| `requirements.txt` / `setup.py` | Add `requests` if missing | ~1 |

### Unchanged

- All experiment scripts, notebooks, configs
- Risk, backtest, ML, ABF modules
- All existing tests
- `load_crsp_prices()` stub

---

## 6. Dependencies

| Package | Purpose | Notes |
|---------|---------|-------|
| `requests` | HTTP client for Massive API | Likely already installed (transitive dep of yfinance) |

No new heavy dependencies.

---

## 7. Configuration

| Env Var | Required | Purpose |
|---------|----------|---------|
| `MASSIVE_API_KEY` | No (enables Massive when set) | API authentication |

No config files, no CLI flags. Presence of the env var is the sole switch.