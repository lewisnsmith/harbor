"""
harbor.data.load_prices — Fetch and cache adjusted price data at scale.

Planned functionality (Phase H1):
- Download adjusted daily close prices for arbitrarily large ticker universes.
- Chunked, concurrent fetching via ThreadPoolExecutor to stay within rate limits.
- Exponential-backoff retry on transient HTTP/network errors.
- Persist raw downloads to a local Parquet (or pickle fallback) cache so
  repeated runs do not hit the network.
- Point-in-time S&P 500 constituent snapshots to avoid look-ahead bias.
- Risk-free rate proxy (^IRX 13-week T-bill) aligned to any date range.

Design notes for massive-API usage:
- ``PriceLoader`` is the primary entry-point for bulk work; the module-level
  helper functions (``load_sp500_prices`` etc.) delegate to a default instance.
- ``chunk_size`` controls how many tickers are batched per yfinance call
  (default 100 — empirically safe below rate-limit thresholds).
- ``max_workers`` sets the ThreadPoolExecutor concurrency ceiling.
- All chunk downloads pass through ``_fetch_with_retry`` which applies
  exponential backoff (base 2s, max 5 attempts) before propagating errors.
- Cache keys are derived from a SHA-256 hash of (sorted_tickers, start, end,
  adjusted) so partial-universe re-runs only download the delta.

Dependencies: yfinance, pandas, pathlib (stdlib), concurrent.futures (stdlib)
Consumed by: harbor.risk, harbor.backtest, harbor.abf.q1
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import time
import warnings
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError("yfinance is required for harbor.data loaders.") from exc


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

DateLike = Union[str, dt.date, dt.datetime, pd.Timestamp]
ProgressCallback = Callable[[int, int], None]  # (completed_tickers, total_tickers) -> None


# ---------------------------------------------------------------------------
# Module-level cache helpers (shared with PriceLoader)
# ---------------------------------------------------------------------------

_PARQUET_ENGINE_AVAILABLE = bool(
    importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet")
)
_PARQUET_FALLBACK_WARNED = False

# Fallback universe if both local membership data and web scraping are unavailable.
_STATIC_FALLBACK_TICKERS = [
    "AAPL", "ABBV", "AMZN", "AVGO", "BRK-B", "CRM", "GOOG", "GS",
    "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MSFT",
    "NEE", "NVDA", "PG", "UNH", "V", "WMT", "XOM",
]


# ---------------------------------------------------------------------------
# PriceLoader — stateful, configurable bulk loader
# ---------------------------------------------------------------------------


class PriceLoader:
    """Session-level loader for large ticker universes.

    Centralises rate-limit settings, cache configuration, and retry policy
    so callers do not have to thread kwargs through every call.

    Parameters
    ----------
    cache_dir:
        Root directory for Parquet cache files.  Created on first write if
        absent.  Defaults to ``<repo_root>/data/cache/prices``.
    chunk_size:
        Number of tickers dispatched per yfinance batch request.  Smaller
        values reduce per-request latency variance; larger values reduce
        overhead.  Default 100.
    max_workers:
        Maximum concurrent download threads.  Default 8.
    max_retries:
        Maximum retry attempts on transient errors before raising.  Default 5.
    backoff_base:
        Base wait time (seconds) for exponential backoff between retries.
        Actual wait on attempt *n* is ``backoff_base * 2 ** n``.  Default 2.0.
    timeout:
        Per-request HTTP timeout in seconds forwarded to yfinance.  Default 30.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        *,
        chunk_size: int = 100,
        max_workers: int = 8,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        timeout: float = 30.0,
    ) -> None:
        self.cache_dir: Path = cache_dir or (_repo_root() / "data" / "cache" / "prices")
        self.chunk_size: int = chunk_size
        self.max_workers: int = max_workers
        self.max_retries: int = max_retries
        self.backoff_base: float = backoff_base
        self.timeout: float = timeout

    # ------------------------------------------------------------------
    # Core fetch
    # ------------------------------------------------------------------

    def fetch(
        self,
        tickers: Sequence[str],
        start: DateLike,
        end: DateLike,
        *,
        adjusted: bool = True,
        progress: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Fetch prices for *tickers* over [start, end], chunked and cached.

        Each chunk is fetched concurrently up to ``max_workers`` threads.
        Results are read from cache when available; only missing chunks hit
        the network.

        Parameters
        ----------
        tickers:
            Ticker symbols to fetch.  Duplicates are deduplicated; symbols
            absent from yfinance are dropped with a warning.
        start:
            Inclusive start date.
        end:
            Inclusive end date.
        adjusted:
            Use split- and dividend-adjusted closes when ``True`` (default).
        progress:
            Optional callback invoked after each chunk completes, receiving
            ``(completed_tickers, total_tickers)``.

        Returns
        -------
        pd.DataFrame
            Wide DataFrame — ``pd.DatetimeIndex`` rows × ticker columns.
            Missing values are ``NaN``.

        Raises
        ------
        ValueError
            If ``start >= end`` or *tickers* is empty.
        RuntimeError
            If all retries are exhausted for any chunk.
        """
        normalized = _normalize_tickers(tickers)
        if not normalized:
            raise ValueError("tickers must contain at least one non-empty symbol.")

        start_ts, end_ts = _validate_date_range(start, end)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        chunks = list(self._chunk_tickers(normalized))
        total = len(normalized)
        completed = 0
        frames: list[pd.DataFrame] = []

        def _fetch_one(chunk: list[str]) -> pd.DataFrame:
            cache_path_parquet, cache_path_pickle = self._cache_paths(
                chunk, start_ts, end_ts, adjusted
            )
            cached = _read_cached_frame(cache_path_parquet, cache_path_pickle)
            if cached is not None:
                return _finalize_price_frame(cached)
            frame = self._fetch_with_retry(chunk, start_ts, end_ts, adjusted)
            _write_cached_frame(frame, cache_path_parquet, cache_path_pickle)
            return frame

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {executor.submit(_fetch_one, chunk): chunk for chunk in chunks}
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                frame = future.result()  # raises on unrecoverable error
                frames.append(frame)
                completed += len(chunk)
                if progress is not None:
                    progress(min(completed, total), total)

        if not frames:
            raise ValueError("All requested symbols were unavailable for the selected date range.")

        combined = pd.concat(frames, axis=1).sort_index()
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate_cache(
        self,
        before: Optional[DateLike] = None,
    ) -> int:
        """Remove stale or targeted Parquet/pickle cache files.

        Parameters
        ----------
        before:
            If provided, only remove files whose modification time predates
            this timestamp (useful for rolling-window refreshes).

        Returns
        -------
        int
            Number of cache files removed.
        """
        removed = 0
        cutoff: Optional[float] = None
        if before is not None:
            cutoff = _normalize_date(before).timestamp()

        for path in self.cache_dir.glob("*"):
            if not path.is_file():
                continue
            if cutoff is not None and path.stat().st_mtime >= cutoff:
                continue
            path.unlink()
            removed += 1

        return removed

    def cache_stats(self) -> dict[str, object]:
        """Return a summary of current cache state.

        Returns
        -------
        dict
            Keys: ``"files"``, ``"total_bytes"``, ``"cache_dir"``.
        """
        files = list(self.cache_dir.glob("*")) if self.cache_dir.exists() else []
        total_bytes = sum(p.stat().st_size for p in files if p.is_file())
        return {
            "files": len(files),
            "total_bytes": total_bytes,
            "cache_dir": str(self.cache_dir),
        }

    # ------------------------------------------------------------------
    # Internal helpers (public for subclassing / testing)
    # ------------------------------------------------------------------

    def _chunk_tickers(self, tickers: Sequence[str]) -> Iterator[list[str]]:
        """Yield successive ``chunk_size``-length sublists from *tickers*.

        Parameters
        ----------
        tickers:
            Full ordered sequence of symbols.

        Yields
        ------
        list[str]
            A contiguous slice of *tickers* of length <= ``self.chunk_size``.
        """
        for i in range(0, len(tickers), self.chunk_size):
            yield list(tickers[i : i + self.chunk_size])

    def _fetch_chunk(
        self,
        tickers: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        adjusted: bool,
    ) -> pd.DataFrame:
        """Download one chunk synchronously via yfinance.

        Parameters
        ----------
        tickers:
            Symbols for this chunk (length <= ``chunk_size``).
        start:
            Inclusive start date (already normalised ``pd.Timestamp``).
        end:
            Inclusive end date (already normalised ``pd.Timestamp``).
        adjusted:
            Adjusted close flag forwarded to yfinance.

        Returns
        -------
        pd.DataFrame
            Partial wide price DataFrame for this chunk.
        """
        return _download_price_panel(tickers, start, end, adjusted=adjusted)

    def _fetch_with_retry(
        self,
        tickers: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        adjusted: bool,
    ) -> pd.DataFrame:
        """Wrap ``_fetch_chunk`` with exponential-backoff retry.

        Retries on ``ConnectionError``, ``TimeoutError``, and HTTP 429/503
        (surfaced as ``Exception`` by yfinance).  Raises ``RuntimeError``
        after ``max_retries`` exhausted.

        Parameters
        ----------
        tickers:
            Symbols for this chunk.
        start:
            Inclusive start date.
        end:
            Inclusive end date.
        adjusted:
            Adjusted close flag.

        Returns
        -------
        pd.DataFrame
            Price data for the chunk.

        Raises
        ------
        RuntimeError
            If all retry attempts fail.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._fetch_chunk(tickers, start, end, adjusted)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = self.backoff_base * (2 ** attempt)
                warnings.warn(
                    f"Chunk fetch failed (attempt {attempt + 1}/{self.max_retries}): {exc}. "
                    f"Retrying in {wait:.1f}s.",
                    stacklevel=2,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"All {self.max_retries} retry attempts exhausted for chunk {tickers[:3]}... "
            f"Last error: {last_exc}"
        ) from last_exc

    def _cache_paths(
        self,
        tickers: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        adjusted: bool,
    ) -> tuple[Path, Path]:
        """Compute deterministic Parquet and pickle cache paths for the given parameters.

        Parameters
        ----------
        tickers:
            Ticker symbols (order-independent; hashed as a sorted list).
        start:
            Start date.
        end:
            End date.
        adjusted:
            Adjusted close flag.

        Returns
        -------
        tuple[Path, Path]
            ``(parquet_path, pickle_path)`` under ``self.cache_dir``.
        """
        key = _cache_key({
            "tickers": sorted(tickers),
            "start": str(start.date()),
            "end": str(end.date()),
            "adjusted": adjusted,
        })
        return (
            self.cache_dir / f"prices_{key}.parquet",
            self.cache_dir / f"prices_{key}.pkl",
        )


# ---------------------------------------------------------------------------
# Module-level default loader + configure()
# ---------------------------------------------------------------------------

_default_loader: Optional[PriceLoader] = None


def get_default_loader() -> PriceLoader:
    """Return (or lazily create) the module-level default ``PriceLoader``.

    Returns
    -------
    PriceLoader
        Singleton instance initialised with default settings.
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = PriceLoader()
    return _default_loader


def configure(
    cache_dir: Optional[Path] = None,
    *,
    chunk_size: int = 100,
    max_workers: int = 8,
    max_retries: int = 5,
    backoff_base: float = 2.0,
    timeout: float = 30.0,
) -> None:
    """Replace the module-level default ``PriceLoader`` with custom settings.

    Call once at application start-up to tune rate-limit and cache behaviour
    for a specific environment (e.g. higher ``max_workers`` on a beefy server,
    lower ``chunk_size`` when sharing an IP with other processes).

    Parameters
    ----------
    cache_dir:
        Root directory for Parquet cache files.
    chunk_size:
        Tickers per yfinance batch request.
    max_workers:
        Concurrent download threads.
    max_retries:
        Max retry attempts on transient errors.
    backoff_base:
        Base wait (seconds) for exponential backoff.
    timeout:
        Per-request HTTP timeout (seconds).
    """
    global _default_loader
    _default_loader = PriceLoader(
        cache_dir,
        chunk_size=chunk_size,
        max_workers=max_workers,
        max_retries=max_retries,
        backoff_base=backoff_base,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Public module-level helpers
# ---------------------------------------------------------------------------


def load_sp500_tickers(as_of: Optional[DateLike] = None) -> list[str]:
    """Return S&P 500 constituent tickers valid on a given date.

    Preferred data source is ``data/universe/sp500_membership.csv`` with columns:
    ``ticker,start_date,end_date`` (``end_date`` can be blank).

    If the membership file is missing, the loader falls back to scraping current
    constituents from Wikipedia and warns that this introduces survivorship bias.
    If web loading fails, a small static list is returned as a last resort.

    Parameters
    ----------
    as_of:
        Point-in-time date for the universe snapshot.  ``None`` returns the
        current list without look-ahead protection.

    Returns
    -------
    list[str]
        Sorted list of uppercase ticker symbols.

    Raises
    ------
    ValueError
        If ``as_of`` precedes the earliest available universe date.
    """
    as_of_ts = _normalize_date(as_of) if as_of is not None else pd.Timestamp.today().normalize()

    membership_path = _repo_root() / "data" / "universe" / "sp500_membership.csv"
    if membership_path.exists():
        membership = _read_membership_table(membership_path)
        unique_tickers = membership["ticker"].nunique()
        if unique_tickers < 400:
            warnings.warn(
                f"Membership file {membership_path} only contains {unique_tickers} unique symbols. "
                "Use WRDS/CRSP historical constituents for survivorship-bias-free studies.",
                stacklevel=2,
            )
        earliest = membership["start_date"].min()
        if pd.notna(earliest) and as_of_ts < earliest:
            raise ValueError(
                f"as_of {as_of_ts.date()} predates earliest membership date "
                f"{earliest.date()} in {membership_path}."
            )
        mask = (membership["start_date"] <= as_of_ts) & (
            membership["end_date"].isna() | (membership["end_date"] >= as_of_ts)
        )
        tickers = sorted(membership.loc[mask, "ticker"].unique().tolist())
        if not tickers:
            raise ValueError(f"No constituents found for as_of={as_of_ts.date()}.")
        return tickers

    warnings.warn(
        "Using current-constituent fallback for S&P 500 universe because "
        "data/universe/sp500_membership.csv is missing. This is survivorship-biased.",
        stacklevel=2,
    )
    scraped = _fetch_current_sp500_tickers()
    if scraped:
        return scraped

    warnings.warn(
        "Wikipedia constituent scrape failed. Falling back to a static seed universe.",
        stacklevel=2,
    )
    return sorted(_STATIC_FALLBACK_TICKERS)


def load_sp500_prices(
    tickers: Sequence[str],
    start: DateLike,
    end: DateLike,
    *,
    adjusted: bool = True,
    cache_dir: Optional[Path] = None,
    chunk_size: int = 100,
    max_workers: int = 8,
    max_retries: int = 5,
    backoff_base: float = 2.0,
    progress: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """Download (or load from cache) daily prices for an arbitrarily large universe.

    Delegates to ``PriceLoader.fetch`` with a temporary loader scoped to the
    provided kwargs.  For repeated calls (e.g. in a loop), prefer constructing
    a ``PriceLoader`` once and calling ``.fetch()`` directly to amortise setup.

    Parameters
    ----------
    tickers:
        Ticker symbols to fetch.  Symbols absent from yfinance are dropped
        with a warning.
    start:
        Inclusive start date, e.g. ``"2010-01-01"``.
    end:
        Inclusive end date.
    adjusted:
        Use split- and dividend-adjusted closes (default ``True``).
    cache_dir:
        Parquet cache directory.  ``None`` uses the repo default.
    chunk_size:
        Tickers per yfinance batch request.  Tune down if hitting 429s.
    max_workers:
        Concurrent download threads.
    max_retries:
        Max retry attempts on transient network errors.
    backoff_base:
        Base wait (seconds) for exponential backoff between retries.
    progress:
        Optional ``(completed, total)`` callback for progress reporting.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame — ``pd.DatetimeIndex`` rows × ticker columns.
        Missing values are ``NaN``.

    Raises
    ------
    ValueError
        If ``start >= end`` or *tickers* is empty.
    RuntimeError
        If retries are exhausted for any chunk.
    """
    loader = PriceLoader(
        cache_dir,
        chunk_size=chunk_size,
        max_workers=max_workers,
        max_retries=max_retries,
        backoff_base=backoff_base,
    )
    return loader.fetch(tickers, start, end, adjusted=adjusted, progress=progress)


def load_risk_free_rate(
    start: DateLike,
    end: DateLike,
    *,
    proxy_ticker: str = "^IRX",
    annualized: bool = True,
) -> pd.Series:
    """Fetch a daily risk-free rate proxy aligned to a business-day index.

    For ``^IRX`` this converts quoted percentage yield to a decimal and
    forward-fills over non-trading days.

    Parameters
    ----------
    start:
        Inclusive start date.
    end:
        Inclusive end date.
    proxy_ticker:
        yfinance symbol for the rate proxy.  Defaults to ``"^IRX"``
        (13-week US T-bill annualised yield, in percent).
    annualized:
        ``True`` (default) → return annualised decimal (e.g. 0.05 for 5 %).
        ``False`` → convert to daily rate (``annualized / 252``).

    Returns
    -------
    pd.Series
        Daily rate values with a ``pd.DatetimeIndex``, forward-filled over
        market holidays.

    Raises
    ------
    ValueError
        If no rate data is returned for the proxy ticker.
    """
    start_ts, end_ts = _validate_date_range(start, end)
    cache_root = _repo_root() / "data" / "cache" / "rates"
    cache_root.mkdir(parents=True, exist_ok=True)

    cache_key = _cache_key({
        "ticker": proxy_ticker,
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "annualized": annualized,
    })
    cache_parquet_path = cache_root / f"risk_free_{cache_key}.parquet"
    cache_pickle_path = cache_root / f"risk_free_{cache_key}.pkl"

    cached = _read_cached_frame(cache_parquet_path, cache_pickle_path)
    if cached is not None:
        series = cached.iloc[:, 0]
        series.name = "risk_free_rate"
        return series

    raw = yf.download(
        tickers=proxy_ticker,
        start=str(start_ts.date()),
        end=str((end_ts + pd.Timedelta(days=1)).date()),
        progress=False,
        auto_adjust=False,
        actions=False,
        threads=True,
    )
    if raw.empty:
        raise ValueError(f"No rate data returned for proxy_ticker={proxy_ticker!r}.")

    if "Close" in raw.columns:
        series = raw["Close"]
    elif "Adj Close" in raw.columns:
        series = raw["Adj Close"]
    else:
        raise ValueError(
            f"Could not find Close/Adj Close columns for proxy_ticker={proxy_ticker!r}."
        )

    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series = pd.Series(series, copy=False).astype(float)
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series.sort_index()

    if proxy_ticker.upper().startswith("^"):
        series = series / 100.0

    if not annualized:
        series = series / 252.0

    business_index = pd.date_range(start_ts, end_ts, freq="B")
    series = series.reindex(business_index).ffill().bfill()
    series.name = "risk_free_rate"

    _write_cached_frame(series.to_frame(), cache_parquet_path, cache_pickle_path)
    return series


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_cached_frame(parquet_path: Path, pickle_path: Path) -> Optional[pd.DataFrame]:
    if _PARQUET_ENGINE_AVAILABLE and parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Failed reading cache {parquet_path}: {exc}; falling back to download.",
                stacklevel=2,
            )

    if pickle_path.exists():
        try:
            return pd.read_pickle(pickle_path)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Failed reading fallback cache {pickle_path}: {exc}", stacklevel=2)

    return None


def _write_cached_frame(frame: pd.DataFrame, parquet_path: Path, pickle_path: Path) -> None:
    global _PARQUET_FALLBACK_WARNED

    if not _PARQUET_ENGINE_AVAILABLE:
        if not _PARQUET_FALLBACK_WARNED:
            warnings.warn(
                "No parquet engine detected (pyarrow/fastparquet). Using pickle cache fallback.",
                stacklevel=2,
            )
            _PARQUET_FALLBACK_WARNED = True
        frame.to_pickle(pickle_path)
        return

    try:
        frame.to_parquet(parquet_path)
        return
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"Failed writing Parquet cache {parquet_path}: {exc}. "
            f"Writing pickle fallback to {pickle_path}.",
            stacklevel=2,
        )

    try:
        frame.to_pickle(pickle_path)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Failed writing pickle fallback cache {pickle_path}: {exc}", stacklevel=2)


def _download_price_panel(
    tickers: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    adjusted: bool,
) -> pd.DataFrame:
    raw = yf.download(
        tickers=" ".join(tickers),
        start=str(start.date()),
        end=str((end + pd.Timedelta(days=1)).date()),
        progress=False,
        auto_adjust=False,
        actions=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise ValueError("No price data returned by yfinance for requested symbols/date range.")

    field_priority = ["Adj Close", "Close"] if adjusted else ["Close", "Adj Close"]

    if isinstance(raw.columns, pd.MultiIndex):
        selected_field = next(
            (field for field in field_priority if field in raw.columns.levels[0]), None
        )
        if selected_field is None:
            available = sorted(set(raw.columns.get_level_values(0)))
            raise ValueError(f"Price columns missing. Available fields: {available}")
        prices = raw[selected_field]
    else:
        selected_field = next(
            (field for field in field_priority if field in raw.columns), None
        )
        if selected_field is None:
            raise ValueError(f"Price columns missing. Found columns: {list(raw.columns)}")
        prices = raw[selected_field]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    return _finalize_price_frame(prices)


def _finalize_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.sort_index()
    frame.columns = [str(col).upper().replace(".", "-") for col in frame.columns]

    missing = frame.columns[frame.isna().all()].tolist()
    if missing:
        warnings.warn(
            f"Dropping symbols with no available price history in the selected range: {missing}",
            stacklevel=2,
        )
        frame = frame.drop(columns=missing)

    if frame.empty:
        raise ValueError("All requested symbols were unavailable for the selected date range.")

    return frame


def _read_membership_table(path: Path) -> pd.DataFrame:
    membership = pd.read_csv(path)
    required = {"ticker", "start_date"}
    missing = required - set(membership.columns)
    if missing:
        raise ValueError(f"Membership file {path} missing required columns: {sorted(missing)}")

    frame = membership.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce").dt.normalize()

    if "end_date" not in frame.columns:
        frame["end_date"] = pd.NaT
    else:
        frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce").dt.normalize()

    frame = frame.dropna(subset=["ticker", "start_date"]).sort_values(["ticker", "start_date"])
    return frame


def _fetch_current_sp500_tickers() -> list[str]:
    try:
        table = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", flavor="lxml"
        )[0]
    except Exception:  # pragma: no cover
        return []

    if "Symbol" not in table.columns:
        return []

    symbols = (
        table["Symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    )
    return sorted(symbols.dropna().unique().tolist())


def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
    cleaned = [
        str(ticker).strip().upper().replace(".", "-")
        for ticker in tickers
        if str(ticker).strip()
    ]
    return list(dict.fromkeys(cleaned))  # deduplicate, preserve order


def _cache_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _normalize_date(value: DateLike) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Could not parse date value: {value!r}")
    return timestamp.normalize()


def _validate_date_range(start: DateLike, end: DateLike) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = _normalize_date(start)
    end_ts = _normalize_date(end)
    if start_ts >= end_ts:
        raise ValueError(
            f"start must be before end. Received start={start_ts.date()} end={end_ts.date()}"
        )
    return start_ts, end_ts


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
