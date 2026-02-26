"""
harbor.v1_paper_trading — v1 Paper Trading engine (refactored from notebook).

Refactored from ``notebooks/v1_HARBOR_Paper_Trading_Algorithm.ipynb`` with the
following fixes applied:

1. Risk metrics guarded (min 20 observations).
2. Ticker universe deduplicated and validated against Alpaca tradability.
3. Live stat-test index alignment normalised to date objects.
4. Backtest uses the same universe as live trading (no fallback).
5. API keys read from environment variables.
6. Transaction costs added to backtest.
7. Value signal uses earnings yield (fundamental) with inverse-vol fallback.
"""

from __future__ import annotations

import os
import time
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

# ---------------------------------------------------------------------------
# Alpaca import (optional — tests mock this)
# ---------------------------------------------------------------------------
try:
    import alpaca_trade_api as tradeapi
except ImportError:  # pragma: no cover
    tradeapi = None  # type: ignore[assignment]


# ============================================================================
# Helpers
# ============================================================================

def validate_universe(tickers: List[str], api: Any = None) -> List[str]:
    """De-duplicate *tickers* and optionally drop non-tradable ones via Alpaca.

    Parameters
    ----------
    tickers : list[str]
        Raw list of ticker symbols (may contain duplicates).
    api : alpaca_trade_api.REST or None
        If provided, each ticker is checked via ``api.get_asset()``.

    Returns
    -------
    list[str]
        Clean, unique, tradable ticker list.
    """
    # Deduplicate preserving order
    seen: dict[str, None] = {}
    for t in tickers:
        seen.setdefault(t.upper().strip(), None)
    unique = list(seen.keys())

    removed_dupes = len(tickers) - len(unique)
    if removed_dupes:
        print(f"  ℹ Removed {removed_dupes} duplicate ticker(s)")

    if api is None:
        return unique

    tradable: List[str] = []
    for t in unique:
        try:
            asset = api.get_asset(t)
            if getattr(asset, "tradable", False):
                tradable.append(t)
            else:
                warnings.warn(f"Ticker {t} is not tradable on Alpaca — dropped")
        except Exception:
            warnings.warn(f"Ticker {t} not found on Alpaca — dropped")

    print(f"  ℹ {len(tradable)}/{len(unique)} tickers are tradable")
    return tradable


def compute_risk_metrics(
    returns: pd.Series | np.ndarray,
    *,
    risk_free_rate: float = 0.045,
    periods_per_year: int = 252,
    min_obs: int = 20,
) -> Optional[Dict[str, float]]:
    """Compute risk-adjusted metrics with a minimum-observation guard.

    Returns ``None`` (with a warning) if fewer than *min_obs* finite
    return observations are available.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) < min_obs:
        warnings.warn(
            f"Only {len(arr)} return observations available (need ≥ {min_obs}). "
            "Risk metrics skipped."
        )
        return None

    avg = arr.mean()
    vol = arr.std(ddof=1)

    annual_return = avg * periods_per_year
    annual_vol = vol * np.sqrt(periods_per_year) if vol > 1e-12 else float("nan")

    sharpe = (
        (annual_return - risk_free_rate) / annual_vol
        if annual_vol > 0 and np.isfinite(annual_vol)
        else float("nan")
    )

    downside = arr[arr < 0]
    if len(downside) > 1:
        down_vol = downside.std(ddof=1) * np.sqrt(periods_per_year)
        sortino = (
            (annual_return - risk_free_rate) / down_vol
            if down_vol > 0
            else float("nan")
        )
    else:
        sortino = float("nan")

    cumulative = np.cumprod(1 + arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(drawdown.min())

    calmar = annual_return / abs(max_dd) if max_dd != 0 else float("nan")
    wr = float((arr > 0).mean())

    return {
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": wr,
        "observations": len(arr),
    }


# ============================================================================
# Alpaca Paper Trading Engine
# ============================================================================

class AlpacaPaperTrading:
    """Automated paper trading via Alpaca API."""

    def __init__(self, api_key: str | None = None, secret_key: str | None = None):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API keys required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables, or pass them directly."
            )

        if tradeapi is None:
            raise ImportError("alpaca-trade-api is required: pip install alpaca-trade-api")

        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url="https://paper-api.alpaca.markets",
            api_version="v2",
        )
        print("✓ Connected to Alpaca Paper Trading")

    # ── Account ────────────────────────────────────────────────────────
    def get_account(self):
        account = self.api.get_account()
        print(f"\n{'='*70}")
        print("ALPACA PAPER TRADING ACCOUNT")
        print(f"{'='*70}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Cash: ${float(account.cash):,.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"Account Status: {account.status}")
        return account

    # ── Positions ──────────────────────────────────────────────────────
    def get_positions(self) -> Dict[str, Dict[str, float]]:
        positions = self.api.list_positions()
        if not positions:
            print("\nNo current positions")
            return {}

        print(f"\n{'='*70}")
        print("CURRENT POSITIONS")
        print(f"{'='*70}")

        positions_dict: Dict[str, Dict[str, float]] = {}
        for p in positions:
            ticker = p.symbol
            shares = float(p.qty)
            price = float(p.current_price)
            mv = float(p.market_value)
            upl = float(p.unrealized_pl)
            uplpc = float(p.unrealized_plpc)
            positions_dict[ticker] = {
                "shares": shares,
                "price": price,
                "value": mv,
                "unrealized_pl": upl,
                "unrealized_plpc": uplpc,
            }
            print(f"{ticker}: {shares} shares @ ${price:.2f} = ${mv:,.2f} ({uplpc:+.2%})")

        return positions_dict

    # ── Orders ─────────────────────────────────────────────────────────
    def place_order(self, ticker: str, shares: float, side: str):
        try:
            order = self.api.submit_order(
                symbol=ticker,
                qty=abs(shares),
                side=side.lower(),
                type="market",
                time_in_force="day",
            )
            print(f"✓ {side} order placed: {abs(shares)} shares of {ticker}")
            return order
        except Exception as e:
            print(f"  Order failed for {ticker}: {e}")
            return None

    def close_all_positions(self):
        print(f"\n{'='*70}")
        print("CLOSING ALL POSITIONS")
        print(f"{'='*70}")
        try:
            self.api.close_all_positions()
            print("✓ All positions closed")
        except Exception as e:
            print(f"  Error closing positions: {e}")

    # ── Rebalance ──────────────────────────────────────────────────────
    def rebalance_to_target(self, target_weights: Dict[str, float]):
        print(f"\n{'='*70}")
        print("REBALANCING TO TARGET ALLOCATION")
        print(f"{'='*70}")

        account = self.api.get_account()
        total_value = float(account.portfolio_value)
        print(f"\nTotal Portfolio Value: ${total_value:,.2f}")

        current_positions = self.get_positions()

        tickers = list(set(list(target_weights.keys()) + list(current_positions.keys())))

        print("\nTarget Allocation:")
        for ticker, weight in sorted(target_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ticker}: {weight:.2%} (${total_value * weight:,.2f})")

        price_data = yf.download(tickers, period="1d", progress=False)
        current_prices: Dict[str, float] = {}
        if len(tickers) == 1:
            current_prices[tickers[0]] = float(price_data["Close"].iloc[-1])
        else:
            for t in tickers:
                try:
                    current_prices[t] = float(price_data["Close"][t].iloc[-1])
                except Exception:
                    current_prices[t] = 0

        trades: list[dict] = []
        for ticker, target_weight in target_weights.items():
            target_value = total_value * target_weight
            current_shares = current_positions.get(ticker, {}).get("shares", 0)
            price = current_prices.get(ticker, 0)
            if price == 0:
                print(f"  Skipping {ticker}: No price data")
                continue
            target_shares = int(target_value / price)
            diff = target_shares - current_shares
            if abs(diff) > 0:
                side = "buy" if diff > 0 else "sell"
                trades.append({"ticker": ticker, "shares": abs(diff), "side": side, "priority": 0 if side == "sell" else 1})

        for ticker in current_positions:
            if ticker not in target_weights:
                trades.append({"ticker": ticker, "shares": current_positions[ticker]["shares"], "side": "sell", "priority": 0})

        print("\nExecuting Trades:")
        for trade in sorted(trades, key=lambda x: x["priority"]):
            self.place_order(trade["ticker"], trade["shares"], trade["side"])

        print("\n✓ Rebalancing complete")
        time.sleep(3)
        self.get_positions()


# ============================================================================
# Strategy Engine
# ============================================================================

class V1StrategyEngine:
    """Generate trading signals (momentum + quality + value composite).

    Value signal uses earnings yield from yfinance fundamentals with a
    fallback to inverse-volatility when fundamental data is unavailable.
    """

    def __init__(self, tickers: List[str], lookback_days: int = 252):
        self.tickers = tickers
        self.lookback_days = lookback_days

    # ── Fundamental value signal ───────────────────────────────────────
    @staticmethod
    def _earnings_yield(ticker: str) -> Optional[float]:
        """Return earnings yield (1 / trailing PE) or None."""
        try:
            info = yf.Ticker(ticker).info
            pe = info.get("trailingPE")
            if pe is not None and pe > 0:
                return 1.0 / pe
        except Exception:
            pass
        return None

    # ── Main signal calculation ────────────────────────────────────────
    def calculate_signals(self) -> Dict[str, float]:
        print(f"\n{'='*70}")
        print(f"V1 STRATEGY ENGINE — Analyzing {len(self.tickers)} stocks")
        print(f"{'='*70}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days + 50)

        price_data = yf.download(self.tickers, start=start_date, end=end_date, progress=False)

        # Handle column formats
        if isinstance(price_data.columns, pd.MultiIndex):
            if "Adj Close" in price_data.columns.get_level_values(0):
                price_data = price_data["Adj Close"]
            else:
                price_data = price_data["Close"]
        elif "Adj Close" in price_data.columns:
            price_data = price_data["Adj Close"]
        else:
            price_data = price_data["Close"]

        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame(name=self.tickers[0])

        price_data = price_data.ffill().bfill()
        returns = price_data.pct_change()

        # Fetch fundamental earnings yields (batch)
        print("  Fetching fundamental data for value signal...")
        ey_map: Dict[str, Optional[float]] = {}
        for t in self.tickers:
            ey_map[t] = self._earnings_yield(t)

        scores: Dict[str, float] = {}
        for ticker in self.tickers:
            try:
                prices = price_data[ticker]
                rets = returns[ticker].dropna()

                # Momentum (40%)
                mom_6m = prices.pct_change(126).iloc[-1] if len(prices) > 126 else 0
                mom_12m = prices.pct_change(252).iloc[-1] if len(prices) > 252 else 0
                momentum = 0.6 * mom_6m + 0.4 * mom_12m

                # Quality (30%) — Sharpe proxy
                mean_ret = rets.mean() * 252
                vol = rets.std() * np.sqrt(252)
                quality = mean_ret / vol if vol > 0 else 0

                # Value (30%) — earnings yield (fundamental), fallback to inverse vol
                ey = ey_map.get(ticker)
                if ey is not None:
                    value = ey
                else:
                    value = 1 / vol if vol > 0 else 0

                composite = 0.4 * momentum + 0.3 * quality + 0.3 * value
                scores[ticker] = composite
            except Exception as e:
                print(f"  Error analyzing {ticker}: {e}")
                scores[ticker] = -999

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        print("\nStock Rankings:")
        for i, (ticker, score) in enumerate(ranked[:8], 1):
            src = "EY" if ey_map.get(ticker) is not None else "IV"
            print(f"  {i}. {ticker}: {score:.4f} [value={src}]")

        # Select top 10
        top_tickers = [t for t, _ in ranked[:10]]

        # Inverse-variance weighting
        selected_returns = returns[top_tickers].iloc[-63:].dropna()
        variances = selected_returns.var().replace(0, np.nan).dropna()
        if variances.empty:
            warnings.warn("All selected tickers have zero variance — using equal weight")
            weights = pd.Series(1.0 / len(top_tickers), index=top_tickers)
        else:
            inv_var = 1 / variances
            raw_weights = inv_var / inv_var.sum()
            weights = raw_weights * 0.5  # half-Kelly
            weights = np.clip(weights, 0, 0.10)
            weights = weights / weights.sum()

        target_allocation = dict(zip(top_tickers, weights))

        print("\nTarget Allocation:")
        for ticker, weight in sorted(target_allocation.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ticker}: {weight:.2%}")

        return target_allocation


# ============================================================================
# Live entrypoints
# ============================================================================

# Default v1 universe
V1_UNIVERSE = [
    "MSFT", "AMZN", "GOOG", "CEG", "NEE", "OKLO", "ASML", "AVGO", "VRT", "ETN",
    "AWK", "VST", "DUK", "XEL", "DTE", "AEP", "BEPC", "GEV", "SBGSY", "SIEGY",
    "PWR", "HUBB", "ABBNY", "MPWR", "ET", "CAT", "BIP", "AES", "EXC", "D", "XYL",
    "BWXT", "SMR", "CCJ", "UEC", "NXE", "ENPH", "SEDG", "BEP", "TSLA", "FLNC",
    "WULF", "IREN", "DLR", "EQIX", "BE", "XLU", "ICLN", "SPY", "COST", "JPM",
]


def run_v1_live_paper_trading(
    api_key: str | None = None,
    secret_key: str | None = None,
    tickers: List[str] | None = None,
) -> AlpacaPaperTrading:
    """Execute live paper trading cycle with v1 strategy."""
    print("\n" + "=" * 70)
    print("v1 HARBOR LIVE PAPER TRADING — ALPACA API")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    tickers = tickers or V1_UNIVERSE

    alpaca = AlpacaPaperTrading(api_key=api_key, secret_key=secret_key)

    # Validate universe
    tickers = validate_universe(tickers, api=alpaca.api)

    alpaca.get_account()

    strategy = V1StrategyEngine(tickers, lookback_days=252)
    target_allocation = strategy.calculate_signals()

    alpaca.rebalance_to_target(target_allocation)
    alpaca.get_account()

    print(f"\n{'='*70}")
    print("✓ Trading complete — Run performance cell to view results")
    print(f"{'='*70}")

    return alpaca


# ============================================================================
# Performance history
# ============================================================================

def get_v1_performance_history(
    api_key: str | None = None,
    secret_key: str | None = None,
) -> Optional[Dict[str, float]]:
    """Fetch and plot portfolio history from Alpaca."""
    try:
        alpaca = AlpacaPaperTrading(api_key=api_key, secret_key=secret_key)
    except Exception as e:
        print(f"Failed to connect to Alpaca: {e}")
        return None

    try:
        ph = alpaca.api.get_portfolio_history(period="1M", timeframe="1D")
        print(f"\n{'='*70}")
        print("PERFORMANCE SUMMARY (Last 30 Days)")
        print(f"{'='*70}")

        equity = list(ph.equity) if ph.equity is not None else []
        if not equity:
            print("No portfolio history data available.")
            return None

        initial_value = next((float(v) for v in equity if v is not None and v != 0), None)
        if initial_value is None or initial_value == 0:
            print("Initial portfolio value is zero or None. Cannot calculate returns.")
            return None

        final_value = float(equity[-1]) if equity[-1] is not None else 0.0
        total_return = (final_value - initial_value) / initial_value

        print(f"Initial Value: ${initial_value:,.2f}")
        print(f"Current Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:+.2%}")

        import matplotlib.pyplot as plt

        timestamps = list(ph.timestamp) if ph.timestamp is not None else list(range(len(equity)))
        pct_returns = [
            (float(e) - initial_value) / initial_value * 100 if e is not None else 0.0
            for e in equity
        ]

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, pct_returns, linewidth=2, color="#2E86C1")
        plt.title("v1 Alpaca Paper Trading Performance", fontsize=14, fontweight="bold")
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Return (%)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("v1_alpaca_performance.png", dpi=300)
        print("\n✓ Saved: v1_alpaca_performance.png")
        plt.show()

        return {"initial": initial_value, "final": final_value, "return": total_return}

    except Exception as e:
        import traceback
        print(f"Error fetching performance: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


# ============================================================================
# Live account progress + risk metrics (guarded)
# ============================================================================

def _portfolio_history_df(
    api_key: str | None,
    secret_key: str | None,
    period: str = "3M",
    timeframe: str = "1D",
) -> Tuple[pd.DataFrame, AlpacaPaperTrading]:
    alpaca = AlpacaPaperTrading(api_key=api_key, secret_key=secret_key)
    ph = alpaca.api.get_portfolio_history(period=period, timeframe=timeframe)

    equity = np.array(ph.equity)
    timestamps = getattr(ph, "timestamp", None)
    if timestamps is None:
        idx = pd.RangeIndex(len(equity))
    else:
        idx = pd.to_datetime(pd.Series(list(timestamps)), unit="s")

    df = pd.DataFrame({"equity": equity}, index=idx).sort_index()
    df["returns"] = df["equity"].pct_change()
    df = df.dropna()
    return df, alpaca


def v1_live_account_progress(
    api_key: str | None = None,
    secret_key: str | None = None,
    period: str = "3M",
    timeframe: str = "1D",
    risk_free_rate: float = 0.045,
    min_obs: int = 20,
) -> Tuple[Optional[Dict[str, float]], pd.DataFrame]:
    """Live account summary with *guarded* risk metrics."""
    df, alpaca = _portfolio_history_df(api_key, secret_key, period=period, timeframe=timeframe)
    if df.empty:
        print("No portfolio history available yet.")
        return None, df

    account = alpaca.api.get_account()
    start_equity = df["equity"].iloc[0]
    end_equity = df["equity"].iloc[-1]
    period_return = (end_equity - start_equity) / start_equity if start_equity != 0 else 0.0

    print(f"\n{'='*70}")
    print("v1 LIVE PAPER ACCOUNT PROGRESS")
    print(f"{'='*70}")
    print(f"Account Status: {account.status}")
    print(f"Equity (Now): ${float(account.equity):,.2f}")
    print(f"Cash: ${float(account.cash):,.2f}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")
    print(f"Period: {period} | Timeframe: {timeframe}")
    print(f"Period Return: {period_return:.2%}")
    print(f"Observations: {len(df)}")

    # Positions snapshot
    try:
        positions = alpaca.api.list_positions()
        if positions:
            pos_rows = sorted(
                [(p.symbol, float(getattr(p, "market_value", 0)), float(p.qty)) for p in positions],
                key=lambda x: x[1],
                reverse=True,
            )
            print("\nTop Positions (by market value):")
            for sym, mv, qty in pos_rows[:10]:
                print(f"  {sym:<6} | Qty: {qty:<8.2f} | Mkt Value: ${mv:,.2f}")
        else:
            print("\nNo open positions.")
    except Exception as e:
        print(f"\nPosition snapshot unavailable: {e}")

    # Risk metrics — guarded
    metrics = compute_risk_metrics(
        df["returns"],
        risk_free_rate=risk_free_rate,
        min_obs=min_obs,
    )

    if metrics is not None:
        print(f"\n{'='*70}")
        print("RISK-ADJUSTED METRICS (LIVE)")
        print(f"{'='*70}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Annual Volatility: {metrics['annual_vol']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe']:.3f}")
        print(f"Sortino Ratio: {metrics['sortino']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Calmar Ratio: {metrics['calmar']:.3f}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
    else:
        print(f"\n⚠ Not enough observations ({len(df)}) for risk metrics (need ≥ {min_obs}).")

    return metrics, df


# ============================================================================
# Live statistical significance tests (with index alignment fix)
# ============================================================================

def run_v1_live_stat_tests(
    api_key: str | None = None,
    secret_key: str | None = None,
    period: str = "3M",
    timeframe: str = "1D",
    risk_free_rate: float = 0.045,
    benchmark: str = "SPY",
    min_obs: int = 20,
) -> Optional[Dict[str, Any]]:
    """Run significance tests on live paper-trading returns vs a benchmark.

    **Fix:** normalises both portfolio and benchmark indices to date objects
    before joining, preventing the zero-overlap issue from v1.
    """
    try:
        df, _ = _portfolio_history_df(api_key, secret_key, period=period, timeframe=timeframe)
    except Exception as e:
        print(f"Error getting live data: {e}")
        return None

    if df.empty or len(df) < 5:
        print("Not enough data to run significance tests.")
        return None

    # ── FIX: normalise index to date ──
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.normalize()
    else:
        try:
            df.index = pd.to_datetime(df.index).normalize()
        except Exception as e:
            print(f"Failed to convert portfolio index: {e}")
            return None

    start_date = df.index.min().date()
    end_date = df.index.max().date()

    bench = yf.download(benchmark, start=start_date, end=end_date + timedelta(days=1), progress=False)
    if bench.empty:
        print("Benchmark data unavailable.")
        return None

    # Extract close price
    if isinstance(bench.columns, pd.MultiIndex):
        if "Adj Close" in bench.columns.get_level_values(0):
            bench = bench.xs("Adj Close", level=0, axis=1)
        elif "Close" in bench.columns.get_level_values(0):
            bench = bench.xs("Close", level=0, axis=1)
    else:
        bench = bench["Adj Close"] if "Adj Close" in bench.columns else bench["Close"]

    if isinstance(bench, pd.DataFrame):
        bench = bench[benchmark] if benchmark in bench.columns else bench.iloc[:, 0]

    bench = bench.rename("bench").pct_change().dropna()

    # ── FIX: normalise benchmark index ──
    if isinstance(bench.index, pd.DatetimeIndex):
        bench.index = bench.index.normalize()

    aligned = pd.DataFrame({"strategy": df["returns"]}).join(bench, how="inner").dropna()

    if len(aligned) < 5:
        print(f"Not enough overlapping data with benchmark ({len(aligned)} rows).")
        return None

    strategy_r = aligned["strategy"].values
    bench_r = aligned["bench"].values
    excess_r = strategy_r - bench_r

    print(f"\n{'='*70}")
    print("v1 LIVE STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*70}")
    print(f"Benchmark: {benchmark}")
    print(f"Samples: {len(aligned)}")
    print(f"Date Range: {aligned.index.min().date()} → {aligned.index.max().date()}")

    t_stat, p_val = stats.ttest_1samp(excess_r, popmean=0.0)
    print("\nOne-sample t-test (Excess returns vs 0):")
    print(f"  t-stat: {t_stat:.3f}")
    print(f"  p-value: {p_val:.4f}")

    t_stat2, p_val2 = stats.ttest_rel(strategy_r, bench_r)
    print("\nPaired t-test (Strategy vs Benchmark):")
    print(f"  t-stat: {t_stat2:.3f}")
    print(f"  p-value: {p_val2:.4f}")

    te = excess_r.std(ddof=1)
    ir = excess_r.mean() / te * np.sqrt(252) if te > 0 else float("nan")
    print("\nInformation Ratio:")
    print(f"  IR: {ir:.3f}")

    rng = np.random.default_rng(42)
    n_boot = 2000
    boot_means = np.array([rng.choice(excess_r, size=len(excess_r), replace=True).mean() for _ in range(n_boot)])
    prob_pos = float((boot_means > 0).mean())
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    print("\nBootstrap Alpha (Excess Return) Summary:")
    print(f"  P(alpha > 0): {prob_pos:.2%}")
    print(f"  95% CI: [{ci_low:.4%}, {ci_high:.4%}]")

    return {
        "p_value_excess": p_val,
        "p_value_pair": p_val2,
        "info_ratio": ir,
        "prob_alpha_pos": prob_pos,
        "boot_ci": (ci_low, ci_high),
        "samples": len(aligned),
    }


# ============================================================================
# 5-Year Backtest (same universe, with transaction costs)
# ============================================================================

def run_v1_5year_backtest(
    tickers: List[str],
    initial_capital: float = 100_000,
    risk_free_rate: float = 0.045,
    train_window: int = 252,
    test_window: int = 63,
    cost_bps: float = 10,
) -> Optional[Dict[str, Any]]:
    """Walk-forward 5-year backtest with transaction costs.

    Parameters
    ----------
    tickers : list[str]
        **Same** universe used for live trading (no fallback).
    cost_bps : float
        Round-trip transaction cost in basis points (default 10).
    """
    if not tickers or len(tickers) < 3:
        print("Provide at least 3 tickers for the backtest universe.")
        return None

    tickers = list(dict.fromkeys(tickers))  # deduplicate

    print(f"\n{'='*70}")
    print("v1 5-YEAR BACKTEST WITH STATISTICAL SIGNIFICANCE")
    print(f"{'='*70}")
    print(f"Universe: {len(tickers)} stocks")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"Transaction Cost: {cost_bps} bps per rebalance turnover")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365 + 100)

    print(f"\nFetching data from {start_date.date()} to {end_date.date()}...")

    stock_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if stock_data.empty:
        print("No stock data returned.")
        return None

    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data = (
            stock_data["Adj Close"]
            if "Adj Close" in stock_data.columns.get_level_values(0)
            else stock_data["Close"]
        )
    if isinstance(stock_data, pd.Series):
        stock_data = stock_data.to_frame()
    stock_data = stock_data.ffill().bfill()

    # SPY benchmark
    spy_raw = yf.download("SPY", start=start_date, end=end_date, progress=False)
    if spy_raw.empty:
        print("SPY data unavailable.")
        return None

    if isinstance(spy_raw.columns, pd.MultiIndex):
        if "Adj Close" in spy_raw.columns.get_level_values(0):
            spy_series = spy_raw.xs("Adj Close", level=0, axis=1)
        elif "Close" in spy_raw.columns.get_level_values(0):
            spy_series = spy_raw.xs("Close", level=0, axis=1)
        else:
            spy_series = spy_raw
    else:
        spy_series = spy_raw.get("Adj Close", spy_raw["Close"])

    if isinstance(spy_series, pd.DataFrame):
        spy_series = spy_series["SPY"] if "SPY" in spy_series.columns else spy_series.iloc[:, 0]

    spy_series = spy_series.reindex(stock_data.index).ffill().bfill()
    print(f"✓ Downloaded {len(stock_data)} days of data")

    portfolio_values = [initial_capital]
    spy_values = [initial_capital]
    dates: list = []
    prev_weights: Optional[pd.Series] = None

    start_idx = train_window

    while start_idx + test_window <= len(stock_data):
        train_data = stock_data.iloc[start_idx - train_window : start_idx]
        train_returns = train_data.pct_change().dropna()
        if train_returns.empty:
            start_idx += test_window
            continue

        scores: Dict[str, float] = {}
        for ticker in train_data.columns:
            try:
                prices = train_data[ticker]
                rets = train_returns[ticker]
                mom = 0.6 * prices.pct_change(63).iloc[-1] + 0.4 * prices.pct_change(126).iloc[-1]
                vol = rets.std() * np.sqrt(252)
                sharpe_proxy = (rets.mean() * 252) / vol if vol > 0 else 0.0
                value = 1 / vol if vol > 0 else 0.0
                scores[ticker] = 0.4 * mom + 0.3 * sharpe_proxy + 0.3 * value
            except Exception:
                scores[ticker] = -999

        top_tickers = [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]]

        selected_returns = train_returns[top_tickers].iloc[-63:]
        if selected_returns.empty:
            start_idx += test_window
            continue

        variances = selected_returns.var().replace(0, np.nan).dropna()
        if variances.empty:
            start_idx += test_window
            continue

        weights = (1 / variances) / (1 / variances).sum()
        weights = np.clip(weights * 0.5, 0, 0.10)
        weights = weights / weights.sum()

        # ── Transaction costs ──────────────────────────────────────────
        if prev_weights is not None:
            # Align to current tickers
            all_tickers = list(set(prev_weights.index.tolist() + weights.index.tolist()))
            pw = prev_weights.reindex(all_tickers, fill_value=0.0)
            cw = weights.reindex(all_tickers, fill_value=0.0)
            turnover = float(abs(cw - pw).sum())
        else:
            turnover = float(weights.sum())  # initial buy
        prev_weights = weights.copy()

        # Out-of-sample returns
        test_data = stock_data[top_tickers].iloc[start_idx : start_idx + test_window]
        test_returns = test_data.pct_change().dropna()
        if test_returns.empty:
            start_idx += test_window
            continue

        period_port = float((1 + (test_returns @ weights)).prod() - 1)
        # Deduct transaction costs
        period_port -= turnover * cost_bps / 10_000

        portfolio_values.append(portfolio_values[-1] * (1 + period_port))

        spy_slice = spy_series.iloc[start_idx : start_idx + test_window].pct_change().dropna()
        period_spy = float((1 + spy_slice).prod() - 1) if not spy_slice.empty else 0.0
        if isinstance(period_spy, (pd.Series, pd.DataFrame)):
            period_spy = float(period_spy.iloc[0]) if hasattr(period_spy, "iloc") else float(period_spy)
        spy_values.append(float(spy_values[-1] * (1 + period_spy)))

        dates.append(stock_data.index[start_idx + test_window - 1])
        start_idx += test_window

    if len(portfolio_values) < 3:
        print("Not enough backtest periods to compute statistics.")
        return None

    portfolio_returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
    spy_returns = np.diff(spy_values) / np.array(spy_values[:-1])

    print(f"\n{'='*70}")
    print("BACKTEST RESULTS")
    print(f"{'='*70}")

    port_total = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    spy_total = (spy_values[-1] - spy_values[0]) / spy_values[0]

    years = len(portfolio_returns) * (test_window / 252)
    if years == 0:
        years = 1e-6
    port_cagr = (portfolio_values[-1] / portfolio_values[0]) ** (1 / years) - 1
    spy_cagr = (spy_values[-1] / spy_values[0]) ** (1 / years) - 1

    port_vol = np.std(portfolio_returns, ddof=1) * np.sqrt(252 / test_window)
    spy_vol = np.std(spy_returns, ddof=1) * np.sqrt(252 / test_window)

    port_sharpe = (port_cagr - risk_free_rate) / port_vol if port_vol > 0 else float("nan")
    spy_sharpe = (spy_cagr - risk_free_rate) / spy_vol if spy_vol > 0 else float("nan")

    cumulative = np.array(portfolio_values) / portfolio_values[0]
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    downside = portfolio_returns[portfolio_returns < 0]
    downside_vol = downside.std(ddof=1) * np.sqrt(252 / test_window) if len(downside) > 0 else 0.0
    sortino = (port_cagr - risk_free_rate) / downside_vol if downside_vol > 0 else float("nan")
    calmar = port_cagr / abs(max_dd) if max_dd != 0 else float("nan")
    wr = (portfolio_returns > 0).mean()

    print("\nStrategy Performance:")
    print(f"  Total Return: {port_total:.2%}")
    print(f"  CAGR: {port_cagr:.2%}")
    print(f"  Volatility: {port_vol:.2%}")
    print(f"  Sharpe Ratio: {port_sharpe:.3f}")
    print(f"  Sortino Ratio: {sortino:.3f}")
    print(f"  Max Drawdown: {max_dd:.2%}")
    print(f"  Calmar Ratio: {calmar:.3f}")
    print(f"  Win Rate: {wr:.2%}")

    print("\nSPY Benchmark:")
    print(f"  Total Return: {spy_total:.2%}")
    print(f"  CAGR: {spy_cagr:.2%}")
    print(f"  Volatility: {spy_vol:.2%}")
    print(f"  Sharpe Ratio: {spy_sharpe:.3f}")

    print("\nOutperformance:")
    print(f"  Alpha: {port_cagr - spy_cagr:+.2%}")
    print(f"  Sharpe Diff: {port_sharpe - spy_sharpe:+.3f}")

    # ── Statistical tests ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*70}")

    if len(portfolio_returns) != len(spy_returns) or len(portfolio_returns) < 2:
        print("Not enough aligned backtest returns for statistical tests.")
        return None

    t_stat, p_value = stats.ttest_rel(portfolio_returns, spy_returns)
    print("\nPaired t-test (Strategy vs SPY):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")

    n = len(portfolio_returns)
    sharpe_diff = port_sharpe - spy_sharpe
    se_diff = np.sqrt((1 + 0.5 * port_sharpe**2) / n + (1 + 0.5 * spy_sharpe**2) / n)
    z_score = sharpe_diff / se_diff if se_diff > 0 else 0.0
    p_sharpe = 2 * (1 - stats.norm.cdf(abs(z_score)))

    print("\nSharpe Ratio Difference Test:")
    print(f"  Sharpe Difference: {sharpe_diff:+.3f}")
    print(f"  z-score: {z_score:.3f}")
    print(f"  p-value: {p_sharpe:.4f}")

    excess_returns = portfolio_returns - spy_returns
    rng = np.random.default_rng(42)
    n_boot = 3000
    boot_means = np.array([rng.choice(excess_returns, size=len(excess_returns), replace=True).mean() for _ in range(n_boot)])
    prob_pos = float((boot_means > 0).mean())
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    print("\nBootstrap Alpha Summary:")
    print(f"  P(alpha > 0): {prob_pos:.2%}")
    print(f"  95% CI: [{ci_low:.4%}, {ci_high:.4%}]")

    return {
        "strategy_cagr": port_cagr,
        "spy_cagr": spy_cagr,
        "alpha": port_cagr - spy_cagr,
        "strategy_sharpe": port_sharpe,
        "spy_sharpe": spy_sharpe,
        "p_value_returns": p_value,
        "p_value_sharpe": p_sharpe,
        "prob_alpha_pos": prob_pos,
        "win_rate": wr,
    }
