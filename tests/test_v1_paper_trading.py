"""Tests for harbor.v1_paper_trading — universe validation, risk metrics guard, index alignment."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from harbor.v1_paper_trading import (
    compute_risk_metrics,
    validate_universe,
)

# ===================================================================
# validate_universe
# ===================================================================

class TestValidateUniverse:
    def test_removes_duplicates(self):
        tickers = ["AAPL", "MSFT", "AAPL", "GOOG", "MSFT"]
        result = validate_universe(tickers)
        assert result == ["AAPL", "MSFT", "GOOG"]

    def test_preserves_order(self):
        tickers = ["GOOG", "AAPL", "MSFT"]
        result = validate_universe(tickers)
        assert result == ["GOOG", "AAPL", "MSFT"]

    def test_strips_and_uppercases(self):
        tickers = [" aapl ", "Msft"]
        result = validate_universe(tickers)
        assert result == ["AAPL", "MSFT"]

    def test_drops_non_tradable_with_api(self):
        mock_api = MagicMock()

        def mock_get_asset(symbol):
            if symbol == "ABBNY":
                asset = SimpleNamespace(tradable=False)
            else:
                asset = SimpleNamespace(tradable=True)
            return asset

        mock_api.get_asset = mock_get_asset
        tickers = ["AAPL", "ABBNY", "GOOG"]
        result = validate_universe(tickers, api=mock_api)
        assert "ABBNY" not in result
        assert "AAPL" in result
        assert "GOOG" in result

    def test_drops_unknown_tickers_with_api(self):
        mock_api = MagicMock()
        mock_api.get_asset.side_effect = Exception("not found")
        result = validate_universe(["AAPL"], api=mock_api)
        assert result == []

    def test_no_api_returns_all(self):
        tickers = ["AAPL", "MSFT"]
        result = validate_universe(tickers, api=None)
        assert result == ["AAPL", "MSFT"]


# ===================================================================
# compute_risk_metrics
# ===================================================================

class TestComputeRiskMetrics:
    def test_returns_none_below_min_obs(self):
        returns = pd.Series([0.01, -0.005, 0.003])
        result = compute_risk_metrics(returns, min_obs=20)
        assert result is None

    def test_returns_dict_with_enough_obs(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0005, 0.01, size=50))
        result = compute_risk_metrics(returns, min_obs=20)
        assert result is not None
        assert isinstance(result, dict)
        expected_keys = {
            "annual_return", "annual_vol", "sharpe", "sortino",
            "max_drawdown", "calmar", "win_rate", "observations",
        }
        assert set(result.keys()) == expected_keys
        assert result["observations"] == 50

    def test_sharpe_is_finite(self):
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.001, 0.02, size=100))
        result = compute_risk_metrics(returns, min_obs=20)
        assert result is not None
        assert np.isfinite(result["sharpe"])

    def test_zero_vol_gives_nan_sharpe(self):
        returns = pd.Series([0.001] * 30)
        result = compute_risk_metrics(returns, min_obs=20)
        assert result is not None
        assert math.isnan(result["sharpe"])

    def test_max_drawdown_negative(self):
        rng = np.random.default_rng(99)
        returns = pd.Series(rng.normal(-0.001, 0.02, size=100))
        result = compute_risk_metrics(returns, min_obs=20)
        assert result is not None
        assert result["max_drawdown"] < 0

    def test_win_rate_bounded(self):
        rng = np.random.default_rng(55)
        returns = pd.Series(rng.normal(0, 0.01, size=100))
        result = compute_risk_metrics(returns, min_obs=20)
        assert result is not None
        assert 0 <= result["win_rate"] <= 1


# ===================================================================
# Index alignment (unit-level)
# ===================================================================

class TestIndexAlignment:
    """Verify that normalising datetime indices to date-only enables joins."""

    def test_normalize_enables_join(self):
        # Simulate Alpaca unix-timestamp-based index (timezone-naive datetime)
        ts = pd.to_datetime(["2026-02-10 20:00:00", "2026-02-11 20:00:00", "2026-02-12 20:00:00"])
        portfolio = pd.DataFrame({"returns": [0.01, -0.005, 0.002]}, index=ts)

        # Simulate yfinance date-based index (midnight-normalised)
        bench_ts = pd.to_datetime(["2026-02-10", "2026-02-11", "2026-02-12"])
        bench = pd.Series([0.005, -0.003, 0.001], index=bench_ts, name="bench")

        # Without normalisation → zero overlap
        joined_raw = portfolio.join(bench, how="inner")
        assert len(joined_raw) == 0  # mismatch

        # With normalisation → full overlap
        portfolio.index = portfolio.index.normalize()
        bench.index = bench.index.normalize()
        joined_fixed = portfolio.join(bench, how="inner")
        assert len(joined_fixed) == 3


# ===================================================================
# Transaction costs (backtest-level, simplified)
# ===================================================================

class TestTransactionCosts:
    """Verify that turnover-based cost deduction reduces returns."""

    def test_cost_reduces_return(self):
        gross_return = 0.05
        turnover = 1.0  # 100% turnover
        cost_bps = 10

        net_return = gross_return - turnover * cost_bps / 10_000
        assert net_return < gross_return
        assert pytest.approx(net_return) == 0.05 - 0.001


# ===================================================================
# Value signal fallback
# ===================================================================

class TestValueSignalFallback:
    """V1StrategyEngine._earnings_yield should return None on failure."""

    def test_returns_none_on_bad_ticker(self):
        from harbor.v1_paper_trading import V1StrategyEngine

        # A ticker that almost certainly won't have valid PE
        result = V1StrategyEngine._earnings_yield("ZZZZZNOTREAL")
        assert result is None
