"""Tests for hangar.backtest.metrics — Sharpe, Sortino, Calmar, max_drawdown, win_rate."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hangar.backtest.metrics import (
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_returns() -> pd.Series:
    """200-day synthetic return series with known properties."""
    rng = np.random.default_rng(123)
    return pd.Series(rng.normal(0.0005, 0.01, size=200))


@pytest.fixture
def all_positive_returns() -> pd.Series:
    """Returns that are always positive."""
    return pd.Series([0.01, 0.005, 0.008, 0.003, 0.012] * 10)


@pytest.fixture
def flat_returns() -> pd.Series:
    """Zero-volatility return series."""
    return pd.Series([0.001] * 50)


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_known_values(self, daily_returns):
        sr = sharpe_ratio(daily_returns, risk_free=0.0, annualization=252)
        assert isinstance(sr, float)
        assert np.isfinite(sr)

    def test_positive_for_positive_mean(self, daily_returns):
        sr = sharpe_ratio(daily_returns, risk_free=0.0)
        # Mean is ~0.0005 per day, so Sharpe should be positive
        assert sr > 0

    def test_too_few_obs_raises(self):
        with pytest.raises(ValueError, match="at least"):
            sharpe_ratio(pd.Series([0.01]))

    def test_zero_vol_returns_nan(self, flat_returns):
        sr = sharpe_ratio(flat_returns, risk_free=0.0)
        # All excess returns identical → std == 0 → nan
        # Actually std(ddof=1) of identical values is 0
        assert math.isnan(sr)


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_known_values(self, daily_returns):
        sr = sortino_ratio(daily_returns, risk_free=0.0)
        assert isinstance(sr, float)
        assert np.isfinite(sr)

    def test_no_downside_returns_nan(self, all_positive_returns):
        sr = sortino_ratio(all_positive_returns, risk_free=0.0)
        # No negative excess returns → nan
        assert math.isnan(sr)

    def test_too_few_obs_raises(self):
        with pytest.raises(ValueError, match="at least"):
            sortino_ratio(pd.Series([0.01]))


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_known_sequence(self):
        # Price: 1.0 → 1.1 → 0.88 → 0.99
        # Returns: +10%, -20%, +12.5%
        returns = pd.Series([0.10, -0.20, 0.125])
        dd = max_drawdown(returns)
        # Peak was 1.1, trough was 0.88 → drawdown = (0.88-1.1)/1.1 = -0.2
        assert pytest.approx(dd, abs=1e-6) == -0.2

    def test_always_negative_or_zero(self, daily_returns):
        dd = max_drawdown(daily_returns)
        assert dd <= 0

    def test_no_drawdown_for_monotone_up(self, all_positive_returns):
        dd = max_drawdown(all_positive_returns)
        assert dd == 0.0

    def test_too_few_obs_raises(self):
        with pytest.raises(ValueError, match="at least"):
            max_drawdown(pd.Series([0.01]))


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    def test_returns_finite(self, daily_returns):
        cr = calmar_ratio(daily_returns)
        assert isinstance(cr, float)
        assert np.isfinite(cr)

    def test_nan_when_no_drawdown(self, all_positive_returns):
        cr = calmar_ratio(all_positive_returns)
        assert math.isnan(cr)


# ---------------------------------------------------------------------------
# win_rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_all_positive(self, all_positive_returns):
        assert win_rate(all_positive_returns) == 1.0

    def test_known_mix(self):
        returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.0])
        # 2 positive out of 5
        assert pytest.approx(win_rate(returns)) == 0.4

    def test_too_few_obs_raises(self):
        with pytest.raises(ValueError, match="at least"):
            win_rate(pd.Series([0.01]))
