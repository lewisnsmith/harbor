"""Tests for harbor.ml.behavior_agents.rewards."""

from __future__ import annotations

import numpy as np

from harbor.ml.behavior_agents.rewards import (
    DispositionEffectShaper,
    LossAversionShaper,
    OverconfidenceShaper,
    ReturnChasingShaper,
    default_behavioral_shaper,
)


def _common_kwargs(
    portfolio_return: float = 0.01,
    weights: np.ndarray = None,
    prev_weights: np.ndarray = None,
) -> dict:
    if weights is None:
        weights = np.array([0.25, 0.25, 0.25, 0.25])
    if prev_weights is None:
        prev_weights = np.array([0.25, 0.25, 0.25, 0.25])
    return dict(
        portfolio_return=portfolio_return,
        weights=weights,
        prev_weights=prev_weights,
        returns_history=np.random.default_rng(42).normal(0, 0.01, (60, 4)),
        portfolio_value=1.05,
        peak_value=1.10,
    )


def test_loss_aversion_penalizes_negative_returns():
    shaper = LossAversionShaper(lambda_la=2.25)

    # Positive return: no penalty
    result_pos = shaper.compute(**_common_kwargs(portfolio_return=0.01))
    assert result_pos == 0.0

    # Negative return: penalty
    result_neg = shaper.compute(**_common_kwargs(portfolio_return=-0.01))
    assert result_neg < 0.0
    assert abs(result_neg - (-2.25 * 0.01)) < 1e-10


def test_loss_aversion_zero_return():
    shaper = LossAversionShaper()
    result = shaper.compute(**_common_kwargs(portfolio_return=0.0))
    assert result == 0.0


def test_overconfidence_penalizes_concentration():
    shaper = OverconfidenceShaper(lambda_oc=0.1)

    # Diversified
    result_div = shaper.compute(
        **_common_kwargs(weights=np.array([0.25, 0.25, 0.25, 0.25]))
    )

    # Concentrated
    result_conc = shaper.compute(
        **_common_kwargs(weights=np.array([0.9, 0.04, 0.03, 0.03]))
    )

    assert result_div < 0.0  # always negative (penalty)
    assert result_conc < result_div  # concentrated is penalized more


def test_return_chasing_detects_momentum_loading():
    shaper = ReturnChasingShaper(lambda_rc=0.05, lookback=21)
    np.random.default_rng(42)  # seed for reproducibility

    # Set up: recent returns are positive for asset 0
    returns_history = np.zeros((60, 4))
    returns_history[-21:, 0] = 0.02  # asset 0 had big positive returns

    # Weight change toward the recent winner
    kwargs = _common_kwargs()
    kwargs["returns_history"] = returns_history
    kwargs["weights"] = np.array([0.5, 0.17, 0.17, 0.16])
    kwargs["prev_weights"] = np.array([0.25, 0.25, 0.25, 0.25])

    result = shaper.compute(**kwargs)
    # Should be negative (penalty for chasing)
    assert result <= 0.0


def test_return_chasing_no_change():
    shaper = ReturnChasingShaper()
    # No weight change → no penalty
    kwargs = _common_kwargs()
    kwargs["weights"] = kwargs["prev_weights"].copy()
    result = shaper.compute(**kwargs)
    assert result == 0.0


def test_disposition_effect_detects_holding_losers():
    shaper = DispositionEffectShaper(lambda_de=0.05)

    returns_history = np.zeros((60, 4))
    returns_history[-1] = np.array([-0.02, 0.02, 0.01, -0.01])

    # Buying the loser (asset 0) and selling the winner (asset 1)
    kwargs = _common_kwargs()
    kwargs["returns_history"] = returns_history
    kwargs["weights"] = np.array([0.4, 0.1, 0.25, 0.25])
    kwargs["prev_weights"] = np.array([0.25, 0.25, 0.25, 0.25])

    result = shaper.compute(**kwargs)
    assert result < 0.0  # penalty


def test_composite_shaper_toggles():
    shaper = default_behavioral_shaper()

    kwargs = _common_kwargs(portfolio_return=-0.01)
    full_result = shaper.compute(**kwargs)

    # Disable loss aversion
    shaper.toggle("loss_aversion", False)
    reduced_result = shaper.compute(**kwargs)

    # With loss aversion disabled, less penalty for negative returns
    assert reduced_result >= full_result


def test_composite_shaper_breakdown():
    shaper = default_behavioral_shaper()
    kwargs = _common_kwargs(portfolio_return=-0.01)

    breakdown = shaper.breakdown(**kwargs)
    assert "loss_aversion" in breakdown
    assert "overconfidence" in breakdown
    assert "return_chasing" in breakdown
    assert "disposition_effect" in breakdown

    # Sum of breakdown should equal composite compute
    total = sum(breakdown.values())
    composite = shaper.compute(**kwargs)
    assert abs(total - composite) < 1e-10


def test_default_behavioral_shaper_all_on():
    shaper = default_behavioral_shaper()
    assert len(shaper.shapers) == 4
    assert all(enabled for _, _, enabled in shaper.shapers)


def test_default_behavioral_shaper_selective():
    shaper = default_behavioral_shaper(
        loss_aversion=True,
        overconfidence=False,
        return_chasing=False,
        disposition_effect=True,
    )
    enabled_names = [n for n, _, e in shaper.shapers if e]
    assert "loss_aversion" in enabled_names
    assert "disposition_effect" in enabled_names
    assert "overconfidence" not in enabled_names
    assert "return_chasing" not in enabled_names
